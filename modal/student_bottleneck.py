"""Frozen-feature variational information bottleneck and matched CTC controls.

Only authorized, unedited help-train training-speaker transcripts are labels.
The KL to one shared standard Gaussian upper-bounds frame information. Model
selection sees whole held-out training speakers, never practice comparisons.
Run with the supervisor on Modal, not on the developer machine.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def grouped_split(rows, fold, folds=3):
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < folds:
        raise ValueError('Insufficient independent training speakers')
    held = set(speakers[fold::folds])
    return ([r for r in rows if r['speaker'] not in held],
            [r for r in rows if r['speaker'] in held])


def cache_rows(path, meta):
    import numpy as np
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty training cache')
    for r in rows:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or
                r.get('excluded') is not False or not r.get('speaker') or
                r.get('kind') != 'none' or r.get('events') or r.get('non_errors')):
            raise ValueError('Ineligible canonical training observation')
        x = np.asarray(r.pop('features'), dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != meta['width'] or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid frozen encoder features')
        if not r.get('pronunciation') or any(p not in meta['tokens'] for p in r['pronunciation']):
            raise ValueError('Missing or invalid canonical phones')
        for p in r.get('controlPronunciations', []):
            if not p or any(c not in meta['tokens'] for c in p) or p == r['pronunciation']:
                raise ValueError('Invalid training-only lexical preservation control')
        r['x'] = x
    return rows


def gaussian_kl(mean, logvar):
    """KL(q(z|frame) || N(0,I)), summed over latent coordinates."""
    return .5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1)


def quadrature(mean, std):
    import torch
    dim = mean.shape[-1]
    points = torch.eye(dim, dtype=mean.dtype, device=mean.device) * math.sqrt(dim)
    points = torch.stack((-points, points), dim=1).reshape(2 * dim, dim)
    return mean.unsqueeze(-2) + points * std.unsqueeze(-2)


def fit(rows, meta, penalty, args, budget, held=None, epochs=None):
    import numpy as np
    import torch
    from torch import nn
    budget()
    width, latent, vocab = meta['width'], args.latent, len(meta['tokens']) + 1
    all_x = np.concatenate([r['x'] for r in rows])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-3)
    fixed_std = math.exp(-1)

    class Readout(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.encoder = nn.Linear(width, 2 * latent)
            self.output = nn.Linear(latent, vocab)
            with torch.no_grad():
                self.encoder.bias[latent:].fill_(-2)
                self.output.bias[0] = 2

        def distribution(self, x):
            mu, second = self.encoder(x).split(latent, dim=-1)
            if self.kind == 'bottleneck':
                logvar = second.clamp(-8, 4)
                return mu, (.5 * logvar).exp(), gaussian_kl(mu, logvar)
            mu = mu + second.tanh()
            return mu, torch.full_like(mu, fixed_std), torch.zeros_like(mu[..., 0])

        def forward(self, x, integrate=False):
            mu, std, kl = self.distribution(x)
            if self.kind == 'deterministic':
                return self.output(mu).log_softmax(-1), kl
            if integrate:
                # One frame's independent latent is integrated before complete CTC.
                values = self.output(quadrature(mu, std)).log_softmax(-1)
                return values.logsumexp(-2) - math.log(2 * latent), kl
            return self.output(mu + std * torch.randn_like(mu)).log_softmax(-1), kl

    arms = []
    for kind in ('bottleneck', 'deterministic', 'fixedNoise'):
        torch.manual_seed(args.seed)
        arms.append(Readout(kind))
    counts = [sum(p.numel() for p in a.parameters()) for a in arms]
    assert len(set(counts)) == 1
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=1e-3) for a in arms]
    ctc = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    rng = np.random.default_rng(args.seed)

    def batch_loss(arm, batch, integrate=False):
        xs = [torch.from_numpy((r['x'] - center) / scale) for r in batch]
        ys = [torch.tensor([meta['tokens'].index(p)+1 for p in r['pronunciation']], dtype=torch.long) for r in batch]
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
        probs, kl = arm(nn.utils.rnn.pad_sequence(xs), integrate)
        mask = torch.arange(probs.shape[0])[:, None] < lengths[None, :]
        loss = ctc(probs.contiguous(), torch.cat(ys), lengths, target_lengths) / target_lengths.sum()
        information = kl[mask].mean()
        return loss, information

    def evaluate():
        reports = []
        with torch.no_grad():
            for arm in arms:
                total, phones, wins, controls, information, frame_count = 0., 0, 0, 0, 0., 0
                for r in held:
                    budget()
                    loss, kl = batch_loss(arm, [r], True)
                    total += float(loss) * len(r['pronunciation'])
                    phones += len(r['pronunciation'])
                    information += float(kl) * len(r['x'])
                    frame_count += len(r['x'])
                    if r.get('controlPronunciations'):
                        x = torch.from_numpy((r['x']-center)/scale)
                        probs, _ = arm(x, True)
                        def score(text):
                            y = torch.tensor([meta['tokens'].index(p)+1 for p in text], dtype=torch.long)
                            return -float(ctc(probs[:, None, :], y, torch.tensor([len(x)]), torch.tensor([len(y)])))
                        spoken = score(r['pronunciation'])
                        for alternative in r['controlPronunciations']:
                            wins += spoken > score(alternative)
                            controls += 1
                reports.append({'ctcPerPhone': total/phones, 'klPerFrame': information/frame_count,
                    'preservationWins': wins, 'preservationControls': controls,
                    'preservationRate': wins/controls if controls else None})
        return reports

    checkpoints = []
    limit = epochs or max(args.checkpoints)
    for epoch in range(1, limit+1):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), args.recording_batch):
            budget()
            batch = [rows[i] for i in order[start:start+args.recording_batch]]
            for i, (arm, opt) in enumerate(zip(arms, optimizers)):
                # Common random numbers and identical recording order in noisy arms.
                torch.manual_seed(args.seed + epoch*100000 + start)
                opt.zero_grad()
                loss, information = batch_loss(arm, batch)
                objective = loss + (penalty*information if i == 0 else 0)
                if not torch.isfinite(objective):
                    raise ValueError('Impossible or nonfinite transcript CTC objective')
                objective.backward()
                nn.utils.clip_grad_norm_(arm.parameters(), 5)
                opt.step()
        if held is not None and epoch in args.checkpoints:
            reports = evaluate()
            checkpoints.append({'epoch': epoch, 'arms': reports})
            print(json.dumps({'penalty': penalty, 'epoch': epoch, 'validation': reports}), flush=True)

    def serialize(a):
        return {name: tensor.detach().tolist() for name, tensor in (
            ('encoder', a.encoder.weight), ('encoderBias', a.encoder.bias),
            ('output', a.output.weight), ('outputBias', a.output.bias))}
    model = {k: meta[k] for k in ('schema','inventory','tokens','width','featureOutput','primarySha256')}
    model.update(center=center.tolist(), scale=scale.tolist(), latent=latent, fixedNoiseStd=fixed_std,
                 **{a.kind: serialize(a) for a in arms})
    report = {'penalty': penalty, 'epochs': limit, 'recordings': len(rows),
              'speakers': sorted({r['speaker'] for r in rows}), 'parametersPerArm': counts[0],
              'checkpoints': checkpoints}
    if held is not None:
        report['validationSpeakers'] = sorted({r['speaker'] for r in held})
    else:
        with torch.no_grad():
            probe = torch.from_numpy(all_x[:8])
            report['parity'] = {'features': probe.tolist(), 'logRows':
                [a((probe-torch.from_numpy(center))/torch.from_numpy(scale), True)[0].tolist() for a in arms]}
            report['trainingKlPerFrame'] = float(arms[0].distribution(
                torch.from_numpy((all_x-center)/scale))[2].mean())
    return model, report


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    began = time.monotonic()
    def budget():
        if time.monotonic()-began > args.seconds:
            raise TimeoutError('Bounded training budget reached; no bundle published')
    meta = json.loads(Path(args.metadata).read_text())
    rows = cache_rows(args.cache, meta)
    trials = []
    # Predeclared compression grid and checkpoint schedule, independent of practice.
    for penalty in (.001, .01, .1):
        folds = []
        for fold in range(3):
            training, held = grouped_split(rows, fold)
            _, report = fit(training, meta, penalty, args, budget, held)
            folds.append(report)
        for epoch in args.checkpoints:
            reports = [next(c['arms'][0] for c in f['checkpoints'] if c['epoch']==epoch) for f in folds]
            if any(r['preservationRate'] is None for r in reports):
                raise ValueError('Training fold lacks substitution preservation controls')
            loss = sum(r['ctcPerPhone'] + 1-r['preservationRate'] for r in reports)/len(reports)
            trials.append({'penalty': penalty, 'epoch': epoch, 'selectionLoss': loss, 'folds': folds})
    selected = min(trials, key=lambda t: t['selectionLoss'])
    model, final = fit(rows, meta, selected['penalty'], args, budget, epochs=selected['epoch'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'bottleneck-readouts.json').write_text(json.dumps(model, allow_nan=False))
    report = {'schema':1, 'selection':'speaker-grouped training CTC plus lexical preservation error',
              'trials':trials, 'selected':{k:selected[k] for k in ('penalty','epoch','selectionLoss')},
              'final':final, 'wallSeconds':time.monotonic()-began, 'schedule':vars(args)}
    (out/'bottleneck-training.json').write_text(json.dumps(report, allow_nan=False, indent=2))
    print(json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('cache','metadata','out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--latent', type=int, default=32)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seconds', type=int, default=3300)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[20,40,80])
    p.add_argument('--recording-batch', type=int, default=4)
    args = p.parse_args()
    if not 0 < args.seconds <= 86400 or not 0 < args.threads <= 8 or not 0 < args.latent <= 64 or min(args.checkpoints)<1:
        p.error('Invalid one-day training budget or readout geometry')
    train(args)
