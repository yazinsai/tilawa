"""CPU-bounded sparse-factor CTC side readout and matched controls.

Input is deployment-exact cached features, never practice data. Mistake takes
are reconstruction-only; only eligible none takes supply canonical CTC labels.
Selection splits whole speakers before fitting normalization or dictionaries.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def grouped_split(rows, fold, folds=3):
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < folds:
        raise ValueError('Insufficient independent training speakers')
    # Stable assignment, independent of recording count, vocabulary and practice.
    speakers.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    held = set(speakers[fold::folds])
    return ([r for r in rows if r['speaker'] not in held],
            [r for r in rows if r['speaker'] in held])


def cache_rows(path, meta):
    import numpy as np
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty training cache')
    for r in rows:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker'):
            raise ValueError('Cache includes ineligible training data')
        x = np.asarray(r['features'], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != meta['width'] or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid full-width encoder features')
        r['x'] = x
        del r['features']
        if r.get('pronunciation'):
            if r.get('kind') != 'none' or r.get('events') or r.get('non_errors'):
                raise ValueError('Performed edits cannot supply canonical labels')
            if any(p not in meta['tokens'] for p in r['pronunciation']):
                raise ValueError('Out-of-inventory canonical phone')
    return rows


def fit(rows, meta, penalty, args, budget, evaluate=None):
    import numpy as np
    import torch
    from torch import nn
    budget()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    width, hidden, vocab = meta['width'], 2 * meta['width'], len(meta['tokens']) + 1
    all_x = np.concatenate([r['x'] for r in rows])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-3)
    frames = torch.from_numpy((all_x - center) / scale)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(width, hidden)
            self.decoder = nn.Linear(hidden, width)

        def forward(self, x):
            z = self.encoder(x).relu()
            return self.decoder(z), z

        def normalize_dictionary(self):
            # Prevent shrinking codes and inflating decoder weights to evade L1.
            with torch.no_grad():
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0).clamp_min(1e-8))

    torch.manual_seed(args.seed)
    sparse = Autoencoder()
    torch.manual_seed(args.seed)
    dense = Autoencoder()
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=1e-4) for a in (sparse, dense)]
    for a in (sparse, dense):
        a.normalize_dictionary()
    for _ in range(args.ae_epochs):
        order = rng.permutation(len(frames))
        for start in range(0, len(order), args.frame_batch):
            budget()
            batch = frames[order[start:start + args.frame_batch]]
            for ae, opt, strength in zip((sparse, dense), optimizers, (penalty, 0.0)):
                opt.zero_grad()
                prediction, z = ae(batch)
                loss = (prediction - batch).square().mean() + strength * z.abs().mean()
                loss.backward()
                opt.step()
                ae.normalize_dictionary()

    class Readout(nn.Module):
        def __init__(self, encoder=None):
            super().__init__()
            self.encoder = nn.Linear(width, hidden)
            self.output = nn.Linear(hidden, vocab)
            if encoder is not None:
                self.encoder.load_state_dict(encoder.state_dict())
                self.encoder.requires_grad_(False)

        def forward(self, x):
            return self.output(self.encoder(x).relu()).log_softmax(-1)

    arms = []
    for encoder in (sparse.encoder, dense.encoder, None):
        torch.manual_seed(args.seed + 1)
        arms.append(Readout(encoder))
    frozen = [[p.detach().clone() for p in a.encoder.parameters()] for a in arms[:2]]
    optimizers = [torch.optim.AdamW((p for p in a.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-3) for a in arms]
    supervised = [r for r in rows if r.get('pronunciation')]
    if not supervised:
        raise ValueError('No eligible none-labelled takes')
    ctc = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)

    def batch_loss(arm, batch):
        xs = [torch.from_numpy((r['x'] - center) / scale) for r in batch]
        labels = [torch.tensor([meta['tokens'].index(p) + 1 for p in r['pronunciation']], dtype=torch.long) for r in batch]
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        target_lengths = torch.tensor([len(y) for y in labels], dtype=torch.long)
        logits = arm(nn.utils.rnn.pad_sequence(xs)).contiguous()
        return ctc(logits, torch.cat(labels), lengths, target_lengths) / target_lengths.sum()

    for _ in range(args.ctc_epochs):
        order = rng.permutation(len(supervised))
        for start in range(0, len(order), args.recording_batch):
            budget()
            batch = [supervised[i] for i in order[start:start + args.recording_batch]]
            for arm, opt in zip(arms, optimizers):
                opt.zero_grad()
                loss = batch_loss(arm, batch)
                if not torch.isfinite(loss):
                    raise ValueError('Impossible or nonfinite full-pronunciation CTC supervision')
                loss.backward()
                nn.utils.clip_grad_norm_((p for p in arm.parameters() if p.requires_grad), 5)
                opt.step()

    diagnostics = {'penalty': penalty, 'recordings': len(rows), 'noneTakes': len(supervised),
                   'speakers': sorted({r['speaker'] for r in rows}), 'inferenceParameters': hidden * (width + 1) + vocab * (hidden + 1)}
    diagnostics['frozenEncodersUnchanged'] = all(torch.equal(before, after) for a, saved in zip(arms[:2], frozen)
        for before, after in zip(saved, a.encoder.parameters()))
    if not diagnostics['frozenEncodersUnchanged']:
        raise AssertionError('CTC fitting changed a frozen autoencoder')
    with torch.no_grad():
        sample = frames[:min(len(frames), 8192)]
        diagnostics['reconstruction'] = []
        for ae in (sparse, dense):
            prediction, z = ae(sample)
            diagnostics['reconstruction'].append({'mse': float((prediction - sample).square().mean()),
                'activeFraction': float((z > 0).float().mean()), 'meanActivation': float(z.mean())})
        if evaluate is not None:
            held = [r for r in evaluate if r.get('pronunciation')]
            if not held:
                raise ValueError('Speaker fold has no canonical validation take')
            diagnostics['validation'] = [float(sum(batch_loss(a, [r]) * len(r['pronunciation']) for r in held) /
                sum(len(r['pronunciation']) for r in held)) for a in arms]
            diagnostics['validationSpeakers'] = sorted({r['speaker'] for r in evaluate})
    def serialize(a):
        return {name: tensor.detach().tolist() for name, tensor in (
            ('encoder', a.encoder.weight), ('encoderBias', a.encoder.bias),
            ('output', a.output.weight), ('outputBias', a.output.bias))}
    model = {k: meta[k] for k in ('schema', 'inventory', 'tokens', 'width', 'featureOutput', 'primarySha256')}
    model.update(center=center.tolist(), scale=scale.tolist(), hidden=hidden,
        sparseFactor=serialize(arms[0]), dense=serialize(arms[1]), control=serialize(arms[2]))
    if evaluate is None:
        diagnostics['dictionaries'] = [ae.decoder.weight.detach().tolist() for ae in (sparse, dense)]
        probe = torch.from_numpy(all_x[:8])
        diagnostics['parity'] = {'features': probe.tolist(), 'logRows':
            [a((probe - torch.from_numpy(center)) / torch.from_numpy(scale)).detach().tolist() for a in arms]}
    return model, diagnostics


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    start = time.process_time()
    def budget():
        if time.process_time() - start > args.cpu_seconds:
            raise TimeoutError('One CPU day training budget reached; no bundle published')
    meta = json.loads(Path(args.metadata).read_text())
    rows = cache_rows(args.cache, meta)
    trials = []
    # This grid is fixed before seeing any warning comparisons. All arms share
    # folds, recordings, initializations and update schedules within every trial.
    for penalty in (0.01, 0.1):
        folds = []
        for fold in range(3):
            training, held = grouped_split(rows, fold)
            _, report = fit(training, meta, penalty, args, budget, held)
            folds.append(report)
            print(json.dumps(report), flush=True)
        trials.append({'penalty': penalty, 'folds': folds,
            'loss': sum(f['validation'][0] for f in folds) / len(folds)})
    selected = min(trials, key=lambda t: t['loss'])['penalty']
    model, final = fit(rows, meta, selected, args, budget)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sparse-factor-readouts.json').write_text(json.dumps(model, allow_nan=False))
    report = {'schema': 1, 'selection': 'speaker-grouped training-only sparse CTC loss', 'trials': trials,
        'final': final, 'cpuSeconds': time.process_time() - start, 'schedule': vars(args)}
    (out / 'sparse-factor-training.json').write_text(json.dumps(report, allow_nan=False, indent=2))
    print(json.dumps({'selectedPenalty': selected, 'cpuSeconds': report['cpuSeconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cpu-seconds', type=int, default=7200)
    p.add_argument('--ae-epochs', type=int, default=8)
    p.add_argument('--ctc-epochs', type=int, default=30)
    p.add_argument('--frame-batch', type=int, default=512)
    p.add_argument('--recording-batch', type=int, default=4)
    args = p.parse_args()
    if not 0 < args.cpu_seconds <= 7200 or not 0 < args.threads <= 8:
        p.error('Training CPU budget must be positive and at most 7200 seconds')
    train(args)
