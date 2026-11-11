"""Matched ordinary/adversarial pronunciation-comparison heads on frozen features.

Only none-labelled help-train training recordings supply phone supervision.
Boundary occupancies and word crops are inferred from complete canonical takes.
Script-labelled substitutions are weak preservation controls, never transcripts.
Every split, scale, projection and training choice is frozen before practice use.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

DIMENSIONS, HIDDEN, EPOCHS = 32, 48, 18
SCALES = (.1, .3, .6)


def topology(ids):
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s + 1] if s + 1 < len(labels) else []) +
             ([s + 2] if s + 2 < len(labels) and labels[s + 2] and labels[s + 2] != p else [])
             for s, p in enumerate(labels)]
    return labels, edges


def sequence_score(lp, ids):
    """Differentiable normalized CTC timing marginal; matches SDK exactly."""
    import torch
    if not ids or len(lp) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return lp.sum() * 0 - torch.inf
    labels, edges = topology(ids)
    negative = lp.new_tensor(-1e30)
    label_ids = torch.tensor(labels, device=lp.device)
    cost = lp.new_tensor([math.log(len(e)) for e in edges])
    skip = torch.tensor([False, False] + [p != 0 and p != labels[i-2] for i, p in enumerate(labels[2:], 2)], device=lp.device)
    a = torch.stack([lp[0, p] if s < 2 else negative for s, p in enumerate(labels)])
    prior = torch.stack([lp.new_tensor(0.) if s < 2 else negative for s in range(len(labels))])
    def forward(v):
        v = v - cost
        one = torch.cat((negative.expand(1), v[:-1]))
        two = torch.cat((negative.expand(2), v[:-2]))
        return torch.logsumexp(torch.stack((v, one, torch.where(skip, two, negative))), 0)
    for t in range(1, len(lp)):
        a = forward(a) + lp[t, label_ids]
        prior = forward(prior)
    return torch.logsumexp(a[-2:], 0) - torch.logsumexp(prior[-2:], 0)


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def loss(model, x, example, reverse=False):
    import torch.nn.functional as F
    lp = model(x).log_softmax(-1)
    candidates = [example['correct'], example['rival']]
    if reverse:
        candidates.reverse()
    a, b = [sequence_score(lp, ids) for ids in candidates]
    # Changing candidate order and the target together leaves the same antisymmetric objective.
    return F.softplus((a - b) if reverse else (b - a))


def variation(rows, center, scale):
    """Within-phone residual PCA, estimated from training recordings only.

Speaker/word frequency affects neither a text-dependent basis nor a candidate-
specific perturbation: one shared acoustic covariance applies to all examples.
"""
    import numpy as np
    xs, labels = [], []
    for r in rows:
        xs.append((np.asarray(r['features']) - center) / scale)
        labels.extend(r['frameLabels'])
    x, labels = np.concatenate(xs), np.asarray(labels)
    means = {p: x[labels == p].mean(0) for p in set(labels)}
    residual = x - np.stack([means[p] for p in labels])
    cov = residual.T @ residual / max(1, len(residual))
    values, vectors = np.linalg.eigh(cov)
    keep = np.argsort(values)[-8:]
    return (vectors[:, keep] * np.sqrt(np.maximum(values[keep], 1e-6))).astype('float32')


def perturb(model, x, example, basis, radius, reverse=False):
    """Projected ascent finds feature changes favoring the incorrect pronunciation.

Coefficient Frobenius norm / sqrt(frames) is bounded by radius. Its shared PCA
basis expresses only variation observed within training-phone realizations.
"""
    import torch
    q = torch.zeros((len(x), basis.shape[1]), dtype=x.dtype, requires_grad=True)
    bound = radius * len(x) ** .5
    for _ in range(3):
        objective = loss(model, x + q @ basis.T, example, reverse)
        grad, = torch.autograd.grad(objective, q)
        q = q.detach() + (bound / 2) * grad / grad.norm().clamp_min(1e-9)
        q = (q * min(1., bound / max(float(q.norm()), 1e-9))).requires_grad_()
    return (q.detach() @ basis.T).detach()


def fit(rows, examples, vocab, radius):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    basis = torch.tensor(variation(rows, center, scale))
    ordinary = head(vocab)
    robust = copy.deepcopy(ordinary)
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=.01) for h in (ordinary, robust)]
    if not examples:
        raise ValueError('No recorded training contrasts survive speaker separation')
    prepared = [(e, torch.tensor((np.asarray(e['features']) - center) / scale, dtype=torch.float32)) for e in examples]
    rng = np.random.default_rng(1729)
    for epoch in range(EPOCHS):
        order = rng.permutation(len(prepared))
        reversals = rng.integers(0, 2, len(prepared)).astype(bool)
        for i in order:
            e, x = prepared[i]
            reverse = bool(reversals[i])
            for model, opt in zip((ordinary, robust), optimizers):
                opt.zero_grad()
                delta = perturb(model, x, e, basis, radius, reverse) if model is robust else 0
                value = loss(model, x + delta, e, reverse)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
        print(json.dumps(dict(epoch=epoch + 1, radius=radius, examples=len(examples))), flush=True)
    return ordinary.eval(), robust.eval(), center, scale, basis.numpy()


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def preference(model, x, a, b, center, scale):
    import numpy as np
    import torch
    with torch.no_grad():
        lp = model(torch.tensor((np.asarray(x) - center) / scale, dtype=torch.float32)).log_softmax(-1)
        return float(sequence_score(lp, a) - sequence_score(lp, b))


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    for r in data:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or ineligible training-speaker provenance')
        if r['kind'] not in ('none', 'substituted_word'):
            raise ValueError('Only none takes and weak substitution preservation controls accepted')
        features = np.asarray(r['features'])
        if features.ndim != 2 or features.shape[1] != DIMENSIONS or not np.isfinite(features).all():
            raise ValueError('Invalid frozen features')
    rows = [r for r in data if r['kind'] == 'none']
    examples = [dict(e, speaker=r['speaker']) for r in rows for e in r['examples']]
    for r in rows:
        if len(r['frameLabels']) != len(r['features']):
            raise ValueError('Canonical boundary labels do not match the frozen frame grid')
    vocab = len(metadata['tokens']) + 1
    for e in examples:
        if not e['correct'] or not e['rival'] or any(p < 1 or p >= vocab for p in e['correct'] + e['rival']):
            raise ValueError('Unknown or empty training pronunciation')
        if any(len(e['features']) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) for ids in (e['correct'], e['rival'])):
            raise ValueError('Training crop cannot carry both complete candidates')
    controls = [dict(e, speaker=r['speaker']) for r in data if r['kind'] != 'none' for e in r['controls']]
    controls = [e for e in controls if all(ids and all(0 < p < vocab for p in ids) and
        len(e['features']) >= len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) for ids in (e['written'], e['rival']))]
    speakers = sorted(set(r['speaker'] for r in rows))
    if len(speakers) < 3 or not examples:
        raise ValueError('At least three training speakers and attested contrasts required')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    validation = []
    for radius in SCALES:
        correct, preserved = [], []
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            available = {tuple(e['correct']) for e in examples if folds[e['speaker']] != fold}
            contrasts = [e for e in examples if folds[e['speaker']] != fold and tuple(e['rival']) in available]
            ordinary, robust, center, scale, _ = fit(training, contrasts, len(metadata['tokens']) + 1, radius)
            per_speaker = {}
            for e in examples:
                if folds[e['speaker']] != fold or tuple(e['rival']) not in available:
                    continue
                value = preference(robust, e['features'], e['correct'], e['rival'], center, scale) > 0
                per_speaker.setdefault(e['speaker'], []).append(value)
            correct.extend(float(np.mean(v)) for v in per_speaker.values())
            for e in controls:
                if folds.get(e['speaker']) != fold:
                    continue
                a = preference(ordinary, e['features'], e['written'], e['rival'], center, scale)
                b = preference(robust, e['features'], e['written'], e['rival'], center, scale)
                # Preserve ordinary-head rival preference; scripts select control recordings,
                # never provide a verified acoustic transcript or frame target.
                if a < 0:
                    preserved.append(b < 0)
        validation.append(dict(radius=radius, meanSpeakerAccuracy=float(np.mean(correct)),
                               preservationControls=len(preserved), retained=sum(preserved)))
    eligible = [v for v in validation if v['preservationControls'] and v['retained'] == v['preservationControls']]
    if not eligible:
        raise ValueError('No training-only radius preserves the labelled substitution controls')
    selected = max(eligible, key=lambda v: (v['meanSpeakerAccuracy'], -v['radius']))['radius']
    ordinary, robust, center, scale, basis = fit(rows, examples, len(metadata['tokens']) + 1, selected)
    metadata.update(center=center.tolist(), scale=scale.tolist(), robust=serialize(robust), control=serialize(ordinary))
    out = Path(args.out)
    asset = out / 'adversarial-feature-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=speakers, examples=len(examples), controls=len(controls),
                  folds=folds, excludedSpeakers=sorted(excluded), validation=validation, radius=selected,
                  basis=basis.tolist(), epochs=EPOCHS, inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out / 'adversarial-feature-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out / 'adversarial-feature.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out / 'adversarial-feature-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
