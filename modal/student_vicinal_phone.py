"""Frozen-encoder soft-occupancy phone heads, with a matched vicinal control.

Train only on none-labelled help-train training speakers. Scripted substitutions
select preservation windows but never supply performed transcripts. All model
selection and interpolation pairs are confined to speaker-grouped training folds.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN, EPOCHS = 32, 48, 18
STRENGTHS = (.25, .5, 1.)
REGULARIZATION = (.0001, .01)


def occupancies(log_probs, ids):
    """Exact forward/backward soft CTC state and phone occupancies.

    Context keys describe surrounding transcript phones of the most probable
    state (including its central identity), while supervision retains the
    entire posterior distribution.
    """
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or not np.isfinite(lp).all() or any(p <= 0 or p >= lp.shape[1] for p in ids):
        raise ValueError('Invalid canonical transcript or posterior rows')
    labels = np.array([0] + [p for i in ids for p in (i, 0)])
    if not ids or len(lp) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return None
    n, t = len(labels), len(lp)
    incoming = [[s] + ([s-1] if s else []) +
                ([s-2] if s > 1 and labels[s] and labels[s] != labels[s-2] else []) for s in range(n)]
    outgoing = [[] for _ in labels]
    for s, edges in enumerate(incoming):
        for before in edges:
            outgoing[before].append(s)
    alpha = np.full((t, n), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for f in range(1, t):
        for s in range(n):
            alpha[f, s] = np.logaddexp.reduce(alpha[f-1, incoming[s]]) + lp[f, labels[s]]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full((t, n), -np.inf)
    beta[-1, -2:] = 0
    for f in range(t-2, -1, -1):
        for s in range(n):
            nxt = outgoing[s]
            beta[f, s] = np.logaddexp.reduce(beta[f+1, nxt] + lp[f+1, labels[nxt]])
    states = np.exp(alpha + beta - z)
    states /= states.sum(1, keepdims=True)
    targets = np.zeros_like(lp)
    for s, phone in enumerate(labels):
        targets[:, phone] += states[:, s]
    context = []
    for s in states.argmax(1):
        # Blank intervals use their immediately adjacent phones. Articulations
        # use the previous/next articulation, including genuine doubled phones.
        pos = (int(s)-1)//2
        if s % 2:
            context.append((ids[pos-1] if pos else 0, ids[pos], ids[pos+1] if pos+1 < len(ids) else 0))
        else:
            context.append((ids[s//2-1] if s else 0, 0, ids[s//2] if s//2 < len(ids) else 0))
    return targets.astype('float32'), context


def prepare(rows):
    import numpy as np
    prepared, skipped = [], []
    for r in rows:
        found = occupancies(r['logProbs'], r['pronunciation'])
        if found is None:
            skipped.append(r['id'])
            continue
        targets, contexts = found
        prepared.append(dict(r, targets=targets, contexts=contexts))
    return prepared, skipped


def interpolation_pairs(rows):
    """Choose donors uniformly by recording, then frame, in exact shared contexts.

    Neither the speaker nor lexical identity is an inference feature. Limiting
    each recording's contribution prevents long takes from dominating donors.
    """
    import numpy as np
    groups, record, keys = {}, [], []
    at = 0
    for r in rows:
        for context in r['contexts']:
            key = tuple(context)
            groups.setdefault(key, {}).setdefault(r['id'], []).append(at)
            record.append(r['id'])
            keys.append(key)
            at += 1
    rng = np.random.default_rng(1729)
    left, right = [], []
    for i, (key, origin) in enumerate(zip(keys, record)):
        donors = sorted(k for k in groups[key] if k != origin)
        if donors:
            chosen = donors[int(rng.integers(len(donors)))]
            left.append(i)
            right.append(int(rng.choice(groups[key][chosen])))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def speaker_folds(rows, speakers):
    """Spread the none-labelled speakers first so every held-out fold has anchors."""
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    ordered = anchors + sorted(set(speakers) - set(anchors))
    return {speaker: i % 3 for i, speaker in enumerate(ordered)}


def fit(rows, vocab, strength, regularization):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    x = torch.tensor((raw-center)/scale)
    y = torch.tensor(np.concatenate([r['targets'] for r in rows]))
    left, right = interpolation_pairs(rows)
    if not len(left):
        raise ValueError('No cross-recording compatible-context interpolation pairs')
    ordinary, vicinal = head(vocab), None
    vicinal = copy.deepcopy(ordinary)
    opts = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=regularization) for h in (ordinary, vicinal)]
    # Equal recording mass in the ordinary soft-target loss.
    weights = np.concatenate([np.full(len(r['features']), 1/len(r['features'])) for r in rows])
    weights /= weights.sum()
    pair_weights = weights[left] / weights[left].sum()
    rng = np.random.default_rng(1729)
    steps = max(1, len(x)//256)
    for _ in range(EPOCHS):
        for _ in range(steps):
            batch = rng.choice(len(x), 256, p=weights)
            pick = rng.choice(len(left), 256, p=pair_weights)
            a, b = left[pick], right[pick]
            lam = torch.tensor(rng.uniform(.25, .75, (256, 1)), dtype=torch.float32)
            mixed_x, mixed_y = lam*x[a]+(1-lam)*x[b], lam*y[a]+(1-lam)*y[b]
            for model, opt in zip((ordinary, vicinal), opts):
                opt.zero_grad()
                original = -(y[batch]*model(x[batch]).log_softmax(-1)).sum(-1).mean()
                if model is vicinal:
                    augmentation = -(mixed_y*model(mixed_x).log_softmax(-1)).sum(-1).mean()
                else:
                    # Identical donor samples, coefficient draws and loss weight;
                    # this control sees the endpoints without interpolating them.
                    augmentation = -(lam*y[a]*model(x[a]).log_softmax(-1) +
                                     (1-lam)*y[b]*model(x[b]).log_softmax(-1)).sum(-1).mean()
                ((original+strength*augmentation)/(1+strength)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
    return ordinary.eval(), vicinal.eval(), center, scale, len(left)


def sequence_score(lp, ids):
    """Same complete CTC marginal and normalized timing prior as the SDK."""
    import numpy as np
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
             ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else []) for s, p in enumerate(labels)]
    a = np.full(len(labels), -np.inf)
    prior = a.copy()
    a[:2] = lp[0, labels[:2]]
    prior[:2] = 0
    for frame in lp[1:]:
        b, q = np.full_like(a, -np.inf), np.full_like(a, -np.inf)
        for s, nxt in enumerate(edges):
            cost = np.log(len(nxt))
            for j in nxt:
                b[j] = np.logaddexp(b[j], a[s]-cost)
                q[j] = np.logaddexp(q[j], prior[s]-cost)
        a, prior = b + frame[labels], q
    likelihood, timing = np.logaddexp.reduce(a[-2:]), np.logaddexp.reduce(prior[-2:])
    return float(likelihood-timing) if np.isfinite(likelihood) and np.isfinite(timing) else -np.inf


def outputs(model, features, center, scale):
    import numpy as np
    import torch
    with torch.no_grad():
        return model(torch.tensor((np.asarray(features, dtype='float32')-center)/scale)).log_softmax(-1).numpy()


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be present and unique')
    for r in data:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or ineligible training-speaker provenance')
        if r['kind'] not in ('none', 'substituted_word'):
            raise ValueError('Unsupervised script kinds cannot train these phone heads')
        features = np.asarray(r['features'])
        if features.ndim != 2 or features.shape[1] != DIMENSIONS or not np.isfinite(features).all():
            raise ValueError('Invalid frozen features')
    rows, skipped = prepare([r for r in data if r['kind'] == 'none'])
    speakers = sorted(set(r['speaker'] for r in data))
    folds = speaker_folds(rows, speakers)
    controls = [dict(e, speaker=r['speaker']) for r in data if r['kind'] == 'substituted_word' for e in r['controls']]
    vocab = len(metadata['tokens']) + 1
    for r in rows:
        if len(r['features']) != len(r['targets']) or r['targets'].shape[1] != vocab:
            raise ValueError('Soft occupancies differ from native features or phone inventory')
    for e in controls:
        if any(not ids or any(p < 1 or p >= vocab for p in ids) for ids in (e['written'], e['rival'])):
            raise ValueError('Invalid complete preservation pronunciation')
    validation = []
    for strength in STRENGTHS:
        for reg in REGULARIZATION:
            errors, preservation = {}, []
            for fold in range(3):
                training = [r for r in rows if folds[r['speaker']] != fold]
                control, treatment, center, scale, _ = fit(training, vocab, strength, reg)
                for r in rows:
                    if folds[r['speaker']] == fold:
                        lp = outputs(treatment, r['features'], center, scale)
                        errors.setdefault(r['speaker'], []).append(float(-(r['targets']*lp).sum(1).mean()))
                for e in controls:
                    if folds[e['speaker']] != fold:
                        continue
                    a, b = [outputs(h, e['features'], center, scale) for h in (control, treatment)]
                    scores = [sequence_score(lp, e['written'])-sequence_score(lp, e['rival']) for lp in (a, b)]
                    if all(np.isfinite(scores)):
                        preservation.append(scores[0] < 0 and scores[1] > 0)
            validation.append(dict(strength=strength, regularization=reg,
                meanSpeakerSoftNll=float(np.mean([np.mean(v) for v in errors.values()])),
                preservationComparisons=len(preservation), preservationReversals=sum(preservation)))
            print(json.dumps(validation[-1]), flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No real substitution preservation comparisons available')
    # Preservation participates in selection without imposing a practice loss veto.
    selected = min(validation, key=lambda v: (v['preservationReversals']/max(1,v['preservationComparisons']),
                                              v['meanSpeakerSoftNll'], v['strength'], v['regularization']))
    control, treatment, center, scale, pairs = fit(rows, vocab, selected['strength'], selected['regularization'])
    metadata.update(center=center.tolist(), scale=scale.tolist(), vicinal=serialize(treatment), control=serialize(control))
    out = Path(args.out)
    asset = out/'vicinal-phone-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=speakers, controls=len(controls), folds=folds,
        excludedSpeakers=sorted(excluded), impossibleAlignments=skipped, validation=validation,
        selected=selected, interpolationPairs=pairs, epochs=EPOCHS,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'vicinal-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'vicinal-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'vicinal-phone-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
