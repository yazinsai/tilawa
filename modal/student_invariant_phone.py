"""Invariant-risk phone readout with an identical-data, identical-capacity pooled control.

Only none-labelled training takes supply transcripts. Real substitution scripts
select preservation requests; their performed phone transcripts are never assumed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN, EPOCHS = 32, 48, 18
GROUP_COUNTS = (2, 3)
PENALTIES = (.1, 1., 10.)
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


def condition_groups(rows, count):
    """Deterministic k-means on text-independent recording descriptors.

    Every call fits normalization and centroids only on its training fold. No
    speaker IDs, transcript phones or target windows enter this grouping.
    """
    import numpy as np
    raw = np.asarray([r['descriptor'] for r in rows], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all() or len(raw) < count:
        raise ValueError('Invalid recording descriptors or insufficient recordings')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    x = (raw-center)/scale
    centers = [x[np.argmax(np.sum(x*x, axis=1))]]
    while len(centers) < count:
        distances = ((x[:, None]-np.array(centers))**2).sum(2).min(1)
        centers.append(x[np.argmax(distances)])
    centers = np.array(centers)
    for _ in range(50):
        labels = ((x[:, None]-centers)**2).sum(2).argmin(1)
        if len(set(labels)) != count:
            raise ValueError('Recording conditions do not support distinct groups')
        updated = np.array([x[labels == g].mean(0) for g in range(count)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels, dict(center=center.tolist(), scale=scale.tolist(), centroids=centers.tolist())


def balanced_groups(rows, labels, vocab):
    """Equal recording mass, then equal soft phone distributions across groups.

    Importance weights are applied to soft targets, never to argmax labels.
    Only phones with support in every group participate in either arm's risk.
    """
    import numpy as np
    groups = []
    for g in sorted(set(labels)):
        selected = [r for r, label in zip(rows, labels) if label == g]
        x = np.concatenate([r['features'] for r in selected]).astype('float32')
        y = np.concatenate([r['targets'] for r in selected])
        w = np.concatenate([np.full(len(r['features']), 1/len(r['features'])/len(selected)) for r in selected])
        groups.append(dict(x=x, y=y, w=w, mass=(y*w[:, None]).sum(0)))
    mass = np.array([g['mass'] for g in groups])
    shared = np.all(mass > 1e-6, axis=0)
    if shared.sum() < 3:
        raise ValueError('Condition groups lack common phone support')
    target = np.where(shared, mass.mean(0), 0.)
    target /= target.sum()
    for g in groups:
        ratio = np.divide(target, g['mass'], out=np.zeros(vocab), where=g['mass'] > 0)
        g['y'] = (g['y']*ratio).astype('float32')
    return groups, target


def classifier_penalty(risk, model):
    """Stationarity of the SAME final classifier in each environment.

    This differentiates through the classifier gradient into the nonlinear
    representation. It is neither a worst-group loss nor a group-ID adversary.
    """
    import torch
    grads = torch.autograd.grad(risk, tuple(model[2].parameters()), create_graph=True)
    return sum(g.square().sum() for g in grads)


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def fit(rows, vocab, count, penalty, regularization):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    labels, grouping = condition_groups(rows, count)
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    groups, distribution = balanced_groups(rows, labels, vocab)
    for g in groups:
        g['x'] = torch.tensor((g['x']-center)/scale)
        g['y'] = torch.tensor(g['y'])
    pooled = head(vocab)
    invariant = copy.deepcopy(pooled)
    opts = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=regularization) for h in (pooled, invariant)]
    rng = np.random.default_rng(1729)
    steps = max(1, len(raw)//(128*count))
    for epoch in range(EPOCHS):
        for _ in range(steps):
            batches = [(g, rng.choice(len(g['x']), 128, p=g['w'])) for g in groups]
            for model, opt in zip((pooled, invariant), opts):
                opt.zero_grad()
                risks = [-(g['y'][b]*model(g['x'][b]).log_softmax(-1)).sum(-1).mean() for g, b in batches]
                average = torch.stack(risks).mean()
                if model is invariant:
                    stationarity = torch.stack([classifier_penalty(r, model) for r in risks]).mean()
                    # Ordinary shared warm-up precedes the frozen penalty weight.
                    loss = average + (penalty if epoch >= 3 else 0)*stationarity
                else:
                    loss = average
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
    return pooled.eval(), invariant.eval(), center, scale, grouping, labels, distribution


def speaker_folds(rows, speakers):
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    return {s: i % 3 for i, s in enumerate(anchors + sorted(set(speakers)-set(anchors)))}


def stationarity_report(models, rows, labels, vocab, center, scale):
    import numpy as np
    import torch
    groups, _ = balanced_groups(rows, labels, vocab)
    rng = np.random.default_rng(2718)
    batches = [(g, rng.choice(len(g['x']), 1024, p=g['w'])) for g in groups]
    report = []
    for model in models:
        values = []
        for g, batch in batches:
            x = torch.tensor((g['x'][batch]-center)/scale)
            y = torch.tensor(g['y'][batch])
            risk = -(y*model(x).log_softmax(-1)).sum(-1).mean()
            values.append(dict(risk=risk.item(), classifierGradientSquared=classifier_penalty(risk, model).item()))
        report.append(values)
    return dict(pooled=report[0], invariant=report[1])


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be present and unique')
    rows, controls, skipped = [], [], []
    vocab = len(metadata['tokens']) + 1
    for r in data:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or ineligible training-speaker provenance')
        x = np.asarray(r['features'])
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all():
            raise ValueError('Invalid frozen features')
        if r['kind'] == 'none':
            found = occupancies(r['logProbs'], r['pronunciation'])
            if found is None:
                skipped.append(r['id'])
                continue
            y, _ = found
            if len(y) != len(x) or y.shape[1] != vocab:
                raise ValueError('Occupancy geometry differs from frozen encoder')
            rows.append(dict(r, targets=y))
        elif r['kind'] == 'substituted_word':
            for e in r['controls']:
                features = np.asarray(e['features'])
                if features.ndim != 2 or features.shape[1] != DIMENSIONS or not np.isfinite(features).all() or any(
                    not ids or any(p < 1 or p >= vocab for p in ids) for ids in (e['written'], e['rival'])):
                    raise ValueError('Invalid complete training-only preservation request')
            controls.extend(dict(e, speaker=r['speaker']) for e in r['controls'])
        else:
            raise ValueError('Scripted deviations cannot supply phone transcripts')
    folds = speaker_folds(rows, {r['speaker'] for r in data})
    validation = []
    for count in GROUP_COUNTS:
        for penalty in PENALTIES:
            for reg in REGULARIZATION:
                errors, preservation = {}, []
                for fold in range(3):
                    training = [r for r in rows if folds[r['speaker']] != fold]
                    a, b, center, scale, grouping, _, distribution = fit(training, vocab, count, penalty, reg)
                    for r in rows:
                        if folds[r['speaker']] != fold:
                            continue
                        lp = outputs(b, r['features'], center, scale)
                        errors.setdefault(r['speaker'], []).append(float(-(r['targets']*lp).sum(1).mean()))
                    for e in controls:
                        if folds[e['speaker']] != fold:
                            continue
                        scores = [sequence_score(outputs(h, e['features'], center, scale), e['written'])-
                                  sequence_score(outputs(h, e['features'], center, scale), e['rival']) for h in (a, b)]
                        if np.isfinite(scores).all():
                            preservation.append(scores[0] < 0 and scores[1] > 0)
                result = dict(groups=count, penalty=penalty, regularization=reg,
                    meanSpeakerSoftNll=float(np.mean([np.mean(v) for v in errors.values()])),
                    preservationComparisons=len(preservation), preservationReversals=sum(preservation))
                validation.append(result)
                print(json.dumps(result), flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No training-speaker substitution preservation comparisons')
    selected = min(validation, key=lambda v: (v['preservationReversals']/max(1, v['preservationComparisons']),
        v['meanSpeakerSoftNll'], v['groups'], v['penalty'], v['regularization']))
    a, b, center, scale, grouping, labels, distribution = fit(rows, vocab, selected['groups'], selected['penalty'], selected['regularization'])
    metadata.update(center=center.tolist(), scale=scale.tolist(), invariant=serialize(b), control=serialize(a))
    out = Path(args.out)
    asset = out/'invariant-phone-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=sorted({r['speaker'] for r in rows}), folds=folds,
        excludedSpeakers=sorted(excluded), impossibleAlignments=skipped, validation=validation, selected=selected,
        grouping=grouping, trainingGroups={r['id']:int(g) for r,g in zip(rows,labels)},
        balancedPhoneDistribution=distribution.tolist(), epochs=EPOCHS,
        stationarity=stationarity_report((a,b), rows, labels, vocab, center, scale),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'invariant-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'invariant-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen feature graph')
    (out/'invariant-phone-asset.json').write_text(json.dumps(dict(schema=1, files=[dict(file=p.name,
        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())


