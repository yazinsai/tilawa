"""Native frozen-feature spherical identity readout and matched affine control.

No target/practice input is accepted. Only none-labelled training-speaker takes
supply soft phone supervision; real scripted swaps are diagnostic controls.
"""
import argparse
import hashlib
import json
from pathlib import Path

EPOCHS = 18
REGULARIZATION = (.0001, .01)
SEED = 1729


def occupancies(log_probs, ids):
    """Exact forward/backward soft CTC state and phone occupancies.

    Supervision sums every legal alignment, retaining uncertainty about
    phoneme identity and boundaries rather than greedy word segments.
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
    return targets.astype('float32')


def prepare(rows):
    import numpy as np
    prepared, skipped = [], []
    for r in rows:
        targets = occupancies(r['logProbs'], r['pronunciation'])
        if targets is None:
            skipped.append(r['id'])
            continue
        x = np.asarray(r['features'], dtype='float32')
        if len(x) != len(targets):
            raise ValueError('Feature and occupancy clocks differ')
        mass = targets[:, 1:].sum(1)
        valid = (mass > 0) & (np.linalg.norm(x, axis=1) > 0)
        if not valid.any():
            skipped.append(r['id'])
            continue
        prepared.append(dict(r, x=x[valid], y=targets[valid, 1:]/mass[valid, None], mass=mass[valid]))
    return prepared, skipped


def speaker_folds(rows, speakers):
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    ordered = anchors + sorted(set(speakers) - set(anchors))
    return {s: i % 3 for i, s in enumerate(ordered)}


def fit(rows, width, classes, regularization):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    raw = np.concatenate([r['x'] for r in rows])
    x = torch.tensor(raw)
    y = torch.tensor(np.concatenate([r['y'] for r in rows]))
    # Same frame draws for both arms; each recording has equal total loss mass.
    weights = np.concatenate([r['mass']/r['mass'].sum()/len(rows) for r in rows]).astype('float64')
    weights /= weights.sum()
    initial = F.normalize(torch.randn(classes, width), dim=1)
    median_norm = float(np.median(np.linalg.norm(raw, axis=1)))

    class Spherical(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vectors = torch.nn.Parameter(initial.clone())
            self.log_kappa = torch.nn.Parameter(torch.tensor(np.log(10.), dtype=torch.float32))

        def forward(self, values):
            return self.log_kappa.exp() * F.linear(F.normalize(values, dim=1), F.normalize(self.vectors, dim=1))

    angular = Spherical()
    control = torch.nn.Linear(width, classes)
    with torch.no_grad():
        control.weight.copy_(initial * 10./median_norm)
        control.bias.zero_()
    heads = (angular, control)
    opts = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=regularization) for h in heads]
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        for _ in range(max(1, int(np.ceil(len(x)/256)))):
            batch = rng.choice(len(x), 256, p=weights)
            for h, opt in zip(heads, opts):
                opt.zero_grad()
                loss = -(y[batch]*h(x[batch]).log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite phone training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 5.)
                opt.step()
    return angular.eval(), control.eval()


def outputs(model, features):
    import numpy as np
    import torch
    with torch.no_grad():
        return model(torch.tensor(np.asarray(features, dtype='float32'))).log_softmax(-1).numpy()


def reconstruct(identity, primary):
    """Keep blank and existing nonblank mass, without adding a probability floor."""
    import numpy as np
    blank = np.asarray(primary, dtype='float64')[:, :1]
    if np.isnan(blank).any() or (blank > 0).any():
        raise ValueError('Invalid primary blank probability')
    with np.errstate(divide='ignore', invalid='ignore'):
        speech = np.where(blank < -np.log(2), np.log1p(-np.exp(blank)), np.log(-np.expm1(blank)))
    return np.concatenate([blank, speech + identity], axis=1)


def sequence_score(lp, ids):
    import numpy as np
    if not len(lp) or not ids:
        return -np.inf
    labels = [0] + [v for p in ids for v in (p, 0)]
    a = np.full(len(labels), -np.inf)
    a[:2] = lp[0, labels[:2]]
    for frame in lp[1:]:
        b = np.full_like(a, -np.inf)
        for s, p in enumerate(labels):
            previous = [s] + ([s-1] if s else [])
            if s > 1 and p and p != labels[s-2]:
                previous.append(s-2)
            b[s] = np.logaddexp.reduce(a[previous]) + frame[p]
        a = b
    return float(np.logaddexp.reduce(a[-2:]))


def provenance(data, excluded, width, classes):
    import numpy as np
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be present and unique')
    for r in data:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or ineligible training-speaker provenance')
        if r.get('kind') not in ('none', 'substituted_word'):
            raise ValueError('Scripted deviations cannot supervise phone identities')
        x = np.asarray(r['features'])
        if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
            raise ValueError('Invalid native feature geometry')
        if r['kind'] == 'none':
            lp = np.asarray(r['logProbs'])
            if lp.shape != (len(x), classes+1):
                raise ValueError('Invalid primary frame inventory')


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    width, classes = metadata['dimensions'], len(metadata['tokens'])
    provenance(data, excluded, width, classes)
    rows, skipped = prepare([r for r in data if r['kind'] == 'none'])
    speakers = sorted({r['speaker'] for r in data})
    folds = speaker_folds(rows, speakers)
    controls = [dict(c, speaker=r['speaker']) for r in data if r['kind'] == 'substituted_word' for c in r['controls']]
    for c in controls:
        if len(c['features']) != len(c['logProbs']) or any(not ids or any(p < 1 or p > classes for p in ids) for ids in (c['written'], c['rival'])):
            raise ValueError('Invalid complete preservation window')
    validation = []
    for reg in REGULARIZATION:
        errors = [{}, {}]
        preservation = []
        for fold in range(3):
            heads = fit([r for r in rows if folds[r['speaker']] != fold], width, classes, reg)
            for r in rows:
                if folds[r['speaker']] != fold:
                    continue
                for arm, h in enumerate(heads):
                    lp = outputs(h, r['x'])
                    nll = float(np.average(-(r['y']*lp).sum(1), weights=r['mass']))
                    errors[arm].setdefault(r['speaker'], []).append(nll)
            for c in controls:
                if folds[c['speaker']] != fold:
                    continue
                margins = []
                for h in heads:
                    lp = reconstruct(outputs(h, c['features']), c['logProbs'])
                    margins.append(sequence_score(lp, c['written'])-sequence_score(lp, c['rival']))
                if np.isfinite(margins).all():
                    preservation.append(margins[0] > 0 and margins[1] < 0)
        nll = [float(np.mean([np.mean(v) for v in arm.values()])) for arm in errors]
        validation.append(dict(regularization=reg, angularSpeakerSoftNll=nll[0], affineSpeakerSoftNll=nll[1],
            matchedMeanSoftNll=float(np.mean(nll)), preservationComparisons=len(preservation), preservationReversals=sum(preservation)))
        print(json.dumps(validation[-1]), flush=True)
    # Shared regularization selected from training-speaker conditional soft NLL only.
    # Weak labelled swaps are reported, never an acoustic training transcript.
    selected = min(validation, key=lambda v: (v['matchedMeanSoftNll'], v['regularization']))
    angular, control = fit(rows, width, classes, selected['regularization'])
    norms = np.concatenate([np.linalg.norm(r['x'][r['mass'] > 1e-5], axis=1) for r in rows])
    metadata.update(angular=dict(vectors=F.normalize(angular.vectors, dim=1).detach().tolist(),
                                concentration=float(angular.log_kappa.exp().detach())),
                    control=dict(weights=control.weight.detach().tolist(), bias=control.bias.detach().tolist()),
                    normRange=[float(norms.min()), float(norms.max())])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(recordings=len(rows), speakers=speakers, controls=len(controls), folds=folds,
        excludedSpeakers=sorted(excluded), impossibleAlignments=skipped, validation=validation, selected=selected,
        epochs=EPOCHS, seed=SEED, dimensions=width, angularParameters=classes*width+1,
        affineParameters=classes*width+classes, normRange=metadata['normRange'],
        concentration=metadata['angular']['concentration'],
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'angular-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    asset = out/'angular-phone-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    files = [asset, out/'angular-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen native feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'angular-phone-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
