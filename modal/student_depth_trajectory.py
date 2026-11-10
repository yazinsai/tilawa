"""CPU phone-conditioned likelihood on frozen encoder-depth trajectories.

Input consists only of help-train training-speaker none-labelled takes. Both
arms have 16 dimensions and identical full-Gaussian emissions/CTC topology.
Regularization is selected by grouped speaker validation before any target use.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 16
STRENGTHS = (4, 16, 64)
EPOCHS = 8


def topology(ids):
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
        ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else [])
        for s, p in enumerate(labels)]
    return labels, edges


def boundary(values, ids, posterior=False):
    import numpy as np
    labels, edges = topology(ids)
    labels = np.asarray(labels)
    tmax, n = len(values), len(labels)
    if not tmax or not ids:
        if posterior:
            raise ValueError('Recording cannot carry the complete canonical pronunciation')
        return -np.inf
    cost = np.log([len(targets) for targets in edges])
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    def forward(previous):
        outgoing = previous-cost
        current = outgoing.copy()
        current[1:] = np.logaddexp(current[1:], outgoing[:-1])
        current[2:] = np.logaddexp(current[2:], np.where(skip, outgoing[:-2], -np.inf))
        return current
    a = np.full((tmax, n), -np.inf)
    prior = np.full(n, -np.inf)
    a[0, :2] = values[0, labels[:2]]
    prior[:2] = 0
    for t in range(1, tmax):
        a[t] = forward(a[t-1])+values[t, labels]
        prior = forward(prior)
    z = np.logaddexp(*a[-1, -2:])
    timing = np.logaddexp(*prior[-2:])
    score = z-timing if np.isfinite(z) and np.isfinite(timing) else -np.inf
    if not posterior:
        return float(score)
    if not np.isfinite(z):
        raise ValueError('Recording cannot carry the complete canonical pronunciation')
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        incoming = b[t+1]+values[t+1, labels]
        current = incoming.copy()
        current[:-1] = np.logaddexp(current[:-1], incoming[1:])
        current[:-2] = np.logaddexp(current[:-2], np.where(skip, incoming[2:], -np.inf))
        b[t] = current-cost
    occupancy = np.zeros_like(values)
    for s, p in enumerate(labels):
        occupancy[:, p] += np.exp(a[:, s]+b[:, s]-z)
    return float(score), occupancy


def emissions(arm, x):
    import numpy as np
    columns = []
    for e in arm['phones']:
        z = (x-e['mean']) @ np.array(e['precision']).T
        columns.append(-.5*(DIMENSIONS*np.log(2*np.pi)+e['logDet']+(z*z).sum(1)))
    return np.array(columns).T


def update(count, sums, cross, strength):
    import numpy as np
    total = count.sum()
    mu = sums.sum(0)/total
    cov = cross.sum(0)/total-np.outer(mu, mu)
    cov = (cov+cov.T)/2 + .01*np.eye(DIMENSIONS)
    phones = []
    for p, n in enumerate(count):
        mean = (sums[p]+strength*mu)/(n+strength)
        scatter = cross[p]-np.outer(sums[p], mean)-np.outer(mean, sums[p])+n*np.outer(mean, mean)
        d = mu-mean
        sigma = (scatter+strength*(cov+np.outer(d, d)))/(n+strength)
        sigma = (sigma+sigma.T)/2 + .01*np.eye(DIMENSIONS)
        l = np.linalg.cholesky(sigma)
        phones.append(dict(mean=mean.tolist(), precision=np.linalg.inv(l).tolist(),
            logDet=float(2*np.log(l.diagonal()).sum()), count=float(n)))
    return phones


def fit(rows, key, vocab, strength):
    import numpy as np
    raw = np.concatenate([r[key] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    def statistics(posts):
        count = np.zeros(vocab)
        sums = np.zeros((vocab, DIMENSIONS))
        cross = np.zeros((vocab, DIMENSIONS, DIMENSIONS))
        for r, w in zip(rows, posts):
            x = (r[key]-center)/scale
            count += w.sum(0)
            sums += w.T @ x
            cross += np.einsum('tp,ti,tj->pij', w, x, x)
        return update(count, sums, cross, strength)
    # The unchanged parent provides only an EM starting distribution. Subsequent
    # iterations marginalize boundaries under the fitted frozen-feature likelihood.
    posts = [boundary(r['log_probs'], r['ids'], True)[1] for r in rows]
    arm = dict(center=center.tolist(), scale=scale.tolist(), phones=statistics(posts))
    for _ in range(EPOCHS):
        posts = [boundary(emissions(arm, (r[key]-center)/scale), r['ids'], True)[1] for r in rows]
        arm['phones'] = statistics(posts)
    return arm


def train(args):
    import numpy as np
    meta = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False:
            raise ValueError('Only explicitly eligible help-train none takes are accepted')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or release-held-out speaker')
        for k in ('trajectory', 'control', 'log_probs'):
            r[k] = np.array(r[k], dtype=float)
        if r['trajectory'].shape != r['control'].shape or r['trajectory'].shape[1] != DIMENSIONS:
            raise ValueError('Unmatched arm capacity or timing')
        if any(not np.isfinite(r[k]).all() for k in ('trajectory','control','log_probs')):
            raise ValueError('Nonfinite observations')
        r['ids'] = [meta['tokens'].index(p)+1 for p in r['pronunciation']]
        rows.append(r)
    speakers = sorted(set(r['speaker'] for r in rows))
    if len(speakers) < 3 or not rows:
        raise ValueError('At least three training speakers are required')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    validation, chosen = {}, {}
    for key in ('trajectory', 'control'):
        result = []
        for strength in STRENGTHS:
            speaker_scores = []
            for fold in range(3):
                training = [r for r in rows if folds[r['speaker']] != fold]
                held = [r for r in rows if folds[r['speaker']] == fold]
                arm = fit(training, key, len(meta['tokens'])+1, strength)
                scores = {}
                for r in held:
                    value = boundary(emissions(arm,(r[key]-arm['center'])/arm['scale']),r['ids'])/len(r[key])
                    scores.setdefault(r['speaker'], []).append(value)
                speaker_scores += [float(np.mean(v)) for v in scores.values()]
            result.append(dict(strength=strength, meanSpeakerLogP=float(np.mean(speaker_scores))))
        selected = max(result,key=lambda r:r['meanSpeakerLogP'])['strength']
        validation[key] = result
        chosen[key] = selected
        meta[key] = fit(rows, key, len(meta['tokens'])+1, selected)
    meta.update(schema=1, dimensions=DIMENSIONS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'depth-trajectory-emissions.json'
    asset.write_text(json.dumps(meta,allow_nan=False))
    report = dict(recordings=len(rows), speakers=speakers, speakerFolds=folds,
        excludedSpeakers=sorted(excluded), validation=validation, selected=chosen,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        assetSha256=hashlib.sha256(asset.read_bytes()).hexdigest(), epochs=EPOCHS)
    (out/'depth-trajectory-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    extractor = out/'depth-trajectory.onnx'
    if not extractor.is_file():
        raise ValueError('Training export directory lacks the verified frozen extractor')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset, extractor)])
    (out/'depth-trajectory-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    train(p.parse_args())
