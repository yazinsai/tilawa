"""CPU pronunciation likelihood on frozen attention means and value variances.

Input consists only of help-train training-speaker none-labelled takes. Both
arms have 16 dimensions and identical full-Gaussian emissions/CTC topology.
Regularization is selected by grouped speaker validation before any target use.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 16
STRENGTHS = (8, 32)
EPOCHS = 3


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


def matched_features(rows, candidate, projection, control_projection):
    import numpy as np
    result = []
    for r in rows:
        raw = r['candidates'][candidate]
        mean, variance = np.array(raw['mean']), np.array(raw['variance'])
        if mean.ndim != 2 or mean.shape != variance.shape or len(mean) != len(r['log_probs']):
            raise ValueError('Attention means and variances must share the primary clock')
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
            raise ValueError('Invalid attention statistics')
        result.append(dict(r, dispersion=np.concatenate([mean, variance], axis=1) @ projection.T,
            control=mean @ control_projection.T))
    return result


def training_contrasts(row, supported):
    # Deterministic absent pronunciations drawn solely from the training transcript.
    # No practice pronunciation, label, target or window participates in selection.
    import numpy as np
    seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    rivals = []
    for _ in range(3):
        rival = list(row['ids'])
        at = int(rng.integers(len(rival)))
        other = [p for p in supported if p != rival[at]]
        if not other:
            continue
        rival[at] = int(rng.choice(other))
        rivals.append(rival)
    return rivals


def validation_loss(arm, rows, key):
    import numpy as np
    supported = [p for p, e in enumerate(arm['phones']) if p and e['count'] > 0]
    speakers = {}
    for r in rows:
        values = emissions(arm, (r[key]-arm['center'])/arm['scale'])
        correct = boundary(values, r['ids'])
        comparisons = [boundary(values, rival) for rival in training_contrasts(r, supported)]
        if not np.isfinite(correct) or not comparisons or not all(np.isfinite(comparisons)):
            raise ValueError('Training validation cannot score complete pronunciation')
        loss = float(np.mean([np.logaddexp(0., wrong-correct) for wrong in comparisons]))
        speakers.setdefault(r['speaker'], []).append(loss)
    return float(np.mean([np.mean(v) for v in speakers.values()]))


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
        if not isinstance(r.get('id'), str) or not r['id']:
            raise ValueError('Training observation lacks identity')
        r['log_probs'] = np.array(r['log_probs'], dtype=float)
        if not np.isfinite(r['log_probs']).all():
            raise ValueError('Nonfinite parent observations')
        if len(r['candidates']) != len(meta['heads']):
            raise ValueError('Missing candidate heads')
        r['ids'] = [meta['tokens'].index(p)+1 for p in r['pronunciation']]
        rows.append(r)
    speakers = sorted(set(r['speaker'] for r in rows))
    if len(speakers) < 3 or not rows:
        raise ValueError('At least three training speakers are required')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    validation = []
    rng = np.random.default_rng(1729)
    selected = None
    for h, head in enumerate(meta['heads']):
        width = head['width']
        # Both arms have exactly 16 Gaussian dimensions/identical parameter count.
        # Shared mean basis; dispersion adds value variance through an independent fixed basis.
        mean_projection = rng.normal(size=(DIMENSIONS,width))/np.sqrt(width)
        variance_projection = rng.normal(size=(DIMENSIONS,width))/np.sqrt(width)
        projection = np.concatenate([mean_projection,variance_projection],axis=1)/np.sqrt(2)
        features = matched_features(rows,h,projection,mean_projection)
        strengths, losses = {}, {}
        for key in ('dispersion','control'):
            records = []
            for strength in STRENGTHS:
                grouped = []
                for fold in range(3):
                    training = [r for r in features if folds[r['speaker']] != fold]
                    held = [r for r in features if folds[r['speaker']] == fold]
                    arm = fit(training,key,len(meta['tokens'])+1,strength)
                    grouped.append(validation_loss(arm,held,key))
                records.append(dict(strength=strength,foldLosses=grouped,loss=float(np.mean(grouped))))
            best = min(records,key=lambda r:r['loss'])
            strengths[key], losses[key] = best['strength'], best['loss']
            validation.append(dict(head=h,arm=key,results=records))
            print(json.dumps(validation[-1]),flush=True)
        # Select a single head by both arms' grouped training discrimination.
        loss = (losses['dispersion']+losses['control'])/2
        if selected is None or loss < selected['loss']:
            selected = dict(head=h,loss=loss,strengths=strengths,projection=projection,
                controlProjection=mean_projection,features=features)
    meta.update(schema=1,dimensions=DIMENSIONS,selected=selected['head'],
        dispersionProjection=selected['projection'].tolist(),controlProjection=selected['controlProjection'].tolist())
    for key in ('dispersion','control'):
        meta[key] = fit(selected['features'],key,len(meta['tokens'])+1,selected['strengths'][key])
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    asset = out/'attention-dispersion-emissions.json'
    asset.write_text(json.dumps(meta,allow_nan=False))
    report = dict(recordings=len(rows),speakers=speakers,speakerFolds=folds,
        excludedSpeakers=sorted(excluded),validation=validation,selectedHead=selected['head'],
        selectedStrengths=selected['strengths'],selectionLoss=selected['loss'],
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        assetSha256=hashlib.sha256(asset.read_bytes()).hexdigest(),epochs=EPOCHS)
    (out/'attention-dispersion-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    extractor = out/'attention-dispersion.onnx'
    if not extractor.is_file():
        raise ValueError('Training export directory lacks the verified frozen extractor')
    descriptor = dict(schema=1,files=[dict(file=p.name,bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,extractor)])
    (out/'attention-dispersion-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    train(p.parse_args())
