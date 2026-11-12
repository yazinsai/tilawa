"""Frozen final-projection null-space pronunciation evidence; CPU-only fitting.

No practice labels enter fitting or selection. Both arms use the same complete
CTC topology, boundary marginalization, Gaussian capacity and regularization.
"""
import argparse
import hashlib
import json
from pathlib import Path

EPOCHS = 4

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
    dimensions = len(arm['center'])
    import numpy as np
    columns = []
    for e in arm['phones']:
        z = (x-e['mean']) @ np.array(e['precision']).T
        columns.append(-.5*(dimensions*np.log(2*np.pi)+e['logDet']+(z*z).sum(1)))
    return np.array(columns).T


def update(count, sums, cross, strength):
    dimensions = sums.shape[1]
    import numpy as np
    total = count.sum()
    mu = sums.sum(0)/total
    cov = cross.sum(0)/total-np.outer(mu, mu)
    cov = (cov+cov.T)/2 + .01*np.eye(dimensions)
    phones = []
    for p, n in enumerate(count):
        mean = (sums[p]+strength*mu)/(n+strength)
        scatter = cross[p]-np.outer(sums[p], mean)-np.outer(mean, sums[p])+n*np.outer(mean, mean)
        d = mu-mean
        sigma = (scatter+strength*(cov+np.outer(d, d)))/(n+strength)
        sigma = (sigma+sigma.T)/2 + .01*np.eye(dimensions)
        l = np.linalg.cholesky(sigma)
        phones.append(dict(mean=mean.tolist(), precision=np.tril(np.linalg.inv(l)).tolist(),
            logDet=float(2*np.log(l.diagonal()).sum()), count=float(n)))
    return phones


def fit(rows, key, vocab, strength):
    dimensions = rows[0][key].shape[1]
    import numpy as np
    raw = np.concatenate([r[key] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    def statistics(posts):
        count = np.zeros(vocab)
        sums = np.zeros((vocab, dimensions))
        cross = np.zeros((vocab, dimensions, dimensions))
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


def validation_loss(arm, rows, key):
    import numpy as np
    grouped = {}
    for r in rows:
        values = emissions(arm, (r[key]-arm['center'])/arm['scale'])
        weights = boundary(r['log_probs'], r['ids'], True)[1][:, 1:]
        logits = values[:, 1:]
        peak = logits.max(1, keepdims=True)
        log_probs = logits-peak-np.log(np.exp(logits-peak).sum(1, keepdims=True))
        loss = -float((weights*log_probs).sum()/max(weights.sum(), 1e-12))
        grouped.setdefault(r['speaker'], []).append(loss)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def preservation_controls(arms, rows, tokens):
    """Held-training-speaker real speech versus counterfactual lexical references.

    A canonical none take is the actually spoken rival. Changing only its written
    reference supplies a substitution-preservation control without assuming that
    scripted learner edits are verified acoustic transcripts. Never used to select.
    """
    total, vetoes, control_spoken, null_spoken = 0, 0, 0, 0
    for r in rows:
        values = {key: emissions(arm, (r[key]-arm['center'])/arm['scale'])
            for key, arm in arms.items()}
        spoken = {key: boundary(v, r['ids']) for key, v in values.items()}
        for text in r.get('controlPronunciations', []):
            ids = [tokens.index(p)+1 for p in text]
            written = {key: boundary(v, ids) for key, v in values.items()}
            total += 1
            control_spoken += written['control'] < spoken['control']
            null_spoken += written['nullspace'] < spoken['nullspace']
            vetoes += written['nullspace'] > spoken['nullspace'] and written['control'] < spoken['control']
    return dict(comparisons=total, vetoes=int(vetoes), controlPrefersSpoken=int(control_spoken),
        nullspacePrefersSpoken=int(null_spoken))


def train(args):
    import numpy as np
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False:
            raise ValueError('Only eligible help-train training-speaker none takes may fit emissions')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or release-held-out speaker')
        r['ids'] = [metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        for key, width in [('log_probs',len(metadata['tokens'])+1),('nullspace',metadata['dimensions']),('control',metadata['dimensions'])]:
            r[key] = np.asarray(r[key], dtype=np.float64)
            if r[key].ndim != 2 or r[key].shape[1] != width or not len(r[key]) or not np.isfinite(r[key]).all():
                raise ValueError('Invalid frozen observations')
        if not len(r['log_probs']) == len(r['nullspace']) == len(r['control']):
            raise ValueError('Arms have different acoustic clocks')
        rows.append(r)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three independent training speakers required')
    folds = {s:i%3 for i,s in enumerate(speakers)}
    results = []
    for strength in (1.,16.,64.):
        losses = {key:[] for key in ('nullspace','control')}
        controls = []
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            held = [r for r in rows if folds[r['speaker']] == fold]
            arms = {}
            for key in losses:
                arm = fit(training,key,len(metadata['tokens'])+1,strength)
                arms[key] = arm
                losses[key].append(validation_loss(arm,held,key))
            controls.append(preservation_controls(arms,held,metadata['tokens']))
        item = dict(strength=strength, **{key:float(np.mean(v)) for key,v in losses.items()})
        item['matchedLoss'] = (item['nullspace']+item['control'])/2
        item['preservationControls'] = controls
        results.append(item)
        print(json.dumps(item), flush=True)
    selected = min(results,key=lambda r:(r['matchedLoss'],r['strength']))
    fitted = {**metadata, **{key:fit(rows,key,len(metadata['tokens'])+1,selected['strength']) for key in ('nullspace','control')}}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    extractor = out/'logit-nullspace.onnx'
    if not extractor.is_file():
        raise ValueError('Verified extractor missing')
    asset = out/'logit-nullspace-emissions.json'
    asset.write_text(json.dumps(fitted,allow_nan=False))
    files = [dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,extractor)]
    (out/'logit-nullspace-asset.json').write_text(json.dumps(dict(schema=1,files=files)))
    report = dict(recordings=len(rows),speakers=speakers,speakerFolds=folds,excludedSpeakers=sorted(excluded),
        validation=results,selected=selected,epochs=EPOCHS,files=files,
        controlLimit='Counterfactual written references on real none takes; not verified scripted substitutions; controls do not select regularization.',
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'logit-nullspace-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    train(p.parse_args())
