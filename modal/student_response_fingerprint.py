"""Frozen listener response-fingerprint pronunciation evidence; CPU-only fitting.

No practice labels enter fitting or selection. Both arms use the same complete
CTC topology, boundary marginalization, Gaussian capacity and regularization.
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

EPOCHS = 4
DEADLINE = float('inf')

def budget():
    if time.time() > DEADLINE:
        raise TimeoutError('Response extraction and fitting exceeded the one-day budget')


def topology(ids):
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
        ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else [])
        for s, p in enumerate(labels)]
    return labels, edges


def boundary(values, ids, posterior=False):
    import numpy as np
    budget()
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
            budget()
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
    total, vetoes, control_spoken, response_spoken = 0, 0, 0, 0
    for r in rows:
        values = {key: emissions(arm, (r[key]-arm['center'])/arm['scale'])
            for key, arm in arms.items()}
        spoken = {key: boundary(v, r['ids']) for key, v in values.items()}
        for text in r.get('controlPronunciations', []):
            ids = [tokens.index(p)+1 for p in text]
            written = {key: boundary(v, ids) for key, v in values.items()}
            total += 1
            control_spoken += written['control'] < spoken['control']
            response_spoken += written['response'] < spoken['response']
            vetoes += written['response'] > spoken['response'] and written['control'] < spoken['control']
    return dict(comparisons=total, vetoes=int(vetoes), controlPrefersSpoken=int(control_spoken),
        responsePrefersSpoken=int(response_spoken))


def train(args):
    import numpy as np
    global DEADLINE
    DEADLINE = args.deadline / 1000
    budget()
    if not math.isfinite(args.deadline) or DEADLINE > time.time()+86400:
        raise ValueError('Shared deadline must be no more than one day away')
    metadata = json.loads(Path(args.metadata).read_text())
    tokens = metadata.get('tokens', '')
    if metadata.get('schema') != 1 or not tokens or len(set(tokens)) != len(tokens) or \
            metadata.get('dimensions') != 2 * (len(tokens)+1) or metadata.get('epsilon') != .05 or \
            metadata.get('radius') != 4 or len(metadata.get('basis', [])) != 2 or \
            any(len(b) != 80 or not all(math.isfinite(v) for v in b) for b in metadata['basis']):
        raise ValueError('Invalid fixed response-fingerprint metadata')
    expected_basis = []
    for center in (.28, .72):
        distances = [abs(i/79-center)/.2 for i in range(80)]
        expected_basis.append([(1+math.cos(math.pi*d))/2 if d < 1 else 0. for d in distances])
    if not np.allclose(metadata['basis'], expected_basis, rtol=0., atol=1e-12):
        raise ValueError('Spectral perturbation basis differs from the fixed extractor')
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False:
            raise ValueError('Only eligible help-train training-speaker none takes may fit emissions')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or release-held-out speaker')
        if r.get('primarySha256') != metadata['primarySha256']:
            raise ValueError('Cached observations came from a different primary graph')
        if r.get('deadlineMs') != args.deadline:
            raise ValueError('Cached extraction deadline differs from the fitting deadline')
        r['ids'] = [metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        for key, width in [('log_probs',len(metadata['tokens'])+1),('response',metadata['dimensions']),('control',metadata['dimensions'])]:
            r[key] = np.asarray(r[key], dtype=np.float64)
            if r[key].ndim != 2 or r[key].shape[1] != width or not len(r[key]) or not np.isfinite(r[key]).all():
                raise ValueError('Invalid frozen observations')
        if not len(r['log_probs']) == len(r['response']) == len(r['control']):
            raise ValueError('Arms have different acoustic clocks')
        rows.append(r)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three independent training speakers required')
    folds = {s:i%3 for i,s in enumerate(speakers)}
    results = []
    for strength in (1.,16.,64.):
        losses = {key:[] for key in ('response','control')}
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
        item['matchedLoss'] = (item['response']+item['control'])/2
        item['preservationControls'] = controls
        results.append(item)
        print(json.dumps(item), flush=True)
    selected = min(results,key=lambda r:(r['matchedLoss'],r['strength']))
    fitted = {**metadata, **{key:fit(rows,key,len(metadata['tokens'])+1,selected['strength']) for key in ('response','control')}}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    primary = Path(args.primary_bundle)
    manifest = json.loads((primary/'manifest.json').read_text())
    graph = primary/manifest['file']
    if manifest['sha256'] != metadata['primarySha256'] or \
            hashlib.sha256(graph.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Training extractor differs from the bundle primary graph')
    if out.resolve() == primary.resolve():
        raise ValueError('Export to a new directory; never edit the supplied bundle')
    # Copy the complete primary bundle, including the existing seam-insertion teacher
    # and external data. The manifest and primary bytes remain unchanged.
    for file in primary.iterdir():
        if file.is_file() and not file.name.startswith('response-fingerprint-'):
            shutil.copy2(file, out/file.name)
    asset = out/'response-fingerprint-emissions.json'
    asset.write_text(json.dumps(fitted,allow_nan=False))
    files = [dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,)]
    (out/'response-fingerprint-asset.json').write_text(json.dumps(dict(schema=1,files=files)))
    copied = [dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(out.iterdir()) if p.is_file() and not p.name.startswith('response-fingerprint-')]
    report = dict(recordings=len(rows),speakers=speakers,speakerFolds=folds,excludedSpeakers=sorted(excluded),
        validation=results,selected=selected,epochs=EPOCHS,files=files,
        primaryBundleFiles=copied,
        controlLimit='Counterfactual written references on real none takes; not verified scripted substitutions; controls do not select regularization.',
        deadlineMs=args.deadline,primarySha256=manifest['sha256'],
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'response-fingerprint-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--primary-bundle',required=True)
    p.add_argument('--deadline',required=True,type=float,help='Shared extraction/fitting deadline, epoch milliseconds')
    train(p.parse_args())
