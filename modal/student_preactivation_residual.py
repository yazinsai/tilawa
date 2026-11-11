"""Training-only frozen-activation residual phone likelihood. No primary weights change."""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 16
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
        phones.append(dict(mean=mean.tolist(), precision=np.tril(np.linalg.inv(l)).tolist(),
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


def design(post, site, predictor):
    import numpy as np
    z = (post-predictor['center'])/predictor['scale']
    return np.concatenate([np.ones((len(z), 1)), z,
        np.tanh(z @ np.array(site['nonlinearProjection']).T)], axis=1)


def predictor_fit(rows, site, number, strength):
    import numpy as np
    post = np.concatenate([r['pairs'][number]['post'] for r in rows])
    center, scale = post.mean(0), np.maximum(post.std(0), .01)
    predictor = dict(center=center.tolist(), scale=scale.tolist(), strength=strength)
    width = site['width']+65
    gram, cross = np.zeros((width, width)), np.zeros((width, 8))
    for r in rows:
        pair = r['pairs'][number]
        x = design(pair['post'], site, predictor)
        y = pair['pre'] @ np.array(site['preProjection']).T
        # Every recording has equal regression weight, regardless of duration.
        gram += x.T @ x/len(x)/len(rows)
        cross += x.T @ y/len(x)/len(rows)
    penalty = np.eye(width)*strength
    penalty[0, 0] = 0
    predictor['coefficients'] = np.linalg.solve(gram+penalty, cross).T.tolist()
    return predictor


def observations(rows, site, number, predictor):
    import numpy as np
    result = []
    for r in rows:
        pair = r['pairs'][number]
        target = pair['pre'] @ np.array(site['preProjection']).T
        residual = target-design(pair['post'], site, predictor) @ np.array(predictor['coefficients']).T
        control = pair['post'] @ np.array(site['postProjection']).T
        ratio = len(control)//len(r['log_probs'])
        def native(v):
            return v.reshape(len(r['log_probs']), ratio, -1).mean(1)
        result.append({**r, 'residual': native(np.concatenate([control[:, :8], residual], axis=1)),
            'control': native(control)})
    return result


def validation_loss(arm, rows, key):
    import numpy as np
    grouped = {}
    for r in rows:
        values = emissions(arm, (r[key]-arm['center'])/arm['scale'])
        # Compare sites using conditional identity CE, not raw Gaussian density
        # in different feature coordinates. Targets marginalize canonical timing.
        weights = boundary(r['log_probs'], r['ids'], True)[1][:, 1:]
        logits = values[:, 1:]
        peak = logits.max(1, keepdims=True)
        log_probs = logits-peak-np.log(np.exp(logits-peak).sum(1, keepdims=True))
        loss = -float((weights*log_probs).sum()/max(weights.sum(), 1e-12))
        grouped.setdefault(r['speaker'], []).append(loss)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def train(args):
    import numpy as np
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False:
            raise ValueError('Only explicitly eligible help-train training-speaker none takes are accepted')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or release-held-out speaker')
        r['log_probs'] = np.array(r['log_probs'], dtype=float)
        r['ids'] = [metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        if r['log_probs'].ndim != 2 or not len(r['log_probs']) or r['log_probs'].shape[1] != len(metadata['tokens'])+1 or not np.isfinite(r['log_probs']).all():
            raise ValueError('Invalid parent rows')
        if len(r['pairs']) != len(metadata['sites']):
            raise ValueError('Missing diagnostic activation site')
        for site, pair in zip(metadata['sites'], r['pairs']):
            pair.update({k: np.array(pair[k], dtype=float) for k in ('pre','post')})
            if pair['pre'].shape != pair['post'].shape or pair['pre'].ndim != 2 or pair['pre'].shape[1] != site['width'] or \
                len(pair['pre']) < len(r['log_probs']) or len(pair['pre']) % len(r['log_probs']) or \
                not all(np.isfinite(pair[k]).all() for k in ('pre','post')):
                raise ValueError('Invalid actual activation pairs')
        rows.append(r)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three independent training speakers are required')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    # Joint choice of site and regression regularization, frozen before targets.
    # Both arms use the same fixed Gaussian shrinkage and update count.
    for number, site in enumerate(metadata['sites']):
        for strength in (.1, 10.):
            losses, control_losses, mse = [], [], []
            for fold in range(3):
                training = [r for r in rows if folds[r['speaker']] != fold]
                held = [r for r in rows if folds[r['speaker']] == fold]
                predictor = predictor_fit(training, site, number, strength)
                tr, va = observations(training, site, number, predictor), observations(held, site, number, predictor)
                arms = {key: fit(tr, key, len(metadata['tokens'])+1, 16) for key in ('residual','control')}
                losses.append(validation_loss(arms['residual'], va, 'residual'))
                control_losses.append(validation_loss(arms['control'], va, 'control'))
                for r in held:
                    pair = r['pairs'][number]
                    error = pair['pre'] @ np.array(site['preProjection']).T-design(pair['post'], site, predictor) @ np.array(predictor['coefficients']).T
                    mse.append(float(np.mean(error**2)))
            item = dict(site=number, predictorStrength=strength, identityLoss=float(np.mean(losses)),
                controlIdentityLoss=float(np.mean(control_losses)), predictionMse=float(np.mean(mse)))
            results.append(item)
            print(json.dumps(item), flush=True)
    selected = min(results, key=lambda v: (v['identityLoss'], v['site'], v['predictorStrength']))
    number, strength = selected['site'], selected['predictorStrength']
    predictor = predictor_fit(rows, metadata['sites'][number], number, strength)
    prepared = observations(rows, metadata['sites'][number], number, predictor)
    model = {**metadata, 'site': number, 'predictor': predictor,
        **{key: fit(prepared, key, len(metadata['tokens'])+1, 16) for key in ('residual','control')}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = out/'preactivation-residual.onnx'
    if not extractor.is_file():
        raise ValueError('Verified diagnostic graph missing from output bundle')
    asset = out/'preactivation-residual-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,extractor)])
    (out/'preactivation-residual-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=speakers, speakerFolds=folds, excludedSpeakers=sorted(excluded),
        validation=results, selected=selected, gaussianStrength=16, epochs=EPOCHS,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), files=descriptor['files'])
    (out/'preactivation-residual-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
