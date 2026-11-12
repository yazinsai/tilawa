"""Paired-observation errors-in-variables phone likelihood; CPU fitting only.

x_v = y + e_v, y|phone ~ N(mu, Sigma), independent e_v ~ N(0, N).
N is estimated from identical-occurrence replicate differences. The constrained
EM uses posterior latent sufficient statistics, never covariance subtraction.
Only authorized training speakers select projection/regularization.
"""
import argparse
import hashlib
import json
from pathlib import Path

EPOCHS = 6
FLOOR = .01

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


def positive(matrix, floor=FLOOR):
    import numpy as np
    values, vectors = np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values, floor))@vectors.T


def gaussian(x, mean, covariance):
    import numpy as np
    l = np.linalg.cholesky(covariance)
    z = np.linalg.solve(l, (x-mean).T).T
    return -.5*(x.shape[1]*np.log(2*np.pi)+2*np.log(l.diagonal()).sum()+(z*z).sum(1))


def latent_posterior(average, mean, covariance, noise):
    """Exact y | (x0+x1)/2; the difference is phone independent."""
    import numpy as np
    gain = np.linalg.solve(covariance+noise/2, covariance).T
    expectation = mean+(average-mean)@gain.T
    uncertainty = positive(covariance-gain@covariance, 1e-10)
    return expectation, uncertainty


def projection(rows, dimensions):
    import numpy as np
    width = rows[0]['view0'].shape[1]
    if not 0 < dimensions <= width:
        raise ValueError('Invalid projection dimension')
    center = np.mean([(r['view0'].mean(0)+r['view1'].mean(0))/2 for r in rows], axis=0)
    covariance = sum(((r['view0']-center).T@(r['view0']-center)+
        (r['view1']-center).T@(r['view1']-center))/(2*len(r['view0'])*len(rows)) for r in rows)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:dimensions]
    matrix = (vectors[:,order]/np.sqrt(np.maximum(values[order], FLOOR))).T
    return dict(featureCenter=center.tolist(), projection=matrix.tolist(), dimensions=dimensions,
        trainingRecordings=len(rows))


def projected(rows, mapping):
    import numpy as np
    matrix = np.asarray(mapping['projection'])
    return [{**r, **{k:(r[k]-mapping['featureCenter'])@matrix.T for k in ('view0','view1','clean')}} for r in rows]


def noise_covariance(rows):
    """E[(x0-x1)(x0-x1)' / 2] = N, equal recording weight.

    Do not subtract temporal means: an independent view's recording-wide offset
    is observation noise too. Native noise shared by views stays in latent speech.
    """
    import numpy as np
    if not rows or any(r['view0'].shape != r['view1'].shape for r in rows):
        raise ValueError('Replicates require identical occurrence clocks')
    return positive(sum((r['view0']-r['view1']).T@(r['view0']-r['view1'])/
        (2*len(r['view0'])*len(rows)) for r in rows), 1e-6)


def regularized(count, sums, cross, strength, prior_mean, prior_covariance):
    import numpy as np
    means, covariances = [], []
    for n, total, moment in zip(count, sums, cross):
        mean = (total+strength*prior_mean)/(n+strength)
        scatter = moment-np.outer(total,mean)-np.outer(mean,total)+n*np.outer(mean,mean)
        delta = prior_mean-mean
        covariance = positive((scatter+strength*(prior_covariance+np.outer(delta,delta)))/(n+strength))
        means.append(mean); covariances.append(covariance)
    return np.asarray(means), np.asarray(covariances)


def fit(rows, vocab, strength, noise, latent):
    import numpy as np
    d = rows[0]['view0'].shape[1]
    raw = np.concatenate([r[k] for r in rows for k in ('view0','view1')])
    prior_mean = raw.mean(0)
    prior_covariance = positive(np.atleast_2d(np.cov(raw, rowvar=False, bias=True)))
    initial = [boundary(r['log_probs'],r['ids'],True)[1] for r in rows]
    def moments(posts, parameters=None):
        count=np.zeros(vocab); sums=np.zeros((vocab,d)); cross=np.zeros((vocab,d,d))
        for r, weights in zip(rows,posts):
            if parameters is None or not latent:
                # Both views are exact observations in the ordinary control.
                for x in (r['view0'],r['view1']):
                    count+=weights.sum(0); sums+=weights.T@x
                    cross+=np.einsum('tp,ti,tj->pij',weights,x,x)
            else:
                means,covariances=parameters
                average=(r['view0']+r['view1'])/2
                for p in range(vocab):
                    expectation,uncertainty=latent_posterior(average,means[p],covariances[p],noise)
                    w=weights[:,p]; n=w.sum()
                    # Two-view weight keeps the regularization strength matched.
                    count[p]+=2*n; sums[p]+=2*(w@expectation)
                    cross[p]+=2*((expectation*w[:,None]).T@expectation+n*uncertainty)
        return count,sums,cross
    count,sums,cross=moments(initial)
    parameters=regularized(count,sums,cross,strength,prior_mean,prior_covariance)
    for _ in range(EPOCHS):
        posts=[]
        for r in rows:
            if latent:
                average=(r['view0']+r['view1'])/2
                difference=r['view0']-r['view1']
                neutral=gaussian(difference,np.zeros(d),2*noise)
                values=np.array([gaussian(average,m,c+noise/2)+neutral for m,c in zip(*parameters)]).T
            else:
                values=np.array([gaussian(r['view0'],m,c)+gaussian(r['view1'],m,c) for m,c in zip(*parameters)]).T
            posts.append(boundary(values,r['ids'],True)[1])
        count,sums,cross=moments(posts,parameters)
        parameters=regularized(count,sums,cross,strength,prior_mean,prior_covariance)
    phones=[]
    for n,mean,covariance in zip(count,*parameters):
        l=np.linalg.cholesky(covariance)
        phones.append(dict(mean=mean.tolist(),precision=np.tril(np.linalg.inv(l)).tolist(),
            logDet=float(2*np.log(l.diagonal()).sum()),count=float(n)))
    return dict(center=[0.]*d,scale=[1.]*d,phones=phones)


def emissions(arm, x):
    import numpy as np
    columns=[]
    for e in arm['phones']:
        z=(x-e['mean'])@np.asarray(e['precision']).T
        columns.append(-.5*(x.shape[1]*np.log(2*np.pi)+e['logDet']+(z*z).sum(1)))
    return np.array(columns).T


def validation_loss(arm, rows):
    import numpy as np
    grouped={}
    for r in rows:
        values=emissions(arm,r['clean'])[:,1:]
        weights=boundary(r['log_probs'],r['ids'],True)[1][:,1:]
        peak=values.max(1,keepdims=True)
        probabilities=values-peak-np.log(np.exp(values-peak).sum(1,keepdims=True))
        loss=-float((weights*probabilities).sum()/max(weights.sum(),1e-12))
        grouped.setdefault(r['speaker'],[]).append(loss)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def controls(arms, rows, tokens):
    total=0; reversals=0
    for r in rows:
        values={k:emissions(arm,r['clean']) for k,arm in arms.items()}
        spoken={k:boundary(v,r['ids']) for k,v in values.items()}
        for text in r.get('controlPronunciations',[]):
            ids=[tokens.index(p)+1 for p in text]
            alternative={k:boundary(v,ids) for k,v in values.items()}
            total+=1
            reversals+=alternative['latent']>spoken['latent'] and alternative['ordinary']<spoken['ordinary']
    return dict(comparisons=total,incorrectVetoes=int(reversals))


def train(args):
    import numpy as np
    import time
    metadata=json.loads(Path(args.metadata).read_text())
    def budget():
        if time.time()-metadata['budgetStartedAt']>args.max_seconds:
            raise TimeoutError('One-day extraction and fitting budget exceeded')
    budget()
    excluded=set(json.loads(Path(args.excluded_speakers).read_text()))
    rows=[]
    for line in Path(args.input).read_text().splitlines():
        budget();r=json.loads(line)
        if r.get('source')!='help-train' or r.get('kind')!='none' or r.get('split')!='train' or r.get('excluded') is not False:
            raise ValueError('Only authorized none-labelled training takes may fit emissions')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or release-held-out speaker')
        r['ids']=[metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        for key,width in [('view0',metadata['width']),('view1',metadata['width']),('clean',metadata['width']),('log_probs',len(metadata['tokens'])+1)]:
            r[key]=np.asarray(r[key],dtype=np.float64)
            if r[key].ndim!=2 or r[key].shape[1]!=width or not len(r[key]) or not np.isfinite(r[key]).all():
                raise ValueError('Invalid frozen observations')
        if len({len(r[k]) for k in ('view0','view1','clean','log_probs')})!=1:
            raise ValueError('Replicates have different clocks')
        rows.append(r)
    speakers=sorted({r['speaker'] for r in rows})
    if len(speakers)<3:
        raise ValueError('At least three independent training speakers required')
    folds={s:i%3 for i,s in enumerate(speakers)}
    partitions=[]
    for fold in range(3):
        training=[r for r in rows if folds[r['speaker']]!=fold]
        held=[r for r in rows if folds[r['speaker']]==fold]
        mapping=projection(training,args.dimensions)
        training=projected(training,mapping);held=projected(held,mapping)
        partitions.append((training,held,noise_covariance(training)))
    results=[]
    vocab=len(metadata['tokens'])+1
    for strength in (1.,16.,64.):
        losses={k:[] for k in ('ordinary','latent')};preservation=[]
        for training,held,noise in partitions:
            budget()
            arms={k:fit(training,vocab,strength,noise,k=='latent') for k in losses}
            for k,arm in arms.items():losses[k].append(validation_loss(arm,held))
            preservation.append(controls(arms,held,metadata['tokens']))
        item=dict(strength=strength,**{k:float(np.mean(v)) for k,v in losses.items()},preservation=preservation)
        item['matchedLoss']=(item['ordinary']+item['latent'])/2
        results.append(item);print(json.dumps(item),flush=True)
    selected=min(results,key=lambda r:(r['matchedLoss'],r['strength']))
    mapping=projection(rows,args.dimensions);training=projected(rows,mapping);noise=noise_covariance(training)
    fitted={**metadata,**mapping,'noiseCovariance':noise.tolist(),
        **{k:fit(training,vocab,selected['strength'],noise,k=='latent') for k in ('ordinary','latent')}}
    budget();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    extractor=out/'deconvolved-phone.onnx'
    if not extractor.is_file():raise ValueError('Verified extractor missing')
    asset=out/'deconvolved-phone-emissions.json';asset.write_text(json.dumps(fitted,allow_nan=False))
    files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset,extractor)]
    (out/'deconvolved-phone-asset.json').write_text(json.dumps(dict(schema=1,files=files)))
    # Shared nonlinear encoder bias is explicitly measured, never called removable noise.
    bias=np.mean([((r['view0']+r['view1'])/2-r['clean']).mean(0) for r in training],axis=0)
    report=dict(recordings=len(rows),speakers=speakers,speakerFolds=folds,excludedSpeakers=sorted(excluded),
        validation=results,selected=selected,epochs=EPOCHS,dimensions=args.dimensions,files=files,
        meanReplicateBias=bias.tolist(),noiseTrace=float(np.trace(noise)),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        limitations='Independent additive view errors are approximated as zero-mean Gaussian in encoder space. Native recording noise is shared and remains latent. Clean runtime receives no added training degradation. Counterfactual lexical controls are diagnostic only.')
    (out/'deconvolved-phone-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',required=True);p.add_argument('--input',required=True)
    p.add_argument('--excluded-speakers',required=True);p.add_argument('--out',required=True)
    p.add_argument('--dimensions',type=int,default=12);p.add_argument('--max-seconds',type=int,default=23*3600)
    train(p.parse_args())
