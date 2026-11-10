"""CPU pronunciation distributions from path signatures, with a matched time control.

Inputs are 80-band SDK observations on its native 40 ms clock. Only none-labelled
help-train training speakers supply transcripts; no scripted edit is a transcript.
All phone/silence boundaries are marginalized. Model selection never reads practice.
"""
import argparse
import hashlib
import json
from pathlib import Path


def dimensions(bands, order):
    return bands + sum(bands**k for k in range(1, order+1))


def signature(path, order):
    import numpy as np
    bands = path.shape[1]
    levels = [np.ones(1)] + [np.zeros(bands**k) for k in range(1, order+1)]
    for dx in np.diff(path, axis=0):
        levels = extend_signature(levels, dx)
    return np.concatenate([path[0], *levels[1:]])


def extend_signature(levels, dx):
    import numpy as np
    exp = [np.ones(1)]
    for k in range(1,len(levels)):
        exp.append(np.kron(exp[-1],dx)/k)
    return [levels[0]]+[sum(np.kron(levels[j],exp[k-j]) for j in range(k+1))
        for k in range(1,len(levels))]


def control(path, order):
    import numpy as np
    size = dimensions(path.shape[1], order)//path.shape[1]
    positions = np.linspace(0, len(path)-1, size)
    return np.stack([np.interp(positions, np.arange(len(path)), path[:,d])
        for d in range(path.shape[1])], axis=1).ravel()


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if (r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False
                or r['speaker'] in excluded or not r['speaker']):
            raise ValueError('Only none-labelled non-excluded training-speaker takes may be fitted')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != 80 or not np.isfinite(x).all() or len(x)<len(ids) or not ids:
            raise ValueError('Invalid SDK features or impossible complete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return out


def trajectories(rows, bands, order, max_segment, center, scale):
    import numpy as np
    result = []
    for r in rows:
        x = np.stack([r['x'][:,d*80//bands:(d+1)*80//bands].mean(1)
            for d in range(bands)], axis=1)
        x = (x-center)/scale
        y = np.zeros((2, len(x), max_segment+1, dimensions(bands,order)))
        for a in range(len(x)):
            levels = [np.ones(1)] + [np.zeros(bands**k) for k in range(1,order+1)]
            for length in range(1,min(max_segment,len(x)-a)+1):
                path = x[a:a+length]
                if length>1:
                    levels = extend_signature(levels,path[-1]-path[-2])
                y[0,a,length] = np.concatenate([path[0],*levels[1:]])
                y[1,a,length] = control(path,order)
        result.append(dict(r, y=y))
    return result


def lattice(emission, ids, posterior=False):
    """Positive phone durations, optional silence, uniform conditional timing prior."""
    import numpy as np
    states = [0]
    for p in ids:
        states.extend([p,0])
    tmax, max_segment = emission.shape[1], emission.shape[2]-1
    a = np.full((len(states)+1,tmax+1), -np.inf)
    prior = a.copy()
    a[0,0] = prior[0,0] = 0
    for s,p in enumerate(states):
        if p==0:
            a[s+1] = a[s]
            prior[s+1] = prior[s]
        for length in range(1,min(max_segment,tmax)+1):
            a[s+1,length:] = np.logaddexp(a[s+1,length:],
                a[s,:tmax+1-length]+emission[p,:tmax+1-length,length])
            prior[s+1,length:] = np.logaddexp(prior[s+1,length:],prior[s,:tmax+1-length])
    value = a[-1,-1]-prior[-1,-1]
    if not posterior:
        return value
    b = np.full_like(a,-np.inf)
    b[-1,-1] = 0
    for s in range(len(states)-1,-1,-1):
        p=states[s]
        if p==0:
            b[s]=b[s+1]
        for length in range(1,min(max_segment,tmax)+1):
            b[s,:tmax+1-length]=np.logaddexp(b[s,:tmax+1-length],
                emission[p,:tmax+1-length,length]+b[s+1,length:])
    weights = np.zeros_like(emission)
    for s,p in enumerate(states):
        for length in range(1,min(max_segment,tmax)+1):
            weights[p,:tmax+1-length,length] += np.exp(a[s,:tmax+1-length]+
                emission[p,:tmax+1-length,length]+b[s+1,length:]-a[-1,-1])
    return value, weights


def emission(arm, y):
    import numpy as np
    mu, var = np.array(arm['means']), np.array(arm['variances'])
    return -.5*(np.log(2*np.pi*var)[:,None,None,:]+(y[None]-mu[:,None,None,:])**2/
        var[:,None,None,:]).sum(-1)


def update(counts, sums, seconds, shrinkage):
    import numpy as np
    total=max(counts.sum(),1)
    pooled_mu=sums.sum(0)/total
    pooled_second=seconds.sum(0)/total
    mu=(sums+shrinkage*pooled_mu)/(counts[:,None]+shrinkage)
    second=(seconds+shrinkage*pooled_second)/(counts[:,None]+shrinkage)
    # A shared training-derived variance floor keeps deterministic coordinates finite.
    floor=np.maximum(pooled_second-pooled_mu**2,1e-6)*.05
    return dict(means=mu.tolist(),variances=np.maximum(second-mu**2,floor).tolist())


def fit(rows, tokens, bands, order, shrinkage, max_segment=32, epochs=6):
    import numpy as np
    x=np.concatenate([r['x'] for r in rows])
    grouped=np.stack([x[:,d*80//bands:(d+1)*80//bands].mean(1) for d in range(bands)],1)
    center,scale=grouped.mean(0),np.maximum(grouped.std(0),.1)
    data=trajectories(rows,bands,order,max_segment,center,scale)
    n=len(tokens)+1
    arms=[]
    history=[]
    for arm_index in range(2):
        counts=np.zeros(n); sums=np.zeros((n,dimensions(bands,order))); seconds=sums.copy()
        # Uniform phone segmentation initializes only; EM never freezes a boundary.
        for r in data:
            bounds=np.linspace(0,len(r['x']),len(r['ids'])+1).astype(int)
            for p,a,b in zip(r['ids'],bounds[:-1],bounds[1:]):
                length=min(b-a,max_segment)
                z=r['y'][arm_index,a,length]
                counts[p]+=1; sums[p]+=z; seconds[p]+=z*z
            for a in [0,len(r['x'])-1]:
                z=r['y'][arm_index,a,1]
                counts[0]+=1; sums[0]+=z; seconds[0]+=z*z
        arm=update(counts,sums,seconds,shrinkage)
        values=[]
        for _ in range(epochs):
            counts.fill(0); sums.fill(0); seconds.fill(0)
            values=[]
            for r in data:
                y=r['y'][arm_index]
                value,w=lattice(emission(arm,y),r['ids'],True)
                if not np.isfinite(value):
                    raise ValueError('Complete training pronunciation exceeds bounded segment lattice')
                values.append(float(value)/len(r['ids']))
                counts+=w.sum((1,2))
                sums+=np.einsum('ptl,tld->pd',w,y)
                seconds+=np.einsum('ptl,tld->pd',w,y*y)
            arm=update(counts,sums,seconds,shrinkage)
        arms.append(arm); history.append(float(np.mean(values)))
    model=dict(schema=1,tokens=tokens,frameMs=40,bands=bands,order=order,
        center=center.tolist(),scale=scale.tolist(),maxSegment=max_segment,maxFrames=2048,
        counts=np.bincount([p for r in rows for p in r['ids']],minlength=n).tolist(),
        signature=arms[0],control=arms[1])
    return model,history


def validation(model, rows):
    import numpy as np
    data=trajectories(rows,model['bands'],model['order'],model['maxSegment'],model['center'],model['scale'])
    by_speaker={}
    for r in data:
        # Predict the recorded transcript against generic same-length phone contrasts.
        # Predictive contrast loss is comparable across feature dimensionalities;
        # raw continuous density would reward a change of units or dimension.
        digest=hashlib.sha256((''.join(map(str,r['ids']))+r['speaker']).encode()).digest()
        rival=r['ids'].copy()
        at=int.from_bytes(digest[:4],'big')%len(rival)
        available=[p for p in range(1,len(model['counts'])) if model['counts'][p]>0 and p!=rival[at]]
        if not available:
            raise ValueError('Training partition lacks a pronunciation contrast')
        rival[at]=available[int.from_bytes(digest[4:8],'big')%len(available)]
        values=[]
        for i,arm in enumerate(['signature','control']):
            e=emission(model[arm],r['y'][i])
            written=lattice(e,r['ids'])
            alternative=lattice(e,rival)
            values.append(-float(np.logaddexp(0,alternative-written)))
        by_speaker.setdefault(r['speaker'],[]).append(values)
    # Equal speaker and equal arm weights. Both models select one shared configuration.
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def main():
    parser=argparse.ArgumentParser()
    for name in ['input','out','tokens','inventory','excluded-speakers']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    excluded=set(json.loads(Path(args.excluded_speakers).read_text()))
    rows=prepare([json.loads(line) for line in Path(args.input).open()],args.tokens,excluded)
    speakers=sorted({r['speaker'] for r in rows},key=lambda s:hashlib.sha256(s.encode()).digest())
    folds={s:i%3 for i,s in enumerate(speakers)}
    results=[]
    for bands,order in [(2,2),(3,2),(2,3)]:
        for shrinkage in [8,32]:
            values=[]
            for fold in range(3):
                model,_=fit([r for r in rows if folds[r['speaker']]!=fold],args.tokens,bands,order,shrinkage)
                values.append(validation(model,[r for r in rows if folds[r['speaker']]==fold]))
            result=dict(bands=bands,order=order,shrinkage=shrinkage,foldLogP=values,meanLogP=sum(values)/3)
            results.append(result)
            print(json.dumps(result),flush=True)
    selected=max(results,key=lambda r:(r['meanLogP'],-dimensions(r['bands'],r['order']),r['shrinkage']))
    model,history=fit(rows,args.tokens,selected['bands'],selected['order'],selected['shrinkage'])
    model['inventory']=args.inventory
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    asset=out/'gesture-emissions.json'
    asset.write_text(json.dumps(model,allow_nan=False))
    descriptor=dict(schema=1,file=asset.name,bytes=asset.stat().st_size,sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'gesture-asset.json').write_text(json.dumps(descriptor))
    report=dict(recordings=len(rows),speakers=len(speakers),speakerFolds=folds,excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),selected=selected,validation=results,history=history,asset=descriptor)
    (out/'gesture-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
