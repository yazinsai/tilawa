"""CPU-only vowel micropitch EM. Fit exclusively none-labelled training speakers.

Inputs contain raw pitch tracks, not listener outputs. Both arms marginalize all
phone/phase boundaries. Contour separation and shrinkage are selected jointly by
speaker-grouped training-only predictive likelihood, before target inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path

VOWELS = "َُِاۦۥ"
CONSONANTS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي"


def states(pronunciation):
    out = []
    for i, p in enumerate(pronunciation):
        if p not in VOWELS:
            out.append(("other", "other", 1, False))
            continue
        left = pronunciation[i-1] if i>0 and pronunciation[i-1] in CONSONANTS else "_"
        right = pronunciation[i+1] if i+1<len(pronunciation) and pronunciation[i+1] in CONSONANTS else "_"
        for phase in range(3):
            c = left if phase == 0 else right if phase == 2 else "_"
            control = p+":"+str(phase)
            out.append((control+":"+c, control, phase, phase != 1 and c != "_"))
    return out


def residuals(frames, radius, reliable):
    import numpy as np
    out = []
    half = round(radius*100)
    # Match the SDK's upper median, including for even-sized neighbourhoods.
    median = lambda v: sorted(v)[len(v)//2]
    for t, f in enumerate(frames):
        neighbours = [j for j in range(max(0, t-half), min(len(frames), t+half+1))
                      if abs(j-t)>6 and frames[j]['reliability']>=reliable]
        if f['reliability']<reliable or len(neighbours)<8 or not any(j<t for j in neighbours) or not any(j>t for j in neighbours):
            out.append([0., 0.]); continue
        center = median([frames[j]['logPitch'] for j in neighbours])
        spread = max(.02, median([abs(frames[j]['logPitch']-center) for j in neighbours])*3)
        keep = [j for j in neighbours if abs(frames[j]['logPitch']-center)<=spread]
        x = np.array([(j-t)/100 for j in keep]); w = np.array([frames[j]['reliability'] for j in keep])
        y = np.array([frames[j]['logPitch'] for j in keep])
        sw, sx, sy, sxx, sxy = w.sum(), w@x, w@y, w@(x*x), w@(x*y)
        det = sw*sxx-sx*sx
        delta = f['logPitch']-(sy*sxx-sxy*sx)/det if det>1e-8 else 1.
        out.append([delta, 1.] if abs(delta)<=.25 else [0., 0.])
    return np.array(out)


def chain(model, x, pronunciation, conditioned, posterior=False):
    import numpy as np
    ss = states(pronunciation); n = len(ss); tmax = len(x)
    mu = np.array([model['means'][s[0]] if conditioned else model['control'][s[1]] for s in ss])
    y, reliable = x[:, 0]/model['scale'], x[:, 1]
    emission = -.5*(np.log(2*np.pi*model['variance'])+(y[:, None]-mu)**2/model['variance'])*reliable[:, None]-np.log(model['scale'])*reliable[:, None]
    stay = np.log(np.array(model['stay'])[[s[2] for s in ss]])
    step = np.log1p(-np.array(model['stay'])[[s[2] for s in ss]])
    a = np.full((tmax, n), -np.inf); prior = np.full(n, -np.inf)
    a[0, 0] = emission[0, 0]; prior[0] = 0
    for t in range(1, tmax):
        a[t] = np.logaddexp(a[t-1]+stay, np.r_[-np.inf, a[t-1, :-1]+step[:-1]])+emission[t]
        prior = np.logaddexp(prior+stay, np.r_[-np.inf, prior[:-1]+step[:-1]])
    value = a[-1, -1]-prior[-1]
    if not posterior: return float(value)
    if not np.isfinite(value): raise ValueError('No complete pronunciation placement')
    b = np.full_like(a, -np.inf); b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        b[t] = np.logaddexp(stay+emission[t+1]+b[t+1], np.r_[step[:-1]+emission[t+1, 1:]+b[t+1, 1:], -np.inf])
    gamma = np.exp(a+b-a[-1, -1])
    return float(value), gamma


def fit(rows, tokens, inventory, radius, reliable, shrinkage, epochs=5):
    import numpy as np
    rows = [dict(r, x=residuals(r['pitch'], radius, reliable)) for r in rows]
    values = np.concatenate([r['x'][r['x'][:, 1]>0, 0] for r in rows])
    if len(values)<100: raise ValueError('Insufficient reliable training pitch')
    scale = max(float(values.std()), .005)
    controls = ['other']+[v+':'+str(q) for v in VOWELS for q in range(3)]
    keys = ['other']+[v+':'+str(q)+':'+c for v in VOWELS for q in range(3)
                      for c in ('_'+CONSONANTS if q != 1 else '_')]
    model = dict(schema=1, tokens=tokens, inventory=inventory, radius=radius, reliability=reliable,
                 shrinkage=shrinkage, scale=scale, variance=1., stay=[.65, .9, .65],
                 means=dict.fromkeys(keys, 0.), control=dict.fromkeys(controls, 0.), counts=dict.fromkeys(keys, 0.))
    history = []
    # Shared variance and duration law remain fixed: only the identity dependency
    # differs. Means are learned with soft occupancy, including during validation.
    for conditioned in [False, True]:
        if conditioned:
            model['means'] = {k: model['control'][':'.join(k.split(':')[:2])] if k!='other' else model['control'][k] for k in keys}
        for epoch in range(epochs):
            count, sums, likelihoods = {}, {}, []
            for r in rows:
                value, gamma = chain(model, r['x'], r['pronunciation'], conditioned, True)
                likelihoods.append(value/len(r['x']))
                weighted = gamma*r['x'][:, 1, None]
                cc, sy = weighted.sum(0), weighted.T@(r['x'][:, 0]/scale)
                for i, s in enumerate(states(r['pronunciation'])):
                    k = s[0] if conditioned else s[1]
                    count[k] = count.get(k, 0.)+cc[i]; sums[k] = sums.get(k, 0.)+sy[i]
            if conditioned:
                model['means'] = {k: (sums.get(k, 0.)+shrinkage*model['control'][':'.join(k.split(':')[:2]) if k!='other' else k])/(count.get(k, 0.)+shrinkage) for k in keys}
                model['counts'] = {k:float(count.get(k, 0.)) for k in keys}
            else:
                model['control'] = {k:float(sums.get(k, 0.)/(count.get(k, 0.)+shrinkage)) for k in controls}
            history.append(dict(conditioned=conditioned, epoch=epoch, logP=float(np.mean(likelihoods))))
    return model, history


def prepare(records, tokens, excluded):
    import math
    rows = []
    for r in records:
        if r.get('kind')!='none' or r.get('split')!='train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled, non-excluded training speakers may fit')
        p, frames = r['pronunciation'], r['pitch']
        if not p or any(c not in tokens for c in p) or not any(c in VOWELS for c in p):
            raise ValueError('Invalid complete training pronunciation')
        if len(frames)<len(states(p)) or len(frames)>6000 or any(not all(math.isfinite(f[k]) for k in ['logPitch','reliability']) or not 0<=f['reliability']<=1 for f in frames):
            raise ValueError('Invalid pitch track')
        rows.append(r)
    if len({r['speaker'] for r in rows})<3: raise ValueError('At least three training speakers required')
    return rows


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    for key in ['input','out','tokens','inventory','excluded-speakers']: parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s:hashlib.sha256(s.encode()).digest())
    folds = {s:i%3 for i,s in enumerate(speakers)}
    validation = []
    # Frozen finite grid; never receives practice paths or incident identifiers.
    for radius, reliable in [(.2,.8),(.35,.8),(.35,.9)]:
        for shrinkage in [16.,64.]:
            scores = []
            for fold in range(3):
                model,_ = fit([r for r in rows if folds[r['speaker']]!=fold], args.tokens,args.inventory,radius,reliable,shrinkage,3)
                grouped = {}
                for r in rows:
                    if folds[r['speaker']]!=fold: continue
                    x = residuals(r['pitch'],radius,reliable)
                    score = sum(chain(model,x,r['pronunciation'],c) for c in [False,True])/2/max(1,x[:,1].sum())
                    grouped.setdefault(r['speaker'],[]).append(score)
                scores.append(float(np.mean([np.mean(v) for v in grouped.values()])))
            validation.append(dict(radius=radius,reliability=reliable,shrinkage=shrinkage,foldLogP=scores,logP=float(np.mean(scores))))
    chosen = max(validation,key=lambda v:(v['logP'],v['shrinkage']))
    model,history = fit(rows,args.tokens,args.inventory,chosen['radius'],chosen['reliability'],chosen['shrinkage'])
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    asset = out/'micropitch-emissions.json';asset.write_text(json.dumps(model,allow_nan=False))
    descriptor = dict(schema=1,file=asset.name,bytes=asset.stat().st_size,sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'micropitch-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows),speakers=len(speakers),speakerFolds=folds,excludedSpeakers=sorted(excluded),
                  inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),validation=validation,selected=chosen,history=history,asset=descriptor)
    (out/'micropitch-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
