"""Frozen-encoder local orbit metric, fitted only on authorized help-train speakers.

Matched SpecAugment draws retain the same native acoustic frame coordinates.
An SPD metric field discounts within-occurrence changes and penalizes collapse
between distinct transcript-supported phones. Projection, geometry and selection
are rebuilt inside each speaker fold; practice audio never enters fitting.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


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
        return (-np.inf, np.zeros_like(values)) if posterior else -np.inf
    cost = np.log([len(e) for e in edges])
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    def forward(prev):
        out = prev-cost
        cur = out.copy()
        cur[1:] = np.logaddexp(cur[1:], out[:-1])
        cur[2:] = np.logaddexp(cur[2:], np.where(skip, out[:-2], -np.inf))
        return cur
    a = np.full((tmax, n), -np.inf)
    prior = np.full(n, -np.inf)
    a[0, :2] = values[0, labels[:2]]
    prior[:2] = 0
    for t in range(1, tmax):
        a[t] = forward(a[t-1])+values[t, labels]
        prior = forward(prior)
    z = np.logaddexp(*a[-1, -2:])
    timing = np.logaddexp(*prior[-2:])
    result = float(z-timing) if np.isfinite(z) and np.isfinite(timing) else -np.inf
    if not posterior:
        return result
    if not np.isfinite(z):
        raise ValueError('Canonical pronunciation cannot complete on the recording clock')
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        v = b[t+1]+values[t+1, labels]
        cur = v.copy()
        cur[:-1] = np.logaddexp(cur[:-1], v[1:])
        cur[:-2] = np.logaddexp(cur[:-2], np.where(skip, v[2:], -np.inf))
        b[t] = cur-cost
    occupancy = np.zeros_like(values)
    for s, p in enumerate(labels):
        occupancy[:, p] += np.exp(a[:, s]+b[:, s]-z)
    return result, occupancy


def encode(session, manifest, metadata, features):
    """SDK-exact chunk, state, padding and end clock; extractor preserves primary graph."""
    import numpy as np
    states = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype']=='int64' else np.float32)
              for s in manifest['states']}
    names = ['log_probs', metadata['featureOutput']] + ['new_'+s['name'] for s in manifest['states']]
    total = max(0, ((len(features)-7)//2+1)//2)
    h, lp, collected = [], [], 0
    for start in range(0, len(features), manifest['shift']):
        if collected >= total:
            break
        x = np.full((1, manifest['window'], manifest['fbank']['mels']), manifest['fbank']['padValue'], np.float32)
        size = min(manifest['window'], len(features)-start)
        x[0, :size] = features[start:start+size]
        out = session.run(names, dict(x=x, **states))
        lp.append(out[0].reshape(-1, len(manifest['tokens'])+1))
        h.append(out[1].reshape(-1, metadata['width']))
        if len(h[-1]) != len(lp[-1]): raise ValueError('Feature/primary clock mismatch')
        collected += len(h[-1])
        states = {s['name']: v for s, v in zip(manifest['states'], out[2:])}
    return np.concatenate(h)[:total], np.concatenate(lp)[:total]


def prepare(args):
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import torchaudio
    from student_train import spec_augment
    metadata = json.loads(Path(args.metadata).read_text())
    manifest = json.loads((Path(args.model)/'manifest.json').read_text())
    if metadata['primarySha256'] != manifest['sha256']:
        raise ValueError('Extractor does not belong to this primary')
    labels = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line]
    # Exclude an entire speaker if ANY of their takes belongs to a non-training split.
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    excluded.update(r['speaker'] for r in labels if r.get('split') != 'train' and r.get('speaker'))
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(Path(args.metadata).parent/'acoustic-orbit.onnx'), options,
                                   providers=['CPUExecutionProvider'])
    torch.set_num_threads(1)
    rows = []
    reference = json.loads(Path(args.reference).read_text())
    # Canonical training transcripts are supplied by SDK preparation, never mistake labels.
    transcripts = reference['transcripts']
    for r in labels:
        if time.time()-metadata['budgetStartedAt'] > 23*3600:
            raise TimeoutError('One-day budget exceeded')
        if (r.get('kind') != 'none' or r.get('split') != 'train' or not r.get('speaker') or
            r['speaker'] in excluded or r.get('exclude') or r.get('excluded') or
            r.get('scored') is False or r.get('events') or r.get('non_errors')):
            continue
        transcript = transcripts.get(r['id'])
        if not transcript:
            raise ValueError('Missing authorized canonical transcript')
        wav, rate = sf.read(Path(args.labels).parent/r['wav'], dtype='float32')
        if rate != 16000 or wav.ndim != 1 or not np.isfinite(wav).all():
            raise ValueError('Only mono 16 kHz takes are authorized')
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wav).unsqueeze(0), num_mel_bins=80,
                    sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=-400)
        clean, lp = encode(session, manifest, metadata, feats.numpy())
        views = [clean]
        # Existing shipped training augmentation, not a newly designed practice channel.
        rng = np.random.default_rng(1729+len(rows))
        for _ in range(3):
            h, _ = encode(session, manifest, metadata, spec_augment(feats, rng).numpy())
            if h.shape != clean.shape:
                raise ValueError('Augmentation changed occurrence correspondence')
            views.append(h)
        ids = [metadata['tokens'].index(p)+1 for p in transcript['pronunciation']]
        _, occupancy = boundary(lp, ids, True)
        row = dict(source='help-train', kind='none', split='train', excluded=False, speaker=r['speaker'], id=r['id'],
                   **transcript, views=np.asarray(views), log_probs=lp, occupancy=occupancy)
        rows.append(row)
        print(json.dumps(dict(prepared=len(rows), frames=len(clean))), flush=True)
    # Pickle-free dense arrays, with text metadata in a separate JSON entry.
    arrays = {}
    records = []
    for i, r in enumerate(rows):
        records.append({k:v for k,v in r.items() if k not in ('views', 'log_probs', 'occupancy')})
        for k in ('views', 'log_probs', 'occupancy'):
            arrays[f'{i}_{k}'] = r[k]
    arrays['records'] = np.array(json.dumps(records))
    arrays['excluded'] = np.array(json.dumps(sorted(excluded)))
    np.savez_compressed(args.input, **arrays)


def mapping(rows, dimensions):
    import numpy as np
    center = np.mean([r['views'][0].mean(0) for r in rows], axis=0)
    covariance = sum((r['views'][0]-center).T@(r['views'][0]-center)/len(r['views'][0]) for r in rows)/len(rows)
    eig, basis = np.linalg.eigh(covariance)
    projection = basis[:, np.argsort(eig)[::-1][:dimensions]].T
    # One scalar unit scale preserves Euclidean geometry; no global whitening here.
    unit = max(float(np.trace(projection@covariance@projection.T)/dimensions), 1e-8)**.5
    return center, projection/unit


def neighborhoods(x, centers, scale):
    import numpy as np
    distances = ((x[:, None]-centers[None])**2).sum(2)
    nearest = np.argsort(distances, axis=1, kind='stable')[:, :3]
    mask = np.zeros_like(distances)
    np.put_along_axis(mask, nearest, 1., axis=1)
    weights = np.exp(-(distances-distances.min(1, keepdims=True))/(2*scale))*mask
    return weights/weights.sum(1, keepdims=True)


def fit(rows, dimensions, regularization, steps=120):
    import numpy as np
    import torch
    if not rows or not 1 <= dimensions <= min(32, rows[0]["views"].shape[2]):
        raise ValueError("Invalid compact metric dimension")
    center, projection = mapping(rows, dimensions)
    projected = [{**r, 'views':(r['views']-center)@projection.T} for r in rows]
    vocab = rows[0]['occupancy'].shape[1]
    counts = sum(r['occupancy'].sum(0)/len(r['occupancy']) for r in projected)
    means = sum(r['occupancy'].T@r['views'].mean(0)/len(r['occupancy']) for r in projected)/np.maximum(counts[:, None], 1e-8)
    # Equal recording contribution, matched original acoustic coordinates across views.
    samples = [np.linspace(0,len(r['views'][0])-1,160,dtype=int) for r in projected]
    x = np.concatenate([r['views'][0][at] for r,at in zip(projected,samples)])
    deltas = np.concatenate([(r['views'][1:]-r['views'][0])[:,at,:].transpose(1,0,2)
                            for r,at in zip(projected,samples)])
    occupancy = np.concatenate([r['occupancy'][at] for r,at in zip(projected,samples)])
    rng = np.random.default_rng(1729)
    centers = x[rng.choice(len(x), min(16,len(x)), replace=False)].copy()
    for _ in range(12):
        groups = ((x[:,None]-centers[None])**2).sum(2).argmin(1)
        for k in range(len(centers)):
            if np.any(groups==k): centers[k] = x[groups==k].mean(0)
    scale = max(float(np.median(((x[:,None]-centers[None])**2).sum(2).min(1))), 1e-4)
    weights = neighborhoods(x,centers,scale)
    # Explicit distinct-phone acoustic contrasts; labels come only from the canonical soft alignment.
    confident = np.flatnonzero((occupancy.max(1)>.7)&(occupancy.argmax(1)!=0))
    first = rng.choice(confident, min(4096,len(confident)), replace=True) if len(confident) else np.array([],dtype=int)
    second = rng.choice(confident, len(first), replace=True) if len(confident) else first
    valid = occupancy[first].argmax(1)!=occupancy[second].argmax(1)
    first,second = first[valid],second[valid]
    if len(first)<32:
        raise ValueError('Insufficient transcript-supported different-phone contrasts')
    negative = x[first]-x[second]
    margin = np.minimum((negative**2).sum(1), float(dimensions))*.5
    td = torch.tensor(deltas, dtype=torch.float64)
    tw = torch.tensor(weights, dtype=torch.float64)
    tn = torch.tensor(negative, dtype=torch.float64)
    nw = torch.tensor((weights[first]+weights[second])/2, dtype=torch.float64)
    target = torch.tensor(margin, dtype=torch.float64)
    raw = torch.nn.Parameter(torch.zeros((len(centers),dimensions,dimensions),dtype=torch.float64))
    optimizer = torch.optim.Adam([raw], lr=.025)
    for _ in range(steps):
        # Positive diagonal triangular factor; never an indefinite discount matrix.
        factor = torch.tril(raw,-1)+torch.diag_embed(torch.exp(raw.diagonal(dim1=1,dim2=2)))
        metric = factor.transpose(1,2)@factor
        orbit = torch.einsum('nvd,kde,nve,nk->',td,metric,td,tw)/(len(x)*deltas.shape[1])
        distance = torch.einsum('nd,kde,ne,nk->n',tn,metric,tn,nw)
        contrast = torch.nn.functional.softplus(target-distance).mean()
        penalty = (metric.diagonal(dim1=1,dim2=2).sum(1)-2*raw.diagonal(dim1=1,dim2=2).sum(1)-dimensions).mean()
        loss = orbit+contrast+regularization*penalty
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    residual = sum(sum((view-means[p]).T@((view-means[p])*r['occupancy'][:,p,None])
                        for p in range(vocab))/len(view)
                   for r in projected for view in r['views'])/(len(projected)*len(projected[0]['views']))
    residual += np.eye(dimensions)*max(float(np.trace(residual)/dimensions)*.05,1e-4)
    global_factor = np.linalg.inv(np.linalg.cholesky(residual))
    local = []
    with torch.no_grad():
        factor = torch.tril(raw,-1)+torch.diag_embed(torch.exp(raw.diagonal(dim1=1,dim2=2)))
    for f in factor.detach().numpy():
        precision = f.T@f
        local.append(np.linalg.inv(np.linalg.cholesky(np.linalg.inv(precision))))
    fitted = dict(featureCenter=center.tolist(),projection=projection.tolist(),dimensions=dimensions,
                trainingRecordings=len(rows),means=means.tolist(),counts=counts.tolist(),
                globalPrecision=global_factor.tolist(),regions=[dict(center=c.tolist(),precision=f.tolist()) for c,f in zip(centers,local)],
                neighborhoodScale=scale,temperature=1.)
    # Refit the shared pronunciation means with boundaries marginalized under
    # the average of the three matched geometries and all same-clock views.
    # No arm receives a different transcript, timing model or prototype budget.
    for _ in range(2):
        normal = np.zeros((vocab,dimensions,dimensions))
        rhs = np.zeros((vocab,dimensions)); effective = np.zeros(vocab)
        for original, r in zip(rows,projected):
            values = sum(emissions(fitted,view,arm) for view in original['views']
                         for arm in ('local','euclidean','global'))/(3*len(r['views']))
            _, post = boundary(values,r['ids'],True)
            effective += post.sum(0)/len(post)
            for view in r['views']:
                weights = neighborhoods(view,centers,scale)
                local_metric = sum(weights[:,k,None,None]*(np.asarray(region['precision']).T@np.asarray(region['precision']))
                                   for k,region in enumerate(fitted['regions']))
                shared = (local_metric + global_factor.T@global_factor + np.eye(dimensions))/3
                mass = post/(len(post)*len(r['views']))
                normal += np.einsum('np,nij->pij',mass,shared)
                rhs += np.einsum('np,nij,nj->pi',mass,shared,view)
        ridge = 1e-4*np.eye(dimensions)
        means = np.array([np.linalg.solve(n+ridge,b+ridge@m) for n,b,m in zip(normal,rhs,means)])
        fitted.update(means=means.tolist(),counts=effective.tolist())
    return fitted


def emissions(model, raw, arm):
    import numpy as np
    x = (raw-model['featureCenter'])@np.asarray(model['projection']).T
    delta = x[:,None]-np.asarray(model['means'])[None]
    if arm=='euclidean': distance = (delta**2).sum(2)
    elif arm=='global': distance = ((delta@np.asarray(model['globalPrecision']).T)**2).sum(2)
    else:
        centers=np.asarray([r['center'] for r in model['regions']])
        weights=neighborhoods(x,centers,model['neighborhoodScale'])
        distance=sum(((delta@np.asarray(r['precision']).T)**2).sum(2)*weights[:,k,None] for k,r in enumerate(model['regions']))
    values=-.5*distance/model['temperature']
    values[:,np.asarray(model['counts'])<=0]=-np.inf
    return values-np.logaddexp.reduce(values,axis=1)[:,None]


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata=json.loads(Path(args.metadata).read_text())
    def budget():
        if time.time()-metadata['budgetStartedAt']>23*3600:raise TimeoutError('One-day budget exceeded')
    archive=np.load(args.input,allow_pickle=False)
    excluded=set(json.loads(str(archive['excluded'])))|set(json.loads(Path(args.excluded_speakers).read_text()))
    rows=[]
    for i,r in enumerate(json.loads(str(archive['records']))):
        if r.get('source')!='help-train' or r.get('kind')!='none' or r.get('split')!='train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Unauthorized training take')
        for k in ('views','log_probs','occupancy'):r[k]=archive[f'{i}_{k}'].astype(np.float64)
        if r['views'].ndim!=3 or r['views'].shape[0]!=4 or r['views'].shape[2]!=metadata['width'] or len(r['occupancy'])!=r['views'].shape[1] or not all(np.isfinite(r[k]).all() for k in ('views','log_probs','occupancy')):
            raise ValueError('Invalid occurrence correspondence')
        if (r['log_probs'].shape != r['occupancy'].shape or r['occupancy'].shape[1] != len(metadata['tokens'])+1 or
            np.any(r['occupancy']<0) or not np.allclose(r['occupancy'].sum(1),1.,atol=1e-6)):
            raise ValueError('Invalid canonical soft phone occupancy')
        r['ids']=[metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        if not r['ids']: raise ValueError('Empty canonical pronunciation')
        rows.append(r)
    speakers=sorted({r['speaker'] for r in rows})
    if len(speakers)<3:raise ValueError('At least three independent training speakers required')
    folds={s:i%3 for i,s in enumerate(speakers)}
    validation=[]
    for regularization in (.1,1.,10.):
        losses=[];controls=[]
        for fold in range(3):
            budget()
            fitted=fit([r for r in rows if folds[r['speaker']]!=fold],args.dimensions,regularization)
            grouped={}
            for r in rows:
                if folds[r['speaker']]!=fold:continue
                v=emissions(fitted,r['views'][0],'local')
                known=np.asarray(fitted['counts'])>0
                mass=r['occupancy'][:,known]
                loss=-float((mass*v[:,known]).sum()/max(mass.sum(),1e-12))
                grouped.setdefault(r['speaker'],[]).append(loss)
                ids=[metadata['tokens'].index(p)+1 for p in r['pronunciation']]
                values={a:emissions(fitted,r['views'][0],a) for a in ('local','euclidean','global')}
                spoken={a:boundary(v,ids) for a,v in values.items()}
                for text in r.get('controlPronunciations',[]):
                    alt=[metadata['tokens'].index(p)+1 for p in text]
                    rival={a:boundary(v,alt) for a,v in values.items()}
                    controls.append(rival['local']>spoken['local'] and all(rival[a]<spoken[a] for a in ('euclidean','global')))
            losses.extend(float(np.mean(v)) for v in grouped.values())
        item=dict(regularization=regularization,loss=float(np.mean(losses)),counterfactualComparisons=len(controls),incorrectVetoes=sum(controls))
        validation.append(item);print(json.dumps(item),flush=True)
    selected=min(validation,key=lambda r:(r['loss'],r['regularization']))
    budget();fitted={**metadata,**fit(rows,args.dimensions,selected['regularization'])};budget()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    asset=out/'acoustic-orbit-emissions.json';asset.write_text(json.dumps(fitted,allow_nan=False))
    paths=[asset,out/'acoustic-orbit.onnx']
    files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (out/'acoustic-orbit-asset.json').write_text(json.dumps(dict(schema=1,files=files)))
    report=dict(recordings=len(rows),speakers=speakers,speakerFolds=folds,excludedSpeakers=sorted(excluded),validation=validation,selected=selected,
                augmentation='student_train.spec_augment; clean plus three same-clock draws',dimensions=args.dimensions,sharedBoundaryRefits=2,
                inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),files=files,
                limitations='Same-frame augmentation paths approximate local nuisance orbits. SpecAugment can hide identity cues; different-phone contrasts do not prove preservation of genuine mistakes.')
    (out/'acoustic-orbit-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true');p.add_argument('--model');p.add_argument('--labels');p.add_argument('--reference')
    p.add_argument('--metadata',required=True);p.add_argument('--input',required=True);p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True);p.add_argument('--dimensions',type=int,default=12)
    args=p.parse_args()
    prepare(args) if args.prepare else train(args)
