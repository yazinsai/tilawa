"""Matched frozen-encoder overlapping-diphone and single-phone acoustic readouts.

Only none-labelled training speakers supervise the heads. Transcript-constrained
CTC latent state posteriors locate successive articulations. A middle articulation
contributes equally to its incoming and outgoing diphone; endpoints contribute
only to their one incident diphone. All boundaries remain latent at inference.
The matched control uses every output parameter but marginalizes the unlabelled
right partner, training solely on single-phone occupancies.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN, EPOCHS = 32, 48, 18
REGULARIZATION = (.0001, .01)

def occupancies(log_probs, ids):
    """Exact transcript-conditioned soft CTC state and phone occupancies."""
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or not np.isfinite(lp).all() or any(p <= 0 or p >= lp.shape[1] for p in ids):
        raise ValueError('Invalid canonical transcript or posterior rows')
    labels = np.array([0] + [p for i in ids for p in (i, 0)])
    if not ids or len(lp) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return None
    n, t = len(labels), len(lp)
    alpha = np.full((t, n), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    skip = (labels != 0) & (labels != np.roll(labels, 2))
    skip[:2] = False
    for f in range(1, t):
        prev = alpha[f-1]
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[-np.inf, -np.inf, prev[:-2]]
        alpha[f] = np.logaddexp(np.logaddexp(prev, one), np.where(skip, two, -np.inf)) + lp[f, labels]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full((t, n), -np.inf)
    beta[-1, -2:] = 0
    for f in range(t-2, -1, -1):
        nxt = beta[f+1] + lp[f+1, labels]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[nxt[2:], -np.inf, -np.inf]
        beta[f] = np.logaddexp(np.logaddexp(nxt, one), np.where(np.r_[skip[2:], False, False], two, -np.inf))
    states = np.exp(alpha + beta - z)
    states /= states.sum(1, keepdims=True)
    targets = np.zeros_like(lp)
    for s, phone in enumerate(labels):
        targets[:, phone] += states[:, s]
    return targets.astype('float32'), states.astype('float32')


def sequence_score(lp, ids):
    """Same complete CTC marginal and normalized timing prior as the SDK."""
    import numpy as np
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
             ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else []) for s, p in enumerate(labels)]
    a = np.full(len(labels), -np.inf)
    prior = a.copy()
    a[:2] = lp[0, labels[:2]]
    prior[:2] = 0
    for frame in lp[1:]:
        b, q = np.full_like(a, -np.inf), np.full_like(a, -np.inf)
        for s, nxt in enumerate(edges):
            cost = np.log(len(nxt))
            for j in nxt:
                b[j] = np.logaddexp(b[j], a[s]-cost)
                q[j] = np.logaddexp(q[j], prior[s]-cost)
        a, prior = b + frame[labels], q
    likelihood, timing = np.logaddexp.reduce(a[-2:]), np.logaddexp.reduce(prior[-2:])
    return float(likelihood-timing) if np.isfinite(likelihood) and np.isfinite(timing) else -np.inf


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())

def pair_ids(ids, phones):
    return [1+(a-1)*phones+b-1 for a, b in zip(ids, ids[1:])]


def pair_targets(states, ids, phones):
    import numpy as np
    labels = pair_ids(ids, phones)
    targets = np.zeros((len(states), phones*phones+1), dtype='float32')
    targets[:, 0] = states[:, ::2].sum(1)
    for i in range(len(ids)):
        incident = ([labels[i-1]] if i else []) + ([labels[i]] if i < len(labels) else [])
        for label in incident:
            targets[:, label] += states[:, 2*i+1]/len(incident)
    return targets


def contextual(features, center, scale):
    import numpy as np
    x = (np.asarray(features, dtype='float32')-center)/scale
    return np.concatenate((np.vstack((x[:1],x[:-1])), x, np.vstack((x[1:],x[-1:]))), axis=1)


def phone_rows(lp, phones):
    import numpy as np
    return np.concatenate((lp[:, :1], np.logaddexp.reduce(lp[:, 1:].reshape(len(lp), phones, phones), axis=2)), axis=1)


def fit(rows, phones, regularization):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    x = torch.tensor(np.concatenate([contextual(r['features'], center, scale) for r in rows]))
    y = torch.tensor(np.concatenate([r['phoneTargets'] for r in rows]))
    pairs = torch.tensor(np.concatenate([r['pairTargets'] for r in rows]))
    treatment = torch.nn.Sequential(torch.nn.Linear(3*DIMENSIONS,HIDDEN),torch.nn.Tanh(),torch.nn.Linear(HIDDEN,phones*phones+1))
    control = copy.deepcopy(treatment)
    opts = [torch.optim.AdamW(h.parameters(),lr=.002,weight_decay=regularization) for h in (treatment,control)]
    weights = np.concatenate([np.full(len(r['features']),1/len(r['features'])) for r in rows])
    weights /= weights.sum()
    rng = np.random.default_rng(1729)
    steps = max(1,len(x)//256)
    for _ in range(EPOCHS):
        for _ in range(steps):
            batch = rng.choice(len(x),256,p=weights)
            for h,opt in zip((treatment,control),opts):
                opt.zero_grad()
                lp = h(x[batch]).log_softmax(-1)
                # Both evaluate exactly the same output matrix and normalization.
                if h is control:
                    lp = torch.cat((lp[:,:1],lp[:,1:].reshape(-1,phones,phones).logsumexp(-1)),dim=1)
                    target = y[batch]
                else:
                    target = pairs[batch]
                loss = -(target*lp).sum(-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(),5.)
                opt.step()
    return treatment.eval(),control.eval(),center,scale,steps*EPOCHS


def outputs(model, features, center, scale):
    import torch
    with torch.no_grad():
        return model(torch.tensor(contextual(features,center,scale))).log_softmax(-1).numpy()


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be unique')
    phones = len(metadata['tokens'])
    rows, skipped = [], []
    for r in data:
        if r.get('source')!='help-train' or r.get('split')!='train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Ineligible training-speaker provenance')
        if r['kind']!='none':
            continue
        x = np.asarray(r['features'])
        if x.ndim!=2 or x.shape[1]!=DIMENSIONS or not np.isfinite(x).all():
            raise ValueError('Invalid frozen features')
        ids = r['pronunciation']
        found = occupancies(r['logProbs'],ids) if len(ids)>=2 else None
        if found is None:
            skipped.append(r['id'])
            continue
        targets, states = found
        if len(targets)!=len(x) or targets.shape[1]!=phones+1:
            raise ValueError('Native clock or vocabulary mismatch')
        rows.append(dict(r,phoneTargets=targets,pairTargets=pair_targets(states,ids,phones)))
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers)<3:
        raise ValueError('Need three none-labelled training speakers for grouped validation')
    folds = {s:i%3 for i,s in enumerate(speakers)}
    validation = []
    for reg in REGULARIZATION:
        errors, controls = {}, {}
        for fold in range(3):
            treatment,control,center,scale,_ = fit([r for r in rows if folds[r['speaker']]!=fold],phones,reg)
            for r in rows:
                if folds[r['speaker']]!=fold:
                    continue
                lp = outputs(treatment,r['features'],center,scale)
                cp = phone_rows(outputs(control,r['features'],center,scale),phones)
                errors.setdefault(r['speaker'],[]).append(float(-(r['pairTargets']*lp).sum(1).mean()))
                controls.setdefault(r['speaker'],[]).append(float(-(r['phoneTargets']*cp).sum(1).mean()))
        result=dict(regularization=reg,meanSpeakerPairNll=float(np.mean([np.mean(v) for v in errors.values()])),
                    meanSpeakerPhoneNll=float(np.mean([np.mean(v) for v in controls.values()])))
        validation.append(result)
        print(json.dumps(result),flush=True)
    selected = min(validation,key=lambda r:(r['meanSpeakerPairNll'],r['regularization']))
    treatment,control,center,scale,updates = fit(rows,phones,selected['regularization'])
    metadata.update(center=center.tolist(),scale=scale.tolist(),diphone=serialize(treatment),control=serialize(control))
    out=Path(args.out)
    asset=out/'diphone-heads.json'
    asset.write_text(json.dumps(metadata,allow_nan=False))
    report=dict(recordings=len(rows),speakers=speakers,excludedSpeakers=sorted(excluded),folds=folds,
                skippedAlignments=skipped,validation=validation,selected=selected,updatesPerHead=updates,epochs=EPOCHS,
                parametersPerHead=sum(p.numel() for p in treatment.parameters()),
                inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'diphone-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    files=[asset,out/'diphone.onnx']
    descriptor=dict(schema=1,files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'diphone-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    train(p.parse_args())

