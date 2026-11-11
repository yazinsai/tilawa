"""Frozen-encoder episodic readout: learn initialization and a ridge regularizer.

Every support set is an earlier accepted prefix of one none-labelled recording.
Query labels train the outer loss only; solve() has no query argument. Speaker
folds and input provenance are checked before any training or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 32
EPOCHS = 60
REGULARIZERS = (.01, .1, 1.)
SUPPORT_COLLAR = 8  # The runtime's 300 ms exclusion, rounded up on the 40 ms clock.

def occupancies(log_probs, ids):
    """Exact forward/backward soft CTC state and phone occupancies."""
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or not np.isfinite(lp).all() or any(p <= 0 or p >= lp.shape[1] for p in ids):
        raise ValueError('Invalid canonical transcript or posterior rows')
    labels = np.array([0] + [p for i in ids for p in (i, 0)])
    if not ids or len(lp) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return None
    n, t = len(labels), len(lp)
    incoming = [[s] + ([s-1] if s else []) +
                ([s-2] if s > 1 and labels[s] and labels[s] != labels[s-2] else []) for s in range(n)]
    outgoing = [[] for _ in labels]
    for s, edges in enumerate(incoming):
        for before in edges:
            outgoing[before].append(s)
    alpha = np.full((t, n), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for f in range(1, t):
        for s in range(n):
            alpha[f, s] = np.logaddexp.reduce(alpha[f-1, incoming[s]]) + lp[f, labels[s]]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full((t, n), -np.inf)
    beta[-1, -2:] = 0
    for f in range(t-2, -1, -1):
        for s in range(n):
            nxt = outgoing[s]
            beta[f, s] = np.logaddexp.reduce(beta[f+1, nxt] + lp[f+1, labels[nxt]])
    states = np.exp(alpha + beta - z)
    states /= states.sum(1, keepdims=True)
    targets = np.zeros_like(lp)
    for s, phone in enumerate(labels):
        targets[:, phone] += states[:, s]
    context = []
    for s in states.argmax(1):
        # Blank intervals use their immediately adjacent phones. Articulations
        # use the previous/next articulation, including genuine doubled phones.
        pos = (int(s)-1)//2
        if s % 2:
            context.append((ids[pos-1] if pos else 0, ids[pos], ids[pos+1] if pos+1 < len(ids) else 0))
        else:
            context.append((ids[s//2-1] if s else 0, 0, ids[s//2] if s//2 < len(ids) else 0))
    return targets.astype('float32'), context



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



def ridge_update(initialization, regularizer, x, targets, weights):
    """Differentiable normalized normal equations, identical to the SDK solve."""
    import torch
    diagonal = torch.diag(regularizer)
    gram = x.T @ (weights[:, None]*x) + diagonal
    rhs = x.T @ (weights[:, None]*targets.clamp_min(1e-6).log()) + diagonal @ initialization
    return torch.linalg.solve(gram, rhs)


def speaker_folds(rows):
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    return {s: i % 3 for i, s in enumerate(speakers)}


def prepare(rows, center, scale):
    import numpy as np
    import torch
    result = []
    for r in rows:
        x = np.asarray(r['features'], dtype='float64')
        x = torch.tensor(np.column_stack(((x-center)/scale, np.ones(len(x)))))
        found = occupancies(r['logProbs'], r['pronunciation'])
        if found is None:
            continue
        y = torch.tensor(found[0], dtype=torch.float64)
        support = []
        for s in r['accepted']:
            a, b = s['fromFrame'], s['toFrame']
            aligned = occupancies(s['logProbs'], s['pronunciation'])
            if aligned is not None:
                support.append(dict(x=x[a:b], y=torch.tensor(aligned[0], dtype=torch.float64),
                                    start=a, end=b, available=s['availableFrame']))
        episodes = []
        # All chronological support prefixes with a later query. Each word has
        # equal total mass, rather than letting a held vowel dominate adaptation.
        for cutoff in sorted({s['available'] for s in support}):
            chosen = [s for s in support if s['available'] <= cutoff and s['end'] <= cutoff]
            if not chosen:
                continue
            # Query complete later words, with the same outside-window collar
            # as inference. Their transcript-conditioned targets are used only
            # in the outer loss, never in the ridge update.
            later = [s for s in support if s['start'] >= cutoff and
                     s['start'] >= max(p['end'] for p in chosen) + SUPPORT_COLLAR]
            if not later:
                continue
            sx, sy = torch.cat([s['x'] for s in chosen]), torch.cat([s['y'] for s in chosen])
            weights = torch.cat([torch.full((len(s['x']),), 1/len(chosen)/len(s['x']), dtype=torch.float64) for s in chosen])
            qx = torch.cat([x[s['start']:s['end']] for s in later])
            qy = torch.cat([y[s['start']:s['end']] for s in later])
            episodes.append((sx, sy, weights, qx, qy))
        if episodes:
            result.append(dict(speaker=r['speaker'], episodes=episodes))
    return result


def fit(rows, vocab, initial_regularizer):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    records = prepare(rows, center, scale)
    if not records:
        raise ValueError('No causal support/query episodes in training fold')
    initialization = torch.nn.Parameter(torch.zeros((DIMENSIONS+1, vocab), dtype=torch.float64))
    rho = torch.nn.Parameter(torch.full((DIMENSIONS+1,), float(np.log(np.expm1(initial_regularizer))), dtype=torch.float64))
    opt = torch.optim.Adam([initialization, rho], lr=.03)
    rng = np.random.default_rng(1729)
    speakers = sorted({r['speaker'] for r in records})
    for epoch in range(EPOCHS):
        for _ in range(len(records)):
            speaker = speakers[rng.integers(len(speakers))]
            candidates = [r for r in records if r['speaker'] == speaker]
            episode = candidates[rng.integers(len(candidates))]['episodes']
            sx, sy, weights, qx, qy = episode[rng.integers(len(episode))]
            opt.zero_grad()
            reg = torch.nn.functional.softplus(rho) + 1e-4
            adapted = ridge_update(initialization, reg, sx, sy, weights)
            query = -(qy*(qx@adapted).log_softmax(-1)).sum(-1).mean()
            base = -(qy*(qx@initialization).log_softmax(-1)).sum(-1).mean()
            (query + .25*base).backward()
            torch.nn.utils.clip_grad_norm_([initialization,rho],5.)
            opt.step()
    return initialization.detach(), (torch.nn.functional.softplus(rho)+1e-4).detach(), center, scale


def evaluate(fit_result, rows):
    import numpy as np
    import torch
    w, reg, center, scale = fit_result
    errors = {}
    with torch.no_grad():
        for r in prepare(rows, center, scale):
            values = []
            for sx, sy, weights, qx, qy in r['episodes']:
                adapted = ridge_update(w,reg,sx,sy,weights)
                values.append(float(-(qy*(qx@adapted).log_softmax(-1)).sum(-1).mean()))
            errors.setdefault(r['speaker'],[]).append(float(np.mean(values)))
    if not errors:
        raise ValueError('Held-speaker fold has no causal query episodes')
    return float(np.mean([np.mean(v) for v in errors.values()]))


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    vocab = len(metadata['tokens'])+1
    if not rows or len({r.get('id') for r in rows}) != len(rows):
        raise ValueError('Unique recording provenance required')
    for r in rows:
        if r.get('source')!='help-train' or r.get('split')!='train' or r.get('kind')!='none' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Only none-labelled training-speaker recordings are authorized')
        x, lp = np.array(r['features']), np.array(r['logProbs'])
        if x.ndim!=2 or x.shape[1]!=DIMENSIONS or lp.shape!=(len(x),vocab) or not np.isfinite(x).all() or not np.isfinite(lp).all():
            raise ValueError('Invalid frozen encoder or occupancy geometry')
        for s in r['accepted']:
            if not 0 <= s['fromFrame'] < s['toFrame'] <= s['availableFrame'] <= len(x):
                raise ValueError('Non-causal accepted-speech interval')
            if len(s['logProbs']) != s['toFrame'] - s['fromFrame']:
                raise ValueError('Accepted occupancy rows differ from their feature interval')
    folds = speaker_folds(rows)
    validation = []
    for strength in REGULARIZERS:
        losses=[]
        for fold in range(3):
            training=[r for r in rows if folds[r['speaker']]!=fold]
            held=[r for r in rows if folds[r['speaker']]==fold]
            losses.append(evaluate(fit(training,vocab,strength),held))
        result=dict(initialRegularizer=strength,heldSpeakerQueryNll=float(np.mean(losses)),foldQueryNll=losses)
        validation.append(result)
        print(json.dumps(result),flush=True)
    selected=min(validation,key=lambda r:r['heldSpeakerQueryNll'])
    w, reg, center, scale=fit(rows,vocab,selected['initialRegularizer'])
    metadata.update(initialization=w.tolist(),regularizer=reg.tolist(),center=center.tolist(),scale=scale.tolist())
    out=Path(args.out)
    head=out/'session-readout-heads.json'
    head.write_text(json.dumps(metadata,allow_nan=False))
    report=dict(recordings=len(rows),speakers=sorted(folds),folds=folds,excludedSpeakers=sorted(excluded),
                epochs=EPOCHS,validation=validation,selected=selected,
                episodes=sum(len(r['episodes']) for r in prepare(rows,center,scale)),
                inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'session-readout-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    files=[head,out/'session-readout.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Missing frozen feature graph')
    (out/'session-readout-asset.json').write_text(json.dumps(dict(schema=1,files=[dict(file=p.name,bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--excluded-speakers',required=True)
    p.add_argument('--out',required=True)
    train(p.parse_args())
