"""Training-only estimator for rhythm.ts. No practice inputs or location-specific parameters.

python3 packages/core/src/rhythm-fit.py --data "$LAB_DATA/help-train" --out "$LAB_OUT/rhythm-model.json"
Model selection uses three deterministic speaker folds and a matched independent-duration control.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

K = 8
HORIZON = 16
SHRINKAGES = (10, 100, 1000)


def classes(ref):
    out = []
    i = 0
    while i < len(ref):
        j = i + 1
        while j < len(ref) and ref[j] == ref[i]:
            j += 1
        p = ref[i]
        c = (1 if p in "َُِ" else 2 if p in "اۦۥ" else
             4 if p in "ں۾" or (p in "نم" and j-i >= 3) else
             3 if p in "نم" else 5 if p in "لريو" else
             6 if p in "فثذزسشصظهحخغعء" else 7)
        out.extend([c] * (j-i))
        i = j
    return np.array(out)


def sub(ref, obs):
    if ref == obs:
        return 0.
    pairs = ("هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس")
    return .5 if ref+obs in [s for p in pairs for s in (p, p[::-1])] or (ref in "َُِ" and obs in "َُِ") else 1.


def indel(ref, at, p, insertion=False):
    def run(k):
        if k < 0 or k >= len(ref) or ref[k] != p:
            return 0
        a, b = k, k+1
        while a > 0 and ref[a-1] == p:
            a -= 1
        while b < len(ref) and ref[b] == p:
            b += 1
        return b-a
    n = max(run(at-1), run(at))+1 if insertion else run(at)
    held = (p in "اۦۥں۾" and n >= 2) or (p in "نم" and n >= 3)
    return .2 if held else .5 if p in "َُِ" else 1.


def correspondence(ref, obs):
    """Exact Gibbs edit forward/backward. Full transcript, open terminal suffix, charged leading prefix.

    Pair expectations retain every possible position of a deletion between consecutive emissions.
    No alignment from tracker visits or labels is selected as the observed path.
    """
    N, T = len(ref), len(obs)
    cls = classes(ref)
    # U_t maps pre-emission reference endpoint to post-emission endpoint and emitted structural class.
    deletion = np.array([math.exp(-indel(ref, j, ref[j])) for j in range(N)])
    closure = np.eye(N+1)
    for i in range(N+1):
        for j in range(i+1, N+1):
            closure[i, j] = closure[i, j-1]*deletion[j-1]
    kernels = []
    for p in obs:
        u = np.zeros((N+1, N+1, K))
        for i in range(N+1):
            u[i, i, 0] = math.exp(-indel(ref, i, p, True))
            if i < N:
                u[i, i+1, cls[i]] = math.exp(-sub(ref[i], p))
        # Reference-only deletions follow consuming transitions, just as in local.ts.
        kernels.append(np.einsum('ijc,jk->ikc', u, closure))
    alpha = [closure[0].copy()]
    scales = []
    for q in kernels:
        v = alpha[-1] @ q.sum(axis=2)
        scale = v.sum()
        scales.append(scale)
        alpha.append(v/scale)
    beta = [None]*(T+1)
    beta[T] = np.ones(N+1)
    for t in range(T-1, -1, -1):
        beta[t] = kernels[t].sum(axis=2) @ beta[t+1]/scales[t]
    single = []
    pairs = []
    for t, q in enumerate(kernels):
        s = np.einsum('i,ijc,j->c', alpha[t], q, beta[t+1])/scales[t]
        single.append(s/s.sum())
        if t:
            left = np.einsum('i,ijc->jc', alpha[t-1], kernels[t-1])/scales[t-1]
            right = np.einsum('ijc,j->ic', q, beta[t+1])/scales[t]
            pair = left.T @ right
            pairs.append(pair/pair.sum())
    return np.array(single), np.array(pairs)


def fit_marginals(rows):
    n = np.zeros(K)
    sx = np.zeros(K)
    sx2 = np.zeros(K)
    for r in rows:
        n += r['q'].sum(axis=0)
        sx += (r['q']*r['x'][:, None]).sum(axis=0)
        sx2 += (r['q']*r['x'][:, None]**2).sum(axis=0)
    pooled_m = sx.sum()/n.sum()
    pooled_v = sx2.sum()/n.sum()-pooled_m**2
    # Ten pooled pseudo-observations per class; shared in every joint/control model.
    mean = (sx+10*pooled_m)/(n+10)
    var = (sx2+10*(pooled_v+pooled_m**2))/(n+10)-mean**2
    return mean, np.sqrt(np.maximum(var, 1e-6))


def fit_rho(rows, mean, sd, shrink):
    n = np.zeros((K, K))
    xy = np.zeros_like(n)
    xx = np.zeros_like(n)
    yy = np.zeros_like(n)
    for r in rows:
        z = (r['x'][:, None]-mean)/sd
        for t, q in enumerate(r['pair']):
            n += q
            xy += q*z[t, :, None]*z[t+1, None, :]
            xx += q*z[t, :, None]**2
            yy += q*z[t+1, None, :]**2
    denom = math.sqrt(xx.sum()*yy.sum())
    pooled = xy.sum()/denom if denom else 0.
    # Shrink class-pair correlations to the pooled correlation; sparse classes cannot set large preferences.
    rho = (xy+shrink*pooled)/np.sqrt((xx+shrink)*(yy+shrink))
    return np.clip(rho, -.95, .95)


def likelihood(r, mean, sd, rho):
    """Marginal likelihood under the transcript-conditioned class chain, including the SAME marginals.

    The chain uses posterior pair correspondences, retaining all class alternatives. This approximation shares
    the exact one- and two-token correspondence marginals; runtime uses the full reference edit graph.
    """
    z = (r['x'][:, None]-mean)/sd
    log_m = -.5*z*z-np.log(sd)
    with np.errstate(divide='ignore'):
        state = np.log(r['q'][0])+log_m[0]
    scale = np.logaddexp.reduce(state)
    state -= scale
    total = scale
    for t, pair in enumerate(r['pair']):
        transition = pair/np.maximum(pair.sum(axis=1, keepdims=True), 1e-300)
        v = 1-rho*rho
        copula = -.5*np.log(v)+(2*rho*z[t, :, None]*z[t+1, None, :]
                  -rho*rho*(z[t, :, None]**2+z[t+1, None, :]**2))/(2*v)
        with np.errstate(divide='ignore'):
            matrix = np.log(transition)+copula
        state = np.logaddexp.reduce(state[:, None]+matrix, axis=0)+log_m[t+1]
        scale = np.logaddexp.reduce(state)
        state -= scale
        total += scale
    return total


def train(data, out, ts_out=None):
    asset = json.loads((Path(__file__).parent.parent/'assets/ref-v2.json').read_text())
    refs = {f"{a['s']}:{a['a']}": a['p'] for a in asset['ayahs']}
    labels = [json.loads(l) for l in (data/'labels.jsonl').read_text().splitlines() if l]
    held_out = {r['speaker'] for r in labels if r.get('split') in ('eval', 'test', 'heldout')}
    eligible = [r for r in labels if r.get('split') == 'train' and r.get('kind') == 'none'
                and r['speaker'] not in held_out and not r.get('events')
                and not r.get('non_errors') and not r.get('exclude')]
    if not eligible:
        raise ValueError('No explicitly training-speaker none-labelled recordings; refusing any fallback split')
    rows = []
    missing = []
    ids = []
    provenance = hashlib.sha256(json.dumps(asset, sort_keys=True).encode())
    for r in eligible:
        p = data/Path(r['wav']).with_suffix('.evidence.json')
        if not p.exists():
            missing.append(str(p.relative_to(data)))
            continue
        contents = p.read_bytes()
        provenance.update(json.dumps(r, sort_keys=True).encode())
        provenance.update(contents)
        ev = json.loads(contents)
        if ev.get('inventory') != asset['inventory']['id']:
            raise ValueError(f'Inventory mismatch: {p}')
        ph = ev['phonemes']
        ref = ''.join(refs[a] for a in r['ayahs'])
        q, pairs = correspondence(ref, [p['p'] for p in ph])
        ids.append(r['id'])
        duration = np.array([p['end']-p['start']+1 for p in ph], dtype=float)
        if np.any(duration <= 0):
            raise ValueError(f'Invalid emitted-run duration: {p}')
        # Overlapping horizon windows give all adjacent relative durations equal access to their full context.
        for end in range(HORIZON, len(ph)+1):
            start = end-HORIZON
            x = np.log(duration[start:end])
            x -= x.mean()
            rows.append(dict(speaker=r['speaker'], x=x, q=q[start:end], pair=pairs[start:end-1]))
        print(f"aligned training recording {len(ids)}/{len(eligible)}", flush=True)
    if missing:
        raise ValueError(f'{len(missing)} eligible training decodes missing: {missing[:3]}; decode with deployed listener first')
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('Need at least three training speakers for speaker-grouped model selection')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    scores = {v: [] for v in SHRINKAGES}
    control_scores = []
    for f in range(3):
        tr = [r for r in rows if folds[r['speaker']] != f]
        va = [r for r in rows if folds[r['speaker']] == f]
        mean, sd = fit_marginals(tr)
        control_scores.append(sum(likelihood(r, mean, sd, np.zeros((K, K))) for r in va))
        for v in SHRINKAGES:
            rho = fit_rho(tr, mean, sd, v)
            scores[v].append(sum(likelihood(r, mean, sd, rho) for r in va))
    selected = max(SHRINKAGES, key=lambda v: (sum(scores[v]), v))
    mean, sd = fit_marginals(rows)
    rho = fit_rho(rows, mean, sd, selected)
    model = dict(mean=mean.tolist(), sd=sd.tolist(), rho=rho.tolist())
    report = dict(recordings=len(ids), speakers=len(speakers), windows=len(rows), horizon=HORIZON,
                  inputSha256=provenance.hexdigest(),
                  selection='three speaker-grouped training folds; largest shrinkage breaks ties',
                  selectedShrinkage=selected, validation={str(v): x for v, x in scores.items()},
                  independentValidation=control_scores, model=model)
    out.write_text(json.dumps(report, indent=2)+'\n')
    if ts_out:
        ts_out.write_text('import type { RhythmModel } from "./rhythm.js";\n\n'
                          '/** Frozen help-train training-speaker model; see rhythm-fit.py. */\n'
                          'export const RELATIVE_RHYTHM_MODEL: RhythmModel = '
                          +json.dumps(model, indent=2)+';\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ts-out', type=Path, help='Write the frozen runtime model ONLY after training succeeds')
    args = parser.parse_args()
    train(args.data, args.out, args.ts_out)
