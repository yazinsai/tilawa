"""Fit a word-internal nucleus timing copula on help-train TRAINING speakers only.

python3 packages/core/src/word-rhythm-fit.py --data "$LAB_DATA/help-train" --out "$LAB_OUT/word-rhythm.json"
No fallback split, missing decode, or practice input is accepted. See .scratch/lab/word-internal-rhythm/design.md.
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


def training_labels(labels):
    """Use only explicitly training, clean scripts, excluding an entire held-out speaker.

    `scored=False` marks extra/unresolved mistakes in help-train; `excluded` is the importer’s audit exclusion.
    Missing split/kind is never inferred from a filename or supplied passage.
    """
    held_out = {r['speaker'] for r in labels if r.get('split') in ('eval', 'test', 'heldout')}
    return [r for r in labels if r.get('split') == 'train' and r.get('kind') == 'none'
            and r['speaker'] not in held_out and not r.get('events')
            and not r.get('non_errors') and not r.get('unsupported')
            and not r.get('excluded') and not r.get('exclude') and r.get('scored') is not False]



def structure(ref, words):
    cls = np.zeros(len(ref), dtype=int)
    for i, p in enumerate(ref):
        a, b = i, i+1
        while a and ref[a-1] == p:
            a -= 1
        while b < len(ref) and ref[b] == p:
            b += 1
        cls[i] = (1 if p in "َُِ" else 2 if p in "اۦۥ" else
                  4 if p in "ں۾" or (p in "نم" and b-a >= 3) else
                  3 if p in "نم" else 5 if p in "لريو" else
                  6 if p in "فثذزسشصظهحخغعء" else 7)
    word = np.full(len(ref), -1, dtype=int)
    nucleus = np.full(len(ref), -1, dtype=int)
    starts, links = [], []
    for wi, (lo, hi) in enumerate(words):
        word[lo:hi] = wi
        previous, previous_end, i = -1, lo, lo
        while i < hi:
            if ref[i] not in "َُِاۦۥ":
                i += 1
                continue
            end = i+1
            while end < hi and ref[end] in "َُِاۦۥ":
                end += 1
            long = any(p in "اۦۥ" for p in ref[i:end])
            n = len(starts)
            starts.append(i)
            links.append(-1 if previous < 0 else int(cls[starts[previous]] == 2)*4
                         +int(long)*2+int(i-previous_end > 1))
            cls[i:end] = 2 if long else 1
            nucleus[i:end] = n
            previous, previous_end, i = n, end, end
    return dict(cls=cls, word=word, nucleus=nucleus, starts=starts, links=links)


def graph(ref, obs):
    N = len(ref)
    deletion = np.array([math.exp(-indel(ref, j, ref[j])) for j in range(N)])
    closure = np.eye(N+1)
    for i in range(N+1):
        for j in range(i+1, N+1):
            closure[i, j] = closure[i, j-1]*deletion[j-1]
    insert = np.array([[math.exp(-indel(ref, i, p, True)) for i in range(N+1)] for p in obs])
    match = np.array([[math.exp(-sub(p, q)) for p in ref] for q in obs])
    alpha, scales = [closure[0].copy()], []
    for ins, mat in zip(insert, match):
        v = alpha[-1]*ins
        v[1:] += alpha[-1][:-1]*mat
        v = v @ closure
        scale = v.sum()
        scales.append(scale)
        alpha.append(v/scale)
    beta = [None]*len(alpha)
    beta[-1] = np.ones(N+1)
    for t in range(len(obs)-1, -1, -1):
        tail = closure @ beta[t+1]
        v = insert[t]*tail
        v[:-1] += match[t]*tail[1:]
        beta[t] = v/scales[t]
    return closure, insert, match, alpha, beta, scales


def add(cell, key, weight):
    if weight:
        cell[key] = cell.get(key, 0.)+weight


def advance(states, t, st, insert, match, closure, factor=None, pairs=None, tail=None, scale=1.):
    """Consuming transitions then reference-only closure, preserving latent nucleus placement.

    A missing nucleus needs no duration observation; its induced copula correlation is integrated separately.
    Crossing a word or restarting clears the previous nucleus. Expanded symbols within one nucleus add no pair.
    """
    N = len(st['cls'])
    emitted = [{} for _ in range(N+1)]
    for i, cell in enumerate(states):
        for (old, ot), weight in cell.items():
            marginal, copula = factor(0, -1, old, ot) if factor else (1., 1.)
            add(emitted[i], (old, ot), weight*insert[i]*marginal)
            if i == N:
                continue
            same = old >= 0 and st['word'][i] == st['word'][st['starts'][old]]
            old = old if same else -1
            ni, c = int(st['nucleus'][i]), int(st['cls'][i])
            marginal, copula = factor(c, ni, old, ot) if factor else (1., 1.)
            key = (ni, t) if ni >= 0 else (old, ot if old >= 0 else -1)
            add(emitted[i+1], key, weight*match[i]*marginal*copula)
            if pairs is not None and ni > old >= 0:
                prob = weight*match[i]*tail[i+1]/scale
                add(pairs, (ot, t, old, ni), prob)
    result = emitted
    # Incremental reference-only deletion closure, identical to the local matcher's recurrence.
    for j in range(1, N+1):
        for (old, ot), weight in result[j-1].items():
            keep = old >= 0 and st['word'][j-1] == st['word'][st['starts'][old]]
            add(result[j], (old, ot) if keep else (-1, -1), weight*closure[j-1, j])
    return result


def recording(ref, words, ph, speaker):
    obs = [p['p'] for p in ph]
    closure, ins, mat, alpha, beta, scales = graph(ref, obs)
    st = structure(ref, words)
    q = np.zeros((len(obs), K))
    pairs = {}
    states = [{(-1, -1): v} for v in alpha[0]]
    for t in range(len(obs)):
        tail = closure @ beta[t+1]
        q[t, 0] = (alpha[t]*ins[t]*tail).sum()/scales[t]
        for i, c in enumerate(st['cls']):
            q[t, c] += alpha[t][i]*mat[t, i]*tail[i+1]/scales[t]
        states = advance(states, t, st, ins[t], mat[t], closure, pairs=pairs, tail=tail, scale=scales[t])
        for cell in states:
            for key in cell:
                cell[key] /= scales[t]
    duration = np.array([p['end']-p['start']+1 for p in ph], dtype=float)
    if not np.all(duration > 0):
        raise ValueError('Invalid token duration')
    windows = []
    for end in range(HORIZON, len(ph)+1):
        start = end-HORIZON
        x = np.log(duration[start:end])
        x -= x.mean()
        windows.append(dict(speaker=speaker, x=x, q=q[start:end], st=st, closure=closure,
                            ins=ins[start:end], mat=mat[start:end], initial=alpha[start],
                            pair={(a-start, b-start, n, m): w for (a, b, n, m), w in pairs.items()
                                  if start <= a < b < end}))
    return windows


def marginals(rows):
    n, sx, sx2 = (np.zeros(K) for _ in range(3))
    for r in rows:
        n += r['q'].sum(axis=0)
        sx += (r['q']*r['x'][:, None]).sum(axis=0)
        sx2 += (r['q']*r['x'][:, None]**2).sum(axis=0)
    pooled_m = sx.sum()/n.sum()
    pooled_v = sx2.sum()/n.sum()-pooled_m**2
    mean = (sx+10*pooled_m)/(n+10)
    var = (sx2+10*(pooled_v+pooled_m**2))/(n+10)-mean**2
    return mean, np.sqrt(np.maximum(var, 1e-6))


def correlations(rows, mean, sd, shrink):
    xy, xx, yy = (np.zeros(8) for _ in range(3))
    for r in rows:
        st = r['st']
        z = (r['x'][:, None]-mean)/sd
        for (a, b, n, m), w in r['pair'].items():
            # Estimate directly observed adjacent reference-nucleus edges. Missing-emission paths are still
            # marginalized in the posterior and validation; their induced correlation is the edge product.
            if m != n+1:
                continue
            link = st['links'][m]
            left, right = z[a, st['cls'][st['starts'][n]]], z[b, st['cls'][st['starts'][m]]]
            xy[link] += w*left*right
            xx[link] += w*left**2
            yy[link] += w*right**2
    # Zero-centered shrinkage is the grouping-free control; no target-derived fallback correlation.
    return np.clip(xy/np.sqrt((xx+shrink)*(yy+shrink)), -.95, .95)


def likelihood(r, mean, sd, rho):
    st = r['st']
    z = (r['x'][:, None]-mean)/sd
    states = [{(-1, -1): v} for v in r['initial']]
    total = 0.
    for t in range(HORIZON):
        def factor(c, ni, old, ot):
            marginal = math.exp(-.5*z[t, c]**2)/sd[c]
            corr = 0.
            if ni > old >= 0 and st['word'][st['starts'][ni]] == st['word'][st['starts'][old]]:
                corr = float(np.prod([rho[st['links'][i]] for i in range(old+1, ni+1)]))
            if not corr:
                return marginal, 1.
            u, v = z[ot, st['cls'][st['starts'][old]]], z[t, c]
            variance = 1-corr*corr
            copula = -.5*math.log(variance)+(2*corr*u*v-corr*corr*(u*u+v*v))/(2*variance)
            return marginal, math.exp(copula)
        states = advance(states, t, st, r['ins'][t], r['mat'][t], r['closure'], factor=factor)
        mass = sum(sum(c.values()) for c in states)
        if not mass > 0 or not math.isfinite(mass):
            raise ValueError('Invalid timing partition')
        total += math.log(mass)
        for cell in states:
            for key in cell:
                cell[key] /= mass
    return total


def train(data, out, ts_out):
    asset = json.loads((Path(__file__).parent.parent/'assets/ref-v2.json').read_text())
    refs = {f"{a['s']}:{a['a']}": a for a in asset['ayahs']}
    labels = [json.loads(l) for l in (data/'labels.jsonl').read_text().splitlines() if l]
    eligible = training_labels(labels)
    if not eligible:
        raise ValueError('No explicit training-speaker clean labels; no fallback allowed')
    rows, ids = [], []
    provenance = hashlib.sha256(json.dumps(asset, sort_keys=True).encode())
    for label in eligible:
        path = data/Path(label['wav']).with_suffix('.evidence.json')
        contents = path.read_bytes()  # Missing eligible decodes stop the fit, never silently reduce training.
        ev = json.loads(contents)
        if ev['inventory'] != asset['inventory']['id']:
            raise ValueError(f'Inventory mismatch: {path}')
        ref, words = '', []
        for key in label['ayahs']:
            a = refs[key]
            base = len(ref)
            words.extend((base+lo, base+(a['o'][i+1] if i+1 < len(a['o']) else len(a['p'])))
                         for i, lo in enumerate(a['o']))
            ref += a['p']
        provenance.update(json.dumps(label, sort_keys=True).encode())
        provenance.update(contents)
        rows.extend(recording(ref, words, ev['phonemes'], label['speaker']))
        ids.append(label['id'])
        print(f'aligned clean training recording {len(ids)}/{len(eligible)}', flush=True)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('Need at least three training speakers')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    scores = {v: [] for v in SHRINKAGES}
    control = []
    for f in range(3):
        tr = [r for r in rows if folds[r['speaker']] != f]
        va = [r for r in rows if folds[r['speaker']] == f]
        mean, sd = marginals(tr)
        control.append(sum(likelihood(r, mean, sd, np.zeros(8)) for r in va))
        for v in SHRINKAGES:
            rho = correlations(tr, mean, sd, v)
            scores[v].append(sum(likelihood(r, mean, sd, rho) for r in va))
        print(f'validated speaker fold {f+1}/3', flush=True)
    selected = max(SHRINKAGES, key=lambda v: (sum(scores[v]), v))
    mean, sd = marginals(rows)
    rho = correlations(rows, mean, sd, selected)
    model = dict(mean=mean.tolist(), sd=sd.tolist(), rho=rho.tolist())
    report = dict(recordings=len(ids), speakers=len(speakers), windows=len(rows), horizon=HORIZON,
                  inputSha256=provenance.hexdigest(), selectedShrinkage=selected,
                  validation={str(v): s for v, s in scores.items()}, independentValidation=control, model=model)
    out.write_text(json.dumps(report, indent=2)+'\n')
    if ts_out:
        ts_out.write_text('import type { WordRhythmModel } from "./word-rhythm.js";\n\n'
                          '/** Frozen help-train training-speaker fit; see word-rhythm-fit.py. */\n'
                          'export const WORD_RHYTHM_MODEL: WordRhythmModel = '+json.dumps(model, indent=2)+';\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ts-out', type=Path)
    args = parser.parse_args()
    train(args.data, args.out, args.ts_out)
