"""Fit the tracker seam channel on none-labelled, training-speaker student posteriors.

CTC forward/backward supplies fractional transcript boundary labels. Tracker visits are
never labels. Select regularization and probability calibration by speaker partitions.
The input directory must contain labels.jsonl; --evidence holds unchanged student dumps.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def rows(e, size):
    p = e["posteriors"]
    ids = np.array(p["ids"]).reshape(-1, p["k"])
    probs = np.array(p["probs"]).reshape(ids.shape)
    out = np.zeros((len(ids), size + 1))
    for k in range(p["k"]):
        out[np.arange(len(ids)), ids[:, k]] += probs[:, k]
    rest = np.maximum(0, 1 - out.sum(axis=1)) / np.maximum(1, (out == 0).sum(axis=1))
    out = np.where(out == 0, rest[:, None], out)
    out /= out.sum(axis=1)[:, None]
    return np.log(np.maximum(1e-6, out))


def features(logp, a, b, ms):
    left = max(a["start"], a["end"] - 3)
    right = min(b["end"] + 1, b["start"] + 4)
    if b["start"] <= a["end"] or left < 0 or right > len(logp):
        return None
    def summary(lo, hi):
        x = logp[lo:hi]
        if not len(x):
            return np.zeros(logp.shape[1]), 0, 0, 0, 0
        p = np.exp(x)
        return p.mean(axis=0), p[:, 0].mean(), -(p*x).sum(axis=1).mean()/np.log(p.shape[1]), p[:, 1:].max(axis=1).mean(), p[:, 0].max()
    av, ab, ah, ap, _ = summary(left, a["end"] + 1)
    _, gb, gh, gp, gm = summary(a["end"] + 1, b["start"])
    bv, bb, bh, bp, _ = summary(b["start"], right)
    av, bv = av[1:], bv[1:]
    return [np.log1p((b["start"] - a["end"] - 1) * ms/1000), ab, gb, bb, ah, gh, bh, ap, gp, bp,
            np.minimum(av, bv).sum(), av@bv/max(1e-12, np.linalg.norm(av)*np.linalg.norm(bv)), gm]


def ctc_boundaries(logp, labels, starts):
    """Return likelihood and the marginalized first entry time of each next word."""
    states = np.zeros(2*len(labels) + 1, dtype=int)
    states[1::2] = labels
    emission = logp[:, states]
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    alpha = np.full(emission.shape, -np.inf)
    alpha[0, :2] = emission[0, :2]
    def incoming(prev):
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[-np.inf, -np.inf, prev[:-2]]
        return np.logaddexp(np.logaddexp(prev, one), np.where(skip, two, -np.inf))
    for t in range(1, len(logp)):
        alpha[t] = incoming(alpha[t-1]) + emission[t]
    z = np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if not np.isfinite(z):
        return z, np.zeros(len(logp))
    beta = np.full(emission.shape, -np.inf)
    beta[-1, -2:] = 0
    for t in range(len(logp)-2, -1, -1):
        nxt = beta[t+1] + emission[t+1]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[np.where(skip[2:], nxt[2:], -np.inf), -np.inf, -np.inf]
        beta[t] = np.logaddexp(np.logaddexp(nxt, one), two)
    boundary = np.zeros(len(logp))
    for label_at in starts[1:]:
        s = 2*label_at + 1
        enter = alpha[:-1, s-1].copy()
        if skip[s]:
            enter = np.logaddexp(enter, alpha[:-1, s-2])
        boundary[1:] += np.exp(enter + emission[1:, s] + beta[1:, s] - z)
    return z, boundary


def transcript_options(ref, ayahs):
    by_key = {f'{a["s"]}:{a["a"]}': a for a in ref["ayahs"]}
    words = []
    for key in ayahs:
        a = by_key[key]
        words.extend(a["p"][at:(a["o"][i+1] if i+1 < len(a["o"]) else len(a["p"]))] for i, at in enumerate(a["o"]))
    options = []
    for isolated in (False, True):
        ws = words.copy()
        if isolated:
            ws[-1] = by_key[ayahs[-1]]["iso"][-1]
        for short in (False, True):
            if short:
                ws = [''.join(c for i, c in enumerate(w) if i == 0 or c != w[i-1]) for w in ws]
            offset = 0
            starts = []
            for w in ws:
                if w:
                    starts.append(offset)
                    offset += len(w)
            seq = ''.join(ws)
            item = (seq, tuple(starts))
            if item not in options:
                options.append(item)
    return options


def fit(x, y, strength):
    def loss(w):
        z = x@w
        return np.mean(np.logaddexp(0, z) - y*z) + strength/2*np.dot(w[1:], w[1:]), x.T@(expit(z)-y)/len(y) + np.r_[0, strength*w[1:]]
    w = np.zeros(x.shape[1])
    w[0] = np.log(y.mean()/(1-y.mean()))
    return minimize(loss, w, jac=True, method="L-BFGS-B").x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--help-dir', required=True)
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ref = json.loads((Path(__file__).parents[2]/'assets/ref-v2.json').read_text())
    token_id = {c: i+1 for i, c in enumerate(ref['inventory']['tokens'])}
    labels = [json.loads(l) for l in (Path(args.help_dir)/'labels.jsonl').read_text().splitlines()]
    # Held-out speakers cannot enter even if they also have another train-labelled take.
    held = {l['speaker'] for l in labels if l.get('split') not in ('train', 'training')}
    selected = [l for l in labels if l.get('split') in ('train', 'training') and l['speaker'] not in held and l['kind'] == 'none'
                and not l.get('events') and not l.get('non_errors') and not l.get('exclude')]
    xs, ys, speakers, used = [], [], [], []
    for l in selected:
        file = Path(args.evidence)/(l['id'] + '.evidence.json')
        if not file.exists():
            raise FileNotFoundError(file)
        e = json.loads(file.read_text())
        if e['inventory'] != ref['inventory']['id']:
            raise ValueError('inventory mismatch')
        logp = rows(e, len(token_id))
        alternatives = [ctc_boundaries(logp, [token_id[c] for c in seq], starts) for seq, starts in transcript_options(ref, l['ayahs'])]
        z = logsumexp([z for z, _ in alternatives])
        if not np.isfinite(z):
            continue
        marginal = sum(np.exp(ll-z)*b for ll, b in alternatives)
        ph = e['phonemes']
        # Project a latent first-entry frame to its nearest observed spike onset.
        onsets = np.array([p['start'] for p in ph])
        bins = np.searchsorted((onsets[1:]+onsets[:-1])/2, np.arange(len(logp)))
        soft = np.bincount(bins, weights=marginal, minlength=len(ph))
        for i in range(1, len(ph)):
            x = features(logp, ph[i-1], ph[i], e['frameMs'])
            if x is not None:
                xs.append(x); ys.append(min(1, soft[i])); speakers.append(l['speaker'])
        used.append({'id': l['id'], 'speaker': l['speaker'], 'evidenceSha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        print('aligned', l['id'], 'phones', len(ph), 'boundary mass', float(marginal.sum()), flush=True)
    x, y = np.array(xs), np.array(ys)
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(unique) < 2:
        raise ValueError('At least two authorized none-labelled speakers required for grouped validation')
    fold_count = min(3, len(unique))
    fold = {s: i % fold_count for i, s in enumerate(unique)}
    groups = np.array([fold[s] for s in speakers])
    losses = []
    for strength in (0.001, 0.01, 0.1, 1.0):
        oof = np.zeros(len(y))
        for f in range(fold_count):
            train, valid = groups != f, groups == f
            mean, scale = x[train].mean(axis=0), np.maximum(1e-6, x[train].std(axis=0))
            w = fit(np.c_[np.ones(train.sum()), (x[train]-mean)/scale], y[train], strength)
            oof[valid] = np.c_[np.ones(valid.sum()), (x[valid]-mean)/scale]@w
        loss = np.mean(np.logaddexp(0, oof) - y*oof)
        losses.append((loss, strength, oof))
    loss, strength, oof = min(losses, key=lambda t: t[0])
    def calibration(t):
        z = oof/np.exp(t[0])
        return np.mean(np.logaddexp(0, z)-y*z)
    temperature = float(np.exp(minimize(calibration, [0.], bounds=[(-3, 3)]).x[0]))
    mean, scale = x.mean(axis=0), np.maximum(1e-6, x.std(axis=0))
    w = fit(np.c_[np.ones(len(y)), (x-mean)/scale], y, strength)
    model = {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': w[1:].tolist(), 'intercept': float(w[0]), 'prior': float(y.mean()), 'temperature': temperature}
    report = {'recordings': len(used), 'speakers': len(unique), 'seams': len(y), 'boundaryMass': float(y.sum()), 'heldExcluded': len(held),
              'regularization': strength, 'speakerOOFLogLoss': float(loss), 'pooledLogLoss': float(-np.mean(y*np.log(y.mean())+(1-y)*np.log1p(-y.mean()))),
              'temperature': temperature, 'partitions': {s: fold[s] for s in unique}, 'inputs': used, 'model': model}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'fit.json').write_text(json.dumps(report, indent=2))
    (out/'word-boundary-model.ts').write_text('import type { BoundaryModel } from "./word-boundary.js";\n\n/** Frozen help-train speaker-partition calibration; regenerate with training/word_boundary.py. */\nexport const WORD_BOUNDARY_MODEL: BoundaryModel = '+json.dumps(model, indent=2)+';\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'model', 'partitions')}, indent=2))


if __name__ == '__main__':
    main()
