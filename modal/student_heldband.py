"""CPU conditional acoustic training, with no practice input or fitted warning threshold.

Preparation (Node primary listener and disjoint spectral observations):
  pnpm --filter @tilawa/sdk exec tsx /app/modal/student_heldband_prepare.ts --data /lab/data/help-train \
      --model /lab/data/models/spliced-clean-int8 --out /scratch/listener-heldband-predictive-evidence/prepared
Training/export:
  python3 modal/student_heldband.py --prepared .../prepared --primary /lab/data/models/spliced-clean-int8 --out .../bundle

Canonical transcripts are anchors only for none-labelled recordings. Student CTC boundary
posteriors initialize acoustic EM; they are neither training features nor the final evidence.
Each conditional predictor reads only its declared visible bands. Boundary prediction at
inference likewise excludes withheld data, and the visible partition normalizes its prior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def topology(labels, stay, ordinary_ctc=False):
    ids = np.zeros(2 * len(labels) + 1, dtype=int)
    ids[1::2] = labels
    n = len(ids)
    self_p = np.ones(n)
    one = np.zeros(n)
    two = np.zeros(n)
    for s in range(n - 1):
        if ordinary_ctc:
            self_p[s] = one[s] = 1
            if s % 2 and s + 2 < n and ids[s] != ids[s + 2]:
                two[s] = 1
        elif s % 2 == 0:
            self_p[s] = one[s] = .5
        else:
            self_p[s] = stay
            direct = s + 2 < n and ids[s] != ids[s + 2]
            one[s] = (1 - stay) / (2 if direct else 1)
            if direct:
                two[s] = one[s]
    return ids, self_p, one, two


def forward_backward(lp, labels, stay, posterior=False, ordinary_ctc=False):
    """Complete CTC chain with scaling. Tiny exhaustive path tests compare this with the SDK."""
    if not len(lp) or not len(labels):
        return -np.inf, None
    ids, self_p, one, two = topology(labels, stay, ordinary_ctc)
    scores = lp[:, ids]
    offset = scores.max(axis=1)
    e = np.exp(np.maximum(scores - offset[:, None], -700))
    alpha = np.zeros_like(e)
    norm = np.zeros(len(e))
    a = np.zeros(len(ids))
    a[:2] = 1 if ordinary_ctc else .5
    for t in range(len(e)):
        a *= e[t]
        norm[t] = a.sum()
        if not norm[t] > 0:
            return -np.inf, None
        a /= norm[t]
        alpha[t] = a
        if t + 1 < len(e):
            b = a * self_p
            b[1:] += a[:-1] * one[:-1]
            b[2:] += a[:-2] * two[:-2]
            a = b
    accept = a[-2:].sum()
    if not accept > 0:
        return -np.inf, None
    value = (np.log(norm) + offset).sum() + np.log(accept)
    if not posterior:
        return float(value), None
    b = np.zeros(len(ids))
    b[-2:] = 1 / accept
    gamma = np.zeros_like(lp)
    for t in reversed(range(len(e))):
        g = alpha[t] * b
        g /= g.sum()
        np.add.at(gamma[t], ids, g)
        if t:
            q = b * e[t]
            b = q * self_p
            b[:-1] += q[1:] * one[:-1]
            b[:-2] += q[2:] * two[:-2]
            b /= norm[t]
    return float(value), gamma


def gaussians(x, mean, variance):
    return -.5 * (np.log(2 * np.pi * variance)[None, :, :] +
                  (x[:, None, :] - mean[None, :, :]) ** 2 / variance[None, :, :]).sum(axis=2)


def moments(x, weights):
    count = weights.sum(axis=0)
    global_mean, global_var = x.mean(axis=0), x.var(axis=0)
    # One effective phone observation of pooled covariance regularizes sparse classes.
    mean = (weights.T @ x + global_mean) / (count[:, None] + 1)
    second = (weights.T @ (x * x) + global_var + global_mean ** 2) / (count[:, None] + 1)
    variance = np.maximum(second - mean * mean, .05 * global_var)
    return mean, np.maximum(variance, 1e-6)


def fit_em(rows, phones):
    x = np.concatenate([r['x'] for r in rows])
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    z = (x - mean) / scale
    weights = []
    for r in rows:
        _, g = forward_backward(r['lp'], r['labels'], .8, posterior=True, ordinary_ctc=True)
        if g is None:
            raise ValueError('Canonical training transcript cannot be completely aligned')
        # Full posterior initialization, with no selected boundary or hard frame transcript.
        weights.append(g[np.minimum(np.arange(len(r['x'])) // 4, len(g) - 1)])
    weights = np.concatenate(weights)
    mu, var = moments(z, weights)
    # Duration prior derives from training recording lengths and complete transcript lengths.
    ratio = sum(len(r['x']) for r in rows) / sum(len(r['labels']) for r in rows)
    stay = float(np.clip(1 - 1 / max(2, ratio - 2), .5, .98))
    for _ in range(6):
        responsibilities, off = [], 0
        for r in rows:
            n = len(r['x'])
            _, g = forward_backward(gaussians(z[off:off+n], mu, var), r['labels'], stay, posterior=True)
            if g is None:
                raise ValueError('Acoustic EM lost complete transcript support')
            responsibilities.append(g)
            off += n
        weights = np.concatenate(responsibilities)
        mu, var = moments(z, weights)
    return dict(mean=mean, scale=scale, phoneMean=mu, phoneVariance=var,
                phoneCount=weights.sum(axis=0), stay=stay), z, weights


def predictors(z, weights, visible, held, ridge):
    x = np.column_stack([np.ones(len(z)), z[:, visible]])
    y = z[:, held]
    dim = x.shape[1]
    regularizer = np.eye(dim)
    base = np.linalg.solve(x.T @ x + ridge * len(x) * regularizer, x.T @ y)
    base_var = np.maximum(((y - x @ base) ** 2).mean(axis=0), 1e-6)
    result = []
    for p in range(weights.shape[1]):
        w = weights[:, p]
        count = w.sum()
        if p == 0 or count == 0:
            b, variance = base, base_var
        else:
            wx = x * w[:, None]
            penalty = ridge * max(1, count) * regularizer
            # Conditional phone predictors shrink to the pronunciation-independent predictor.
            b = np.linalg.solve(x.T @ wx + penalty, wx.T @ y + penalty @ base)
            variance = (((y - x @ b) ** 2).T @ w + ridge * count * base_var) / ((1 + ridge) * count)
            variance = np.maximum(variance, 1e-6)
        result.append(dict(weights=b.tolist(), variance=variance.tolist()))
    return dict(visible=visible, held=held, background=dict(weights=base.tolist(), variance=base_var.tolist()), phones=result)


def conditional(z, q, partition):
    x = np.column_stack([np.ones(len(z)), z[:, partition['visible']]])
    delta = z[:, partition['held']] - x @ np.asarray(q['weights'])
    variance = np.asarray(q['variance'])
    return -.5 * (np.log(2 * np.pi * variance) + delta ** 2 / variance).sum(axis=1)


def predictive(model, raw, labels, partition):
    z = (raw - model['mean']) / model['scale']
    visible = partition['visible']
    v = gaussians(z[:, visible], model['phoneMean'][:, visible], model['phoneVariance'][:, visible])
    c = np.column_stack([conditional(z, p, partition) for p in partition['phones']])
    joint, _ = forward_backward(v + c, labels, model['stay'])
    prior, _ = forward_backward(v, labels, model['stay'])
    return joint - prior - conditional(z, partition['background'], partition).sum()


def partitions(bands, block):
    a = [i for i in range(bands) if (i // block) % 2 == 0]
    b = [i for i in range(bands) if i not in a]
    return [(a, b), (b, a)]


def train(prepared, primary, out):
    meta = json.loads((prepared / 'input.json').read_text())
    manifest = json.loads((primary / 'manifest.json').read_text())
    if manifest['inventory'] != meta['inventory'] or manifest['tokens'] != meta['tokens']:
        raise ValueError('Training inventory differs from the exported primary listener')
    if hashlib.sha256((primary / manifest['file']).read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Primary listener checksum does not match its supplied manifest')
    token_id = {p: i + 1 for i, p in enumerate(meta['tokens'])}
    rows = []
    for path in sorted(prepared.glob('*.json')):
        if path.name == 'input.json':
            continue
        row = json.loads(path.read_text())
        row['labels'] = [token_id[p] for p in row['transcript']]
        row['x'], row['lp'] = np.asarray(row['x']), np.asarray(row['lp'])
        if len(row['x']) <= len(row['labels']):
            raise ValueError('Training recording shorter than its complete pronunciation')
        rows.append(row)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < 3:
        raise ValueError('Insufficient training speakers')
    hold = set(speakers[::5])
    fit = [r for r in rows if r['speaker'] not in hold]
    validation = [r for r in rows if r['speaker'] in hold]
    model, z, weights = fit_em(fit, len(token_id) + 1)
    trials = []
    # Fixed training-only model-selection grid; no practice reading or warning outcomes enter it.
    for block in [1, 3, 12]:
        for ridge in [.1, 1., 10.]:
            ps = [predictors(z, weights, a, b, ridge) for a, b in partitions(z.shape[1], block)]
            scores = [sum(predictive(model, r['x'], r['labels'], p) for p in ps) / len(r['x']) for r in validation]
            # Equal speaker weighting, so a prolific learner cannot choose the acoustic partition.
            value = float(np.mean([np.mean([v for r, v in zip(validation, scores) if r['speaker'] == s]) for s in sorted(hold)]))
            trials.append(dict(block=block, ridge=ridge, heldoutPredictiveGain=value))
            print(trials[-1], flush=True)
    chosen = max(trials, key=lambda t: t['heldoutPredictiveGain'])
    if not np.isfinite(chosen['heldoutPredictiveGain']):
        raise ValueError('No finite complete held-out prediction')
    # Hyperparameters are now frozen; refit on all authorized training speakers, including the selection fold.
    model, z, weights = fit_em(rows, len(token_id) + 1)
    ps = [predictors(z, weights, a, b, chosen['ridge']) for a, b in partitions(z.shape[1], chosen['block'])]
    bundle = dict(schema=1, inventory=meta['inventory'], tokens=meta['tokens'], edges=meta['edges'],
                  **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}, partitions=ps)
    out.mkdir(parents=True, exist_ok=False)
    # Preserve the primary graph, manifest and every already-declared auxiliary asset byte for byte.
    shutil.copytree(primary, out, dirs_exist_ok=True)
    for source in primary.rglob('*'):
        if source.is_file() and hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256((out / source.relative_to(primary)).read_bytes()).digest():
            raise ValueError('Primary bundle copy changed an existing asset')
    dest = out / 'held-band.json'
    dest.write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    report = dict(recordings=len(rows), speakers=len(speakers), selectionSpeakers=len(hold), trials=trials, chosen=chosen,
                  trainingDigest=hashlib.sha256(''.join(p.read_text() for p in sorted(prepared.glob('*.json'))).encode()).hexdigest(),
                  bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out.parent / 'training-report.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    train(args.prepared, args.primary, args.out)
