"""Differentiable structured correction on the existing local matching graph, training speakers only.

Input comes from reference-edit-build.ts; both members retain identical decoded observations. Movement edges come from the
unchanged local search. A conditional path partition trains location and lexical-event identity, while paired
filter distributions must agree on exactly corresponding unchanged words. No practice input is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
STEPS = 80
SHRINKAGES = (.1, 1., 10.)


def prepare(g, common):
    result = {k: torch.tensor(g[k], dtype=torch.float64) for k in ('ins', 'sub', 'del', 'entry', 'fixed')}
    result.update({k: torch.tensor(g[k], dtype=torch.long) for k in ('source', 'jump')})
    result['allowed'] = torch.tensor(g['allowed'], dtype=torch.bool)
    result['owners'] = torch.tensor([x if x >= 0 else common for x in g['owners']], dtype=torch.long)
    result['initial'] = g['initial']
    result['common'] = common
    return result


def forward(g, theta, restricted=False):
    """Exact sum over paths through the rolling graph. Reference-only closure uses logcumsumexp.

    Jump sources are those exposed by the unchanged search, not transcript-supplied inference destinations.
    The loss differentiates through every retained insertion, match, deletion and jump path.
    """
    scale = theta.exp()
    n = len(g['del'])
    a = torch.full((n,), -10000., dtype=torch.float64)
    a[g['initial']] = 0.
    cumulative = torch.cumsum(g['del'] * scale[2] + g['entry'], dim=0)
    logz = theta.sum() * 0
    occupancy = []
    for t in range(len(g['source'])):
        ins = a - g['ins'][t] * scale[1]
        sub = torch.cat((a.new_full((1,), -10000.), a[:-1])) - g['sub'][t] * scale[0] - g['entry']
        feature = g['jump'][t]
        jump = a[g['source'][t]] - 2 * scale[feature.clamp(min=0)] - g['fixed'][t] - g['sub'][t] * scale[0] - g['entry']
        valid = feature >= 0
        if restricted:
            # The none transcript plus one controlled reference change supplies a lexical edit, not a passage move.
            valid = valid & (feature == 3)
        jump = torch.where(valid, jump, -10000.)
        b = torch.logsumexp(torch.stack((ins, sub, jump)), dim=0)
        if restricted:
            b = torch.where(g['allowed'], b, -10000.)
        a = torch.logcumsumexp(b + cumulative, dim=0) - cumulative
        if restricted:
            a = torch.where(g['allowed'], a, -10000.)
        z = torch.logsumexp(a, dim=0)
        logz = logz + z
        a = a - z
        q = torch.zeros(g['common'] + 1, dtype=torch.float64).scatter_add(0, g['owners'], a.exp())
        occupancy.append(q)
    return logz, torch.stack(occupancy)


def objective(pair, theta):
    za, qa = forward(pair['original'], theta)
    zb, qb = forward(pair['edited'], theta)
    ca, _ = forward(pair['original'], theta, True)
    cb, _ = forward(pair['edited'], theta, True)
    mean = (qa + qb) / 2
    # JS is symmetric; changed/unmapped positions occupy one residual bin, leaving only exact word correspondences.
    js = .5 * (qa * ((qa + 1e-12).log() - (mean + 1e-12).log())
               + qb * ((qb + 1e-12).log() - (mean + 1e-12).log())).sum() / len(qa)
    return (za - ca + zb - cb) / (2 * len(qa)) + js


def fit(rows, shrinkage):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    theta = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=.03)
    for step in range(STEPS):
        batch = [rows[i] for i in rng.choice(len(rows), min(4, len(rows)), replace=False)]
        opt.zero_grad()
        loss = torch.stack([objective(r, theta) for r in batch]).mean() + shrinkage * theta.square().mean()
        loss.backward()
        opt.step()
    return theta.detach()


def train(path, out, ts_out=None):
    contents = path.read_bytes()
    data = json.loads(contents)
    rows = [{**p, 'original': prepare(p['original'], p['common']), 'edited': prepare(p['edited'], p['common'])}
            for p in data['pairs']]
    speakers = sorted({p['speaker'] for p in rows})
    if len(speakers) < 3:
        raise ValueError('Need three training speakers; no fallback to release or practice data')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    validation = {str(v): [] for v in SHRINKAGES}
    zero = torch.zeros(6, dtype=torch.float64)
    with torch.no_grad():
        baseline = float(torch.stack([objective(r, zero) for r in rows]).mean())
    for fold in range(3):
        tr = [r for r in rows if folds[r['speaker']] != fold]
        va = [r for r in rows if folds[r['speaker']] == fold]
        for shrinkage in SHRINKAGES:
            theta = fit(tr, shrinkage)
            with torch.no_grad():
                loss = float(torch.stack([objective(r, theta) for r in va]).mean())
            validation[str(shrinkage)].append(loss)
            print('VALIDATION', fold, shrinkage, loss, theta.tolist(), flush=True)
    selected = min(SHRINKAGES, key=lambda v: (sum(validation[str(v)]), -v))
    theta = fit(rows, selected)
    with torch.no_grad():
        trained = float(torch.stack([objective(r, theta) for r in rows]).mean())
    report = dict(model=dict(logScale=theta.tolist()), recordings=len(data['provenance']), speakers=len(speakers),
                  pairs=len(rows), selectedShrinkage=selected, validation=validation, baselineLoss=baseline,
                  fittedLoss=trained, inputSha256=hashlib.sha256(contents).hexdigest(),
                  listener=data['listener'], listenerSha256=data['listenerSha256'], excluded=data['excluded'])
    out.write_text(json.dumps(report, indent=2) + '\n')
    if ts_out:
        ts_out.write_text('import type { ReferenceEditModel } from "./reference-edit.js";\n\n'
                          '/** Frozen on help-train training speakers; see the experiment report. */\n'
                          'export const REFERENCE_EDIT_MODEL: ReferenceEditModel = '
                          + json.dumps(report['model']) + ';\n')
    print('FROZEN_MODEL', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ts-out', type=Path, help='Emit runtime weights only after training and speaker selection succeed')
    args = parser.parse_args()
    train(args.pairs, args.out, args.ts_out)
