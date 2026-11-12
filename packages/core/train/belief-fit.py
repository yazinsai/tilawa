"""Distill full-sequence location posteriors into a causal candidate RNN and a same-size instantaneous MLP.

Only belief-build's explicitly training-speaker export is accepted. No practice input or score enters fitting,
speaker partitioning, regularization selection or score calibration. Runtime and PyTorch use the same equations.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
HIDDEN = 8
FEATURES = 12
STEPS = 100
SHRINKAGES = (.01, .1)
SCALES = (0., .25, .5, 1., 2.)


class Student(torch.nn.Module):
    def __init__(self, nonrecurrent=False):
        super().__init__()
        torch.manual_seed(17)
        self.input = torch.nn.Parameter(torch.randn(HIDDEN, FEATURES) * .1)
        self.recurrent = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN) * .1)
        self.bias = torch.nn.Parameter(torch.zeros(HIDDEN))
        self.output_bias = torch.nn.Parameter(torch.zeros(HIDDEN))
        self.output = torch.nn.Parameter(torch.randn(HIDDEN) * .1)
        self.nonrecurrent = nonrecurrent

    def forward(self, frames, reset=False):
        previous = torch.zeros(1, HIDDEN)
        logits = []
        for f in frames:
            p = f['predecessor']
            valid_parent = (p >= 0) & (p < len(previous))
            parent = previous[p.clamp(min=0, max=len(previous) - 1)] * valid_parent[:, None]
            if reset:
                parent = parent * 0
            if self.nonrecurrent:
                inner = torch.tanh(f['x'] @ self.input.T + self.bias)
                current = torch.tanh(inner @ self.recurrent.T + self.output_bias)
                score = current @ self.output
            else:
                current = torch.tanh(f['x'] @ self.input.T + parent @ self.recurrent.T + self.bias)
                score = torch.tanh(current + self.output_bias) @ self.output
            previous = current * f['valid'][:, None]
            logits.append(score)
        return logits

    def export(self, scale):
        result = dict(version=1, input=self.input.detach().tolist(), recurrent=self.recurrent.detach().tolist(),
                      bias=self.bias.detach().tolist(), outputBias=self.output_bias.detach().tolist(),
                      output=self.output.detach().tolist(), scale=scale)
        if self.nonrecurrent:
            result['control'] = 'nonrecurrent'
        return result


def prepare(row):
    return {**row, 'frames': [{
        'x': torch.tensor(f['x'], dtype=torch.float32),
        'predecessor': torch.tensor(f['predecessor'], dtype=torch.long),
        'valid': torch.tensor(f['valid'], dtype=torch.bool),
        'costs': torch.tensor(f['costs'], dtype=torch.float32),
        'target': torch.tensor(f['target'], dtype=torch.float32),
    } for f in row['frames']]}


def loss(frames, scores, scale=1.):
    terms = []
    for f, score in zip(frames, scores):
        if not f['target'].sum():
            continue
        logp = torch.log_softmax((-f['costs'] + scale * score).masked_fill(~f['valid'], -1e6), 0)
        terms.append(-(f['target'] * logp).sum())
    return torch.stack(terms).mean()


def fit(rows, nonrecurrent, shrinkage):
    m = Student(nonrecurrent)
    rng = np.random.default_rng(29)
    opt = torch.optim.Adam(m.parameters(), lr=.02)
    for _ in range(STEPS):
        r = rows[int(rng.integers(len(rows)))]
        opt.zero_grad()
        objective = loss(r['frames'], m(r['frames'])) + shrinkage * torch.stack([p.square().mean() for p in m.parameters()]).mean()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
        opt.step()
    return m


def evaluate(m, rows):
    with torch.no_grad():
        results = []
        for r in rows:
            scores = m(r['frames'])
            for scale in SCALES:
                errors = [1. - float(f['target'][int(torch.argmax(-f['costs'] + scale * s))])
                          for f, s in zip(r['frames'], scores) if f['target'].sum()]
                results.append(dict(speaker=r['speaker'], kind=r['kind'], scale=scale,
                                    loss=float(loss(r['frames'], scores, scale)), error=float(np.mean(errors))))
        return results


def train(path, out):
    contents = path.read_bytes()
    data = json.loads(gzip.decompress(contents))
    if data.get('schema') != 1 or not data.get('listenerSha256') or not data.get('labelSha256'):
        raise ValueError('Not an authorized belief-build export')
    if not data.get('provenance') or any(p.get('split') != 'train' or p.get('kind') != 'none' for p in data['provenance']):
        raise ValueError('Only explicitly training-speaker none recordings are allowed')
    rows = [prepare(r) for r in data['rows']]
    speakers = sorted({r['speaker'] for r in rows})
    authorized = {p['speaker'] for p in data['provenance']}
    if not set(speakers).issubset(authorized):
        raise ValueError('A stream uses a speaker absent from authorized provenance')
    if len(speakers) < 3:
        raise ValueError('Need at least three training speakers')
    fold = {s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 3 for s in speakers}
    if len(set(fold.values())) < 3:
        fold = {s: i % 3 for i, s in enumerate(speakers)}
    if any(not any(r['kind'] == 'none' and fold[r['speaker']] == f for r in rows) for f in range(3)):
        raise ValueError('Every training-speaker fold needs ordinary-reading controls')
    models, selection = {}, {}
    for control in (False, True):
        name = 'nonrecurrent' if control else 'recurrent'
        candidates = {}
        for shrinkage in SHRINKAGES:
            reports = []
            for f in range(3):
                training = [r for r in rows if fold[r['speaker']] != f]
                validation = [r for r in rows if fold[r['speaker']] == f]
                m = fit(training, control, shrinkage)
                reports += evaluate(m, validation)
                print('VALIDATION', name, shrinkage, f, flush=True)
            candidates[shrinkage] = reports
        shrinkage = min(SHRINKAGES, key=lambda a: np.mean([v['loss'] for v in candidates[a] if v['scale'] == 1.]))
        reports = candidates[shrinkage]
        clean_base = np.mean([v['error'] for v in reports if v['kind'] == 'none' and v['scale'] == 0.])
        allowed = [s for s in SCALES if np.mean([v['error'] for v in reports if v['kind'] == 'none' and v['scale'] == s]) <= clean_base + 1e-8]
        scale = min(allowed, key=lambda s: (np.mean([v['loss'] for v in reports if v['scale'] == s]), s))
        models[name] = fit(rows, control, shrinkage).export(scale)
        selection[name] = dict(shrinkage=shrinkage, scale=scale, validation=reports, cleanBaselineError=float(clean_base))
    report = dict(models=models, selection=selection, speakerFolds=fold,
                  parameterCount=sum(p.numel() for p in Student().parameters()),
                  streams=len(rows), speakers=len(speakers), recordings=len(data['provenance']),
                  exclusions=data['exclusions'], provenance=data['provenance'],
                  inputSha256=hashlib.sha256(contents).hexdigest(), labelSha256=data['labelSha256'],
                  listener=data['listener'], listenerSha256=data['listenerSha256'])
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('FROZEN', json.dumps({k: report[k] for k in ('streams', 'speakers', 'recordings', 'parameterCount')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    train(args.training, args.out)
