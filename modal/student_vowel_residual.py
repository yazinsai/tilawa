"""CPU-bounded, training-speaker-only vowel residual and matched phone readouts.

The JSONL comes from the deployed streaming extractor, with one acoustic-only
vowel-context distribution shared by every arm. No practice recordings or
warning labels enter this program. Selection minimizes mean validation CTC
across all three equal-capacity arms, never a target-window preference.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path


def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write_json(path, value):
    target = Path(path)
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(target)


def partition(rows, excluded):
    """Provenance validation precedes any acoustic fitting or speaker splitting."""
    speakers = sorted({r['speaker'] for r in rows}, key=stable)
    if len(speakers) < 4 or len(rows) < 8:
        raise ValueError('Need at least four training speakers and eight clean recordings')
    for r in rows:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or
                r.get('kind') != 'none' or r['speaker'] in excluded or r.get('excluded') or
                r.get('events') or r.get('non_errors') or r.get('unsupported')):
            raise ValueError('Unauthorized training provenance')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate recording id')
    valid = set(speakers[:max(1, len(speakers)//5)])
    return [r for r in rows if r['speaker'] not in valid], [r for r in rows if r['speaker'] in valid]


def shuffled_context(rows):
    """Shuffle frame contexts within each training speaker, destroying acoustic
    alignment while preserving that speaker's vowel/phase marginal distribution.
    It is applied only to fit the control predictor; inference uses the same
    unshuffled acoustic distribution as the residual arm.
    """
    import numpy as np
    result = [None] * len(rows)
    for speaker in sorted({r['speaker'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['speaker'] == speaker]
        values = np.concatenate([rows[i]['context'] for i in indices])
        rng = np.random.default_rng(int(stable(speaker)[:8], 16))
        values = values[rng.permutation(len(values))]
        at = 0
        for i in indices:
            n = len(rows[i]['context'])
            result[i] = values[at:at+n]
            at += n
    return result


def nuisance_fit(x, phi, ridge=.01):
    import numpy as np
    # Each recording has equal weight, rather than letting a long ayah dominate.
    xx, xy = np.zeros((10, 10)), np.zeros((10, 32))
    for features, context in zip(x, phi):
        xx += context.T @ context / len(context)
        xy += context.T @ features / len(context)
    return np.linalg.solve(xx + ridge*len(x)*np.eye(10), xy).T


def sequence_log_probability(lp, ids):
    """Independent export parity calculation: sum the complete CTC chain."""
    import numpy as np
    labels = [0, *[x for i in ids for x in (i, 0)]]
    a = np.full(len(labels), -np.inf)
    a[0], a[1] = lp[0, 0], lp[0, ids[0]]
    for row in lp[1:]:
        b = np.full_like(a, -np.inf)
        for s, label in enumerate(labels):
            p = a[s]
            if s:
                p = np.logaddexp(p, a[s-1])
            if s > 1 and label and label != labels[s-2]:
                p = np.logaddexp(p, a[s-2])
            b[s] = p + row[label]
        a = b
    return float(np.logaddexp(a[-1], a[-2]))


def fit(args):
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text())
    extraction = json.loads(Path(args.extraction_report).read_text())
    spent = metadata['exportCpuSeconds'] + extraction['cpuSeconds']
    start = time.process_time()

    def budget():
        if spent + time.process_time()-start >= 86400:
            raise TimeoutError('One CPU day exhausted; do not export a partial scorer')

    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    train, valid = partition(rows, excluded)
    for r in rows:
        x, phi = np.asarray(r['features']), np.asarray(r['context'])
        if (x.ndim != 2 or x.shape[1] != 32 or phi.shape != (len(x), 10) or
                not np.isfinite(x).all() or not np.isfinite(phi).all() or not r['ids'] or
                any(i <= 0 or i > len(metadata['tokens']) for i in r['ids']) or
                len(x) < len(r['ids']) + sum(a == b for a, b in zip(r['ids'], r['ids'][1:]))):
            raise ValueError('Invalid streaming features/context or impossible complete transcript')
    # Fit preprocessing solely on the fitting speakers. All arms use the same
    # 32-feature projection and normalizer; each readout has exactly the same size.
    joined = np.concatenate([r['features'] for r in train])
    center = joined.mean(0)
    scale = np.maximum(joined.std(0), .01)
    xs = [(np.asarray(r['features'])-center)/scale for r in train]
    contexts = [np.asarray(r['context']) for r in train]
    predictor = nuisance_fit(xs, contexts)
    shuffled = nuisance_fit(xs, shuffled_context(train))

    def prepare(population, arm):
        data = []
        for r in population:
            x = (np.asarray(r['features'])-center)/scale
            if arm != 'original':
                x -= np.asarray(r['context']) @ (predictor if arm == 'residual' else shuffled).T
            data.append((torch.tensor(x, dtype=torch.float32), torch.tensor(r['ids'], dtype=torch.long)))
        return data

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(32, 32)
            self.output = nn.Linear(32, len(metadata['tokens'])+1)

        def forward(self, x):
            return self.output(self.hidden(x).tanh()).log_softmax(-1)

    loss_fn = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)

    def loss(model, x, y):
        return loss_fn(model(x)[:, None], y, [len(x)], [len(y)]) / len(x)

    arms = ('residual', 'original', 'shuffled')
    training = {a: prepare(train, a) for a in arms}
    validation = {a: prepare(valid, a) for a in arms}
    choices, best = [], None
    # Fixed finite search, common initialization/data order/update counts. Save
    # each candidate at a common epoch, select the joint validation minimum.
    for decay in (.001, .01):
        models, optimizers = {}, {}
        for arm in arms:
            torch.manual_seed(1729)
            models[arm] = Readout()
            optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=decay)
        rng = np.random.default_rng(1729)
        for epoch in range(80):
            for i in rng.permutation(len(train)):
                budget()
                for arm in arms:
                    model, optimizer = models[arm], optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    value = loss(model, *training[arm][i])
                    if not torch.isfinite(value):
                        raise ValueError('Non-finite complete-transcript loss')
                    value.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    scores = {a: float(torch.stack([loss(models[a], x, y) for x, y in validation[a]]).mean()) for a in arms}
                joint = sum(scores.values()) / len(arms)
                choice = dict(decay=decay, epoch=epoch+1, validation=scores, joint=joint)
                choices.append(choice)
                if best is None or joint < best[0]:
                    best = (joint, choice, {a: copy.deepcopy(models[a].state_dict()) for a in arms})
                print(json.dumps(choice), flush=True)
    budget()
    result = {k: metadata[k] for k in ('schema', 'inventory', 'tokens', 'primarySha256', 'featureOutput', 'projection', 'vowels', 'durations')}
    result.update(center=center.tolist(), scale=scale.tolist(), predictor=predictor.tolist(), shuffledPredictor=shuffled.tolist())
    fixtures = []
    for arm in arms:
        model = Readout()
        model.load_state_dict(best[2][arm])
        sd = best[2][arm]
        result[arm] = dict(weight=sd['hidden.weight'].tolist(), bias=sd['hidden.bias'].tolist(),
                           output=sd['output.weight'].tolist(), intercept=sd['output.bias'].tolist())
        for r, (x, y) in zip(valid[:3], validation[arm][:3]):
            with torch.no_grad():
                score = sequence_log_probability(model(x).numpy(), y.tolist())
            fixtures.append(dict(arm=arm, features=r['features'], context=r['context'], ids=r['ids'], score=score))
    write_json(out/'vowel-residual.json', result)
    write_json(out/'vowel-residual-parity.json', fixtures)
    assets = {}
    for name in ('vowel-residual.json', 'vowel-residual-features.onnx'):
        data = (out/name).read_bytes()
        assets[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write_json(out/'vowel-residual-assets.json', assets)
    report = dict(prediction=dict(listener=-1, overall=-1), mechanism='acoustic vowel-context residual',
                  selection='joint speaker-validation CTC, before any practice inspection', selected=best[1], choices=choices,
                  trainingSpeakers=sorted({r['speaker'] for r in train}), validationSpeakers=sorted({r['speaker'] for r in valid}),
                  excludedSpeakers=sorted(excluded), trainingRecordings=len(train), validationRecordings=len(valid),
                  parametersPerReadout=sum(v.numel() for v in Readout().parameters()),
                  predictorParameters=int(predictor.size), cpuSeconds=spent+time.process_time()-start,
                  inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    write_json(out/'vowel-residual-training-report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('input', 'metadata', 'extraction-report', 'excluded-speakers', 'out'):
        parser.add_argument('--'+name, required=True)
    fit(parser.parse_args())
