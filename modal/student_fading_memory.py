"""CPU-bounded, frozen contractive reservoir and equal-capacity CTC controls.

Only canonical help-train training-speaker recordings enter this program.
Text supervises the output head; it never drives the fixed reservoir. Shuffling
destroys historical order while retaining the current frame at its original time.
All arms share initialization, updates and a training-speaker selected checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path


def shuffled_order(n):
    order, seed = list(range(n)), 1729
    for i in range(n - 1, 0, -1):
        seed = (seed * 1664525 + 1013904223) & 0xffffffff
        j = seed % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def summaries(x, matrix, rho):
    import numpy as np
    drive = x.astype(np.float64) @ matrix.T
    state = np.zeros(len(matrix))
    memory = np.empty_like(drive)
    for t in range(len(x)):
        state = np.tanh(drive[t] + rho * np.roll(state, 1))
        memory[t] = state
    state = np.zeros(len(matrix))
    shuffled = np.empty_like(drive)
    for t in shuffled_order(len(x)):
        state = np.tanh(drive[t] + rho * np.roll(state, 1))
        shuffled[t] = state
    return [np.concatenate([x, h], axis=1).astype(np.float32)
        for h in (memory, np.tanh(drive), shuffled)]


def fit(args):
    import numpy as np
    import torch
    from torch import nn
    if not 1 <= args.epochs <= 120:
        raise ValueError('Bounded training requires 1..120 epochs')
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    meta = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = [json.loads(s) for s in Path(args.observations).read_text().splitlines() if s]
    if any(r['source'] != 'help-train' or r['kind'] != 'none' or
           r['split'] != 'train' or r.get('excluded') or r['speaker'] in excluded for r in rows):
        raise ValueError('Only authorized none-labelled training takes are eligible')
    speakers = sorted(set(r['speaker'] for r in rows),
        key=lambda s: hashlib.sha256(('fading-memory:' + s).encode()).hexdigest())
    if len(speakers) < 5:
        raise ValueError('Need at least five independent training speakers')
    validation = set(speakers[:max(1, len(speakers) // 5)])
    train = [r for r in rows if r['speaker'] not in validation]
    dev = [r for r in rows if r['speaker'] in validation]
    all_x = np.concatenate([np.asarray(r['features'], dtype=np.float64) for r in train])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    rng = np.random.default_rng(1729)
    hidden, rho = 64, .8
    matrix = rng.normal(size=(hidden, meta['width'])) * (.5 / np.sqrt(meta['width']))
    tokens, vocab = meta['tokens'], len(meta['tokens']) + 1

    def item(r):
        x = (np.asarray(r['features'], dtype=np.float64) - center) / scale
        y = torch.tensor([tokens.index(p) + 1 for p in r['pronunciation']], dtype=torch.long)
        if not len(y) or len(x) < len(y) + int((y[1:] == y[:-1]).sum()):
            raise ValueError('Complete transcript is impossible under ordinary CTC')
        return [torch.tensor(v) for v in summaries(x, matrix, rho)], y

    train_items, dev_items = [item(r) for r in train], [item(r) for r in dev]
    heads = [nn.Linear(meta['width'] + hidden, vocab) for _ in range(3)]
    effective = np.asarray(meta['effectiveWeight'])
    initial_weight = np.concatenate([(scale[:, None] * effective).T, np.zeros((vocab, hidden))], axis=1)
    initial_bias = center @ effective + np.asarray(meta['effectiveBias'])
    for h in heads:
        with torch.no_grad():
            h.weight.copy_(torch.tensor(initial_weight, dtype=torch.float32))
            h.bias.copy_(torch.tensor(initial_bias, dtype=torch.float32))
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.0003, weight_decay=.01) for h in heads]
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    def loss(h, x, y):
        return ctc(h(x).log_softmax(-1)[:, None, :], y, [len(x)], [len(y)])

    def dev_scores():
        with torch.no_grad():
            scores = [sum(float(loss(h, xs[a], y)) for xs, y in dev_items) / len(dev_items)
                      for a, h in enumerate(heads)]
        if not all(np.isfinite(v) for v in scores):
            raise ValueError('Nonfinite training-speaker validation loss')
        return scores

    initial = dev_scores()
    best, selected = float('inf'), None
    checkpoints, history = None, []
    for epoch in range(1, args.epochs + 1):
        training = [0.] * 3
        for i in rng.permutation(len(train_items)):
            xs, y = train_items[i]
            for a, (h, o) in enumerate(zip(heads, optimizers)):
                o.zero_grad()
                v = loss(h, xs[a], y)
                if not torch.isfinite(v):
                    raise ValueError('Nonfinite transcript CTC loss')
                v.backward()
                nn.utils.clip_grad_norm_(h.parameters(), 5)
                o.step()
                training[a] += float(v.detach()) / len(train_items)
        scores = dev_scores()
        # Common checkpoint preserves the identical update budget across controls.
        if sum(scores) / 3 < best:
            best, selected = sum(scores) / 3, epoch
            checkpoints = [{k: v.detach().clone() for k, v in h.state_dict().items()} for h in heads]
        record = dict(epoch=epoch, train=training, validation=scores)
        history.append(record)
        if epoch % 5 == 0:
            print(json.dumps(record), flush=True)
    for h, state in zip(heads, checkpoints):
        h.load_state_dict(state)
    fitted = {k: meta[k] for k in ('schema', 'inventory', 'tokens', 'width',
        'featureOutput', 'logitOutput', 'primarySha256')}
    fitted.update(center=center.tolist(), scale=scale.tolist(), input=matrix.tolist(), rho=rho)
    for name, h in zip(('memory', 'instant', 'shuffled'), heads):
        fitted[name] = dict(weight=h.weight.detach().tolist(), bias=h.bias.detach().tolist())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fading-memory-emissions.json').write_text(json.dumps(fitted, allow_nan=False))
    raw = np.asarray(dev[0]['features'][:32], dtype=np.float64)
    xs = summaries((raw-center)/scale, matrix, rho)
    with torch.no_grad():
        parity = dict(features=raw.tolist(), expected=[h(torch.tensor(x)).log_softmax(-1).tolist()
            for h, x in zip(heads, xs)])
    (out / 'readout-parity.json').write_text(json.dumps(parity, allow_nan=False))
    report = dict(recordings=len(rows), trainRecordings=len(train), validationRecordings=len(dev),
        excludedSpeakers=sorted(excluded),
        trainingSpeakers=[s for s in speakers if s not in validation], validationSpeakers=sorted(validation),
        parameterCounts=[sum(p.numel() for p in h.parameters()) for h in heads], reservoirWidth=hidden,
        contractionBound=rho, epochs=args.epochs, selectedEpoch=selected,
        initialValidationLoss=initial, selectedValidationLoss=history[selected-1]['validation'], history=history,
        observationsSha256=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest())
    (out / 'training-report.json').write_text(json.dumps(report, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--observations', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=40)
    fit(p.parse_args())
