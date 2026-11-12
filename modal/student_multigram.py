"""Frozen acoustic readouts with segmentation-marginalized CTC supervision.

Only authorized none-labelled help-train observations enter fitting. Unit mining,
normalization and checkpoint selection use disjoint training-speaker groups. The
matched control retains the same negative output slots and parameter count, but
can transcribe only singleton phones. No chosen segmentation or forced timing is
used by either arm. Runtime weights and graph-score parity examples are exported.
"""
import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def graph(pronunciation, units):
    """CTC automaton: blanks at phone prefixes, active states at unit endings."""
    n = len(pronunciation)
    edges = [(i, i + len(u), k + 1) for i in range(n)
             for k, u in enumerate(units) if pronunciation.startswith(u, i)]
    ending = defaultdict(list)
    for k, (_, end, unit) in enumerate(edges):
        ending[end].append((n + 1 + k, unit))
    predecessors = [[i] + [k for k, _ in ending[i]] for i in range(n + 1)]
    emissions = [0] * (n + 1)
    for k, (start, _, unit) in enumerate(edges):
        predecessors.append([n + 1 + k, start] + [j for j, u in ending[start] if u != unit])
        emissions.append(unit)
    accepting = [n] + [j for j, _ in ending[n]]
    return predecessors, emissions, accepting


def compiled_graph(pronunciation, units):
    import torch
    pred, emit, accept = graph(pronunciation, units)
    sentinel = len(pred)
    width = max(map(len, pred))
    return (torch.tensor([p + [sentinel] * (width-len(p)) for p in pred]),
            torch.tensor(emit), torch.tensor(accept))


def graph_score(log_probs, compiled):
    """Differentiable forward sum with safe derivatives for unreachable states."""
    import torch
    pred, emit, accept = compiled
    a = log_probs.new_full((len(emit) + 1,), -float('inf'))
    a[0] = 0
    for row in log_probs:
        incoming = a[pred]
        reachable = torch.isfinite(incoming).any(1)
        # logsumexp(all -inf) has NaN derivatives. Give unreachable rows a
        # finite temporary value, then restore -inf without creating a path.
        safe = torch.where(reachable[:, None], incoming, torch.zeros_like(incoming))
        updated = torch.logsumexp(safe, 1) + row[emit]
        updated = torch.where(reachable, updated, -float('inf'))
        a = torch.cat((updated, log_probs.new_full((1,), -float('inf'))))
    return torch.logsumexp(a[accept], 0)


def mine_units(rows, tokens):
    counts, speakers = Counter(), defaultdict(set)
    for r in rows:
        text = r['pronunciation']
        for length in (2, 3):
            for i in range(len(text)-length+1):
                u = text[i:i+length]
                counts[u] += 1
                speakers[u].add(r['speaker'])
    frequent = sorted((u for u in counts if counts[u] >= 8 and len(speakers[u]) >= 2),
                      key=lambda u: (-counts[u], u))[:48]
    return list(tokens) + frequent


def fit(args):
    if not 1 <= args.epochs <= 120 or not 1 <= args.seconds <= 43200:
        raise ValueError('Budget is one to 120 epochs, at most one CPU day')
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    start = time.monotonic()
    meta = json.loads(Path(args.metadata).read_text())
    rows = [json.loads(s) for s in Path(args.observations).read_text().splitlines() if s]
    if any(r['source'] != 'help-train' or r['kind'] != 'none' or r['split'] != 'train' for r in rows):
        raise ValueError('Only authorized none-labelled training takes are eligible')
    speakers = sorted(set(r['speaker'] for r in rows),
                      key=lambda s: hashlib.sha256(('multigram:'+s).encode()).hexdigest())
    if len(speakers) < 5:
        raise ValueError('Need at least five independent training speakers')
    validation = set(speakers[:max(1, len(speakers)//5)])
    train_rows = [r for r in rows if r['speaker'] not in validation]
    dev_rows = [r for r in rows if r['speaker'] in validation]
    tokens = meta['tokens']
    units = mine_units(train_rows, tokens)
    if len(units) == len(tokens):
        raise ValueError('No training-supported multigrams')
    all_x = np.concatenate([np.asarray(r['features'], dtype=np.float32) for r in train_rows])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    def item(r):
        text = r['pronunciation']
        x = torch.tensor((np.asarray(r['features'], dtype=np.float32)-center)/scale)
        if not text or any(p not in tokens for p in text) or len(x) < len(text)+sum(a == b for a,b in zip(text,text[1:])):
            raise ValueError('Complete control transcript is impossible under CTC')
        return x, [compiled_graph(text, u) for u in (units, list(tokens))], len(text)
    train, dev = [item(r) for r in train_rows], [item(r) for r in dev_rows]
    vocab, hidden = len(units)+1, 64
    heads = [nn.Sequential(nn.Linear(meta['width'], hidden), nn.Tanh(), nn.Linear(hidden, vocab)) for _ in range(2)]
    heads[1].load_state_dict(heads[0].state_dict())
    for h in heads:
        with torch.no_grad():
            h[2].bias.zero_()
            h[2].bias[0] = 2
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=.01) for h in heads]
    def loss(h, x, g, length):
        return -graph_score(h(x).log_softmax(-1), g)/length
    best, selected, checkpoints, history = [float('inf')]*2, [None]*2, [None]*2, []
    rng = np.random.default_rng(1729)
    for epoch in range(1, args.epochs+1):
        if history and time.monotonic()-start >= args.seconds:
            break
        training = [0., 0.]
        for i in rng.permutation(len(train)):
            x, gs, length = train[i]
            for arm, (h, o, g) in enumerate(zip(heads, optimizers, gs)):
                o.zero_grad()
                v = loss(h, x, g, length)
                if not torch.isfinite(v):
                    raise ValueError('Nonfinite CTC training loss')
                v.backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Nonfinite CTC gradient')
                nn.utils.clip_grad_norm_(h.parameters(), 5)
                o.step()
                training[arm] += float(v.detach())/len(train)
        with torch.no_grad():
            scores = [sum(float(loss(h,x,gs[a],length)) for x,gs,length in dev)/len(dev) for a,h in enumerate(heads)]
        for a in range(2):
            if scores[a] < best[a]:
                best[a], selected[a] = scores[a], epoch
                checkpoints[a] = {k:v.detach().clone() for k,v in heads[a].state_dict().items()}
        record = dict(epoch=epoch, train=training, validation=scores)
        history.append(record)
        print(json.dumps(record), flush=True)
    def export(h, state):
        h.load_state_dict(state)
        return dict(input=h[0].weight.detach().tolist(), inputBias=h[0].bias.detach().tolist(),
                    output=h[2].weight.detach().tolist(), outputBias=h[2].bias.detach().tolist())
    fitted = dict(**meta, units=units, center=center.tolist(), scale=scale.tolist(),
                  multigram=export(heads[0],checkpoints[0]), control=export(heads[1],checkpoints[1]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'multigram-heads.json').write_text(json.dumps(fitted, allow_nan=False))
    x = dev[0][0][:8]
    raw = x.numpy()*scale+center
    with torch.no_grad():
        parity = dict(features=raw.tolist(), expected=[h(x).log_softmax(-1).tolist() for h in heads])
        full, gs, _ = dev[0]
        parity['pronunciation'] = dev_rows[0]['pronunciation']
        parity['scoreFeatures'] = (full.numpy()*scale+center).tolist()
        parity['scores'] = [float(graph_score(h(full).double().log_softmax(-1),g)) for h,g in zip(heads,gs)]
    (out/'readout-parity.json').write_text(json.dumps(parity, allow_nan=False))
    report = dict(recordings=len(rows), trainRecordings=len(train), validationRecordings=len(dev),
                  trainingSpeakers=[s for s in speakers if s not in validation], validationSpeakers=sorted(validation),
                  units=units, parameterCounts=[sum(p.numel() for p in h.parameters()) for h in heads],
                  epochs=len(history), selectedEpochs=selected, selectedValidationLoss=best,
                  elapsedSeconds=time.monotonic()-start, cpuDayBoundThreads=2, history=history,
                  observationsSha256=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest())
    (out/'training-report.json').write_text(json.dumps(report, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='history'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--observations', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--seconds', type=int, default=3000)
    fit(p.parse_args())
