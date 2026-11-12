"""Matched frozen-encoder readouts: proper sequence Brier and unbounded log loss.

The transcript is one categorical outcome. Distinct competing collapsed strings
and a remaining-probability bucket partition the rest of the CTC sample space.
The named sequence masses are exact forward sums, never renormalized. All arms
start at the shipped projection and share data, recording weights and updates.
Only training-speaker partitions select checkpoints; no warning labels enter fit.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def prefix_candidates(rows, limit=32, branches=8):
    """Frozen, reference-free bounded prefix search, including the empty string.

    Search scores are only for proposing candidates. Training recomputes every
    named probability by the exact full-vocabulary CTC forward algorithm.
    """
    import numpy as np
    add = np.logaddexp
    beam = {(): (0., -math.inf)}
    for row in rows:
        choices = sorted(set([0, *np.argsort(row)[-branches:].tolist()]))
        nxt = {}
        def put(prefix, blank=-math.inf, nonblank=-math.inf):
            if not math.isfinite(blank) and not math.isfinite(nonblank):
                return
            b, n = nxt.get(prefix, (-math.inf, -math.inf))
            nxt[prefix] = (float(add(b, blank)), float(add(n, nonblank)))
        for prefix, (b, n) in beam.items():
            total = float(add(b, n))
            put(prefix, blank=total+row[0])
            for p in choices:
                if p == 0:
                    continue
                if prefix and p == prefix[-1]:
                    put(prefix, nonblank=n+row[p])
                    put(prefix+(p,), nonblank=b+row[p])
                else:
                    put(prefix+(p,), nonblank=total+row[p])
        beam = dict(sorted(nxt.items(), key=lambda kv: -float(add(*kv[1])))[:limit])
    return list(beam)


def sequence_log_prob(log_probs, sequences):
    """Differentiable exact CTC sum for each distinct collapsed sequence."""
    import torch
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    targets = torch.tensor([p for s in sequences for p in s], dtype=torch.long)
    inputs = log_probs[:, None, :].expand(-1, len(sequences), -1).contiguous()
    return -torch.nn.functional.ctc_loss(inputs, targets,
        torch.full_like(lengths, len(log_probs)), lengths, blank=0,
        reduction='none', zero_infinity=False)


def brier_from_log_probs(sequence_scores):
    import torch
    probabilities = sequence_scores.exp()
    remaining = 1-probabilities.sum()
    # Never rescale or clamp masses: a negative bucket beyond roundoff is a bug.
    if float(remaining.detach()) < -1e-10:
        raise ValueError('Named sequences exceed unit probability (duplicate candidates?)')
    return (1-probabilities[0]).square()+probabilities[1:].square().sum()+remaining.square()


def fit(args):
    import numpy as np
    import torch
    from torch import nn
    if not 1 <= args.epochs <= 120:
        raise ValueError('One-day fitting budget: at most 120 epochs')
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    meta = json.loads(Path(args.metadata).read_text())
    rows = [json.loads(s) for s in Path(args.observations).read_text().splitlines() if s]
    if not rows or any(r['source'] != 'help-train' or r['kind'] != 'none' or
        r['split'] != 'train' or not r['speaker'] or not math.isfinite(r['weight']) or
        r['weight'] <= 0 for r in rows):
        raise ValueError('Only authorized weighted none-labelled training recordings are eligible')
    speakers = sorted(set(r['speaker'] for r in rows),
        key=lambda s: hashlib.sha256(('bounded-sequence:'+s).encode()).hexdigest())
    if len(speakers) < 5:
        raise ValueError('Need five independent training speakers')
    validation = set(speakers[:max(1, len(speakers)//5)])
    train_rows = [r for r in rows if r['speaker'] not in validation]
    dev_rows = [r for r in rows if r['speaker'] in validation]
    all_x = np.concatenate([np.asarray(r['features']) for r in train_rows])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    limits = [8, 32]  # Frozen before acoustic or practice inspection.
    vocab = len(meta['tokens'])+1
    def item(r):
        y = tuple(meta['tokens'].index(p)+1 for p in r['pronunciation'])
        x = torch.tensor((np.asarray(r['features'])-center)/scale, dtype=torch.float64)
        if not y or len(x) < len(y)+sum(a == b for a, b in zip(y, y[1:])):
            raise ValueError('Complete authorized transcript is impossible under CTC')
        proposals = prefix_candidates(np.asarray(r['logProbs']), limits[-1])
        competitors = [s for s in proposals if s != y]
        sequences = [[y, *competitors[:k]] for k in limits]
        if any(len(s) != len(set(s)) for s in sequences):
            raise ValueError('Duplicate categorical outcome')
        return x, sequences, r['weight']
    train, dev = [item(r) for r in train_rows], [item(r) for r in dev_rows]
    # All four arms start at the exact effective deployed projection, including
    # quantization scales. Normalization is absorbed into the same linear head.
    weight, bias = np.asarray(meta['initialWeight']), np.asarray(meta['initialBias'])
    heads = [nn.Linear(meta['width'], vocab).double() for _ in range(4)]
    for h in heads:
        with torch.no_grad():
            h.weight.copy_(torch.tensor(weight*scale))
            h.bias.copy_(torch.tensor(bias+weight@center))
    optimizers = [torch.optim.Adam(h.parameters(), lr=.0003) for h in heads]
    def loss(h, x, seq, arm):
        logp = h(x).log_softmax(-1)
        scores = sequence_log_prob(logp, seq if arm%2 == 0 else seq[:1])
        return brier_from_log_probs(scores) if arm%2 == 0 else -scores[0]
    def evaluate():
        with torch.no_grad():
            total = sum(w for _, _, w in dev)
            return [sum(float(loss(h, x, seq[a//2], a))*w for x, seq, w in dev)/total
                for a, h in enumerate(heads)]
    best, selected = evaluate(), [0]*4
    checkpoints = [{k:v.detach().clone() for k, v in h.state_dict().items()} for h in heads]
    history = [dict(epoch=0, validation=best)]
    # Initial exact mass audit makes Brier saturation visible instead of silently
    # introducing length normalization or a temperature that changes the premise.
    with torch.no_grad():
        initial_transcript_scores = [float(sequence_log_prob(heads[0](x).log_softmax(-1), seq[0][:1])[0])
            for x, seq, _ in train]
    rng = np.random.default_rng(1729)
    for epoch in range(1, args.epochs+1):
        if time.monotonic()-started > 22*3600:
            raise RuntimeError('One-day compute budget reached')
        training, gradients = [0.]*4, [0.]*4
        total = sum(w for _, _, w in train)
        for i in rng.permutation(len(train)):
            x, seq, recording_weight = train[i]
            for arm, (h, optimizer) in enumerate(zip(heads, optimizers)):
                optimizer.zero_grad()
                value = loss(h, x, seq[arm//2], arm)*recording_weight
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite proper sequence training loss')
                value.backward()
                norm = math.sqrt(sum(float(p.grad.square().sum()) for p in h.parameters()))
                if not math.isfinite(norm):
                    raise ValueError('Nonfinite sequence training gradient')
                gradients[arm] += norm/len(train)
                optimizer.step()
                training[arm] += float(value.detach())/total
        scores = evaluate()
        for a in range(4):
            if scores[a] < best[a]:
                best[a], selected[a] = scores[a], epoch
                checkpoints[a] = {k:v.detach().clone() for k,v in heads[a].state_dict().items()}
        record = dict(epoch=epoch, train=training, validation=scores, meanGradientNorm=gradients)
        history.append(record)
        print(json.dumps(record), flush=True)
    def export(h, state):
        h.load_state_dict(state)
        return dict(weight=h.weight.detach().tolist(), bias=h.bias.detach().tolist())
    fitted = {k:v for k,v in meta.items() if k not in ('initialWeight', 'initialBias')}
    fitted.update(center=center.tolist(), scale=scale.tolist(), candidateLimits=limits)
    for name, h, state in zip(('bounded','control','enlargedBounded','enlargedControl'), heads, checkpoints):
        fitted[name] = export(h, state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'bounded-sequence-heads.json').write_text(json.dumps(fitted, allow_nan=False))
    x = dev[0][0][:8]
    with torch.no_grad():
        parity = dict(features=(x.numpy()*scale+center).tolist(),
            expected=[h(x).log_softmax(-1).tolist() for h in heads])
    (out/'readout-parity.json').write_text(json.dumps(parity, allow_nan=False))
    report = dict(recordings=len(rows), trainRecordings=len(train), validationRecordings=len(dev),
        trainingSpeakers=[s for s in speakers if s not in validation], validationSpeakers=sorted(validation),
        parameterCounts=[sum(p.numel() for p in h.parameters()) for h in heads], candidateLimits=limits,
        epochs=args.epochs, selectedEpochs=selected, selectedValidationLoss=best, history=history,
        initialTranscriptLogProbabilityQuantiles=np.quantile(initial_transcript_scores,[0,.25,.5,.75,1]).tolist(),
        observationsSha256=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
        elapsedSeconds=time.monotonic()-started)
    (out/'training-report.json').write_text(json.dumps(report, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='history'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    fit(parser.parse_args())
