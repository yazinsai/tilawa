"""Bounded frozen-encoder CTC readouts, selected using training-speaker groups only.

Both heads have identical layers, parameter counts, input examples and update
budgets. The consonant arm assigns no transcript labels to vowel/duration slots;
those unused slots remain negative softmax classes so capacity is exactly matched.
No alignments, practice examples, warning windows or teacher labels enter fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path


def fit(args):
    if not 1 <= args.epochs <= 120:
        raise ValueError("Training budget is one to 120 epochs")
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    np.random.seed(1729)
    meta = json.loads(Path(args.metadata).read_text())
    rows = [json.loads(s) for s in Path(args.observations).read_text().splitlines() if s]
    if any(r['source'] != 'help-train' or r['kind'] != 'none' or r['split'] != 'train' for r in rows):
        raise ValueError('Only authorized none-labelled training takes are eligible')
    speakers = sorted(set(r['speaker'] for r in rows),
        key=lambda s: hashlib.sha256(('consonant-event:'+s).encode()).hexdigest())
    if len(speakers) < 5:
        raise ValueError('Need at least five independent training speakers')
    validation = set(speakers[:max(1, len(speakers)//5)])
    train = [r for r in rows if r['speaker'] not in validation]
    dev = [r for r in rows if r['speaker'] in validation]
    all_x = np.concatenate([np.asarray(r['features'], dtype=np.float32) for r in train])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    tokens = meta['tokens']
    def item(r):
        x = torch.tensor((np.asarray(r['features'], dtype=np.float32)-center)/scale)
        labels = [torch.tensor([tokens.index(p)+1 for p in r[name]], dtype=torch.long)
            for name in ('consonants', 'pronunciation')]
        for y in labels:
            if not len(y) or len(x) < len(y)+sum(y[1:]==y[:-1]):
                raise ValueError('Complete transcript is impossible under CTC')
        return x, labels
    train, dev = [item(r) for r in train], [item(r) for r in dev]
    vocab, hidden = len(tokens)+1, 64
    heads = [nn.Sequential(nn.Linear(meta['width'], hidden), nn.Tanh(), nn.Linear(hidden, vocab)) for _ in range(2)]
    heads[1].load_state_dict(heads[0].state_dict())
    # A neutral blank prior prevents all-blank collapse during the first updates.
    for h in heads:
        with torch.no_grad():
            h[2].bias.zero_()
            h[2].bias[0] = 2
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=.01) for h in heads]
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
    def loss(h, x, y):
        return ctc(h(x).log_softmax(-1)[:,None,:], y, [len(x)], [len(y)])
    best = [float('inf')]*2
    selected, checkpoints, history = [None]*2, [None]*2, []
    rng = np.random.default_rng(1729)
    for epoch in range(1, args.epochs+1):
        training = [0., 0.]
        for i in rng.permutation(len(train)):
            x, ys = train[i]
            for arm,(h,o,y) in enumerate(zip(heads,optimizers,ys)):
                o.zero_grad()
                v = loss(h,x,y)
                if not torch.isfinite(v): raise ValueError('Nonfinite CTC training loss')
                v.backward()
                nn.utils.clip_grad_norm_(h.parameters(),5)
                o.step()
                training[arm] += float(v.detach())/len(train)
        with torch.no_grad():
            scores = [sum(float(loss(h,x,ys[a])) for x,ys in dev)/len(dev) for a,h in enumerate(heads)]
        for a in range(2):
            if scores[a] < best[a]:
                best[a], selected[a] = scores[a], epoch
                checkpoints[a] = {k:v.detach().clone() for k,v in heads[a].state_dict().items()}
        record = dict(epoch=epoch, train=training, validation=scores)
        history.append(record)
        if epoch%10 == 0: print(json.dumps(record), flush=True)
    def export(h, state):
        h.load_state_dict(state)
        return dict(input=h[0].weight.detach().tolist(), inputBias=h[0].bias.detach().tolist(),
            output=h[2].weight.detach().tolist(), outputBias=h[2].bias.detach().tolist())
    fitted = dict(**meta, center=center.tolist(), scale=scale.tolist(),
        consonant=export(heads[0],checkpoints[0]), control=export(heads[1],checkpoints[1]))
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'consonant-event-heads.json').write_text(json.dumps(fitted,allow_nan=False))
    # Runtime export parity is checked on training-only feature rows, not warning windows.
    x = dev[0][0][:8]
    raw = x.numpy()*scale+center
    with torch.no_grad():
        parity = dict(features=raw.tolist(), expected=[h(x).log_softmax(-1).tolist() for h in heads])
    (out/'readout-parity.json').write_text(json.dumps(parity,allow_nan=False))
    report = dict(recordings=len(rows), trainRecordings=len(train), validationRecordings=len(dev),
        trainingSpeakers=[s for s in speakers if s not in validation], validationSpeakers=sorted(validation),
        parameterCounts=[sum(p.numel() for p in h.parameters()) for h in heads], epochs=args.epochs,
        selectedEpochs=selected, selectedValidationLoss=best, history=history,
        observationsSha256=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest())
    (out/'training-report.json').write_text(json.dumps(report,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--observations',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=120)
    fit(p.parse_args())
