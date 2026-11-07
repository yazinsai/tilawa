"""Weak-label acoustic/text segment CRF on frozen streaming encoder features.

Run directly in the supervisor image. No engine visits, word cuts, teacher
alignments or practice examples enter training. Each read word consumes a
nonempty segment; exactly the scripted missing word consumes zero frames.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def text_features(words, tokens):
    """Order-sensitive, vocabulary-independent word descriptions."""
    ids = {p: i for i, p in enumerate(tokens)}
    k = len(tokens)
    out = np.zeros((len(words), 3 * k + 33), np.float32)
    for i, word in enumerate(words):
        phones = [p for p in word if p in ids]
        if not phones:
            raise ValueError("Empty word pronunciation")
        for p in phones:
            out[i, ids[p]] += 1 / len(phones)
        out[i, k + ids[phones[0]]] = 1
        out[i, 2 * k + ids[phones[-1]]] = 1
        for a, b in zip(phones, phones[1:]):
            out[i, 3 * k + (ids[a] * k + ids[b]) % 32] += 1 / max(1, len(phones) - 1)
        out[i, -1] = math.log(len(phones))
    return out


def advance(alpha, emissions, log_stay, log_exit):
    """Sum every nonempty segment in O(T), including all monotone boundaries.

    Segment [a,b) has additive acoustic score, a geometric duration probability
    exit * stay**(b-a-1), and alpha[a]. No maximum duration or forced cuts.
    """
    prefix = torch.cat((emissions.new_zeros(1), emissions.cumsum(0)))
    t = torch.arange(len(prefix), device=prefix.device, dtype=prefix.dtype)
    starts = alpha[:-1] - prefix[:-1] - t[:-1] * log_stay
    summed = torch.logcumsumexp(starts, 0)
    return torch.cat((prefix.new_full((1,), -1e30),
                      prefix[1:] + (t[1:] - 1) * log_stay + log_exit + summed))


def partition(emissions, log_stay, log_exit, skip_logit, omitted=None, union=False):
    """Complete reading, a labelled zero-frame omission, or their normalized union.

    Union permits at most one omitted word. Its word prior is uniform. Both
    paths use identical acoustic scores, duration model and frame interval.
    """
    n, frames = emissions.shape
    initial = emissions.new_full((frames + 1,), -1e30)
    initial[0] = 0
    full, skipped = initial, emissions.new_full((frames + 1,), -1e30)
    skip = skip_logit - math.log(n)
    for w in range(n):
        if union:
            # The skip transition consumes zero frames; it cannot skip twice.
            skipped = torch.logaddexp(advance(skipped, emissions[w], log_stay[w], log_exit[w]), full + skip)
        if omitted == w:
            full = full + skip
        else:
            full = advance(full, emissions[w], log_stay[w], log_exit[w])
    return torch.logaddexp(full[-1], skipped[-1]) if union else full[-1]


class SegmentScorer(nn.Module):
    def __init__(self, acoustic_dim, text_dim, hidden=16):
        super().__init__()
        self.acoustic = nn.Linear(acoustic_dim, hidden)
        self.text = nn.Linear(text_dim, hidden)
        # Duration expectation scales with phone count and this recording's rate.
        self.duration = nn.Parameter(torch.zeros(2))
        self.skip_logit = nn.Parameter(torch.tensor(0.))

    def potentials(self, x, q, lengths):
        e = torch.tanh(self.text(q)) @ torch.tanh(self.acoustic(x)).T / math.sqrt(self.acoustic.out_features)
        rate = len(x) / lengths.sum()
        mean = 1 + lengths * rate * torch.exp(self.duration[0])
        exit = (1 / mean).clamp(1e-5, 1 - 1e-5)
        # Learned global duration slope remains small under the selected L2.
        stay = torch.log1p(-exit) * torch.exp(self.duration[1])
        return e, stay, torch.log(-torch.expm1(stay))

    def loss(self, x, q, lengths, omitted):
        e, stay, exit = self.potentials(x, q, lengths)
        return partition(e, stay, exit, self.skip_logit, union=True) - partition(e, stay, exit, self.skip_logit, omitted=omitted)


def fit(examples, train, val, decay, epochs, seed, tokens):
    torch.manual_seed(seed)
    mean = np.concatenate([examples[i]['x'] for i in train]).mean(0)
    std = np.maximum(np.concatenate([examples[i]['x'] for i in train]).std(0), .1)
    model = SegmentScorer(len(mean), 3 * len(tokens) + 33)
    model.skip_logit.data.fill_(math.log(max(1, sum(examples[i]['omitted'] is not None for i in train)) /
                                        max(1, sum(examples[i]['omitted'] is None for i in train))))
    data = [(torch.tensor((e['x'] - mean) / std), torch.tensor(text_features(e['words'], tokens)),
             torch.tensor([len(w) for w in e['words']], dtype=torch.float32), e['omitted']) for e in examples]
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(train), max(1, math.ceil(len(train) / 8))):
            opt.zero_grad()
            loss = torch.stack([model.loss(*data[i]) for i in batch]).mean()
            loss = loss + decay * sum(p.square().mean() for p in model.parameters())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
    with torch.no_grad():
        losses = [float(model.loss(*data[i])) for i in val]
    return model, mean, std, losses


def train(args):
    torch.set_num_threads(1)
    root = Path(args.cache)
    meta = json.loads((root / 'examples.json').read_text())
    examples = [{**e, 'x': np.load(root / e['features'])} for e in meta['examples']]
    speakers = sorted({e['speaker'] for e in examples})
    # Hash groups, never recordings. Selection and preprocessing are refitted per fold.
    groups = {s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 3 for s in speakers}
    records = []
    for decay in (.001, .01, .1):
        losses = []
        for fold in range(3):
            tr = [i for i,e in enumerate(examples) if groups[e['speaker']] != fold]
            va = [i for i,e in enumerate(examples) if groups[e['speaker']] == fold]
            if not tr or not va:
                raise ValueError('Need nonempty speaker-disjoint folds')
            _, _, _, held = fit(examples, tr, va, decay, args.epochs, 719, meta['tokens'])
            losses += held
        records.append({'l2': decay, 'heldOutNll': float(np.mean(losses))})
        print(records[-1], flush=True)
    decay = min(records, key=lambda r:r['heldOutNll'])['l2']
    model, mean, std, _ = fit(examples, list(range(len(examples))), [], decay, args.epochs, 719, meta['tokens'])
    state = {k:v.detach().numpy().tolist() for k,v in model.state_dict().items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = dict(schema=1, tokens=meta['tokens'], inventory=meta['inventory'], featureNode=meta['featureNode'],
                 mean=mean.tolist(), std=std.tolist(), hidden=16, weights=state, encoderSha256=meta['encoderSha256'])
    (out/'segment-scorer.json').write_text(json.dumps(asset))
    report = dict(examples=len(examples), speakers=len(speakers), none=sum(e['omitted'] is None for e in examples),
                  omissions=sum(e['omitted'] is not None for e in examples), folds=groups, selection=records,
                  selectedL2=decay, epochs=args.epochs, parameters=sum(p.numel() for p in model.parameters()),
                  assetSha256=hashlib.sha256((out/'segment-scorer.json').read_bytes()).hexdigest())
    (out/'training.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    train(parser.parse_args())
