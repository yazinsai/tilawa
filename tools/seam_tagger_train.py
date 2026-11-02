"""Train the seam tagger and write weights the checker can run.

The forward matches `packages/core/src/seam-tagger.ts`: pre-norm, bias-free attention, tanh GELU,
span score = mean softmax P(added) over the span. Calibration is a later pass in that same runtime.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

TOKENS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۦۥَُِ۪ـٲڇں۾ؙۜ"
D, HEADS, FF, MAXLEN, FEAT = 32, 4, 64, 96, 8
TAG_ADDED = 0


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D, eps=1e-5)
        self.q = torch.nn.Linear(D, D, bias=False)
        self.k = torch.nn.Linear(D, D, bias=False)
        self.v = torch.nn.Linear(D, D, bias=False)
        self.wo = torch.nn.Linear(D, D, bias=False)
        self.ln2 = torch.nn.LayerNorm(D, eps=1e-5)
        self.w1 = torch.nn.Linear(D, FF)
        self.w2 = torch.nn.Linear(FF, D)

    def forward(self, x, pad):
        h = self.ln1(x)
        bsz, t, _ = h.shape
        hd = D // HEADS

        def split(y):
            return y.view(bsz, t, HEADS, hd).transpose(1, 2)

        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
        # True pad entries are real tokens. The exported runtime never pads, so this mask is training-only.
        scores = scores.masked_fill(~pad[:, None, None, :], float("-inf"))
        att = torch.softmax(scores, dim=-1)
        mixed = (att @ v).transpose(1, 2).contiguous().view(bsz, t, D)
        x = x + self.wo(mixed)
        h = self.ln2(x)
        x = x + self.w2(F.gelu(self.w1(h), approximate="tanh"))
        return x


class Tagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(len(TOKENS), D)
        self.pos = torch.nn.Embedding(MAXLEN, D)
        self.feat = torch.nn.Linear(FEAT, D)
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.head = torch.nn.Linear(D, 4)

    def forward(self, ids, feat, span, pad):
        """ids [B,T] long (-1 unknown), feat [B,T,8], span [B,T] bool, pad [B,T] bool (True = real)."""
        b, t = ids.shape
        x = self.feat(feat)
        known = ids >= 0
        emb = self.embed(ids.clamp(min=0))
        x = x + emb * known.unsqueeze(-1)
        pos = torch.arange(t, device=ids.device).clamp(max=MAXLEN - 1)
        x = x + self.pos(pos)[None, :, :]
        for layer in self.layers:
            x = layer(x, pad)
        return self.head(x)


def feat_of(tok):
    own = [0.0, 0.0, 0.0, 0.0]
    own[int(tok["own"])] = 1.0
    return [tok["conf"], tok["pause"], *own, tok["step"], math.log1p(tok["spanLen"]) / 4]


def load(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batch(rows, device):
    t = max(len(r["tok"]) for r in rows)
    b = len(rows)
    ids = torch.full((b, t), -1, dtype=torch.long, device=device)
    feat = torch.zeros(b, t, FEAT, device=device)
    span = torch.zeros(b, t, dtype=torch.bool, device=device)
    y = torch.full((b, t), -1, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        n = len(r["tok"])
        ids[i, :n] = torch.tensor([tok["id"] for tok in r["tok"]])
        feat[i, :n] = torch.tensor([feat_of(tok) for tok in r["tok"]])
        a, c = r["span"]
        span[i, a:c] = True
        y[i, :n] = torch.tensor([tok["y"] for tok in r["tok"]])
    return ids, feat, span, y


def export(model, threshold, path):
    def flat(w):
        return w.detach().cpu().reshape(-1).tolist()

    def t_in_out(linear):
        # pytorch [out, in] -> row-major [in, out]
        return flat(linear.weight.t())

    layers = []
    for layer in model.layers:
        layers.append({
            "ln1g": flat(layer.ln1.weight), "ln1b": flat(layer.ln1.bias),
            "q": t_in_out(layer.q), "k": t_in_out(layer.k), "v": t_in_out(layer.v), "wo": t_in_out(layer.wo),
            "ln2g": flat(layer.ln2.weight), "ln2b": flat(layer.ln2.bias),
            "w1": t_in_out(layer.w1), "b1": flat(layer.w1.bias),
            "w2": t_in_out(layer.w2), "b2": flat(layer.w2.bias),
        })
    doc = {
        "tokens": TOKENS, "d": D, "heads": HEADS, "ff": FF, "maxLen": MAXLEN, "threshold": threshold,
        "embed": flat(model.embed.weight), "pos": flat(model.pos.weight),
        "featW": t_in_out(model.feat), "featB": flat(model.feat.bias),
        "layers": layers, "headW": t_in_out(model.head), "headB": flat(model.head.bias),
    }
    json.dump(doc, open(path, "w"))
    print(f"wrote {path}", flush=True)


def main():
    root = os.environ.get("SEAM_ROOT", f"/scratch/{os.environ.get('AUTOLAB_EXP', 'checker-learned-seam-token-tagger')}")
    out = os.environ.get("LAB_OUT", root)
    paths = sys.argv[1:] or [f"{root}/feats.jsonl"]
    # Real clean seams are held out of the update and set the operating point.
    rows = [r for r in load(paths) if r["split"] == "train" and not r["clean"]]

    def is_pos(r):
        a, b = r["span"]
        return any(t["y"] == TAG_ADDED for t in r["tok"][a:b])

    pos = [r for r in rows if is_pos(r)]
    neg = [r for r in rows if not is_pos(r)]
    print(f"train seams {len(rows)} positive {len(pos)} clean {len(neg)}", flush=True)
    if len(pos) < 30 or len(neg) < 30:
        raise SystemExit(f"not enough seams to train ({len(pos)} positive, {len(neg)} clean)")
    rng = random.Random(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    npos, nneg = max(1, len(pos) // 10), max(1, len(neg) // 10)
    hold = pos[:npos] + neg[:nneg]
    train = pos[npos:] + neg[nneg:]
    device = torch.device("cpu")
    torch.set_num_threads(8)
    model = Tagger().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.zeros(4)
    for r in train:
        for tok in r["tok"]:
            counts[tok["y"]] += 1
    weight = torch.tensor(np.clip(counts.sum() / np.maximum(counts, 1), 0.25, 8.0), dtype=torch.float32)
    print("class weights", weight.tolist(), "counts", counts.tolist(), flush=True)
    best = -1.0
    best_state = None
    for epoch in range(40):
        rng.shuffle(train)
        model.train()
        total = 0.0
        n = 0
        for i in range(0, len(train), 32):
            batch_rows = train[i:i + 32]
            ids, feat, span, y = batch(batch_rows, device)
            logits = model(ids, feat, span, y >= 0)  # pad mask is the real-token mask
            ce = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1), weight=weight, ignore_index=-1)
            prob = torch.softmax(logits, dim=-1)[..., TAG_ADDED]
            denom = span.sum(dim=-1).clamp(min=1)
            score = (prob * span).sum(dim=-1) / denom
            target = ((y == TAG_ADDED) & span).any(dim=-1).float()
            bce = F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6), target)
            loss = ce + bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(batch_rows)
            n += len(batch_rows)
        model.eval()
        with torch.no_grad():
            ids, feat, span, y = batch(hold, device)
            logits = model(ids, feat, span, y >= 0)  # pad mask is the real-token mask
            prob = torch.softmax(logits, dim=-1)[..., TAG_ADDED]
            score = ((prob * span).sum(-1) / span.sum(-1).clamp(min=1)).tolist()
            target = ((y == TAG_ADDED) & span).any(-1).tolist()
        ps = [s for s, t in zip(score, target) if t]
        ns = [s for s, t in zip(score, target) if not t]
        gap = (sum(ps) / len(ps) - sum(ns) / len(ns)) if ps and ns else -1
        above = sum(s > max(ns) for s in ps) if ns else 0
        print(f"epoch {epoch} loss {total / max(n, 1):.4f} hold gap {gap:.3f} positives above max clean {above}/{len(ps)}", flush=True)
        if gap > best:
            best = gap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(out, exist_ok=True)
    export(model, 1.0, f"{out}/seam-weights.json")
    export(model, 1.0, f"{root}/seam-weights.json")


if __name__ == "__main__":
    main()
