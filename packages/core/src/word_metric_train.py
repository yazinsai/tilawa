"""CPU-feasible compositional phone-string dual encoder; run only on authorized training data.

After SDK word-metric-decode.ts, invoke on Modal:
  python3 packages/core/src/word_metric_train.py "$LAB_OUT/word-metric-training.json" "$LAB_OUT"
No practice audio, labels, candidate locations or scorer outputs are read here.
"""
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VOWELS = set("َُِ")
HELD = set("اۦۥں۾")
CONFUSABLE = {p for ab in ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"] for p in (ab, ab[::-1])}


def sub(a, b):
    return 0.0 if a == b else 0.5 if a + b in CONFUSABLE or (a in VOWELS and b in VOWELS) else 1.0


def run_at(s, i):
    a, b = i, i + 1
    while a > 0 and s[a - 1] == s[i]:
        a -= 1
    while b < len(s) and s[b] == s[i]:
        b += 1
    return b - a


def long(p, n):
    return (p in HELD and n >= 2) or (p in "نم" and n >= 3)


def delete(r, i):
    return 0.2 if long(r[i], run_at(r, i)) else 0.5 if r[i] in VOWELS else 1.0


def insert(r, i, p):
    n = max(run_at(r, i - 1) if i > 0 and r[i - 1] == p else 0,
            run_at(r, i) if i < len(r) and r[i] == p else 0)
    return 0.2 if long(p, n + 1) else 0.5 if p in VOWELS else 1.0


def logsum(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def latent_words(words, heard, rng, samples=4):
    """Sample transcript-constrained edit paths, integrating uncertain word boundaries.

    Full reference and observation are consumed. The energy uses the deployed directed edit
    prices at its existing 3 nats/edit. Boundary samples are not taken from tracker locations.
    Duration indels keep their separate price and are excluded from identity examples.
    """
    r = "".join(words)
    owner = [w for w, text in enumerate(words) for _ in text]
    n, m = len(r), len(heard)
    alpha = np.full((n + 1, m + 1), -np.inf)
    alpha[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            terms = []
            if i and j:
                terms.append(alpha[i - 1, j - 1] - 3 * sub(r[i - 1], heard[j - 1]))
            if i:
                terms.append(alpha[i - 1, j] - 3 * delete(r, i - 1))
            if j:
                terms.append(alpha[i, j - 1] - 3 * insert(r, i, heard[j - 1]))
            alpha[i, j] = logsum(terms)
    out = []
    for _ in range(samples):
        i, j = n, m
        steps = []
        while i or j:
            choices = []
            if i and j:
                c = sub(r[i - 1], heard[j - 1])
                choices.append((i - 1, j - 1, owner[i - 1], r[i - 1], heard[j - 1], c, False))
            if i:
                c = delete(r, i - 1)
                choices.append((i - 1, j, owner[i - 1], r[i - 1], "", c, c == 0.2))
            if j:
                c = insert(r, i, heard[j - 1])
                w = owner[max(0, i - 1)]
                choices.append((i, j - 1, w, "", heard[j - 1], c, c == 0.2))
            logits = [alpha[x[0], x[1]] - 3 * x[5] for x in choices]
            maximum = max(logits)
            choice = rng.choices(choices, weights=[math.exp(x - maximum) for x in logits])[0]
            steps.append(choice[2:])
            i, j = choice[:2]
        grouped = [[] for _ in words]
        for w, a, b, c, duration in reversed(steps):
            grouped[w].append((a, b, c, duration))
        for group in grouped:
            a = b = ""
            for k, (x, y, _c, duration) in enumerate(group):
                if not duration:
                    a += x
                    b += y
                # Full words and partial frontiers both occur during streaming. The deterministic
                # prefix sample prevents training only completed-word embeddings.
                if a and b and (k == len(group) - 1 or k == len(group) // 2):
                    out.append((a, b))
    return out


def identity_edit(r, o):
    """Original minimum identity charges, with duration charges excluded only from the target."""
    n, m = len(r), len(o)
    cost = np.zeros((n + 1, m + 1))
    identity = np.zeros_like(cost)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            choices = []
            if i and j:
                c = sub(r[i - 1], o[j - 1])
                choices.append((cost[i - 1, j - 1] + c, identity[i - 1, j - 1] + c))
            if i:
                c = delete(r, i - 1)
                choices.append((cost[i - 1, j] + c, identity[i - 1, j] + (0 if c == 0.2 else c)))
            if j:
                c = insert(r, i, o[j - 1])
                choices.append((cost[i, j - 1] + c, identity[i, j - 1] + (0 if c == 0.2 else c)))
            cost[i, j], identity[i, j] = min(choices)
    return float(identity[n, m] * 10)


def hard_candidates(word, vocabulary, inverted):
    grams = {word[i:i + 2] for i in range(len(word) - 1)}
    counts = {}
    for g in grams:
        for v in inverted.get(g, ()):
            if v != word:
                counts[v] = counts.get(v, 0) + 1
    # The preliminary lexical index only bounds CPU work. Final negatives are chosen by phone
    # distance, never by passages, and include short words without a shared bigram.
    pool = sorted(counts, key=lambda v: (-counts[v], abs(len(v) - len(word)), v))[:48]
    if len(pool) < 8:
        pool += sorted((v for v in vocabulary if v != word and v not in pool),
                       key=lambda v: (abs(len(v) - len(word)), v))[:8]
    return sorted(pool, key=lambda v: (identity_edit(v, word), v))[:8]


class Encoder(nn.Module):
    def __init__(self, phones, dim):
        super().__init__()
        self.phones = phones
        self.convolution = nn.Parameter(torch.randn(dim, 3, phones + 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.projection = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, ids, mask):
        pad = torch.full((ids.shape[0], 1), self.phones, dtype=torch.long)
        padded = torch.cat((pad, ids, pad), dim=1)
        # The right neighbour of each final real symbol is a boundary, including in padded batches.
        padded[:, 1:-1] = torch.where(mask, ids, self.phones)
        hidden = self.bias[None, None, :].expand(ids.shape[0], ids.shape[1], -1)
        for k in range(3):
            hidden = hidden + self.convolution[:, k, :].T[padded[:, k:k + ids.shape[1]]]
        hidden = torch.tanh(hidden)
        mean = (hidden * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        maximum = hidden.masked_fill(~mask[:, :, None], -torch.inf).max(1).values
        return F.normalize(self.projection(torch.cat((mean, maximum), 1)), dim=1)

    def export(self):
        return {"convolution": self.convolution.detach().tolist(), "bias": self.bias.detach().tolist(),
                "projection": self.projection.weight.detach().tolist()}


class Dual(nn.Module):
    def __init__(self, phones, dim):
        super().__init__()
        self.written = Encoder(phones, dim)
        self.observed = copy.deepcopy(self.written)


def batch(strings, inventory):
    length = max(map(len, strings))
    ids = torch.full((len(strings), length), len(inventory), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, text in enumerate(strings):
        ids[i, :len(text)] = torch.tensor([inventory[p] for p in text])
        mask[i, :len(text)] = True
    return ids, mask


def distance(model, pairs, inventory):
    r, o = zip(*pairs)
    a = model.written(*batch(r, inventory))
    b = model.observed(*batch(o, inventory))
    control = model.observed(*batch(r, inventory))
    return (((a - b) ** 2).sum(1) - ((a - control) ** 2).sum(1)).clamp_min(0)


def fit(examples, inventory, dim, epochs, negatives):
    torch.manual_seed(1701)
    rng = random.Random(1701)
    model = Dual(len(inventory), dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            rows = [examples[i] for i in order[start:start + 64]]
            # An identical phone-string control accompanies every acoustic observation.
            rows += [(r, r) for r, _ in rows]
            references = [[r] + negatives[r] for r, _ in rows]
            a = model.written(*batch([s for choices in references for s in choices], inventory)).reshape(len(rows), 9, dim)
            b = model.observed(*batch([o for _, o in rows], inventory))
            logits = (a * b[:, None, :]).sum(-1) / 0.1
            positive = ((a[:, 0] - b) ** 2).sum(1).mean()
            loss = F.cross_entropy(logits, torch.zeros(len(rows), dtype=torch.long)) + positive
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def calibration_pairs(examples, negatives):
    return [(candidate, o) for r, o in examples for candidate in [r] + negatives[r]]


def scale_fit(model, pairs, inventory):
    xx = xy = 0.0
    with torch.no_grad():
        for start in range(0, len(pairs), 256):
            rows = pairs[start:start + 256]
            x = distance(model, rows, inventory).numpy()
            y = np.array([identity_edit(r, o) for r, o in rows])
            xx += float(x @ x)
            xy += float(x @ y)
    if xx <= 0:
        raise ValueError("Degenerate training metric cannot be calibrated")
    return max(0.0, xy / xx)


def validate(model, scale, examples, inventory, negatives):
    errors, squared, count = 0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            rows = examples[start:start + 64]
            pairs = calibration_pairs(rows, negatives)
            scores = distance(model, pairs, inventory).numpy().reshape(-1, 9) * scale
            errors += int((scores[:, 1:].min(1) <= scores[:, 0]).sum())
            truth = np.array([identity_edit(r, o) for r, o in pairs]).reshape(-1, 9)
            squared += float(((scores - truth) ** 2).sum())
            count += len(pairs)
    return {"examples": len(examples), "rankingErrors": errors, "squaredError": squared, "pairs": count}


def main(source, output):
    torch.set_num_threads(2)
    text = Path(source).read_text()
    document = json.loads(text)
    if document.get("trainingOnly") is not True or document.get("folds", 0) < 2:
        raise ValueError("Explicit training-only speaker partitions required")
    inventory = {p: i for i, p in enumerate(document["tokens"])}
    speakers = {}
    examples = []
    for clip in document["clips"]:
        if clip["speaker"] in speakers and speakers[clip["speaker"]] != clip["fold"]:
            raise ValueError("Speaker crosses a validation boundary")
        speakers[clip["speaker"]] = clip["fold"]
        seed = int.from_bytes(hashlib.sha256(json.dumps(clip, sort_keys=True).encode()).digest()[:8], "big")
        pairs = latent_words(clip["written"], clip["heard"], random.Random(seed))
        examples += [(clip["fold"], r, o) for r, o in pairs]
    if not examples or any(p not in inventory for _, r, o in examples for p in r + o):
        raise ValueError("Empty training data or unknown phones")
    vocabulary = document["vocabulary"]
    inverted = {}
    for word in vocabulary:
        for gram in {word[i:i + 2] for i in range(len(word) - 1)}:
            inverted.setdefault(gram, []).append(word)
    negatives = {r: hard_candidates(r, vocabulary, inverted) for _, r, _ in examples}
    if any(len(choices) != 8 for choices in negatives.values()):
        raise ValueError("Not enough distinct Quran phone-string negatives")
    print(json.dumps({"clips": len(document["clips"]), "speakers": len(speakers), "examples": len(examples)}), flush=True)
    selection = []
    # Every checkpoint and distance conversion is selected within training speakers. No target
    # replay may precede this selection and final fit.
    for dim in [8, 16]:
        for epochs in [20, 40]:
            results = []
            for fold in range(document["folds"]):
                train = [(r, o) for f, r, o in examples if f != fold]
                valid = [(r, o) for f, r, o in examples if f == fold]
                if not train or not valid:
                    raise ValueError("Empty speaker fold")
                model = fit(train, inventory, dim, epochs, negatives)
                scale = scale_fit(model, calibration_pairs(train, negatives), inventory)
                results.append({"fold": fold, "scale": scale, **validate(model, scale, valid, inventory, negatives)})
            summary = {"dim": dim, "epochs": epochs, "folds": results,
                       "rankingErrors": sum(x["rankingErrors"] for x in results),
                       "squaredError": sum(x["squaredError"] for x in results)}
            selection.append(summary)
            print(json.dumps(summary), flush=True)
    selected = min(selection, key=lambda x: (x["rankingErrors"], x["squaredError"], x["dim"], x["epochs"]))
    all_pairs = [(r, o) for _, r, o in examples]
    model = fit(all_pairs, inventory, selected["dim"], selected["epochs"], negatives)
    scale = scale_fit(model, calibration_pairs(all_pairs, negatives), inventory)
    artifact = {"version": 1, "inventory": document["inventory"], "tokens": document["tokens"],
                "written": model.written.export(), "observed": model.observed.export(), "scale": scale}
    report = {"trainingOnly": True, "trainingSha256": hashlib.sha256(text.encode()).hexdigest(),
              "provenance": document["provenance"], "boundarySamples": 4, "examples": len(examples),
              "selection": selection, "selected": selected, "scale": scale,
              "boundaryModel": "transcript-constrained forward-summed directed edit energies; four posterior path samples",
              "selectionRule": "fewest speaker-held-out retrieval errors, then cost squared error, then smaller model",
              "costConversion": "nonnegative training least-squares slope, zero identical-string control; weighted edit units",
              "limitations": "finite boundary sampling; hard-negative shortlist; retained original runtime alignments"}
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "word-metric-model.json").write_text(json.dumps(artifact, ensure_ascii=False))
    Path(output, "word-metric-fit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # A cross-language parity fixture comes from training, not practice.
    sample = all_pairs[:16]
    with torch.no_grad():
        distances = distance(model, sample, inventory).tolist()
    Path(output, "word-metric-parity.json").write_text(json.dumps([
        {"written": r, "observed": o, "cost": d * scale} for (r, o), d in zip(sample, distances)
    ], ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
