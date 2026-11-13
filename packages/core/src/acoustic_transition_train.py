"""Train matched original-feature and posterior pronunciation-transition scorers on help-train none-labelled training speakers.

Three stages: boundaries (transcript-constrained latent paths), search (acoustic-transition-candidates.ts),
then fit (speaker-grouped validation, frozen calibration and JS parity fixture).
No practice data is read by this estimator.
"""
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



def boundaries(document):
    """Sample word and prefix boundaries under complete transcript-constrained edit paths.

    The latent alignment is only supervision for segmentation. All acoustic inputs remain
    posterior distributions, including blank frames between observed runs.
    """
    examples = []
    for clip_index, clip in enumerate(document["clips"]):
        r = "".join(clip["written"])
        phones = clip["evidence"]["phonemes"]
        heard = "".join(p["p"] for p in phones)
        owner = [w for w, s in enumerate(clip["written"]) for _ in s]
        n, m = len(r), len(heard)
        alpha = np.full((n + 1, m + 1), -np.inf)
        alpha[0, 0] = 0
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
        seed = int.from_bytes(hashlib.sha256(json.dumps(clip, sort_keys=True).encode()).digest()[:8], "big")
        rng = random.Random(seed)
        for sample in range(4):
            i, j = n, m
            path = []
            while i or j:
                choices = []
                if i and j:
                    choices.append((i - 1, j - 1, r[i - 1], j - 1, owner[i - 1], sub(r[i - 1], heard[j - 1])))
                if i:
                    choices.append((i - 1, j, r[i - 1], None, owner[i - 1], delete(r, i - 1)))
                if j:
                    choices.append((i, j - 1, "", j - 1, owner[max(0, i - 1)], insert(r, i, heard[j - 1])))
                logits = [alpha[x[0], x[1]] - 3 * x[5] for x in choices]
                peak = max(logits)
                chosen = rng.choices(choices, weights=[math.exp(x - peak) for x in logits])[0]
                path.append(chosen)
                i, j = chosen[:2]
            grouped = [[] for _ in clip["written"]]
            for step in reversed(path):
                grouped[step[4]].append(step)
            for wi, group in enumerate(grouped):
                for end in sorted({len(group), max(1, len(group) // 2)}):
                    suffix = []
                    count = 0
                    for step in reversed(group[:end]):
                        suffix.append(step)
                        if step[3] is not None:
                            count += 1
                            if count == 16:
                                break
                    suffix.reverse()
                    obs = [s[3] for s in suffix if s[3] is not None]
                    written = "".join(s[2] for s in suffix)
                    if not obs or not written:
                        continue
                    if obs != list(range(obs[0], obs[-1] + 1)):
                        raise ValueError("Non-contiguous latent observations")
                    preceding_ref = sum(len(s[2]) for s in group[:end]) - len(written)
                    examples.append({"speaker": clip["speaker"], "fold": clip["fold"], "written": written,
                                     "start": clip["starts"][wi] + preceding_ref,
                                     "phones": phones[obs[0]:obs[-1] + 1], "clip": clip_index,
                                     "boundarySample": sample})
    return {**document, "examples": examples}


class TransitionScorer(nn.Module):
    def __init__(self, vocab, dim, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.acoustic = nn.Parameter(torch.randn(dim, 3, len(mean)) * 0.1)
        self.acoustic_bias = nn.Parameter(torch.zeros(dim))
        self.text = nn.Parameter(torch.randn(dim, 2, vocab) * 0.1)
        self.text_bias = nn.Parameter(torch.zeros(dim))
        self.joint = nn.Linear(3 * dim, dim)
        self.output = nn.Linear(dim, 1, bias=False)
        self.position = nn.Parameter(torch.tensor(1.0))
        self.dim = dim

    def forward(self, rows, text):
        # One observation interval, a variable-size set of competing pronunciations.
        rows = (rows - self.mean) / self.scale
        a = torch.tanh(F.conv1d(rows.T[None], self.acoustic.permute(0, 2, 1), self.acoustic_bias, padding=1)[0].T)
        energies = []
        for phones in text:
            padded = F.pad(phones, (1, 0), value=0)
            t = self.text_bias[None].expand(len(phones), -1)
            for k in range(2):
                t = t + self.text[:, k, :].T[padded[k:k + len(phones)]]
            t = torch.tanh(t)
            delta = (torch.arange(len(a)) + 0.5)[None] / len(a) - (torch.arange(len(t)) + 0.5)[:, None] / len(t)
            logits = t @ a.T / math.sqrt(self.dim) - self.position.clamp_min(0) * delta.square()
            attended = logits.softmax(1) @ a
            hidden = torch.tanh(self.joint(torch.cat((t, attended, t * attended), 1)))
            energies.append(self.output(hidden).mean())
        return torch.stack(energies)

    def export(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "acoustic": self.acoustic.detach().tolist(), "acousticBias": self.acoustic_bias.detach().tolist(),
                "text": self.text.detach().tolist(), "textBias": self.text_bias.detach().tolist(),
                "joint": self.joint.weight.detach().tolist(), "jointBias": self.joint.bias.detach().tolist(),
                "output": self.output.weight.detach()[0].tolist(), "positionWeight": float(self.position.detach())}


def tensor_examples(document, source):
    inventory = {p: i + 1 for i, p in enumerate(document["tokens"])}
    speakers = {}
    examples = []
    for e in document["examples"]:
        if e["speaker"] in speakers and speakers[e["speaker"]] != e["fold"]:
            raise ValueError("Speaker crosses validation folds")
        speakers[e["speaker"]] = e["fold"]
        if len(e["choices"]) < 2:
            continue
        choices = [torch.tensor([inventory[p] for p in text], dtype=torch.long) for text in e["choices"]]
        examples.append((e["fold"], torch.tensor(e[source], dtype=torch.float32), choices))
    if not examples or len(speakers) < 2:
        raise ValueError("No usable speaker-grouped acoustic comparisons")
    return examples


def fit(examples, vocab, dim, epochs):
    torch.manual_seed(1701)
    rng = random.Random(1701)
    rows = torch.cat([e[1] for e in examples])
    model = TransitionScorer(vocab, dim, rows.mean(0), rows.std(0).clamp_min(0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    for _ in range(epochs):
        order = list(examples)
        rng.shuffle(order)
        optimizer.zero_grad()
        for k, (_, rows, choices) in enumerate(order):
            logits = model(rows, choices)
            loss = F.cross_entropy(logits[None], torch.tensor([0])) / 16
            loss.backward()
            if (k + 1) % 16 == 0 or k + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad()
    return model.eval()


def validation(model, examples):
    errors = 0
    logits = []
    with torch.no_grad():
        for _, rows, choices in examples:
            energy = model(rows, choices)
            errors += int(energy[1:].max() >= energy[0])
            logits.append(energy.detach())
    return errors, logits


def calibrate(logits):
    # Temperature is determined on out-of-speaker predictions, never practice.
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=40)
    def closure():
        optimizer.zero_grad()
        loss = sum(F.cross_entropy((x / log_temperature.exp())[None], torch.tensor([0])) for x in logits) / len(logits)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Degenerate calibration")
    return temperature


def main(mode, source, output):
    torch.set_num_threads(2)
    raw = Path(source).read_text()
    document = json.loads(raw)
    if document.get("trainingOnly") is not True or document.get("folds", 0) < 2:
        raise ValueError("Explicit training-only speaker partitions required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if mode == "boundaries":
        prepared = boundaries(document)
        out.joinpath("acoustic-transition-boundaries.json").write_text(json.dumps(prepared, ensure_ascii=False))
        print(json.dumps({"latentExamples": len(prepared["examples"])}), flush=True)
        return
    if mode != "fit":
        raise ValueError("Use boundaries or fit")
    document["examples"] = [e for e in document["examples"] if len(e["choices"]) >= 2]
    arms = {name: tensor_examples(document, source) for name, source in [("original", "rows"), ("posterior", "controlRows")]}
    if any(a[1].shape != b[1].shape for a, b in zip(arms["original"], arms["posterior"])):
        raise ValueError("Matched scorer inputs must have identical capacity and intervals")
    selections, best = [], None
    for dim in [8, 16]:
        for epochs in [8, 16]:
            summary = {"dim": dim, "epochs": epochs, "arms": {}}
            for name, examples in arms.items():
                details, logits = [], []
                for fold in range(document["folds"]):
                    train = [e for e in examples if e[0] != fold]
                    valid = [e for e in examples if e[0] == fold]
                    if not train or not valid:
                        raise ValueError("Empty speaker partition")
                    model = fit(train, len(document["tokens"]) + 1, dim, epochs)
                    errors, energies = validation(model, valid)
                    logits += energies
                    details.append({"fold": fold, "examples": len(valid), "rankingErrors": errors})
                temperature = calibrate(logits)
                nll = sum(float(F.cross_entropy((x / temperature)[None], torch.tensor([0]))) for x in logits)
                summary["arms"][name] = {"folds": details, "rankingErrors": sum(x["rankingErrors"] for x in details),
                                          "temperature": temperature, "nll": nll}
            selections.append(summary)
            print(json.dumps(summary), flush=True)
            key = (sum(a["rankingErrors"] for a in summary["arms"].values()), sum(a["nll"] for a in summary["arms"].values()), dim, epochs)
            if best is None or key < best[0]:
                best = key, summary
    selected = best[1]
    for name, examples in arms.items():
        model = fit(examples, len(document["tokens"]) + 1, selected["dim"], selected["epochs"])
        temperature = selected["arms"][name]["temperature"]
        artifact = {"version": 1, "inventory": document["inventory"], "tokens": document["tokens"], "frames": 64,
                    **model.export(), "temperature": temperature}
        out.joinpath(f"acoustic-transition-{name}-model.json").write_text(json.dumps(artifact, ensure_ascii=False))
        with torch.no_grad():
            parity = [{"rows": e[1].tolist(), "choices": document["examples"][i]["choices"],
                       "costs": (-model(e[1], e[2]) / temperature * 10 / 3).tolist()}
                      for i, e in enumerate(examples[:8])]
        out.joinpath(f"acoustic-transition-{name}-parity.json").write_text(json.dumps(parity, ensure_ascii=False))
    report = {"trainingOnly": True, "trainingSha256": hashlib.sha256(raw.encode()).hexdigest(),
              "provenance": document["provenance"], "boundarySamples": 4, "examples": len(arms["original"]),
              "selection": selections, "selected": selected,
              "selectionRule": "sum of speaker-held-out ranking errors in both arms, then calibrated NLL, then smaller architecture",
              "costConvention": "temperature frozen from out-of-speaker training predictions; 3 nats/edit",
              "limitations": "finite latent boundary samples; bounded 64-bin trajectories; retained paths from unchanged search"}
    out.joinpath("acoustic-transition-fit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
