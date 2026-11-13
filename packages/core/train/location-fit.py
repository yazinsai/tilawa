"""Matched-compute adversarial and random-corruption structured margin fit.

Configuration and score units are frozen here before any practice inspection.
Only explicit training-speaker recordings and their observation splices enter.
"""
import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from location_margin import FEATURES, Reference, corruption_channel, inner_search


def seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def selected_examples(examples):
    # Bound one supervisor fit equally in both arms. Round-robin across speakers within each
    # kind, selecting by source hash rather than current performance (six intervals per kind).
    groups = collections.defaultdict(list)
    for e in examples:
        groups[e["kind"], e["speaker"]].append(e)
    selected = []
    for kind in sorted({e["kind"] for e in examples}):
        queues = [sorted(groups[kind, speaker], key=lambda r: seed(json.dumps(r, sort_keys=True)))
                  for speaker in sorted({e["speaker"] for e in examples if e["kind"] == kind}, key=seed)]
        chosen = []
        while any(queues) and len(chosen) < 6:
            for queue in queues:
                if queue and len(chosen) < 6:
                    chosen.append(queue.pop(0))
        selected.extend(chosen)
    return selected


def fit(reference, examples, channel, l2, adversarial, epochs, deadline):
    weights = np.zeros(len(FEATURES))
    evaluations = unreachable = step = 0
    ordered = sorted(examples, key=lambda e: seed(json.dumps(e, sort_keys=True)))
    for epoch in range(epochs):
        for e in ordered:
            if time.monotonic() > deadline:
                raise TimeoutError("Fit budget reached; no interrupted model will be published")
            s = seed(f'{epoch}:{e["source"]}:{e["start"]}:{e["target"]}:{e["kind"]}')
            observations = []
            result, count = inner_search(reference, e, weights, channel, s, adversarial, on_selection=observations.append)
            evaluations += count
            step += 1
            grad = l2 * weights
            if result is None:
                unreachable += 1
            elif result[0] + max(1, 30 / len(observations[0])) > 0:
                grad = grad + result[1]
            weights -= (.05 / np.sqrt(step)) * grad
            # A small linear correction, bounded independently of target or practice results.
            weights /= max(1., np.linalg.norm(weights))
    return weights, {"evaluations": evaluations, "unreachable": unreachable, "updates": step}


def evaluate(reference, examples, weights):
    by_kind = collections.defaultdict(list)
    losses = collections.defaultdict(list)
    unreachable = collections.Counter()
    for e in examples:
        result = reference.margin(e, weights)
        if result is None:
            unreachable[e["kind"]] += 1
        else:
            by_kind[e["kind"]].append(result[0])
            losses[e["kind"]].append(max(0, max(1, 30 / len(e["phones"])) + result[0]))
    return {"groups": {k: {"n": len(v), "gap": float(np.mean(v)),
                            "loss": float(np.mean(losses[k]))} for k, v in sorted(by_kind.items())},
            "unreachable": dict(unreachable)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=50)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    if args.epochs != 3:
        raise ValueError("Epoch count is frozen to three for the matched-compute comparison")
    packed = Path(args.training).read_bytes()
    data = json.loads(gzip.decompress(packed))
    if data["version"] != 1 or data["features"] != FEATURES:
        raise ValueError("Training schema mismatch")
    recordings = data["recordings"]
    if any(r.get("split") != "train" or not r.get("speaker") for r in recordings):
        raise ValueError("Only explicit training-speaker recordings permitted")
    speakers = sorted({r["speaker"] for r in recordings}, key=seed)
    if len(speakers) < 3:
        raise ValueError("Three speaker partitions required")
    folds = {speaker: i % 3 for i, speaker in enumerate(speakers)}
    sources = {(r["speaker"], r["hash"]) for r in recordings}
    if any(not e.get("sources") or any((e["speaker"], h) not in sources for h in e["sources"])
           for e in data["examples"]):
        raise ValueError("Example source is not an authorized training recording")
    examples = data["examples"]
    reference = Reference(data["reference"])
    deadline = time.monotonic() + args.minutes * 60
    configs = []
    for l2 in [.1, 1.]:
        rows = []
        for fold in range(3):
            train = selected_examples([e for e in examples if folds[e["speaker"]] != fold])
            valid = selected_examples([e for e in examples if folds[e["speaker"]] == fold])
            if not train or not valid or any(not any(e["kind"] == k for e in group)
                                            for group in [train, valid] for k in ["clean", "jump", "passage-jump"]):
                raise ValueError("Each speaker fold needs clean readings and positive passage movements")
            channel = corruption_channel([r for r in recordings if folds[r["speaker"]] != fold], reference)
            baseline = evaluate(reference, valid, np.zeros(len(FEATURES)))
            # Freeze validation damage before either arm is fitted. Its channel saw only this fold's
            # training speakers, and both models see exactly the same held-out observations.
            robust_valid = []
            for e in valid:
                observations = []
                inner_search(reference, e, np.zeros(len(FEATURES)), channel,
                             seed(f'validation:{e["source"]}:{e["target"]}:{e["kind"]}'), True,
                             on_selection=observations.append)
                robust_valid.append({**e, "phones": observations[0]})
            arms = {}
            for name, adversarial in [("adversarial", True), ("random", False)]:
                w, compute = fit(reference, train, channel, l2, adversarial, args.epochs, deadline)
                arms[name] = {"compute": compute, "validation": evaluate(reference, valid, w),
                              "robust_validation": evaluate(reference, robust_valid, w)}
            if arms["adversarial"]["compute"]["evaluations"] != arms["random"]["compute"]["evaluations"]:
                raise AssertionError("Unmatched inner-search compute")
            rows.append({"fold": fold, "baseline": baseline,
                         "robust_baseline": evaluate(reference, robust_valid, np.zeros(len(FEATURES))), **arms})
            print("FOLD", l2, fold, json.dumps(rows[-1]), flush=True)
        configs.append({"l2": l2, "folds": rows})
    # Choose one shared configuration: architecture, epochs, data and regularization are matched.
    # Held-out training-speaker clean/jump gaps constrain stickiness in BOTH arms.
    def acceptable(config):
        for row in config["folds"]:
            for name in ["adversarial", "random"]:
                v, baseline = row[name]["validation"], row["baseline"]
                if v["unreachable"] != baseline["unreachable"]:
                    return False
                if row[name]["robust_validation"]["unreachable"] != row["robust_baseline"]["unreachable"]:
                    return False
                for kind in ["clean", "jump", "passage-jump"]:
                    if kind not in v["groups"] or kind not in baseline["groups"]:
                        return False
                    if v["groups"][kind]["gap"] > baseline["groups"][kind]["gap"]:
                        return False
        return True
    def loss(config):
        return sum(g["loss"] for row in config["folds"] for name in ["adversarial", "random"]
                   for g in row[name]["robust_validation"]["groups"].values())
    baseline_loss = sum(g["loss"] for row in configs[0]["folds"] for g in row["robust_baseline"]["groups"].values()) * 2
    allowed = [c for c in configs if acceptable(c) and loss(c) < baseline_loss]
    chosen = min(allowed, key=loss) if allowed else None
    channel = corruption_channel(recordings, reference)
    models, final_compute = {}, {}
    for name, adversarial in [("adversarial", True), ("random", False)]:
        if chosen:
            weights, compute = fit(reference, selected_examples(examples), channel, chosen["l2"], adversarial, args.epochs, deadline)
        else:
            weights, compute = np.zeros(len(FEATURES)), {"evaluations": 0, "unreachable": 0, "updates": 0}
        models[name] = {"version": 1, "features": FEATURES, "weights": weights.tolist()}
        final_compute[name] = compute
    if final_compute["adversarial"]["evaluations"] != final_compute["random"]["evaluations"]:
        raise AssertionError("Unmatched final fit")
    result = {"models": models, "selected": chosen["l2"] if chosen else None,
              "reason": "training-fold controls accepted" if chosen else "no shared configuration passed robustness and clean/jump controls",
              "configs": configs, "folds": folds, "compute": final_compute, "channel": channel,
              "examples": len(examples), "finalOptimizationExamples": len(selected_examples(examples)),
              "kinds": dict(collections.Counter(e["kind"] for e in examples)),
              "trainingSha256": hashlib.sha256(packed).hexdigest(), "provenance": data["provenance"],
              "sourceHashes": sorted({r["hash"] for r in recordings}),
              "fitCodeSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "searchCodeSha256": hashlib.sha256(Path(__file__).with_name("location_margin.py").read_bytes()).hexdigest()}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("FROZEN", result["reason"], result["selected"], flush=True)


if __name__ == "__main__":
    main()
