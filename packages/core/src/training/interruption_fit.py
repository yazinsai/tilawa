"""Freeze an acoustic interruption model on authorized training-speaker examples.

Examples come from SDK training/interruption-examples.ts, never practice evidence.
Construction, L2 grid, speaker partitioning and calibration rule are fixed here.
Final validation is reported once and does not select any model parameter.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def partition(speakers):
    ordered = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(ordered) < 6:
        raise ValueError("Need six authorized speakers for disjoint fit/calibration/validation")
    return {s: i % 5 for i, s in enumerate(ordered)}


def fit(x, y, strength):
    def loss(w):
        z = x @ w
        penalty = strength * np.dot(w[1:], w[1:]) / 2
        gradient = x.T @ (expit(z) - y) / len(y) + np.r_[0, strength * w[1:]]
        return np.mean(np.logaddexp(0, z) - y * z) + penalty, gradient
    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(result.message)
    return result.x


def design(x, mean, scale):
    return np.c_[np.ones(len(x)), (x - mean) / scale]


def metrics(y, p, threshold):
    return {"examples": len(y), "positives": int(y.sum()),
            "logLoss": float(np.mean(-y*np.log(np.maximum(1e-12, p))-(1-y)*np.log(np.maximum(1e-12, 1-p)))),
            "falseInterruptionRate": float(np.mean(p[y == 0] >= threshold)),
            "interruptionRecall": float(np.mean(p[y == 1] >= threshold))}


def train(examples):
    if any(e['split'] != 'train' or e['construction'] != 'interruption-v1' for e in examples):
        raise ValueError("Only authorized interruption-v1 training examples may enter")
    x = np.array([e['features'] for e in examples], dtype=float)
    y = np.array([e['label'] for e in examples], dtype=float)
    if x.ndim != 2 or x.shape[1] != 13 or not np.isfinite(x).all() or not set(y).issubset({0, 1}):
        raise ValueError("Invalid training features or labels")
    assignments = partition([e['speaker'] for e in examples])
    groups = np.array([assignments[e['speaker']] for e in examples])
    fitting, calibration, validation = groups >= 2, groups == 1, groups == 0
    for mask in (fitting, calibration, validation):
        if set(y[mask]) != {0, 1}:
            raise ValueError("Every partition needs both classes")
    losses = []
    for strength in (0.01, 0.1, 1.0):
        fold_losses = []
        for f in sorted(set(groups[fitting])):
            a, b = fitting & (groups != f), fitting & (groups == f)
            if not a.any():
                raise ValueError("Need multiple fitting-speaker folds")
            mean, scale = x[a].mean(axis=0), np.maximum(1e-6, x[a].std(axis=0))
            w = fit(design(x[a], mean, scale), y[a], strength)
            z = design(x[b], mean, scale) @ w
            fold_losses.append(float(np.mean(np.logaddexp(0, z)-y[b]*z)))
        losses.append((float(np.mean(fold_losses)), strength))
    loss, strength = min(losses)
    mean, scale = x[fitting].mean(axis=0), np.maximum(1e-6, x[fitting].std(axis=0))
    w = fit(design(x[fitting], mean, scale), y[fitting], strength)
    logits = design(x, mean, scale) @ w
    def calibrate(t):
        z = logits[calibration] / np.exp(t[0])
        return np.mean(np.logaddexp(0, z)-y[calibration]*z)
    t = minimize(calibrate, [0.0], bounds=[(-3, 3)])
    if not t.success:
        raise ValueError(t.message)
    temperature = float(np.exp(t.x[0]))
    p = expit(logits / temperature)
    # Predeclared specificity rule: above the 99th percentile of complete controls.
    negatives = p[calibration & (y == 0)]
    threshold = min(1-1e-9, max(0.5, float(np.quantile(negatives, 0.99, method='higher')) + 1e-9))
    model = {"mean": mean.tolist(), "scale": scale.tolist(), "weights": w[1:].tolist(),
             "intercept": float(w[0]), "temperature": temperature, "threshold": threshold}
    report = {"construction": "interruption-v1", "regularization": strength, "speakerCVLogLoss": loss,
              "partitions": assignments, "fitting": metrics(y[fitting], p[fitting], threshold),
              "calibration": metrics(y[calibration], p[calibration], threshold),
              "validation": metrics(y[validation], p[validation], threshold), "model": model}
    return model, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--examples', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    examples, hashes = [], {}
    for file in sorted(args.examples):
        data = Path(file).read_bytes()
        hashes[file] = hashlib.sha256(data).hexdigest()
        examples.extend(json.loads(line) for line in data.decode().splitlines())
    model, report = train(examples)
    report['inputSha256'] = hashes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'fit.json').write_text(json.dumps(report, indent=2))
    (out/'acoustic-interruption-model.ts').write_text(
        'import type { InterruptionModel } from "./acoustic-interruption.js";\n'
        '/** Frozen interruption-v1, speaker-disjoint fit/calibration/validation. */\n'
        'export const ACOUSTIC_INTERRUPTION_MODEL: InterruptionModel = '+json.dumps(model, indent=2)+';\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('model', 'partitions', 'inputSha256')}, indent=2))


if __name__ == '__main__':
    main()
