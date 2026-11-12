"""Speaker-separated, practice-free conditional trajectory fit (one-day bound).

L-BFGS trains full-graph conditional likelihood, not a phone channel or per-word ranker.
Regularization is selected by three speaker folds with both clean and jump controls.
"""
import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from trajectory_crf import compile_graph, conditional_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minutes", type=float, default=50)
    args = parser.parse_args()
    deadline = time.monotonic() + min(args.minutes, 24 * 60) * 60
    raw = gzip.open(args.training, "rb").read()
    data = json.loads(raw)
    if data["version"] != 1 or any(r["split"] != "train" for r in data["provenance"]):
        raise ValueError("Training-only provenance required")
    declared_speakers = {r["speaker"] for r in data["provenance"]}
    if any(r["speaker"] not in declared_speakers for r in data["rows"]):
        raise ValueError("Graph speaker is absent from training provenance")
    features = data["features"]
    zero = np.zeros(len(features))
    rows, excluded = [], Counter()
    for row in data["rows"]:
        pair = (compile_graph(row, len(zero)), compile_graph(row, len(zero), True))
        try:
            loss, _, gap = conditional_loss(pair, zero, False)
        except ValueError:
            excluded[row["kind"]] += 1
            continue
        rows.append({"pair": pair, "speaker": row["speaker"], "kind": row["kind"],
                     "n": len(row["frames"]), "baseline": (loss, gap)})
    speakers = sorted({r["speaker"] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < 3:
        raise ValueError("Need three speakers with reachable transcript graphs")
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    clean = {"none", "joined-none"}
    jumps = {"jump", "repeat", "scripted-jump", "scripted-repeat"}

    def fit(selected, l2):
        # Equal stream weight prevents long recordings dominating; every stream is normalized only after log Z.
        def objective(w):
            if time.monotonic() > deadline:
                raise TimeoutError("Training budget expired; no partially fitted model may be installed")
            value, grad = l2 * np.dot(w, w) / 2, l2 * w
            for row in selected:
                loss, g, _ = conditional_loss(row["pair"], w)
                value += loss / row["n"] / len(selected)
                grad = grad + g / row["n"] / len(selected)
            return value, grad
        result = minimize(objective, zero, jac=True, method="L-BFGS-B",
                          options={"maxiter": 60, "ftol": 1e-8, "gtol": 1e-5})
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)) or (not result.success and result.status != 1):
            raise ValueError("Conditional fit failed: " + result.message)
        return result.x, {"iterations": result.nit, "converged": bool(result.success), "objective": float(result.fun)}

    selections = []
    for l2 in [.1, 1.]:  # Predeclared; never expanded after a practice replay.
        validation = []
        for fold in range(3):
            train = [r for r in rows if folds[r["speaker"]] != fold]
            held = [r for r in rows if folds[r["speaker"]] == fold]
            if not any(r["kind"] in clean for r in held) or not any(r["kind"] in jumps for r in held):
                raise ValueError("Each speaker fold needs unchanged-reading and genuine-movement controls")
            w, report = fit(train, l2)
            for row in held:
                loss, _, gap = conditional_loss(row["pair"], w, False)
                validation.append({"kind": row["kind"], "loss": loss / row["n"], "gap": gap / row["n"],
                                   "baseLoss": row["baseline"][0] / row["n"], "baseGap": row["baseline"][1] / row["n"]})
            print("FOLD", l2, fold, report, flush=True)
        metrics = {"loss": float(np.mean([v["loss"] for v in validation])),
                   "baseLoss": float(np.mean([v["baseLoss"] for v in validation]))}
        for name, kinds in [("clean", clean), ("movement", jumps)]:
            group = [v for v in validation if v["kind"] in kinds]
            metrics[name + "Gap"] = float(np.mean([v["gap"] for v in group]))
            metrics[name + "BaseGap"] = float(np.mean([v["baseGap"] for v in group]))
        eligible = metrics["loss"] < metrics["baseLoss"] and all(metrics[n + "Gap"] <= metrics[n + "BaseGap"] + 1e-9 for n in ["clean", "movement"])
        selections.append({"l2": l2, "eligible": eligible, **metrics})
    choices = [s for s in selections if s["eligible"]]
    chosen = min(choices, key=lambda s: s["loss"]) if choices else None
    weights, final = fit(rows, chosen["l2"]) if chosen else (zero, None)
    output = {"version": 1, "model": {"version": 1, "features": features, "weights": weights.tolist()},
              "selected": chosen, "selection": selections, "final": final, "folds": folds,
              "trainingSha256": hashlib.sha256(raw).hexdigest(), "accepted": len(rows), "excluded": dict(excluded),
              "kinds": dict(Counter(r["kind"] for r in rows)), "provenance": data["provenance"],
              "listenerSha256": data["listenerSha256"], "labelSha256": data["labelSha256"]}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print("FROZEN", json.dumps({k: output[k] for k in ["accepted", "excluded", "selection", "selected"]}), flush=True)


if __name__ == "__main__":
    main()
