"""CPU-only fit of acoustic self-similarity, using none-labelled help-train training speakers.

Input rows carry SDK features, full-vocabulary posteriors, canonical phones and
their word indices. Speaker folds freeze regularization before target inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path


def forced_labels(logp, phones):
    """Canonical CTC alignment, with internal blanks assigned to the nearest phone.

    Dense acoustic supervision needs no human boundaries. Repeated canonical
    phones retain distinct articulation states, including their blank separators.
    """
    import numpy as np
    states = [0]
    for p in phones:
        states.extend([p, 0])
    n, length = len(logp), len(states)
    score = np.full(length, -np.inf)
    score[:2] = logp[0, states[:2]]
    back = np.zeros((n, length), dtype=np.int32)
    for t in range(1, n):
        new = np.full(length, -np.inf)
        for s, p in enumerate(states):
            prev = [s]
            if s:
                prev.append(s - 1)
            if s > 1 and p and p != states[s - 2]:
                prev.append(s - 2)
            best = max(prev, key=lambda k: score[k])
            new[s] = score[best] + logp[t, p]
            back[t, s] = best
        score = new
    end = length - 1 if score[-1] >= score[-2] else length - 2
    if not np.isfinite(score[end]):
        return None
    path = [end]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path = np.array(path[::-1])
    peaks = np.flatnonzero(path % 2)
    if not len(peaks):
        return None
    nearest = np.searchsorted(peaks, np.arange(n))
    right = peaks[np.minimum(nearest, len(peaks) - 1)]
    left = peaks[np.maximum(nearest - 1, 0)]
    chosen = np.where(np.arange(n) - left <= right - np.arange(n), left, right)
    return (path[chosen] - 1) // 2


def prepare(rows):
    import numpy as np
    out = []
    for row in rows:
        if row.get("kind") != "none" or row.get("split") != "train" or row.get("excluded"):
            raise ValueError("Only none-labelled training-speaker recordings may be fitted")
        x = np.array(row["features"], dtype=np.float64)
        posts = np.log(np.maximum(np.array(row["posteriors"]), 1e-8))
        n = min(len(x), len(posts))
        placement = forced_labels(posts[:n], row["phones"])
        if placement is None:
            continue
        out.append({"speaker": row["speaker"], "x": x[:n],
                    "ids": np.array(row["phones"])[placement],
                    "words": np.array(row["wordIds"])[placement]})
    return out


def fit(rows, vocab, shrinkage):
    import numpy as np
    all_x = np.concatenate([r["x"] for r in rows])
    center, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    dim = all_x.shape[1]
    counts, sums = np.zeros(vocab), np.zeros((vocab, dim))
    squares = sums.copy()
    pc = np.zeros((vocab, vocab))
    ps, pq = pc.copy(), pc.copy()
    for r in rows:
        x = (r["x"] - center) / scale
        for p in range(vocab):
            part = x[r["ids"] == p]
            counts[p] += len(part)
            sums[p] += part.sum(0)
            squares[p] += (part ** 2).sum(0)
        for word in np.unique(r["words"]):
            keep = r["words"] == word
            z, ids = x[keep], r["ids"][keep]
            if len(z) < 3:
                continue
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6)
            a, b = np.triu_indices(len(z), k=2)
            sim = (z @ z.T)[a, b]
            np.add.at(pc, (ids[a], ids[b]), 1)
            np.add.at(ps, (ids[a], ids[b]), sim)
            np.add.at(pq, (ids[a], ids[b]), sim ** 2)
    # Similarity is symmetric; phoneme placement remains ordered at inference.
    pc += pc.T.copy()
    ps += ps.T.copy()
    pq += pq.T.copy()
    null_mean = ps.sum() / pc.sum()
    null_var = max(pq.sum() / pc.sum() - null_mean ** 2, .01)
    means = sums / (counts[:, None] + shrinkage)
    variances = np.maximum((squares + shrinkage) / (counts[:, None] + shrinkage) - means ** 2, .05)
    pm, pv = np.zeros((vocab, vocab)), np.zeros((vocab, vocab))
    for same in [False, True]:
        mask = np.eye(vocab, dtype=bool) == same
        total = pc[mask].sum()
        mean = ps[mask].sum() / total if total else null_mean
        second = pq[mask].sum() / total if total else null_var + mean ** 2
        pm[mask] = (ps[mask] + shrinkage * mean) / (pc[mask] + shrinkage)
        pv[mask] = np.maximum((pq[mask] + shrinkage * second) / (pc[mask] + shrinkage) - pm[mask] ** 2, .01)
    return {"dimensions": dim, "center": center.tolist(), "scale": scale.tolist(),
            "means": means.tolist(), "variances": variances.tolist(),
            "pairMeans": pm.tolist(), "pairVariances": pv.tolist(),
            "nullMean": float(null_mean), "nullVariance": float(null_var)}


def validation(model, rows):
    import numpy as np
    center, scale = np.array(model["center"]), np.array(model["scale"])
    pm, pv = np.array(model["pairMeans"]), np.array(model["pairVariances"])
    scores = []
    for r in rows:
        x = (r["x"] - center) / scale
        z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        terms = []
        for word in np.unique(r["words"]):
            keep = r["words"] == word
            zz, ids = z[keep], r["ids"][keep]
            a, b = np.triu_indices(len(zz), k=2)
            if not len(a):
                continue
            sim = (zz @ zz.T)[a, b]
            var, mean = pv[ids[a], ids[b]], pm[ids[a], ids[b]]
            terms.append(float(np.mean(.5 * (np.log(2 * np.pi * var) + (sim - mean) ** 2 / var))))
        if terms:
            scores.append(float(np.mean(terms)))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--inventory", required=True)
    args = parser.parse_args()
    rows = prepare([json.loads(line) for line in Path(args.input).open()])
    speakers = sorted({r["speaker"] for r in rows})
    if len(speakers) < 3:
        raise ValueError("At least three none-labelled training speakers are required")
    folds = {s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 3 for s in speakers}
    if len(set(folds.values())) < 3:
        folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for shrinkage in [8, 32, 128]:
        scores = []
        for fold in sorted(set(folds.values())):
            train = [r for r in rows if folds[r["speaker"]] != fold]
            held = [r for r in rows if folds[r["speaker"]] == fold]
            scores.append(validation(fit(train, len(args.tokens) + 1, shrinkage), held))
        results.append({"shrinkage": shrinkage, "foldNll": scores, "mean": sum(scores) / len(scores)})
    selected = min(results, key=lambda r: (r["mean"], -r["shrinkage"]))["shrinkage"]
    model = {"schema": 1, "inventory": args.inventory, "tokens": args.tokens, "frameMs": 40,
             **fit(rows, len(args.tokens) + 1, selected), "weight": 1, "beam": 8, "maxFrames": 256}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out / "acoustic-recurrence.json"
    asset.write_text(json.dumps(model, allow_nan=False))
    report = {"recordings": len(rows), "speakers": len(speakers), "speakerFolds": folds,
              "validation": results, "selectedShrinkage": selected,
              "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
    (out / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
