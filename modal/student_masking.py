"""Training-only simultaneous-masking acoustic emissions; CPU utility.

Input JSONL is extracted with the SDK's maskingFeatures from none-labelled
help-train training speakers and contains full-vocabulary shipped posteriors,
canonical phone ids, speaker, split and exclusion provenance. Mistake scripts
are never interpreted as performed phone transcripts. No practice data is used.
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



def uncertainty(x, strength):
    import numpy as np
    bark_step = (13 * np.arctan(.00076 * 7600) + 3.5 * np.arctan((7600 / 7500) ** 2)) / 24
    out = np.zeros_like(x)
    for d in range(24):
        threshold = np.zeros(len(x))
        for j in range(max(0, d - 3), min(24, d + 4)):
            if d == j:
                continue
            delta = x[:, j] - x[:, d]
            attenuation = 10 ** (-(12 if j < d else 27) * abs(j - d) * bark_step / 10)
            threshold += np.where(delta > 0, np.exp(delta) * attenuation, 0)
        out[:, d] = strength * np.log1p(threshold) ** 2
    return out


def prepare(rows, excluded, vocab):
    import numpy as np
    out = []
    for row in rows:
        if row.get("kind") != "none" or row.get("split") != "train" or row.get("excluded") is not False or row["speaker"] in excluded:
            raise ValueError("Only none-labelled, non-excluded training-speaker recordings may be fitted")
        x, posts = np.array(row["features"], dtype=float), np.array(row["posteriors"], dtype=float)
        phones = np.array(row["phones"], dtype=int)
        if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all() or posts.ndim != 2 or posts.shape[1] != vocab or not np.isfinite(posts).all() or (posts < 0).any():
            raise ValueError("Invalid acoustic observations or posterior inventory")
        if not len(phones) or (phones < 1).any() or (phones >= vocab).any():
            raise ValueError("Canonical pronunciation contains an unsupported phone")
        n = min(len(x), len(posts))
        if not n:
            continue
        placement = forced_labels(np.log(np.maximum(posts[:n], 1e-8)), phones)
        if placement is not None:
            out.append({"speaker": row["speaker"], "x": x[:n], "ids": phones[placement]})
    return out


def fit(rows, vocab, strength, shrinkage):
    import numpy as np
    raw = np.concatenate([r["x"] for r in rows])
    ids = np.concatenate([r["ids"] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    x = (raw - center) / scale
    u = uncertainty(raw, strength) / scale ** 2
    means, variances = np.zeros((vocab, 24)), np.ones((vocab, 24))
    counts = np.bincount(ids, minlength=vocab)
    for p in range(1, vocab):
        if not counts[p]:
            continue
        part, error = x[ids == p], u[ids == p]
        mu, var = part.mean(0), np.maximum(part.var(0), .05)
        # EM for latent unmasked log spectra with observed signal-dependent variance.
        # Pseudocounts regularize toward global standardized mean 0 / variance 1.
        for _ in range(20):
            gain = var / (var + error)
            latent_mean = mu + gain * (part - mu)
            latent_variance = var * error / (var + error)
            mu = latent_mean.sum(0) / (len(part) + shrinkage)
            second = (latent_mean ** 2 + latent_variance).sum(0) + shrinkage
            var = np.maximum(second / (len(part) + shrinkage) - mu ** 2, .05)
        means[p], variances[p] = mu, var
    return {"center": center.tolist(), "scale": scale.tolist(), "means": means.tolist(),
            "variances": variances.tolist(), "counts": counts.tolist()}


def validation(model, rows, strength):
    import numpy as np
    means, var = np.array(model["means"]), np.array(model["variances"])
    by_speaker = {}
    for r in rows:
        x = (r["x"] - model["center"]) / model["scale"]
        v = var[r["ids"]] + uncertainty(r["x"], strength) / np.array(model["scale"]) ** 2
        nll = .5 * (np.log(2 * np.pi * v) + (x - means[r["ids"]]) ** 2 / v)
        by_speaker.setdefault(r["speaker"], []).append(float(nll.mean()))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--excluded-speakers", required=True, help="JSON array from dataset split provenance")
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], excluded, len(args.tokens) + 1)
    speakers = sorted({r["speaker"] for r in rows})
    if len(speakers) < 3:
        raise ValueError("At least three none-labelled training speakers required")
    # Hash order followed by round-robin guarantees three nonempty, speaker-disjoint folds.
    speakers.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for strength in [0, .25, 1, 4]:
        for shrinkage in [8, 32, 128]:
            scores = []
            for fold in range(3):
                train = [r for r in rows if folds[r["speaker"]] != fold]
                held = [r for r in rows if folds[r["speaker"]] == fold]
                scores.append(validation(fit(train, len(args.tokens) + 1, strength, shrinkage), held, strength))
            results.append({"strength": strength, "shrinkage": shrinkage, "foldNll": scores, "meanNll": sum(scores) / 3})
    selected = min(results, key=lambda r: (r["meanNll"], r["strength"], -r["shrinkage"]))
    model = {"schema": 1, "inventory": args.inventory, "tokens": args.tokens, "frameMs": 40, "dimensions": 24,
             "strength": selected["strength"], "maxFrames": 512,
             **fit(rows, len(args.tokens) + 1, selected["strength"], selected["shrinkage"])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out / "masking-emissions.json"
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = {"schema": 1, "file": asset.name, "bytes": asset.stat().st_size,
                  "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
    (out / "masking-asset.json").write_text(json.dumps(descriptor))
    report = {"recordings": len(rows), "speakers": len(speakers), "frames": sum(len(r["x"]) for r in rows),
              "inputSha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              "excludedSpeakers": sorted(excluded), "speakerFolds": folds, "validation": results,
              "selected": selected, "asset": descriptor}
    (out / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
