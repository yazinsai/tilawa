"""CPU occurrence calibration on none-labelled help-train training speakers only.

Use `prepare --help-dir /lab/data/help-train --out ...`, the sibling TypeScript
collector, then `fit --cache ... --out ...`. No practice information enters the fit.
The deployment fragments (including runs cut by a model window) are the sampling
unit. Their correctness labels marginalize complete-transcript CTC placements.
"""
import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, expit


def prepare(help_dir, out):
    reference = json.loads(Path("packages/core/assets/ref-v2.json").read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a for a in reference["ayahs"]}
    labels = (help_dir / "labels.jsonl").read_bytes()
    labels_sha256 = hashlib.sha256(labels).hexdigest()
    rows = [json.loads(s) for s in labels.decode().splitlines() if s.strip()]
    non_training_speakers = {r["speaker"] for r in rows if r.get("split") not in ("train", "training")}
    items = []
    for row in rows:
        # Physical help-train export excludes release speakers; reject explicit non-training splits too.
        if row.get("kind") != "none" or row.get("split") not in ("train", "training"):
            continue
        if (row.get("events") or row.get("errors") or row.get("non_errors") or row.get("nonErrors") or
                row.get("unsupported") or row.get("excluded") or row.get("scored") is False):
            continue
        speaker = row.get("speaker")
        if not speaker:
            raise ValueError("Training speaker identity is required")
        if speaker in non_training_speakers:
            raise ValueError("A speaker appears in training and non-training partitions")
        transcripts = [""]
        for key in row["ayahs"]:
            a = ayahs[key]
            offsets = a["o"] + [len(a["p"])]
            words = [a["p"][offsets[i]:offsets[i + 1]] for i in range(len(a["w"]))]
            # Pronunciation-generated length repeats are one articulation, separately per word.
            collapse = lambda s: "".join(p for p, _ in groupby(s))
            connected = "".join(collapse(w) for w in words)
            paused = "".join(collapse(w) for w in words[:-1]) + collapse(a["iso"][-1])
            variants = sorted(set((connected, paused)))
            transcripts = sorted(set(t + v for t in transcripts for v in variants))
        wav = (help_dir / row["wav"]).resolve()
        if not wav.is_file():
            raise FileNotFoundError(wav)
        items.append(dict(id=row["id"], speaker=speaker, wav=str(wav), transcripts=transcripts,
                          source=dict(kind="none", split="train", labelsSha256=labels_sha256)))
    if len(set(i["speaker"] for i in items)) < 4:
        raise ValueError("Fewer than four training speakers with none-labelled recordings")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False))
    print(json.dumps(dict(recordings=len(items), speakers=len(set(i["speaker"] for i in items)))))


def ctc_graph(lp, labels):
    """Forward/backward over ordinary CTC. No best-alignment boundaries are retained."""
    states = np.zeros(2 * len(labels) + 1, dtype=int)
    states[1::2] = labels
    skip = (states != 0) & (states != np.roll(states, 2))
    skip[:2] = False
    before = np.full((len(lp) + 1, len(states)), -np.inf)
    before[0, 0] = 0
    def advance(previous, frame):
        stay = previous
        one = np.r_[-np.inf, previous[:-1]]
        two = np.r_[-np.inf, -np.inf, previous[:-2]] if len(states) > 1 else np.array([-np.inf])
        return np.logaddexp(np.logaddexp(stay, one), np.where(skip, two, -np.inf)) + frame[states]
    for t in range(len(lp)):
        before[t + 1] = advance(before[t], lp[t])
    after = np.full_like(before, -np.inf)
    after[-1, -min(2, len(states)):] = 0
    for t in range(len(lp) - 1, -1, -1):
        nxt = after[t + 1] + lp[t, states]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[np.where(skip[2:], nxt[2:], -np.inf), -np.inf, -np.inf] if len(states) > 1 else np.array([-np.inf])
        after[t] = np.logaddexp(np.logaddexp(nxt, one), two)
    z = after[0, 0]
    return states, before, after, advance, z


def labelled_runs(cache):
    records = []
    metadata = None
    for path in sorted(cache.rglob("*.json")):
        row = json.loads(path.read_text())
        if row.get("source", {}).get("kind") != "none" or row.get("source", {}).get("split") != "train":
            raise ValueError("Cache contains a recording outside none-labelled training speakers")
        identity = {k: row[k] for k in ("modelSha256", "inventory", "tokens", "source")}
        if metadata is not None and identity != metadata:
            raise ValueError("Cache mixes listeners, inventories or training exports")
        metadata = identity
        lp = np.array(row["logProbs"], dtype=float)
        ids = {p: i + 1 for i, p in enumerate(row["tokens"])}
        graphs = [ctc_graph(lp, [ids[p] for p in transcript]) for transcript in row["transcripts"]]
        norm = logsumexp([g[-1] for g in graphs])
        if not np.isfinite(norm):
            raise ValueError(f"No complete transcript alignment: {row['id']}")
        for token in row["phonemes"]:
            a, b = token["start"], token["end"] + 1
            phone = ids[token["p"]]
            y = occurrence_label(graphs, lp, a, b, phone)
            probabilities = np.round(np.exp(lp[a:b, phone]), 4).tolist()
            records.append(dict(speaker=row["speaker"], recording=row["id"], y=y, p=probabilities))
    if not records:
        raise ValueError("Cache contains no emitted training occurrences")
    return records, metadata


def occurrence_label(graphs, lp, a, b, phone):
    """Probability that a matching transcript class occupies this emitted fragment.

    Sum over forms and their CTC placements, rather than selecting a forced
    alignment. This is a transcript-conditioned soft label, not an independent
    human judgment of correctness.
    """
    if not 0 <= a < b <= len(lp) or phone <= 0:
        raise ValueError("Invalid emitted occurrence")
    norm = logsumexp([g[-1] for g in graphs])
    if not np.isfinite(norm):
        raise ValueError("No complete transcript alignment")
    # Complement event: no matching transcript state occupies the fragment.
    missing = []
    for states, before, after, advance, _ in graphs:
        current = before[a].copy()
        for t in range(a, b):
            current = advance(current, lp[t])
            current[states == phone] = -np.inf
        missing.append(logsumexp(current + after[b]))
    return float(-np.expm1(min(0, logsumexp(missing) - norm)))


def correlation(ps, prior, shrinkage):
    n = len(ps)
    if n < 3:
        return prior
    xs = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
    variance = np.sum((xs - xs.mean()) ** 2)
    rho = np.clip(1 - np.sum(np.diff(xs) ** 2) / (2 * variance), 0, .95) if variance > 1e-12 else prior
    weight = (n - 2) / (n - 2 + shrinkage)
    return weight * rho + (1 - weight) * prior


def design(records, mode, shrinkage, rho):
    x = np.array([np.log(np.clip(max(r["p"]), 1e-6, 1 - 1e-6) / (1 - np.clip(max(r["p"]), 1e-6, 1 - 1e-6))) for r in records])
    opportunities = np.array([1 + (len(r["p"]) - 1) * (1 - correlation(r["p"], rho, shrinkage)) /
                              (1 + correlation(r["p"], rho, shrinkage)) for r in records]) if mode == "selection" else np.ones(len(records))
    return x, opportunities, np.array([r["y"] for r in records])


def fit_model(records, mode, shrinkage, penalty):
    rhos = [correlation(r["p"], 0, 0) for r in records if len(r["p"]) >= 3]
    rho = float(np.mean(rhos)) if rhos else 0
    x, n, y = design(records, mode, shrinkage, rho)
    # Each speaker contributes equal mass; no recording's frames masquerade as independent speakers.
    counts = {s: sum(r["speaker"] == s for r in records) for s in set(r["speaker"] for r in records)}
    weights = np.array([1 / counts[r["speaker"]] for r in records]); weights /= weights.sum()
    prior = float(np.clip(weights @ y, 1e-6, 1 - 1e-6))
    center = float(weights @ x)
    sd = max(.25, float(np.sqrt(weights @ (x - center) ** 2)))
    def loss(theta):
        means, sds = theta[:2], np.exp(theta[2:])
        z = (x[:, None] - means) / sds
        logdensity = -np.log(sds) - z ** 2 / 2 + (n[:, None] - 1) * log_ndtr(z)
        data = -weights @ (y * logdensity[:, 0] + (1 - y) * logdensity[:, 1])
        regularizer = penalty * (np.sum(((means - center) / sd) ** 2) + np.sum((theta[2:] - np.log(sd)) ** 2))
        return data + regularizer
    result = minimize(loss, [center + .5, center - .5, np.log(sd), np.log(sd)],
                      method="L-BFGS-B", bounds=[(-20, 20)] * 2 + [(np.log(.25), np.log(20))] * 2)
    if not result.success:
        raise ValueError(result.message)
    return dict(mode=mode, prior=prior, correct=dict(mean=float(result.x[0]), sd=float(np.exp(result.x[2]))),
                incorrect=dict(mean=float(result.x[1]), sd=float(np.exp(result.x[3]))),
                dependence=dict(correlation=rho, shrinkage=shrinkage))


def predict(records, fit):
    x, n, y = design(records, fit["mode"], fit["dependence"]["shrinkage"], fit["dependence"]["correlation"])
    def density(c):
        z = (x - c["mean"]) / c["sd"]
        return -np.log(c["sd"]) - z ** 2 / 2 + (n - 1) * log_ndtr(z)
    odds = np.log(fit["prior"] / (1 - fit["prior"])) + density(fit["correct"]) - density(fit["incorrect"])
    return expit(odds), y


def fit(cache, out):
    records, metadata = labelled_runs(cache)
    speakers = sorted(set(r["speaker"] for r in records), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < 4:
        raise ValueError("At least four independent training speakers required")
    folds = {s: i % 4 for i, s in enumerate(speakers)}
    reports = {}; fits = {}
    for mode in ("selection", "peak"):
        choices = []
        for shrinkage in ((0, 4, 16) if mode == "selection" else (0,)):
            for penalty in (.001, .01, .1):
                errors = []
                for fold in range(4):
                    train = [r for r in records if folds[r["speaker"]] != fold]
                    valid = [r for r in records if folds[r["speaker"]] == fold]
                    model = fit_model(train, mode, shrinkage, penalty)
                    p, y = predict(valid, model); p = np.clip(p, 1e-9, 1 - 1e-9)
                    cross_entropy = -y * np.log(p) - (1 - y) * np.log1p(-p)
                    errors.extend(float(cross_entropy[[r["speaker"] == s for r in valid]].mean()) for s in set(r["speaker"] for r in valid))
                choices.append((float(np.mean(errors)), shrinkage, penalty))
        best = min(choices)
        model = fit_model(records, mode, best[1], best[2])
        model.update(schema=1, modelSha256=metadata["modelSha256"], inventory=metadata["inventory"])
        fits[mode] = model
        reports[mode] = dict(speaker_cross_entropy=best[0], shrinkage=best[1], penalty=best[2], candidates=choices)
    out.mkdir(parents=True, exist_ok=True)
    for mode, model in fits.items():
        (out / f"{mode}.json").write_text(json.dumps(model, indent=2) + "\n")
    report = dict(modelSha256=metadata["modelSha256"], inventory=metadata["inventory"], source=metadata["source"],
                  recordings=len(set(r["recording"] for r in records)), speakers=len(speakers), occurrences=len(records),
                  expected_correct=sum(r["y"] for r in records), partitions=folds, validation=reports)
    (out / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Cache soft labels separately for reproducibility; not a scorer input or supplied evidence file.
    (out / "occurrences.json").write_text(json.dumps(records))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--help-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("fit"); p.add_argument("--cache", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.help_dir, args.out)
    else:
        fit(args.cache, args.out)
