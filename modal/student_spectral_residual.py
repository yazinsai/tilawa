"""CPU fit of acoustic information discarded by the shipped mel aggregation.

Only help-train training-speaker none-labelled recordings supply phone supervision.
All conditional prediction, covariance and regularization fits are training-only.
No Modal imports or filesystem access at module import time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EPS = np.finfo(np.float32).eps


def clean_help(r):
    return (r.get("split") == "train" and r.get("kind") == "none" and r.get("scored") is True
            and bool(r.get("speaker")) and not any(r.get(k) for k in
            ("events", "non_errors", "unsupported", "excluded")))


def mel_matrix():
    mel = lambda hz: 1127 * np.log1p(hz / 700)
    lo, hi = mel(20), mel(7600)
    edges = np.linspace(lo, hi, 82)
    f = mel(np.arange(256) * 16000 / 512)
    return np.maximum(0, np.minimum((f[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
        (edges[2:, None] - f[None]) / (edges[2:] - edges[1:-1])[:, None]))


def null_projection(w):
    # Exclude bins outside all mel filters: this tests within-band detail only.
    used = np.flatnonzero(w.sum(0) > 0)
    _, s, vh = np.linalg.svd(w[:, used], full_matrices=True)
    rank = np.count_nonzero(s > s[0] * 1e-10)
    q = np.zeros((256, len(used) - rank))
    q[used] = vh[rank:].T
    if np.abs(w @ q).max() > 1e-9:
        raise RuntimeError("Projection leaks observable mel information")
    return q


def frontend(wave, rate):
    if wave.ndim == 2:
        wave = wave.mean(1)
    if rate != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // g, rate // g).astype(np.float32)
    n = max(0, 1 + (len(wave) - 400) // 160)
    frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::160][:n].astype(np.float64)
    frames -= frames.mean(1, keepdims=True)
    pre = frames - .97 * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    window = (.5 - .5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** .85
    power = np.abs(np.fft.rfft(pre * window, n=512))[:, :256] ** 2
    mel = np.log(np.maximum(power @ mel_matrix().T, EPS)).astype(np.float32)
    # Normalization depends only on the already available mel features.
    energies = np.exp(mel.astype(np.float64))
    total = energies.sum(1, keepdims=True)
    return mel, power / total, np.concatenate([np.ones((n, 1)), energies / total], axis=1)


def stream(mel, manifest, session):
    outputs = [o.name for o in session.get_outputs()]
    states = {s["name"]: np.zeros(s["shape"], dtype=s["dtype"]) for s in manifest["states"]}
    total = (int((len(mel) - 7) // 2) + 1) // 2
    result = []
    consumed = emitted = 0
    while emitted < total:
        x = np.full((1, manifest["window"], 80), manifest["fbank"]["padValue"], np.float32)
        part = mel[consumed:consumed + manifest["window"]]
        x[0, :len(part)] = part
        out = dict(zip(outputs, session.run(None, dict(x=x, **states))))
        states = {s["name"]: out["new_" + s["name"]] for s in manifest["states"]}
        lp = out["log_probs"].reshape(-1, len(manifest["tokens"]) + 1)
        lp = lp[:min(len(lp), total - emitted)]
        result.append(lp)
        emitted += len(lp)
        consumed += manifest["shift"]
    return np.concatenate(result)


def forced_labels(lp, labels):
    states = np.array([0] + [x for p in labels for x in (p, 0)])
    tmax, n = len(lp), len(states)
    prev = np.full(n, -np.inf)
    prev[:2] = lp[0, states[:2]]
    back = np.zeros((tmax, n), np.int16)
    for t in range(1, tmax):
        scores = np.stack([prev, np.r_[-np.inf, prev[:-1]], np.r_[-np.inf, -np.inf, prev[:-2]]])
        skip = (states != 0) & (states != np.r_[-1, -1, states[:-2]])
        scores[2, ~skip] = -np.inf
        choice = scores.argmax(0)
        back[t] = choice
        prev = scores[choice, np.arange(n)] + lp[t, states]
    s = n - 2 + int(prev[-1] > prev[-2])
    if not np.isfinite(prev[s]):
        raise ValueError("Impossible complete phone alignment")
    path = np.empty(tmax, np.int32)
    for t in range(tmax - 1, -1, -1):
        path[t] = states[s]
        s -= int(back[t, s])
    return path


def corpus(data, model):
    import soundfile as sf
    import onnxruntime as ort
    base = Path(__file__).resolve().parents[1]
    manifest = json.loads((Path(model) / "manifest.json").read_text())
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 2
    session = ort.InferenceSession(str(Path(model) / manifest["file"]), sess_options=opt, providers=["CPUExecutionProvider"])
    reference = json.loads((base / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {(a["s"], a["a"]): a["p"] for a in reference["ayahs"]}
    tokens = manifest["tokens"]
    root = Path(data) / "help-train"
    records = []
    for r in map(json.loads, (root / "labels.jsonl").read_text().splitlines()):
        if not clean_help(r):
            continue
        text = "".join(ayahs[tuple(map(int, a.split(":")))] for a in r["ayahs"])
        if any(p not in tokens for p in text):
            raise ValueError("Reference inventory differs from student")
        audio = root / r["wav"]
        wave, sr = sf.read(audio, dtype="float32")
        mel, power, condition = frontend(wave, sr)
        lp = stream(mel, manifest, session)
        y = np.repeat(forced_labels(lp, [tokens.index(p) + 1 for p in text]), 4)
        records.append(dict(key=r["id"], group=r["speaker"], power=power[:len(y)], mel=condition[:len(y)], y=y,
                            sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), rate=sr))
        if len(records) % 25 == 0:
            print(json.dumps(dict(extracted=len(records))), flush=True)
    if not records:
        raise ValueError("No eligible none-labelled training recordings")
    return records, tokens


def fit_parameters(records, q, ridge, shrink, vocab):
    m = np.concatenate([r["mel"] for r in records])
    z = np.concatenate([r["power"] @ q for r in records])
    labels = np.concatenate([r["y"] for r in records])
    penalty = np.eye(81) * ridge * len(m); penalty[0, 0] = 0
    predictor = np.linalg.solve(m.T @ m + penalty, m.T @ z)
    residual = z - m @ predictor
    count = np.bincount(labels, minlength=vocab)
    mean = np.zeros((vocab, q.shape[1]))
    np.add.at(mean, labels, residual)
    mean /= (count + shrink)[:, None]
    error = residual - mean[labels]
    cov = error.T @ error / len(error)
    # OAS shared-covariance shrinkage, determined by training sample statistics.
    d = cov.shape[0]; mu = np.trace(cov) / d; alpha = np.mean(cov ** 2)
    strength = min(1., (alpha + mu ** 2) / max((len(error) + 1) * (alpha - mu ** 2 / d), np.finfo(float).eps))
    cov = (1 - strength) * cov + strength * mu * np.eye(d)
    values, vectors = np.linalg.eigh(cov)
    whitening = vectors @ np.diag(1 / np.sqrt(np.maximum(values, np.finfo(float).eps)))
    return dict(projection=(q @ whitening).T.tolist(), predictor=(predictor @ whitening).tolist(),
                scale=np.ones(d).tolist(), means=(mean @ whitening).tolist(), variance=np.ones(d).tolist(),
                covered=(count > 0).tolist()), float(np.log(np.maximum(values, np.finfo(float).eps)).sum())


def validation_nll(records, p, logdet):
    projection = np.array(p["projection"]).T; predictor = np.array(p["predictor"]); means = np.array(p["means"])
    scores = []
    for r in records:
        z = r["power"] @ projection - r["mel"] @ predictor
        error = z - means[r["y"]]
        scores.append(float(.5 * (np.mean(np.sum(error ** 2, axis=1)) + logdet + z.shape[1] * np.log(2 * np.pi))))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--model", required=True); parser.add_argument("--out", required=True)
    a = parser.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    records, tokens = corpus(a.data, a.model)
    q = null_projection(mel_matrix())
    folds = {r["group"]: int(hashlib.sha256(r["group"].encode()).hexdigest()[:8], 16) % 5 for r in records}
    scores = []
    for ridge in (1e-6, 1e-4, .01):
        for shrink in (1, 10, 100, 1000):
            metrics = []
            for fold in range(5):
                train = [r for r in records if folds[r["group"]] != fold]
                valid = [r for r in records if folds[r["group"]] == fold]
                if not train or not valid:
                    raise ValueError("Need five nonempty speaker-disjoint folds")
                p, logdet = fit_parameters(train, q, ridge, shrink, len(tokens) + 1)
                metrics.append(validation_nll(valid, p, logdet))
            entry = dict(ridge=ridge, shrink=shrink, folds=metrics, mean=float(np.mean(metrics)))
            scores.append(entry); print(json.dumps(entry), flush=True)
    best = min(scores, key=lambda s: s["mean"])
    p, _ = fit_parameters(records, q, best["ridge"], best["shrink"], len(tokens) + 1)
    p.update(schema=1, tokens=tokens)
    asset = out / "spectral-residual.json"; asset.write_text(json.dumps(p, separators=(",", ":")))
    audit = dict(recordings=len(records), groups=len(folds), frames=sum(len(r["y"]) for r in records), dimension=q.shape[1],
                 selection=best, candidates=scores, mel_null_error=float(np.abs(mel_matrix() @ q).max()),
                 records=[dict(key=r["key"], group=r["group"], fold=folds[r["group"]], sha256=r["sha256"], rate=r["rate"]) for r in records],
                 asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out / "training.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k not in ("records", "candidates")}), flush=True)


if __name__ == "__main__":
    main()
