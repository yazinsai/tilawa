"""Frozen streaming-encoder head refit in a reproducible paired-gradient subspace.

Runs as an ordinary Python process in the authorized training sandbox, not as a
Modal app. Selection consumes training speakers only; inference is the existing
streaming graph with its final affine phone head replaced. No runtime veto or
change to the CTC alphabet, caches, frame clock, or decoder is involved.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def speaker_partition(speakers, seed=17):
    """Stable group split, independent of input order and recording counts."""
    groups = sorted(set(speakers), key=lambda s: hashlib.sha256(f"{seed}/{s}".encode()).digest())
    if len(groups) < 5:
        raise ValueError("need at least five training speakers for grouped selection")
    validation = set(groups[:max(1, len(groups) // 5)])
    return np.array([s in validation for s in speakers])


def shuffled_pairs(n, seed=19):
    """One-to-one recording derangement; every second view is used once."""
    if n < 2:
        raise ValueError("need at least two independent recording pairs")
    order = np.random.default_rng(seed).permutation(n)
    permutation = np.empty(n, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    return permutation


def gradient_subspaces(a, b, max_rank=16, sketch_dim=128, ridge_fraction=0.05, seed=23):
    """Compare symmetrized cross-view covariance with view-difference noise.

    A shared randomized range bounds memory without forming a parameters-squared
    covariance. Both arms use the same gradient evaluations and parameter range.
    Generalized eigenvectors maximize cross-view signal / (difference noise +
    ridge). Keep paired directions with a ratio above one, up to max_rank. The
    shuffled control uses exactly that rank, including nonpositive directions.
    Return Euclidean-orthonormal bases so projection cannot amplify an update.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 3:
        raise ValueError("gradient pairs must have matching (recordings, parameters) shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite gradient")
    if max_rank < 1 or sketch_dim < max_rank or ridge_fraction <= 0:
        raise ValueError("invalid subspace budget")
    a, b = a - a.mean(0), b - b.mean(0)
    stacked = np.concatenate([a, b])
    dim = min(sketch_dim, stacked.shape[0], stacked.shape[1])
    sketch = stacked.T @ np.random.default_rng(seed).standard_normal((len(stacked), dim))
    u, singular, _ = np.linalg.svd(sketch, full_matrices=False)
    active = singular > (singular[0] * 1e-9 if len(singular) else 0)
    q = u[:, active]
    if not q.shape[1]:
        raise ValueError("no nonzero centered gradient directions")
    ar, br = a @ q, b @ q

    def directions(second):
        cross = (ar.T @ second + second.T @ ar) / (2 * len(a))
        difference = ar - second
        noise = difference.T @ difference / (2 * len(a))
        total = (ar.T @ ar + second.T @ second) / (2 * len(a))
        ridge = max(float(np.trace(total) / len(total)) * ridge_fraction, 1e-14)
        vals, vecs = np.linalg.eigh(noise + np.eye(len(noise)) * ridge)
        inverse_root = (vecs * (1 / np.sqrt(vals))) @ vecs.T
        ratios, vectors = np.linalg.eigh(inverse_root @ cross @ inverse_root)
        order = np.argsort(ratios)[::-1]
        return ratios[order], q @ inverse_root @ vectors[:, order]

    ratios, directions_paired = directions(br)
    rank = min(max_rank, int(np.sum(ratios > 1)))
    shuffled_ratios, directions_shuffled = directions(br[shuffled_pairs(len(br), seed + 1)])
    paired = np.linalg.qr(directions_paired[:, :rank], mode="reduced")[0]
    shuffled = np.linalg.qr(directions_shuffled[:, :rank], mode="reduced")[0]
    return paired, shuffled, {
        "rank": rank, "range_rank": q.shape[1], "paired_ratios": ratios[:max_rank].tolist(),
        "shuffled_ratios": shuffled_ratios[:max_rank].tolist(), "recordings": len(a),
        "parameters": a.shape[1], "ridge_fraction": ridge_fraction,
    }


def project_update(update, basis):
    if basis is None:
        return update
    return basis @ (basis.T @ update)


class StreamingHead:
    """Expose the effective, quantized encoder output of the deployed int8 graph.

    Keeping the head's input quantizer prevents a float export from accidentally
    changing frozen features. The replacement affine head is float; at the parent
    weights it must reproduce the integer head before any refitting is allowed.
    """

    def __init__(self, directory):
        import onnx
        from onnx import helper, numpy_helper

        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        path = self.directory / self.manifest["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["sha256"]:
            raise ValueError("parent model hash mismatch")
        self.graph = onnx.load(path)
        producers = {o: n for n in self.graph.graph.node for o in n.output}
        tensors = {t.name: numpy_helper.to_array(t) for t in self.graph.graph.initializer}
        output = self.graph.graph.output[0].name
        logsoftmax = producers[output]
        if logsoftmax.op_type != "LogSoftmax":
            raise ValueError("expected a streaming affine CTC head")
        add = producers[logsoftmax.input[0]]
        if add.op_type != "Add":
            raise ValueError("expected phone-head bias addition")
        bias_name = next((x for x in add.input if x in tensors), None)
        if bias_name is None:
            raise ValueError("missing phone-head bias")
        scaled = producers[next(x for x in add.input if x != bias_name)]
        cast = next((producers[x] for x in scaled.input if x in producers and producers[x].op_type == "Cast"), None)
        if scaled.op_type != "Mul" or cast is None:
            raise ValueError("expected dynamically quantized phone head")
        mm = producers[cast.input[0]]
        if mm.op_type != "MatMulInteger" or len(mm.input) != 4:
            raise ValueError("expected affine MatMulInteger phone head")
        quant = producers[mm.input[0]]
        if quant.op_type != "DynamicQuantizeLinear":
            raise ValueError("expected retained head input quantizer")
        scale_product = producers[next(x for x in scaled.input if x != cast.output[0])]
        weight_scale = next((tensors[x] for x in scale_product.input if x in tensors), None)
        if scale_product.op_type != "Mul" or weight_scale is None:
            raise ValueError("missing head weight scale")
        self.weight = ((tensors[mm.input[1]].astype(np.float32) - tensors[mm.input[3]]) * weight_scale).astype(np.float32)
        self.bias = tensors[bias_name].astype(np.float32)
        if self.weight.shape[1] != len(self.manifest["tokens"]) + 1 or self.bias.shape != (self.weight.shape[1],):
            raise ValueError("phone inventory mismatch")
        self.output = output
        self.feature_name = "paired_gradient_features"
        feature_nodes = [
            helper.make_node("Cast", [quant.output[0]], ["pg_quant_float"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Cast", [quant.output[2]], ["pg_zero_float"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Sub", ["pg_quant_float", "pg_zero_float"], ["pg_centered"]),
            helper.make_node("Mul", ["pg_centered", quant.output[1]], [self.feature_name]),
        ]
        self.graph.graph.node.extend(feature_nodes)
        self.graph.graph.output.append(helper.make_tensor_value_info(self.feature_name, onnx.TensorProto.FLOAT,
                                                                     [1, self.manifest["chunk"] // 2, self.weight.shape[0]]))
        self.feature_graph = self.graph.SerializeToString()
        self.session = None

    def stream(self, fbank):
        import onnxruntime as ort

        if self.session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.session = ort.InferenceSession(self.feature_graph, sess_options=options, providers=["CPUExecutionProvider"])
        m = self.manifest
        states = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in m["states"]}
        need = max(0, ((len(fbank) - 7) // 2 + 1) // 2)
        windows = max(1, -(-need // (m["chunk"] // 2)))
        total = m["window"] + (windows - 1) * m["shift"]
        padded = np.pad(fbank, ((0, max(0, total - len(fbank))), (0, 0)), constant_values=m["fbank"]["padValue"])
        features, parent = [], []
        names = [self.output, self.feature_name, *["new_" + s["name"] for s in m["states"]]]
        for start in range(0, total - m["window"] + 1, m["shift"]):
            lp, f, *new = self.session.run(names, {"x": padded[None, start:start + m["window"]].astype(np.float32), **states})
            features.append(f[0])
            parent.append(lp[0])
            states = {s["name"]: v for s, v in zip(m["states"], new)}
        f, lp = np.concatenate(features)[:need], np.concatenate(parent)[:need]
        logits = f @ self.weight + self.bias
        check = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
        if len(lp) and (np.max(np.abs(check - lp)) > 0.002 or not np.array_equal(check.argmax(1), lp.argmax(1))):
            raise ValueError("effective head does not reproduce shipped posteriors")
        return f, lp

    def export(self, directory, weight, bias):
        import copy
        import shutil
        import onnx
        from onnx import helper, numpy_helper

        out = Path(directory)
        if out.resolve() == self.directory.resolve():
            raise ValueError("export must not overwrite its parent model")
        if weight.shape != self.weight.shape or bias.shape != self.bias.shape or not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("invalid replacement head")
        out.mkdir(parents=True, exist_ok=True)
        graph = copy.deepcopy(self.graph)
        # Preserve every original encoder/state node and the input quantizer.
        # The old affine head becomes dead code; ONNX Runtime prunes it.
        for node in graph.graph.node:
            for i, name in enumerate(node.output):
                if name == self.output:
                    node.output[i] = "pg_parent_log_probs"
        del graph.graph.output[-1]
        graph.graph.initializer.extend([
            numpy_helper.from_array(np.asarray(weight, dtype=np.float32), "pg_weight"),
            numpy_helper.from_array(np.asarray(bias, dtype=np.float32), "pg_bias"),
        ])
        graph.graph.node.extend([
            helper.make_node("MatMul", [self.feature_name, "pg_weight"], ["pg_linear"]),
            helper.make_node("Add", ["pg_linear", "pg_bias"], ["pg_logits"]),
            helper.make_node("LogSoftmax", ["pg_logits"], [self.output], axis=-1),
        ])
        onnx.checker.check_model(graph)
        path = out / "model.onnx"
        onnx.save(graph, path)
        manifest = dict(self.manifest, id=out.name, file=path.name,
                        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # Auxiliary listeners are part of the installed engine's behavior.
        # Preserve every parent bundle file; fail on missing external tensors.
        for source in self.directory.rglob("*"):
            if source.is_symlink():
                raise ValueError("parent model bundle contains a symlink")
            if source.is_file() and source not in (self.directory / self.manifest["file"], self.directory / "manifest.json"):
                target = out / source.relative_to(self.directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if target.stat().st_size != source.stat().st_size:
                    raise ValueError("incomplete auxiliary asset copy")
        for model in out.rglob("*.onnx"):
            auxiliary = onnx.load(model, load_external_data=False)
            for tensor in auxiliary.graph.initializer:
                for field in tensor.external_data:
                    if field.key == "location" and not (model.parent / field.value).is_file():
                        raise ValueError("export is missing an external ONNX tensor")
        # Verify the exported primary against the selected affine head before
        # making its manifest available for installation.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        original = ort.InferenceSession(self.feature_graph, sess_options=options, providers=["CPUExecutionProvider"])
        exported = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        m = self.manifest
        states = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in m["states"]}
        input_shape = [d.dim_value for d in self.graph.graph.input[0].type.tensor_type.shape.dim]
        rng = np.random.default_rng(41)
        for _ in range(3):
            x = rng.normal(-3, 2, size=input_shape).astype(np.float32)
            old = original.run(None, {"x": x, **states})
            got = exported.run(None, {"x": x, **states})
            features = old[-1]
            logits = features @ weight + bias
            expected = logits - np.logaddexp.reduce(logits, axis=-1)[..., None]
            if np.max(np.abs(expected - got[0])) > .002 or not np.array_equal(expected.argmax(-1), got[0].argmax(-1)):
                raise ValueError("exported head parity check failed")
            for previous, current in zip(old[1:-1], got[1:]):
                if not np.array_equal(previous, current):
                    raise ValueError("refit changed streaming encoder caches")
            states = {s["name"]: value for s, value in zip(m["states"], got[1:])}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        return manifest


def authorized_recordings(lab_data, student_data, max_original=384):
    """Training split only. No scripted-error transcript is treated as performed."""
    import io
    import soundfile as sf
    import pyarrow.parquet as pq
    from student import exclusions, spliced_target

    root = Path(__file__).resolve().parents[1]
    gold = exclusions((root / "docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    # Exclude the whole published held-out speaker group, not just its listed clips.
    held = json.loads((root / "docs/benchmarks/gold-per-v1.json").read_text())
    held_speakers = {c["speaker"] for clips in held["slices"].values() for c in clips if isinstance(c, dict) and "speaker" in c}
    metadata = []
    for path in sorted((Path(student_data) / "s1").glob("*.parquet")):
        columns = ["shard", "row", "bucket", "split", "cluster", "n_samples"]
        for index, r in enumerate(pq.read_table(path, columns=columns).to_pylist()):
            speaker = f"tlog-cluster-{r['cluster']}"
            if (r["cluster"] is not None and int(r["cluster"]) >= 0 and r["split"] == "train" and r["bucket"] == "clean-match" and speaker not in held_speakers
                    and (r["shard"], r["row"]) not in gold and 2 * 16000 <= r["n_samples"] <= 30 * 16000):
                metadata.append((path, index, speaker, f"{r['shard']}:{r['row']}"))
    metadata.sort(key=lambda r: hashlib.sha256(r[3].encode()).digest())
    selected, counts = [], {}
    for row in metadata:
        if counts.get(row[2], 0) < 12:
            selected.append(row)
            counts[row[2]] = counts.get(row[2], 0) + 1
        if len(selected) >= max_original:
            break
    by_file = {}
    for path, index, speaker, key in selected:
        by_file.setdefault(path, []).append((index, speaker, key))
    for path, rows in by_file.items():
        table = pq.read_table(path, columns=["ref", "hyp", "flac"])
        for index, speaker, key in rows:
            r = table.slice(index, 1).to_pylist()[0]
            wave, sr = sf.read(io.BytesIO(r["flac"]), dtype="float32")
            yield {"key": key, "speaker": speaker, "target": spliced_target(r["ref"], r["hyp"]), "wave": wave, "sr": sr, "source": "s1"}
    reference = json.loads((root / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in reference["ayahs"]}
    help_dir = Path(lab_data) / "help-train"
    for line in (help_dir / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] != "train" or r["kind"] != "none" or r["events"] or r.get("non_errors") or r["speaker"] in held_speakers:
            continue
        path = (help_dir / r["wav"]).resolve()
        if not path.is_relative_to(help_dir.resolve()):
            raise ValueError("training audio path escapes help-train")
        wave, sr = sf.read(path, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(1)
        pronunciation = "".join(ayahs[a] for a in r["ayahs"])
        target = "".join(c for i, c in enumerate(pronunciation) if i == 0 or c != pronunciation[i - 1])
        yield {"key": r["id"], "speaker": "help-" + r["speaker"], "target": target, "wave": wave, "sr": sr, "source": "help-none"}


def degrade(wave, rng, noise_bank=None, rir_bank=None):
    """Independent shipped training augmentation draws; preserve sample count."""
    from scipy.signal import fftconvolve
    from student import mix_noise

    out = wave.copy()
    if rir_bank is not None and rng.random() < .2:
        i = int(rng.integers(len(rir_bank["offsets"]) - 1))
        rir = rir_bank["data"][rir_bank["offsets"][i]:rir_bank["offsets"][i + 1]].astype(np.float32)
        out = fftconvolve(out, rir)[:len(out)].astype(np.float32)
    if noise_bank is not None:
        i = int(rng.integers(len(noise_bank["offsets"]) - 1))
        noise = noise_bank["data"][noise_bank["offsets"][i]:noise_bank["offsets"][i + 1]].astype(np.float32)
        out = mix_noise(out, noise, float(rng.uniform(5, 25)), rng)
    else:
        out = mix_noise(out, rng.standard_normal(len(out)).astype(np.float32), float(rng.uniform(5, 25)), rng)
    out *= 10 ** (float(rng.uniform(-6, 6)) / 20)
    peak = np.max(np.abs(out))
    return (out / max(1, peak)).astype(np.float32)


def cache_features(args):
    import os
    import time
    import torch
    import torchaudio
    from scipy.signal import resample_poly
    from student import TOKENS

    if args.max_original < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid recording or shard budget")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    head = StreamingHead(args.model)
    if head.manifest["tokens"] != TOKENS or head.manifest.get("blank") != 0 or head.manifest["frameMs"] != 40:
        raise ValueError("training requires the unchanged shipped phone alphabet and clock")
    banks = [Path(args.student_data) / "aug" / name for name in ("noise.npz", "rir.npz")]
    bank_identity = [{"file": p.name, "bytes": p.stat().st_size, "modified_ns": p.stat().st_mtime_ns}
                     if p.exists() else None for p in banks]
    banks = [dict(np.load(p)) if p.exists() else None for p in banks]
    if banks[0] is None:
        raise ValueError("authorized shipped training noise bank unavailable")
    rows = list(authorized_recordings(args.lab_data, args.student_data, args.max_original))
    validation = speaker_partition([r["speaker"] for r in rows])
    identity = {"parent_sha256": head.manifest["sha256"], "augmentation_seed": 29, "pause_samples": 9600,
                "max_original": args.max_original, "augmentation_banks": bank_identity,
                "recordings": [{k: r[k] for k in ("key", "speaker", "target", "source")} for r in rows]}
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("existing feature cache has a different parent or training protocol")
    if args.shard == 0:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False))
        np.savez(out / "parent.npz", weight=head.weight, bias=head.bias)
    start = time.monotonic()
    metadata = []
    for i, r in enumerate(rows):
        entry = {k: r[k] for k in ("key", "speaker", "target", "source")}
        entry.update(validation=bool(validation[i]), file=f"{i:04}.npz")
        metadata.append(entry)
        if i % args.shards != args.shard:
            continue
        if time.monotonic() - start > 23 * 3600:
            raise TimeoutError("one-day cached-feature budget exhausted")
        if (out / entry["file"]).exists():
            continue
        wave = r["wave"]
        if r["sr"] != 16000:
            import math
            divisor = math.gcd(r["sr"], 16000)
            wave = resample_poly(wave, 16000 // divisor, r["sr"] // divisor).astype(np.float32)
        # Keep completed pauses in every view, with no transcript targets there.
        wave = np.pad(wave, (9600, 9600))
        views = [wave]
        for view in range(2):
            seed = int.from_bytes(hashlib.sha256(f"29/{r['key']}/{view}".encode()).digest()[:8], "little")
            views.append(degrade(wave, np.random.default_rng(seed), *banks))
        cached = {}
        for j, view in enumerate(views):
            fbank = torchaudio.compliance.kaldi.fbank(torch.from_numpy(view).unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0, energy_floor=1e-10, high_freq=-400).numpy()
            features, lp = head.stream(fbank)
            cached[f"features{j}"] = features.astype(np.float32)
            cached[f"parent{j}"] = lp.astype(np.float32)
        temp = out / (entry["file"] + ".tmp")
        with temp.open("wb") as f:
            np.savez(f, **cached)
        os.replace(temp, out / entry["file"])
        print(json.dumps({"cached": i, "key": r["key"], "seconds": round(time.monotonic() - start, 1)}), flush=True)
    (out / f"metadata-{args.shard}.json").write_text(json.dumps(metadata, ensure_ascii=False))


def train_heads(args):
    import time
    import torch
    import torch.nn.functional as F
    from student import phoneme_ids

    if args.epochs < 1:
        raise ValueError("need at least one refit epoch")
    torch.set_num_threads(2)
    torch.manual_seed(31)
    cache = Path(args.cache)
    metadata_files = sorted(cache.glob("metadata-*.json"))
    if not metadata_files:
        raise ValueError("feature cache metadata is missing")
    metadata = json.loads(metadata_files[0].read_text())
    for path in metadata_files[1:]:
        if json.loads(path.read_text()) != metadata:
            raise ValueError("feature cache shards disagree")
    parent = np.load(cache / "parent.npz")
    w0, b0 = torch.from_numpy(parent["weight"]), torch.from_numpy(parent["bias"])
    theta0 = torch.cat([w0.reshape(-1), b0])
    data = []
    for r in metadata:
        views = np.load(cache / r["file"])
        data.append((r, [(torch.from_numpy(views[f"features{j}"]), torch.from_numpy(views[f"parent{j}"])) for j in range(3)],
                     torch.tensor(phoneme_ids(r["target"]), dtype=torch.long)))
    train = [r for r in data if not r[0]["validation"]]
    validation = [r for r in data if r[0]["validation"]]
    if not train or not validation:
        raise ValueError("empty grouped training partition")
    ctc = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=False)

    def view_loss(theta, view, target, anchor):
        features, parent_lp = view
        weight, bias = theta[:w0.numel()].reshape(w0.shape), theta[w0.numel():]
        lp = (features @ weight + bias).log_softmax(-1)
        loss = ctc(lp[:, None], target, torch.tensor([len(lp)]), torch.tensor([len(target)])) / len(lp)
        preservation = F.kl_div(lp, parent_lp.exp(), reduction="sum") / len(lp)
        return loss + anchor * preservation

    def objective(theta, r):
        _, views, target = r
        # Identical objective for estimation, training and every matched arm.
        return (view_loss(theta, views[0], target, 1.) +
                view_loss(theta, views[1], target, .5) + view_loss(theta, views[2], target, .5)) / 3

    a, b = [], []
    start = time.monotonic()
    for i, r in enumerate(train):
        theta = theta0.clone().requires_grad_()
        a.append(torch.autograd.grad(view_loss(theta, r[1][1], r[2], .5), theta)[0].numpy())
        b.append(torch.autograd.grad(view_loss(theta, r[1][2], r[2], .5), theta)[0].numpy())
        if i % 50 == 0:
            print(json.dumps({"gradients": i, "seconds": round(time.monotonic() - start, 1)}), flush=True)
    paired, shuffled, spectrum = gradient_subspaces(a, b)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subspace.json").write_text(json.dumps(spectrum, indent=1))
    np.savez(out / "subspaces.npz", paired=paired, shuffled=shuffled)
    if spectrum["rank"] == 0:
        raise ValueError("no gradient direction has reproducible signal above view-difference noise")
    bases = {"ordinary": None, "paired": torch.from_numpy(paired.astype(np.float32)),
             "shuffled": torch.from_numpy(shuffled.astype(np.float32))}
    theta = {arm: theta0.clone() for arm in bases}
    best = {arm: (float("inf"), 0, None) for arm in bases}
    history = []
    rng = np.random.default_rng(37)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        for begin in range(0, len(train), 8):
            batch = [train[i] for i in order[begin:begin + 8]]
            for arm, basis in bases.items():
                current = theta[arm].detach().requires_grad_()
                loss = sum(objective(current, r) for r in batch) / len(batch)
                gradient = torch.autograd.grad(loss, current)[0]
                if not torch.isfinite(gradient).all():
                    raise ValueError("nonfinite CTC gradient; no checkpoint selected")
                if basis is not None:
                    gradient = basis @ (basis.T @ gradient)
                gradient *= min(1., 1. / max(float(gradient.norm()), 1e-12))
                # Plain SGD keeps projected updates inside the frozen subspace.
                theta[arm] = current.detach() - .1 * gradient
        metrics = {}
        with torch.no_grad():
            for arm in bases:
                per_speaker = {}
                for r in validation:
                    per_speaker.setdefault(r[0]["speaker"], []).append(float(objective(theta[arm], r)))
                metric = float(np.mean([np.mean(values) for values in per_speaker.values()]))
                if not np.isfinite(metric):
                    raise ValueError("nonfinite training-validation objective")
                metrics[arm] = metric
                if metric < best[arm][0]:
                    best[arm] = (metric, epoch, theta[arm].clone())
        history.append({"epoch": epoch, **metrics})
        print(json.dumps(history[-1]), flush=True)
    for arm, (metric, epoch, selected) in best.items():
        np.savez(out / f"{arm}.npz", weight=selected[:w0.numel()].reshape(w0.shape).numpy(), bias=selected[w0.numel():].numpy())
    report = {"selection": {arm: {"objective": metric, "epoch": epoch} for arm, (metric, epoch, _) in best.items()},
              "history": history, "spectrum": spectrum, "recordings": len(data),
              "training_speakers": sorted({r[0]["speaker"] for r in train}),
              "validation_speakers": sorted({r[0]["speaker"] for r in validation}),
              "sources": {source: sum(r[0]["source"] == source for r in data) for source in {r[0]["source"] for r in data}},
              "paired_gradient_evaluations": 2 * len(train), "refit_gradient_evaluations_per_arm": args.epochs * len(train) * 3,
              "protocol": {"epochs": args.epochs, "batch_recordings": 8, "learning_rate": .1,
                           "clean_kl": 1., "degraded_kl": .5, "gradient_clip": 1., "seed": 37},
              "cache_identity_sha256": hashlib.sha256((cache / "identity.json").read_bytes()).hexdigest(),
              "seconds": time.monotonic() - start}
    (out / "selection.json").write_text(json.dumps(report, indent=1))


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--model", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--lab-data", default=os.environ.get("LAB_DATA", "/lab/data"))
    cache.add_argument("--student-data", default="/student")
    cache.add_argument("--max-original", type=int, default=384)
    cache.add_argument("--shard", type=int, default=int(os.environ.get("LAB_SHARD", "0")))
    cache.add_argument("--shards", type=int, default=int(os.environ.get("LAB_SHARDS", "1")))
    train = sub.add_parser("train")
    train.add_argument("--cache", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=12)
    export = sub.add_parser("export")
    export.add_argument("--model", required=True)
    export.add_argument("--head", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "cache":
        cache_features(args)
    elif args.command == "train":
        train_heads(args)
    else:
        head = StreamingHead(args.model)
        weights = np.load(args.head)
        print(json.dumps(head.export(args.output, weights["weight"], weights["bias"])))


if __name__ == "__main__":
    main()
