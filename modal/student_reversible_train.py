"""Matched training/export for reversible upper acoustic blocks.

Run directly inside the supervisor's train image; no Modal client is needed.
Only supplied training speakers enter either partition. Practice audio and its
labels are never opened by this program, including for checkpoint selection.
"""

import hashlib
import io
import json
from pathlib import Path

import torch

SR = 16000
ARMS = ("residual", "reversible")


def partition_records(records):
    train, validation = [], []
    for record in records:
        bucket = int.from_bytes(hashlib.sha256(record["speaker"].encode()).digest()[:4], "big") % 5
        (validation if bucket == 0 else train).append(record)
    return train, validation


def original_records(root, exclusion_file, max_hours, seed):
    import numpy as np
    import pyarrow.parquet as pq
    from student import exclusions, spliced_target

    excluded = exclusions(Path(exclusion_file).read_text())
    paths = sorted(Path(root).glob("*.parquet"))
    blocked_clusters = set()
    for path in paths:
        metadata = pq.read_table(path, columns=["shard", "row", "cluster"])
        for r in metadata.to_pylist():
            if (r["shard"], r["row"]) in excluded:
                if r["cluster"] is None:
                    raise ValueError("Excluded original clip lacks a speaker-cluster assignment")
                blocked_clusters.add(r["cluster"])
    records = []
    columns = ["shard", "row", "bucket", "split", "cluster", "n_samples", "ref", "hyp", "flac",
               "top5_ids", "top5_p"]
    for path in paths:
        table = pq.read_table(path, columns=columns,
                              filters=[("split", "=", "train"), ("bucket", "=", "clean-match")])
        for r in table.to_pylist():
            if (r["shard"], r["row"]) in excluded or r["cluster"] in blocked_clusters or not 400 <= r["n_samples"] <= 30 * SR:
                continue
            if r["cluster"] is None:
                raise ValueError("Original data lacks speaker-cluster assignments")
            target = spliced_target(r["ref"], r["hyp"])
            if target:
                records.append(dict(key=f'{r["shard"]}:{r["row"]}', speaker=f'tlog:{r["cluster"]}',
                                    audio=r["flac"], n_samples=r["n_samples"], target=target,
                                    top_ids=r["top5_ids"], top_p=r["top5_p"]))
    selected, samples = [], 0
    for i in np.random.default_rng(seed).permutation(len(records)):
        selected.append(records[i])
        samples += records[i]["n_samples"]
        if max_hours and samples >= max_hours * 3600 * SR:
            break
    return selected


def eligible_none(record, blocked_speakers):
    return (record.get("split") == "train" and record.get("kind") == "none"
            and record.get("speaker") is not None and record["speaker"] not in blocked_speakers
            and not any(record.get(key) for key in ("exclude", "excluded", "unsupported", "events", "non_errors")))


def learner_records(root, reference_file):
    import soundfile as sf
    from student import ctc_collapse, ids_phonemes, phoneme_ids

    root = Path(root)
    labels = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines() if line]
    # A speaker with any nontraining/excluded row cannot leak via a none take.
    blocked = {r.get("speaker") for r in labels if r.get("split") != "train" or r.get("excluded") or r.get("exclude")}
    reference = json.loads(Path(reference_file).read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    records = []
    for r in labels:
        if not eligible_none(r, blocked):
            continue
        target = ids_phonemes(ctc_collapse(phoneme_ids("".join(ayahs[a] for a in r["ayahs"]))))
        path = (root / r["wav"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Learner audio path leaves the supplied dataset")
        audio = path.read_bytes()
        info = sf.info(io.BytesIO(audio))
        if info.samplerate != SR or info.channels != 1 or not 400 <= info.frames <= 30 * SR or not target:
            continue
        records.append(dict(key=r["id"], speaker=f'help:{r["speaker"]}', target=target,
                            audio=audio, n_samples=info.frames))
    return records


class PairedDataset:
    """Materialize each augmented tensor once; share it between both arms."""

    def __init__(self, records, aug=None, seed=41):
        self.records, self.aug, self.seed, self.epoch = records, aug, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import numpy as np
        import soundfile as sf
        from torchaudio.compliance.kaldi import fbank
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        r = self.records[i]
        wave, sr = sf.read(io.BytesIO(r["audio"]), dtype="float32")
        if sr != SR or wave.ndim != 1:
            raise ValueError("Training input must be mono 16 kHz")
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + i)
        if self.aug is not None:
            if rng.random() < 0.2:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < 0.5:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(5, 25), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = fbank(torch.from_numpy(wave.copy()).unsqueeze(0), num_mel_bins=80, sample_frequency=SR,
                      dither=0.0, energy_floor=1e-10, high_freq=-400)
        if self.aug is not None:
            feats = spec_augment(feats, rng)
        present = "top_ids" in r
        ids = np.frombuffer(r["top_ids"], np.uint8).reshape(-1, 5).astype(np.int64) if present else np.zeros((1, 5), np.int64)
        probs = np.frombuffer(r["top_p"], np.float16).reshape(-1, 5).astype(np.float32) if present else np.zeros((1, 5), np.float32)
        if len(ids) == 0 or ids.shape != probs.shape:
            raise ValueError("Invalid training teacher targets")
        return dict(i=i, feats=feats, target=torch.tensor(phoneme_ids(r["target"]), dtype=torch.long),
                    top_ids=torch.from_numpy(ids), top_p=torch.from_numpy(probs), has_teacher=present)


def collate(items):
    from student_train import collate as ordinary_collate

    result = ordinary_collate(items)
    result["has_teacher"] = torch.tensor([item["has_teacher"] for item in items])
    return result


def acoustic_losses(lp, lens, batch, cfg, reference):
    from student import soft_target_loss
    import torch.nn.functional as F

    valid = torch.arange(lp.shape[1], device=lp.device)[None] < lens[:, None]
    ctc = F.ctc_loss(lp.transpose(0, 1), batch["targets"].to(lp.device), lens,
                     batch["target_lens"].to(lp.device), reduction="sum", zero_infinity=True) / lens.sum().clamp_min(1)
    preservation = (F.kl_div(lp, reference, log_target=True, reduction="none").sum(-1) * valid).sum() / valid.sum().clamp_min(1)
    kd = lp.new_zeros(())
    if cfg.get("kd_weight", 0):
        idx = (torch.arange(lp.shape[1], device=lp.device)[None] - cfg.get("kd_delay", 0)).clamp_min(0)
        idx = torch.minimum(idx, (batch["n_teacher"].to(lp.device) - 1)[:, None])
        ids = batch["top_ids"].to(lp.device).gather(1, idx[..., None].expand(-1, -1, 5))
        probs = batch["top_p"].to(lp.device).gather(1, idx[..., None].expand(-1, -1, 5))
        kd = soft_target_loss(lp, ids, probs, valid & batch["has_teacher"].to(lp.device)[:, None])
    return cfg.get("ctc_weight", 1) * ctc + cfg.get("kd_weight", 0) * kd + 5 * preservation, dict(
        ctc=float(ctc.detach()), kd=float(kd.detach()), preservation=float(preservation.detach()))


def paired_models(checkpoint, seed):
    from student_train import build_model

    cfg = checkpoint["cfg"]
    if cfg.get("upper_blocks") is not None:
        raise ValueError("Initialize from the shipped ordinary acoustic model")
    base = build_model(cfg["size"])
    base.load_state_dict(checkpoint["model"])
    arms = {}
    for mode in ARMS:
        torch.manual_seed(seed)
        model = build_model(cfg["size"], mode)
        start = model.encoder.upper_start
        replacement = tuple(f"encoder.encoders.{i}.encoder." for i in range(start, len(model.encoder.encoders)))
        # The final native stacks are wrapped by temporal down/up-sampling.
        kept = {key: value for key, value in checkpoint["model"].items() if not key.startswith(replacement)}
        result = model.load_state_dict(kept, strict=False)
        if result.unexpected_keys or any(not key.startswith(replacement) for key in result.missing_keys):
            raise RuntimeError("Shipped stem or temporal geometry did not load completely")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(replacement))
        arms[mode] = model
    if sum(p.numel() for p in arms[ARMS[0]].parameters()) != sum(p.numel() for p in arms[ARMS[1]].parameters()):
        raise RuntimeError("Matched arms have different parameter counts")
    for key, value in arms[ARMS[0]].state_dict().items():
        if not torch.equal(value, arms[ARMS[1]].state_dict()[key]):
            raise RuntimeError("Matched arms have different initialization")
    return base, arms


def validate_stream(model, records):
    """Select on real cached streams of speaker-disjoint training audio."""
    from student_export import PAD, LOG_EPS, wrap
    from student import corpus_per, ctc_collapse, ids_phonemes

    model.cpu().eval()
    wrapper = wrap(model, 16, 128)
    dataset, pairs = PairedDataset(records), []
    with torch.no_grad():
        for i, record in enumerate(records):
            f = dataset[i]["feats"]
            need = ((len(f) - 7) // 2 + 1) // 2
            windows = max(1, (need + 7) // 8)
            total = 45 + (windows - 1) * 32
            f = torch.nn.functional.pad(f, (0, 0, 0, max(0, total - len(f))), value=LOG_EPS)
            states, outputs = wrapper.init_states(), []
            for offset in range(0, total - 45 + 1, 32):
                lp, states = wrapper(f[None, offset:offset + 2 * 16 + PAD], states)
                outputs.append(lp[0])
            path = torch.cat(outputs)[:need].argmax(-1).tolist()
            pairs.append((ids_phonemes(ctc_collapse(path)), record["target"]))
    return corpus_per(pairs)


def train_pair(checkpoint_path, training, validation, output, aug, epochs=4, batch_seconds=120,
               max_minutes=50, seed=41):
    import random
    import time
    from student_train import batches

    if not training or not validation or {r["speaker"] for r in training} & {r["speaker"] for r in validation}:
        raise ValueError("Nonempty, disjoint training-speaker partitions are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{arm}.pt").exists() for arm in ARMS):
        raise ValueError("Use a fresh paired output directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base, arms = paired_models(checkpoint, seed)
    base = base.cuda().eval().requires_grad_(False)
    base.streaming_eval()
    opts = {}
    for name, model in arms.items():
        model.cuda()
        opts[name] = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    ds = PairedDataset(training, aug, seed)
    best, step = {name: float("inf") for name in ARMS}, 0
    started = time.monotonic()
    base_count = checkpoint["state"]["step"] * checkpoint["cfg"].get("max_batch_s", 800) / 600
    partitions = {name: sorted({r["speaker"] for r in rows}) for name, rows in (("train", training), ("validation", validation))}
    (output / "partitions.json").write_text(json.dumps(partitions, indent=2))
    with (output / "train.jsonl").open("w") as log:
        def emit(**row):
            line = json.dumps(row)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        emit(event="config", epochs=epochs, batch_seconds=batch_seconds, seed=seed, anchor_weight=5,
             lr=1e-3, frozen="stem, first four stacks, temporal resampling, CTC head",
             cfg=checkpoint["cfg"], parameters={name: sum(p.numel() for p in m.parameters()) for name, m in arms.items()})
        for epoch in range(epochs):
            ds.epoch = epoch
            loader = torch.utils.data.DataLoader(ds, batch_sampler=batches(
                [r["n_samples"] for r in training], batch_seconds, seed + epoch), collate_fn=collate, num_workers=4)
            for batch in loader:
                feats, lens = batch["feats"].cuda(), batch["feat_lens"].cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    reference, ref_lens = base(feats, lens)
                metrics = {}
                for name, model in arms.items():
                    # Evaluation behavior keeps feature dropout and native
                    # resampling stochasticity off in the frozen shipped stem.
                    # F/G have no train/eval-dependent operations; gradients
                    # still update every replacement parameter.
                    model.eval()
                    model.streaming_eval()
                    torch.manual_seed(seed + step)
                    random.seed(seed + step)
                    for module in model.modules():
                        module.batch_count = base_count + step * batch_seconds / 600
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lp, lengths = model(feats, lens)
                    if not torch.equal(lengths, ref_lens):
                        raise RuntimeError("Replacement changed the output clock")
                    loss, metrics[name] = acoustic_losses(lp.float(), lengths, batch, checkpoint["cfg"], reference.float())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite matched training loss")
                    opts[name].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opts[name].step()
                step += 1
                if step % 20 == 0:
                    emit(event="train", epoch=epoch, step=step, **metrics)
                if time.monotonic() - started > max_minutes * 60:
                    break
            # Validate both arms after identical updates, never on practice.
            for name, model in arms.items():
                per = validate_stream(model, validation)
                emit(event="validation", arm=name, epoch=epoch, step=step, per=per)
                if per < best[name]:
                    best[name] = per
                    torch.save(dict(model=model.state_dict(), cfg={**checkpoint["cfg"], "upper_blocks": name},
                                    state=dict(step=step, epoch=epoch), per=per), output / f"{name}.pt")
                model.cuda()
            if time.monotonic() - started > max_minutes * 60:
                break
        emit(event="selected", best=best, paired_steps=step)
    return best


def export_bundle(checkpoint_path, destination, base_bundle, validation):
    """Declare only a complete bundle with the unchanged insertion assets."""
    import shutil
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_train import build_model
    from student_export import export_onnx, manifest, wrap, PAD

    destination, base_bundle = Path(destination), Path(base_bundle)
    assets = ("teacher.onnx", "teacher.onnx.data")
    for asset in assets:
        if not (base_bundle / asset).is_file():
            raise FileNotFoundError(f"Missing unchanged insertion asset: {asset}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]["size"], ck["cfg"]["upper_blocks"])
    model.load_state_dict(ck["model"])
    destination.mkdir(parents=True, exist_ok=True)
    fp32, int8 = destination / "model-fp32.onnx", destination / "model.onnx"
    states = export_onnx(model, str(fp32), 16, 128)
    quantize_dynamic(str(fp32), str(int8), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    wrapper = wrap(model, 16, 128)
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in (("fp32", fp32), ("int8", int8))}
    report = []
    dataset = PairedDataset(validation)
    with torch.no_grad():
        for i in range(min(4, len(dataset))):
            f = dataset[i]["feats"]
            if len(f) < 45 + 2 * 32:
                continue
            caches = {name: {s["name"]: np.zeros(s["shape"], np.int64 if s["dtype"] == "int64" else np.float32)
                             for s in states} for name in sessions}
            pt_states = wrapper.init_states()
            errors = {name: 0.0 for name in sessions}
            for offset in range(0, 3 * 32, 32):
                x = f[None, offset:offset + 32 + PAD]
                expected, pt_states = wrapper(x, pt_states)
                for name, session in sessions.items():
                    got = session.run(None, {"x": x.numpy(), **caches[name]})
                    if len(got) != len(states) + 1 or got[0].shape != tuple(expected.shape):
                        raise RuntimeError("Export changed streaming state count or output clock")
                    for spec, actual, target in zip(states, got[1:], pt_states):
                        if actual.shape != tuple(target.shape) or not np.isfinite(actual).all():
                            raise RuntimeError(f"Invalid exported cache: {spec['name']}")
                        if name == "fp32":
                            np.testing.assert_allclose(actual, target.numpy(), rtol=1e-3, atol=1e-4)
                    if not np.isfinite(got[0]).all():
                        raise RuntimeError("Nonfinite exported posteriors")
                    error = float(np.abs(np.exp(got[0]) - expected.exp().numpy()).max())
                    errors[name] = max(errors[name], error)
                    if name == "fp32":
                        np.testing.assert_allclose(np.exp(got[0]), expected.exp().numpy(), rtol=1e-3, atol=1e-4)
                    caches[name] = {s["name"]: v for s, v in zip(states, got[1:])}
            report.append(dict(key=validation[i]["key"], windows=3, max_probability_error=errors))
    if not report:
        raise RuntimeError("No training-validation clips available for warmed export parity")
    hashes = {}
    for asset in assets:
        with (base_bundle / asset).open("rb") as file:
            hashes[asset] = hashlib.file_digest(file, "sha256").hexdigest()
        shutil.copyfile(base_bundle / asset, destination / asset)
        with (destination / asset).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != hashes[asset]:
                raise RuntimeError("Insertion asset copy failed hash verification")
    (destination / "export-parity.json").write_text(json.dumps(dict(clips=report, auxiliary_sha256=hashes), indent=2))
    exported = manifest(f'upper-{ck["cfg"]["upper_blocks"]}', states, 16, 128, str(int8), ck)
    exported["format"] = "onnx-int8"
    (destination / "manifest.json").write_text(json.dumps(exported, ensure_ascii=False, indent=1))


def main():
    import argparse
    import numpy as np
    from student_export import onnx_deps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--original", default="/student/s1")
    parser.add_argument("--learners", default="/lab/data/help-train")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=50)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.epochs < 1 or args.hours < 0 or args.max_minutes <= 0:
        raise ValueError("Training requires positive epochs and time, and a nonnegative data budget")
    for name in ("model.onnx", "manifest.json", "teacher.onnx", "teacher.onnx.data"):
        if not (Path(args.base_bundle) / name).is_file():
            raise FileNotFoundError(f"Shipped bundle is incomplete before training: {name}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shipped = json.loads((Path(args.base_bundle) / "manifest.json").read_text())
    if shipped.get("trainStep") is not None and shipped["trainStep"] != ck["state"]["step"]:
        raise ValueError("Initialization checkpoint does not match the shipped bundle's training step")
    root = Path(__file__).resolve().parents[1]
    ot, ov = partition_records(original_records(args.original, root / "docs/benchmarks/gold-per-v1-exclude.txt", args.hours, args.seed))
    ht, hv = partition_records(learner_records(args.learners, root / "packages/core/assets/ref-v2.json"))
    if not all((ot, ov, ht, hv)):
        raise ValueError("Both datasets need training and validation speaker partitions")
    training = ot + [ht[i % len(ht)] for i in range(max(len(ht), len(ot) // 4))]
    validation = ov[:128] + hv
    aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
    train_pair(args.checkpoint, training, validation, args.output, aug, epochs=args.epochs,
               max_minutes=args.max_minutes, seed=args.seed)
    onnx_deps()
    for arm in ARMS:
        export_bundle(Path(args.output) / f"{arm}.pt", Path(args.output) / arm, args.base_bundle, validation)


if __name__ == "__main__":
    main()
