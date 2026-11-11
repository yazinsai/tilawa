"""Matched upper-block continuation versus validity-masked training.

Only supplied training recordings are admitted. Both arms receive identical
batches; speaker-held training partitions select checkpoints before export.
"""

import torch

SR, MIN_SAMPLES = 16000, 400 + 8 * 160  # At least nine fbank frames, one embedded frame.


def copy_insertion_assets(source, destination):
    """Retain the existing SDK teacher and its external weights in both bundles."""
    import hashlib
    import shutil
    from pathlib import Path

    source, destination = Path(source), Path(destination)
    assets = ("teacher.onnx", "teacher.onnx.data")
    # Check every input before copying: a partial bundle must never be declared.
    for name in assets:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Existing insertion verification asset is missing: {source / name}")

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    hashes = {}
    for name in assets:
        shutil.copyfile(source / name, destination / name)
        hashes[name] = digest(source / name)
        if digest(destination / name) != hashes[name]:
            raise RuntimeError(f"Copied insertion asset differs: {name}")
    return hashes


def export_bundle(model, checkpoint, destination, name, base_bundle, chunk=16, left=128):
    """Export the unchanged streaming computation with the newly fitted weights."""
    import json
    from pathlib import Path
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import export_onnx, manifest

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    fp32, path = destination / "model-fp32.onnx", destination / "model.onnx"
    states = export_onnx(model, str(fp32), chunk, left)
    quantize_dynamic(str(fp32), str(path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    exported = manifest(name, states, chunk, left, str(path), checkpoint)
    exported["format"] = "onnx-int8"
    auxiliary = copy_insertion_assets(base_bundle, destination)
    parity = verify_streaming_export(model, fp32, path, states, chunk, left)
    parity["auxiliary_sha256"] = auxiliary
    (destination / "export-parity.json").write_text(json.dumps(parity, indent=2))
    (destination / "manifest.json").write_text(json.dumps(exported, ensure_ascii=False, indent=1))
    return exported


def verify_streaming_export(model, fp32, int8, states, chunk, left):
    """Compare every output and warmed cache of the streaming graph to PyTorch."""
    import numpy as np
    import onnxruntime as ort
    from student_export import PAD, wrap

    wrapper = wrap(model, chunk, left)
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in (("fp32", fp32), ("int8", int8))}
    caches = {name: {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
                     for s in states} for name in sessions}
    pt_states = wrapper.init_states()
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(1, 6 * chunk + PAD, 80, generator=generator)
    errors = {name: 0.0 for name in sessions}
    with torch.no_grad():
        for offset in range(0, 6 * chunk, 2 * chunk):
            x = features[:, offset:offset + 2 * chunk + PAD]
            expected, pt_states = wrapper(x, pt_states)
            for name, session in sessions.items():
                output = session.run(None, {"x": x.numpy(), **caches[name]})
                if len(output) != len(states) + 1:
                    raise RuntimeError(f"{name} export changed the streaming state count")
                if output[0].shape != tuple(expected.shape) or not np.isfinite(output[0]).all():
                    raise RuntimeError(f"{name} export changed output shape or emitted nonfinite posteriors")
                errors[name] = max(errors[name], float(np.abs(np.exp(output[0]) - expected.exp().numpy()).max()))
                for spec, actual, target in zip(states, output[1:], pt_states):
                    if actual.shape != tuple(target.shape) or not np.isfinite(actual).all():
                        raise RuntimeError(f"{name} export changed cache {spec['name']}")
                    if name == "fp32":
                        np.testing.assert_allclose(actual, target.numpy(), rtol=1e-3, atol=1e-4)
                caches[name] = {s["name"]: value for s, value in zip(states, output[1:])}
                if name == "fp32":
                    np.testing.assert_allclose(np.exp(output[0]), expected.exp().numpy(), rtol=1e-3, atol=1e-4)
    return dict(windows=3, max_abs_probability_error=errors)


class ReadingDataset:
    """Deterministic paired batches with the shipped waveform and feature chain."""
    def __init__(self, records, aug=None, seed=41):
        self.records, self.aug, self.seed, self.epoch = records, aug, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import io
        import numpy as np
        import soundfile as sf
        from torchaudio.compliance.kaldi import fbank
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        record = self.records[i]
        wave, sr = sf.read(io.BytesIO(record["audio"]), dtype="float32")
        if sr != SR or wave.ndim != 1:
            raise ValueError("Training input must be mono 16 kHz PCM")
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + i)
        if self.aug is not None:
            if rng.random() < 0.2:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < 0.5:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(5, 25), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = fbank(torch.from_numpy(wave.copy()).unsqueeze(0), num_mel_bins=80,
                      sample_frequency=SR, dither=0.0, energy_floor=1e-10, high_freq=-400)
        if self.aug is not None:
            feats = spec_augment(feats, rng)
        return dict(feats=feats, target=torch.tensor(phoneme_ids(record["target"]), dtype=torch.long))


def reading_batch(items):
    from torch.nn.utils.rnn import pad_sequence
    return dict(feats=pad_sequence([x["feats"] for x in items], batch_first=True, padding_value=-23.0259),
                lens=torch.tensor([len(x["feats"]) for x in items]),
                targets=torch.cat([x["target"] for x in items]),
                target_lens=torch.tensor([len(x["target"]) for x in items]))


def original_records(root, exclusion_file, max_hours=10, seed=41):
    """Admit only the prepared training split and retain its speaker-cluster IDs."""
    import numpy as np
    import pyarrow.parquet as pq
    from pathlib import Path
    from student import exclusions, spliced_target

    excluded = exclusions(Path(exclusion_file).read_text())
    records = []
    for path in sorted(Path(root).glob("*.parquet")):
        table = pq.read_table(path, columns=["shard", "row", "bucket", "split", "cluster", "n_samples", "ref", "hyp", "flac"],
                              filters=[("split", "=", "train"), ("bucket", "=", "clean-match")])
        for r in table.to_pylist():
            if (r["shard"], r["row"]) in excluded or r["n_samples"] > 30 * SR or r["n_samples"] < MIN_SAMPLES:
                continue
            if r["cluster"] is None:
                raise ValueError("Original data lacks a speaker-cluster assignment")
            target = spliced_target(r["ref"], r["hyp"])
            if target:
                records.append(dict(key=f'{r["shard"]}:{r["row"]}', speaker=f'tlog:{r["cluster"]}',
                                    audio=r["flac"], n_samples=r["n_samples"], target=target))
    order = np.random.default_rng(seed).permutation(len(records))
    selected, samples = [], 0
    for i in order:
        selected.append(records[i])
        samples += records[i]["n_samples"]
        if samples >= max_hours * 3600 * SR:
            break
    return selected


def partition_records(records):
    """A frozen partition of supplied training speakers, independent of practice."""
    import hashlib
    train, validation = [], []
    for r in records:
        bucket = int.from_bytes(hashlib.sha256(r["speaker"].encode()).digest()[:4], "big") % 5
        (validation if bucket == 0 else train).append(r)
    return train, validation


def learner_records(root, reference_file):
    """None-labelled training speakers only; scripts for planted edits are excluded."""
    import io
    import json
    from pathlib import Path
    import soundfile as sf
    from student import ctc_collapse, ids_phonemes, phoneme_ids

    root = Path(root)
    reference = json.loads(Path(reference_file).read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    records = []
    for line in (root / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] != "train" or r["kind"] != "none" or r.get("excluded") or r.get("exclude"):
            continue
        if r["events"] or r["non_errors"] or r.get("unsupported"):
            continue
        target = ids_phonemes(ctc_collapse(phoneme_ids("".join(ayahs[a] for a in r["ayahs"]))))
        audio = (root / r["wav"]).read_bytes()
        info = sf.info(io.BytesIO(audio))
        if info.samplerate != SR or info.channels != 1 or info.frames < MIN_SAMPLES or info.frames > 30 * SR:
            continue
        records.append(dict(key=r["id"], speaker=f'help:{r["speaker"]}', target=target, audio=audio, n_samples=info.frames))
    return records


def upper_parameters(model, blocks=2):
    """Freeze the embedding/lower stacks; fit only the last two stacks and head."""
    if blocks <= 0 or blocks > len(model.encoder.encoders):
        raise ValueError("Upper-block count exceeds the encoder")
    model.requires_grad_(False)
    for module in [*list(model.encoder.encoders)[-blocks:], model.ctc_output]:
        module.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def check_partitions(train, validation):
    if not train or not validation:
        raise ValueError("Nonempty training and speaker-held validation are required")
    if {r["speaker"] for r in train} & {r["speaker"] for r in validation}:
        raise ValueError("Checkpoint selection speakers overlap training")


def refit_pair(checkpoint, train_records, validation_records, destination, aug, base_bundle,
               epochs=3, batch_seconds=120, seed=41, max_minutes=45):
    """Matched CTC + ordinary-reading preservation; no practice-based selection.

    The treatment's forward computation has item boundaries, the control's is
    the original. All weights, trainable blocks, inputs, augmentations, seeds,
    optimizer settings and update counts otherwise match. The fixed objective
    is CTC + 5 KL from the frozen shipped acoustic model on valid frames.
    """
    import copy
    import hashlib
    import json
    import random
    import time
    from pathlib import Path
    import torch.nn.functional as F
    from student import corpus_per, ctc_collapse, ids_phonemes
    from student_train import batches, build_model
    from student_masked_convolutions import install_validity_masks

    check_partitions(train_records, validation_records)
    if not 1 <= epochs <= 3 or not 0 < max_minutes <= 45:
        raise ValueError("Refit is bounded to three epochs and 45 minutes before validation/export")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use a fresh output directory; paired refits do not resume old arms")
    for name in ("model.onnx", "manifest.json", "teacher.onnx", "teacher.onnx.data"):
        if not (Path(base_bundle) / name).is_file():
            raise FileNotFoundError(f"Incomplete shipped listener bundle: {name}")
    torch.manual_seed(seed)
    random.seed(seed)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck["cfg"].get("masked_training_convolutions", False):
        raise ValueError("The continuation must start at an unmasked shipped checkpoint")
    acoustic = build_model(ck["cfg"]["size"])
    acoustic.load_state_dict(ck["model"], strict=True)
    anchor = copy.deepcopy(acoustic).cuda().eval().requires_grad_(False)
    anchor.streaming_eval()
    arms, optimizers = {}, {}
    for name in ("control", "masked"):
        model = copy.deepcopy(acoustic)
        if name == "masked":
            sites = install_validity_masks(model)
        model.cuda()
        optimizers[name] = torch.optim.AdamW(upper_parameters(model), lr=1e-5, weight_decay=0)
        arms[name] = model
    destination.mkdir(parents=True)
    log = (destination / "train.jsonl").open("w")

    def emit(**row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # The fingerprint records exact admitted content and order, not practice IDs.
    fingerprint = hashlib.sha256()
    for partition, records in (("train", train_records), ("validation", validation_records)):
        for record in records:
            fingerprint.update(json.dumps([partition, record["key"], record["speaker"],
                                           record["target"]]).encode())
            fingerprint.update(hashlib.sha256(record["audio"]).digest())
    emit(event="partitions", train=len(train_records), validation=len(validation_records),
         train_speakers=len({r["speaker"] for r in train_records}),
         validation_speakers=len({r["speaker"] for r in validation_records}),
         data_sha256=fingerprint.hexdigest(), sites=sites, upper_blocks=2,
         lr=1e-5, preservation_weight=5, epochs=epochs, seed=seed)
    dataset = ReadingDataset(train_records, aug, seed)
    validation = ReadingDataset(validation_records)
    base_count = ck["state"]["step"] * ck["cfg"].get("max_batch_s", 800) / 600
    best, best_pair, step = {name: float("inf") for name in arms}, float("inf"), 0
    started = time.monotonic()

    def objective(lp, lengths, ref, b):
        valid = torch.arange(lp.shape[1], device=lp.device)[None] < lengths[:, None]
        ctc = F.ctc_loss(lp.transpose(0, 1), b["targets"].cuda(), lengths,
                         b["target_lens"].cuda(), reduction="sum", zero_infinity=True) / lengths.sum()
        kl = F.kl_div(lp, ref, log_target=True, reduction="none").sum(-1)
        preservation = kl.masked_fill(~valid, 0).sum() / valid.sum()
        return ctc + 5 * preservation, ctc, preservation

    def validate():
        # Singleton validation removes artificial padding from selection. Both
        # arms face the same ordinary readings and fixed preservation anchor.
        totals = {name: 0.0 for name in arms}
        pairs = {name: [] for name in arms}
        frames = 0
        for model in arms.values():
            model.eval()
            model.streaming_eval()
        with torch.no_grad():
            for i in range(len(validation)):
                b = reading_batch([validation[i]])
                feats, lens = b["feats"].cuda(), b["lens"].cuda()
                ref, ref_lengths = anchor(feats, lens)
                frames += int(ref_lengths.sum())
                for name, model in arms.items():
                    lp, lengths = model(feats, lens)
                    if not torch.equal(lengths, ref_lengths):
                        raise RuntimeError("Validation changed the output clock")
                    loss, _, _ = objective(lp.float(), lengths, ref.float(), b)
                    totals[name] += float(loss) * int(lengths.sum())
                    hyp = ids_phonemes(ctc_collapse(lp[0, :lengths[0]].argmax(-1).tolist()))
                    pairs[name].append((hyp, validation_records[i]["target"]))
        return {name: dict(objective=totals[name] / frames, per=corpus_per(pairs[name])) for name in arms}

    emit(event="initial_validation", **validate())
    for epoch in range(epochs):
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches(
            [r["n_samples"] for r in train_records], batch_seconds, seed + epoch),
            collate_fn=reading_batch, num_workers=4)
        for b in loader:
            feats, lens = b["feats"].cuda(), b["lens"].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref, ref_lengths = anchor(feats, lens)
            metrics = {}
            for name, model in arms.items():
                torch.manual_seed(seed + step)
                random.seed(seed + step)
                model.train()
                model.streaming_eval()
                for module in model.modules():
                    module.batch_count = base_count + step * batch_seconds / 600
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, lengths = model(feats, lens)
                if not torch.equal(lengths, ref_lengths):
                    raise RuntimeError("Validity masking changed the output clock")
                loss, ctc, preservation = objective(lp.float(), lengths, ref.float(), b)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite paired training loss")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
                optimizers[name].step()
                metrics[name] = dict(ctc=float(ctc.detach()), preservation=float(preservation.detach()))
            step += 1
            if step % 20 == 0:
                emit(event="train", epoch=epoch, step=step, **metrics)
            if time.monotonic() - started >= max_minutes * 60:
                break
        if step == 0:
            raise RuntimeError("No paired training updates completed")
        results = validate()
        pair_objective = sum(r["objective"] for r in results.values()) / len(results)
        for name, result in results.items():
            emit(event="validation", arm=name, epoch=epoch, step=step, **result)
        # Select both checkpoints at the same update, so the exported control
        # has seen exactly the treatment's data and number of updates.
        if pair_objective < best_pair:
            best_pair = pair_objective
            for name, result in results.items():
                best[name] = result["objective"]
                state = dict(model={k: v.detach().cpu().clone() for k, v in arms[name].state_dict().items()},
                             cfg={**ck["cfg"], "masked_training_convolutions": name == "masked"},
                             state=dict(step=step, epoch=epoch), validation=result,
                             source_step=ck["state"]["step"], data_sha256=fingerprint.hexdigest())
                torch.save(state, destination / f"{name}.pt")
        if time.monotonic() - started >= max_minutes * 60:
            break
    emit(event="selected", objective=best, steps=step)
    log.close()
    for name, model in arms.items():
        selected = torch.load(destination / f"{name}.pt", map_location="cpu", weights_only=False)
        model.cpu().load_state_dict(selected["model"], strict=True)
        export_bundle(model, selected, destination / name, f"masked-convolutions-{name}", base_bundle)
    (destination / "complete.json").write_text(json.dumps(dict(arms=["control", "masked"],
                                                              steps=step, objective=best), indent=2))
    return best


def main():
    import argparse
    from pathlib import Path
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Verified checkpoint of the shipped primary listener")
    parser.add_argument("--base-bundle", required=True, help="Complete shipped bundle including its insertion teacher")
    parser.add_argument("--output", required=True)
    parser.add_argument("--original", default="/student/s1")
    parser.add_argument("--learners", default="/lab/data/help-train")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--max-minutes", type=float, default=45)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = original_records(args.original, root / "docs/benchmarks/gold-per-v1-exclude.txt", args.hours)
    learners = learner_records(args.learners, root / "packages/core/assets/ref-v2.json")
    ot, ov = partition_records(original)
    ht, hv = partition_records(learners)
    if not ot or not ov or not ht or not hv:
        raise ValueError("Both data sources require nonempty speaker-held partitions")
    mixed = ot + [ht[i % len(ht)] for i in range(max(len(ht), len(ot) // 4))]
    validation = ov[:128] + hv
    aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
    refit_pair(args.checkpoint, mixed, validation, args.output, aug, args.base_bundle,
               max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
