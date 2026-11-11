"""Carry BiasNorm's removed log denominator into the next acoustic operation.

The scale and constant-input arms allocate identical zero-initialized projections.
Training data and checkpoint selection use disjoint supplied training speakers.
The exported graph retains the standard 80-mel, 40 ms streaming SDK interface.
"""

import torch
from torch import nn
from scaling import BiasNorm, limit_param_value

SR, WIN = 16000, 400


class ScaleBiasNorm(BiasNorm):
    """BiasNorm with one extra scalar per activation, injected after normalization.

    Keep the biased channel RMS used by BiasNorm, not raw-input energy or
    statistics over time. The numerator remains x (not x - bias). The original
    output gain and its training constraint remain separate from the RMS.
    """

    def forward(self, x):
        assert x.shape[self.channel_dim] == self.num_channels
        channel_dim = self.channel_dim % x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        inverse_rms = ((x - bias) ** 2).mean(channel_dim, keepdim=True).pow(-0.5)
        log_scale = self.log_scale
        if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
            log_scale = limit_param_value(log_scale, min=float(self.log_scale_min),
                                          max=float(self.log_scale_max), training=self.training)
        normalized = x * (inverse_rms * log_scale.exp())
        # This is exactly the denominator removed above, including the learned
        # bias. Reuse it rather than estimating scale from the normalized output.
        log_rms = -inverse_rms.log()
        signal = log_rms if self.scale_input == "scale" else torch.ones_like(log_rms)
        residual = self.scale_projection(signal.movedim(channel_dim, -1)).movedim(-1, channel_dim)
        return normalized + residual


def install_scale_bypass(model, mode="scale"):
    """Upgrade each site in place: original checkpoint names and weights survive."""
    if mode not in ("scale", "constant"):
        raise ValueError("Bypass input must be scale or constant")
    modules = list(model.named_modules())
    if any(isinstance(module, ScaleBiasNorm) for _, module in modules):
        raise ValueError("Scale bypass is already installed")
    sites = [(name, module) for name, module in modules if isinstance(module, BiasNorm)]
    if not sites:
        raise ValueError("Acoustic model contains no BiasNorm sites")
    for _, module in sites:
        module.__class__ = ScaleBiasNorm
        module.scale_input = mode
        module.scale_projection = nn.Linear(1, module.num_channels, bias=False).to(module.bias)
        nn.init.zeros_(module.scale_projection.weight)
    return [name for name, _ in sites]


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
    """All scale operations stay inside the ordinary streaming ONNX graph."""
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


class ScaleDataset:
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


def scale_batch(items):
    from torch.nn.utils.rnn import pad_sequence
    return dict(feats=pad_sequence([x["feats"] for x in items], batch_first=True, padding_value=-23.0259),
                lens=torch.tensor([len(x["feats"]) for x in items]),
                targets=torch.cat([x["target"] for x in items]),
                target_lens=torch.tensor([len(x["target"]) for x in items]))


def train_pair(checkpoint, train_records, validation_records, destination, aug, base_bundle=None,
               projection_epochs=1, joint_epochs=3, batch_seconds=120, seed=41, max_minutes=50):
    """Fit projections first, then allow a small joint update in both arms.

    Checkpoint selection depends only on disjoint training speakers. Each arm
    sees the exact same augmented tensors and stochastic acoustic masks. The
    frozen shipped model anchors ordinary speech and real pause frames alike.
    """
    import copy
    import json
    import random
    import time
    from pathlib import Path
    import torch.nn.functional as F

    if not train_records or not validation_records:
        raise ValueError("Nonempty training and validation partitions are required")
    if {r["speaker"] for r in train_records} & {r["speaker"] for r in validation_records}:
        raise ValueError("Checkpoint selection speakers overlap training")
    if projection_epochs < 1 or joint_epochs < 1:
        raise ValueError("Projection fitting must precede limited joint fitting")
    if base_bundle is None:
        raise ValueError("The shipped bundle is required to retain insertion verification")
    if any(not (Path(base_bundle) / name).is_file() for name in ("teacher.onnx", "teacher.onnx.data")):
        raise FileNotFoundError("The shipped bundle lacks the existing insertion teacher")
    destination = Path(destination)
    if any((destination / f"{name}.pt").exists() for name in ("scale", "constant")):
        raise ValueError("Use a fresh paired output directory; this command does not resume earlier arms")
    from student import corpus_per, ctc_collapse, ids_phonemes
    from student_train import batches, build_model

    torch.manual_seed(seed)
    random.seed(seed)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    acoustic = build_model(ck["cfg"]["size"])
    acoustic.load_state_dict(ck["model"])
    base_batch_count = ck["state"]["step"] * ck["cfg"].get("max_batch_s", 800) / 600
    anchor = copy.deepcopy(acoustic).cuda().eval()
    anchor.streaming_eval()
    anchor.requires_grad_(False)
    arms, opts = {}, {}
    for name in ("constant", "scale"):
        model = copy.deepcopy(acoustic)
        sites = install_scale_bypass(model, name)
        model.cuda()
        projections = [p for n, p in model.named_parameters() if ".scale_projection." in n]
        original = [p for n, p in model.named_parameters() if ".scale_projection." not in n]
        opts[name] = torch.optim.AdamW([{"params": original, "lr": 1e-5},
                                        {"params": projections, "lr": 1e-3}], weight_decay=0)
        arms[name] = model
    destination.mkdir(parents=True, exist_ok=True)
    log = (destination / "train.jsonl").open("a")
    started = time.monotonic()

    def emit(**row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(event="partitions", train=len(train_records), validation=len(validation_records),
         train_speakers=len({r["speaker"] for r in train_records}),
         validation_speakers=len({r["speaker"] for r in validation_records}), sites=sites,
         projection_epochs=projection_epochs, joint_epochs=joint_epochs, seed=seed,
         anchor_weight=5, encoder_lr=1e-5, projection_lr=1e-3)
    ds, val = ScaleDataset(train_records, aug, seed), ScaleDataset(validation_records)
    best, step = {name: float("inf") for name in arms}, 0
    epochs = projection_epochs + joint_epochs
    for epoch in range(epochs):
        ds.epoch = epoch
        joint = epoch >= projection_epochs
        for model in arms.values():
            for name, p in model.named_parameters():
                p.requires_grad_(joint or ".scale_projection." in name)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches(
            [r["n_samples"] for r in train_records], batch_seconds, seed + epoch),
            collate_fn=scale_batch, num_workers=4)
        for b in loader:
            feats, lens = b["feats"].cuda(), b["lens"].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref, ref_lens = anchor(feats, lens)
                ref = ref.float()
            metrics = {}
            for name, model in arms.items():
                torch.manual_seed(seed + step)
                random.seed(seed + step)
                for module in model.modules():
                    module.batch_count = base_batch_count + step * batch_seconds / 600
                model.train()
                model.streaming_eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, out_lens = model(feats, lens)
                lp = lp.float()
                if not torch.equal(out_lens, ref_lens):
                    raise RuntimeError("Scale bypass changed the output clock")
                valid = torch.arange(lp.shape[1], device="cuda")[None] < out_lens[:, None]
                ctc = F.ctc_loss(lp.transpose(0, 1), b["targets"].cuda(), out_lens, b["target_lens"].cuda(),
                                 reduction="sum", zero_infinity=True) / out_lens.sum()
                preservation = (F.kl_div(lp, ref, log_target=True, reduction="none").sum(-1) * valid).sum() / valid.sum()
                loss = ctc + 5 * preservation
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opts[name].step()
                metrics[name] = dict(ctc=float(ctc.detach()), preservation=float(preservation.detach()))
            step += 1
            if step % 20 == 0:
                emit(event="train", epoch=epoch, joint=joint, step=step, **metrics)
            if time.monotonic() - started > max_minutes * 60:
                break
        for name, model in arms.items():
            model.eval()
            model.streaming_eval()
            pairs = []
            with torch.no_grad():
                for i in range(len(val)):
                    b = scale_batch([val[i]])
                    lp, lengths = model(b["feats"].cuda(), b["lens"].cuda())
                    hyp = ids_phonemes(ctc_collapse(lp[0, :lengths[0]].argmax(-1).tolist()))
                    pairs.append((hyp, validation_records[i]["target"]))
            per = corpus_per(pairs)
            emit(event="validation", arm=name, epoch=epoch, joint=joint, step=step, per=per)
            # Both stages must have run before an export can be selected.
            if joint and per < best[name]:
                best[name] = per
                state = dict(model=model.cpu().state_dict(), cfg={**ck["cfg"], "normalization_scale_bypass": name},
                             state=dict(step=step, epoch=epoch), per=per)
                torch.save(state, destination / f"{name}.pt")
                model.cuda()
        if time.monotonic() - started > max_minutes * 60:
            break
    emit(event="selected", best=best, steps=step)
    log.close()
    for name, model in arms.items():
        if not (destination / f"{name}.pt").exists():
            raise RuntimeError("Training budget ended before both stages completed")
        selected = torch.load(destination / f"{name}.pt", map_location="cpu", weights_only=False)
        model.cpu().load_state_dict(selected["model"])
        export_bundle(model, selected, destination / name, f"normalization-scale-{name}", base_bundle)
    return best


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
            if (r["shard"], r["row"]) in excluded or r["n_samples"] > 30 * SR or r["n_samples"] < WIN:
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
        if info.samplerate != SR or info.channels != 1 or info.frames < WIN or info.frames > 30 * SR:
            continue
        records.append(dict(key=r["id"], speaker=f'help:{r["speaker"]}', target=target, audio=audio, n_samples=info.frames))
    return records


def main():
    import argparse
    from pathlib import Path
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-bundle", required=True,
                        help="Shipped primary bundle, containing its unchanged insertion teacher")
    parser.add_argument("--original", default="/student/s1")
    parser.add_argument("--learners", default="/lab/data/help-train")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--max-minutes", type=float, default=50)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = original_records(args.original, root / "docs/benchmarks/gold-per-v1-exclude.txt", args.hours)
    learners = learner_records(args.learners, root / "packages/core/assets/ref-v2.json")
    ot, ov = partition_records(original)
    ht, hv = partition_records(learners)
    if not ht or not hv:
        raise ValueError("None-labelled learners need both training-speaker partitions")
    mixed = ot + [ht[i % len(ht)] for i in range(max(len(ht), len(ot) // 4))]
    validation = ov[:128] + hv
    aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
    train_pair(args.checkpoint, mixed, validation, args.output, aug, base_bundle=args.base_bundle,
               max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
