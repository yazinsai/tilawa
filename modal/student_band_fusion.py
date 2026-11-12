"""Causal, audio-dependent temporal alignment of log-mel frequency regions.

The sampler delays individual regions by at most four existing 10 ms frames;
it never reads future input or changes the encoder's queries. Raw history is
carried separately from encoder state and advances only by the committed shift.
The matched synchronous arm uses the same predictor for instantaneous spectral
contrast, with the same trainable parameter count and identity initialization.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

MELS, REGIONS, BOUND, LOG_EPS = 80, 8, 4, -23.0259


class BandFusion(nn.Module):
    def __init__(self, asynchronous=True, bound=BOUND):
        super().__init__()
        if not isinstance(bound, int) or bound < 1:
            raise ValueError("Delay bound must be a positive integer frame count")
        self.asynchronous, self.bound = asynchronous, bound
        # Three causal frames of regional level and first difference. Groups
        # keep frequency regions separate until the shipped subsampling layer.
        self.hidden = nn.Conv1d(2 * REGIONS, 8 * REGIONS, 3, groups=REGIONS)
        self.readout = nn.Conv1d(8 * REGIONS, REGIONS, 1, groups=REGIONS)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.history = max(bound, 3)

    def initial_state(self, batch=1, device=None):
        # Store energy relative to silence: generic SDK zero initialization is
        # then the same as LOG_EPS padding in full-sequence training.
        return torch.zeros(batch, self.history, MELS, device=device)

    def _predict(self, x):
        level = x.reshape(x.shape[0], x.shape[1], REGIONS, MELS // REGIONS).mean(-1)
        previous = F.pad(level[:, :-1], (0, 0, 1, 0), value=LOG_EPS)
        cues = torch.stack(((level - LOG_EPS) / 10, (level - previous) / 10), -1)
        cues = cues.flatten(2).transpose(1, 2)
        return self.readout(F.silu(self.hidden(F.pad(cues, (2, 0))))).transpose(1, 2).tanh()

    def _sample(self, all_frames, start):
        x = all_frames[:, start:]
        # Fractional offsets stay float32 under encoder bf16 autocast; integer
        # frame indices never pass through floating point, even on long takes.
        signal = self._predict(all_frames)[:, start:].float()
        # Clamp has a nonzero one-sided derivative at the identity. Squaring
        # a zero predictor would prevent the offset head from learning.
        delays = self.bound * signal.clamp(min=0)
        if not self.asynchronous:
            delays = delays * 0
        whole = delays.floor().long()
        latest = (torch.arange(x.shape[1], device=x.device)[None, :, None] + start - whole).clamp(min=0)
        bands = all_frames.reshape(all_frames.shape[0], all_frames.shape[1], REGIONS, MELS // REGIONS)
        shape = (*latest.shape, MELS // REGIONS)
        after = bands.gather(1, latest[..., None].expand(shape))
        # At integral delays use the past interval's derivative as well, so the
        # zero-initialized readouts receive gradients without reading ahead.
        past = (latest - 1).clamp(min=0)
        preceding = bands.gather(1, past[..., None].expand(shape))
        fraction = (delays - whole).unsqueeze(-1)
        output = after + fraction * (preceding - after)
        if not self.asynchronous:
            # Same predictor and sampling operations, active parameters, no
            # displacement. Contrast is zero-mean within each frequency region.
            output = output + 0.1 * signal[..., None] * (output - output.mean(-1, keepdim=True))
        return output.flatten(2), delays

    def forward(self, x, lens=None, mask=None):
        padded = torch.cat((self.initial_state(x.shape[0], x.device) + LOG_EPS, x), 1)
        out, _ = self._sample(padded, self.history)
        if lens is not None:
            valid = torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]
            out = out.masked_fill(~valid[..., None], LOG_EPS)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        return out

    def streaming_forward(self, x, history, shift):
        if not 0 < shift <= x.shape[1]:
            raise ValueError("State must advance by a nonempty committed input prefix")
        raw = torch.cat((history + LOG_EPS, x), 1)
        out, _ = self._sample(raw, self.history)
        new_history = raw[:, shift:shift + self.history] - LOG_EPS
        return out, new_history


class FusedStudent(nn.Module):
    def __init__(self, acoustic, asynchronous=True, bound=BOUND):
        super().__init__()
        self.acoustic = acoustic
        self.fusion = BandFusion(asynchronous, bound)

    def forward(self, x, lens, mask=None):
        return self.acoustic(self.fusion(x, lens, mask), lens)


def streaming_wrapper(model, chunk=16, left=128):
    from student_export import wrap

    class Streaming(nn.Module):
        def __init__(self):
            super().__init__()
            self.acoustic = wrap(model.acoustic, chunk, left)
            self.fusion = model.fusion

        def init_states(self):
            return self.acoustic.init_states() + [self.fusion.initial_state()]

        def forward(self, x, states):
            features, history = self.fusion.streaming_forward(x, states[-1], 2 * chunk)
            lp, new_states = self.acoustic(features, states[:-1])
            return lp, new_states + [history]

    return Streaming()


def export_bundle(model, checkpoint, destination, name, parity_features, chunk=16, left=128):
    """Export and verify actual carried-state inference before declaring a bundle."""
    import json
    from pathlib import Path

    import numpy as np
    import onnx
    import onnxruntime as ort
    from icefall.utils import get_onnx_export_kwargs
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import PAD, manifest, stream_onnx

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    wrapper = streaming_wrapper(model, chunk, left)
    states = wrapper.init_states()
    kinds = ["cached_key", "cached_nonlin_attn", "cached_val1", "cached_val2", "cached_conv1", "cached_conv2"]
    names = [f"{kinds[i % 6]}_{i // 6}" for i in range(len(states) - 3)]
    names += ["embed_states", "processed_lens", "band_history"]
    specs = [dict(name=n, shape=list(t.shape), dtype="int64" if t.dtype == torch.int64 else "float32")
             for n, t in zip(names, states)]
    fp32, path = destination / "fp32.onnx", destination / "model.onnx"
    cls = model.acoustic.encoder.__class__
    plain = cls.forward
    try:
        cls.forward = cls.streaming_forward
        with torch.no_grad():
            torch.onnx.export(wrapper, (torch.rand(1, 2 * chunk + PAD, MELS), states), str(fp32),
                              opset_version=17, input_names=["x", *names],
                              output_names=["log_probs", *[f"new_{n}" for n in names]],
                              **get_onnx_export_kwargs())
    finally:
        cls.forward = plain
    onnx.checker.check_model(str(fp32))
    quantize_dynamic(str(fp32), str(path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    rows = []
    for features in parity_features:
        need = ((len(features) - 7) // 2 + 1) // 2
        count = max(1, math.ceil(need / (chunk // 2)))
        padded = F.pad(features, (0, 0, 0, max(0, 2 * chunk + PAD + (count - 1) * 2 * chunk - len(features))), value=LOG_EPS)
        current, outputs = wrapper.init_states(), []
        with torch.no_grad():
            for i in range(count):
                lp, current = wrapper(padded[None, i * 2 * chunk:i * 2 * chunk + 2 * chunk + PAD], current)
                outputs.append(lp[0])
        expected = torch.cat(outputs).numpy()
        actual = stream_onnx(str(fp32), features, specs, chunk)
        quantized = stream_onnx(str(path), features, specs, chunk)
        error = float(np.abs(np.exp(actual) - np.exp(expected)).max())
        agreement = float((actual.argmax(-1) == expected.argmax(-1)).mean())
        if actual.shape != expected.shape or error > 0.003 or agreement < 0.995:
            raise RuntimeError(f"Streaming export parity failed: probability error={error}, top1={agreement}")
        rows.append(dict(frames=len(expected), max_probability_error=error, fp32_top1=agreement,
                         int8_top1=float((quantized.argmax(-1) == expected.argmax(-1)).mean())))
    if not rows:
        raise ValueError("Bundle verification requires training-only parity recordings")
    # Verify manifest feeds and returned cache names, including sampler state.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if {x.name for x in session.get_inputs()} != {"x", *names}:
        raise RuntimeError("Export pruned a declared streaming state")
    m = manifest(name, specs, chunk, left, str(path), checkpoint)
    m["format"] = "onnx-int8"
    (destination / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    (destination / "parity.json").write_text(json.dumps(rows, indent=1))
    return m


class FeatureDataset:
    """Shipped augmentations with common recording/epoch draws in both arms."""
    def __init__(self, records, aug=None, seed=41):
        self.records, self.aug, self.seed, self.epoch = records, aug, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import io
        import numpy as np
        import soundfile as sf
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment
        from torchaudio.compliance import kaldi

        record = self.records[index]
        wave, rate = sf.read(io.BytesIO(record["audio"]), dtype="float32")
        if rate != 16000 or wave.ndim != 1:
            raise ValueError("Training requires mono 16 kHz audio")
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        if self.aug is not None:
            if rng.random() < 0.2:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < 0.5:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(5, 25), rng)
            wave *= 10 ** (rng.uniform(-10, 6) / 20)
        features = kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
                               num_mel_bins=MELS, sample_frequency=16000, dither=0,
                               energy_floor=1e-10, high_freq=-400)
        mask = (spec_augment(torch.ones_like(features), rng) == 0 if self.aug is not None
                else torch.zeros_like(features, dtype=torch.bool))
        return dict(features=features, mask=mask, target=torch.tensor(phoneme_ids(record["target"])))


def feature_batch(items):
    from torch.nn.utils.rnn import pad_sequence

    return dict(features=pad_sequence([x["features"] for x in items], batch_first=True, padding_value=LOG_EPS),
                mask=pad_sequence([x["mask"] for x in items], batch_first=True),
                lens=torch.tensor([len(x["features"]) for x in items]),
                targets=torch.cat([x["target"] for x in items]),
                target_lens=torch.tensor([len(x["target"]) for x in items]))


def partition_records(records):
    """Frozen speaker partition within the authorized training pool."""
    import hashlib

    train, validation = [], []
    for record in records:
        if not record["speaker"]:
            raise ValueError("Recording lacks a training-speaker identity")
        bucket = int.from_bytes(hashlib.sha256(record["speaker"].encode()).digest()[:4], "big") % 5
        (validation if bucket == 0 else train).append(record)
    return train, validation


def original_records(root, exclusions_file, hours=4, seed=41):
    import numpy as np
    import pyarrow.parquet as pq
    from pathlib import Path
    from student import exclusions, spliced_target

    excluded = exclusions(Path(exclusions_file).read_text())
    records = []
    for path in sorted(Path(root).glob("*.parquet")):
        table = pq.read_table(path, columns=["shard", "row", "bucket", "split", "cluster", "n_samples", "ref", "hyp"],
                              filters=[("split", "=", "train"), ("bucket", "=", "clean-match")])
        for row in table.to_pylist():
            if (row["shard"], row["row"]) in excluded or not 400 <= row["n_samples"] <= 30 * 16000:
                continue
            if row["cluster"] is None:
                raise ValueError("Original training recording lacks a speaker cluster")
            target = spliced_target(row["ref"], row["hyp"])
            if target:
                records.append(dict(key=f'{row["shard"]}:{row["row"]}', speaker=f'tlog:{row["cluster"]}',
                                    file=path, shard=row["shard"], row=row["row"],
                                    n_samples=row["n_samples"], target=target))
    order = np.random.default_rng(seed).permutation(len(records))
    selected, samples = [], 0
    for index in order:
        selected.append(records[index])
        samples += records[index]["n_samples"]
        if samples >= hours * 3600 * 16000:
            break
    # Avoid loading the entire authorized audio pool merely to select a subset.
    from collections import defaultdict
    by_file = defaultdict(list)
    for record in selected:
        by_file[record["file"]].append(record)
    for path, group in by_file.items():
        wanted = {(r["shard"], r["row"]): r for r in group}
        rows = pq.read_table(path, columns=["shard", "row", "flac"],
                             filters=[("row", "in", [r["row"] for r in group])]).to_pylist()
        for row in rows:
            if (row["shard"], row["row"]) in wanted:
                wanted[row["shard"], row["row"]]["audio"] = row["flac"]
    for record in selected:
        del record["file"], record["shard"], record["row"]
        if "audio" not in record:
            raise ValueError("Selected authorized recording has no audio")
    return selected


def learner_records(root, reference_file):
    """Only none-labelled training takes with no labelled edits or repeats."""
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
        row = json.loads(line)
        if row["split"] != "train" or row["kind"] != "none" or row.get("exclude") or row.get("excluded"):
            continue
        if row["events"] or row["non_errors"] or row.get("unsupported"):
            continue
        target = ids_phonemes(ctc_collapse(phoneme_ids("".join(ayahs[a] for a in row["ayahs"]))))
        audio = (root / row["wav"]).read_bytes()
        info = sf.info(io.BytesIO(audio))
        if info.samplerate == 16000 and info.channels == 1 and 400 <= info.frames <= 30 * 16000:
            records.append(dict(key=row["id"], speaker=f'help:{row["speaker"]}', audio=audio,
                                n_samples=info.frames, target=target))
    return records


def train_pair(checkpoint, train_records, validation_records, destination, aug,
               epochs=2, batch_seconds=90, seed=41, max_steps=360):
    """Matched CTC plus ordinary-reading/pause preservation refit.

    Bounds, objective, updates and partition are fixed before any target decode.
    Each arm's checkpoint is selected only by PER on training-pool speakers not
    admitted to its updates. No practice examples are accepted by this runner.
    """
    import copy
    import json
    from pathlib import Path
    from student import corpus_per, ctc_collapse, ids_phonemes
    from student_train import batches, build_model

    ts = {r["speaker"] for r in train_records}
    vs = {r["speaker"] for r in validation_records}
    if not ts or not vs or ts & vs:
        raise ValueError("Training and checkpoint selection require disjoint nonempty speaker partitions")
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    acoustic = build_model(checkpoint["cfg"]["size"])
    acoustic.load_state_dict(checkpoint["model"])
    acoustic.eval()
    anchor = copy.deepcopy(acoustic).cuda().eval()
    anchor.streaming_eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    first = FusedStudent(acoustic)
    arms = {"asynchronous": copy.deepcopy(first).cuda(), "synchronous": copy.deepcopy(first).cuda()}
    arms["synchronous"].fusion.asynchronous = False
    optimizers = {name: torch.optim.AdamW([
        {"params": model.acoustic.parameters(), "lr": 1e-5},
        {"params": model.fusion.parameters(), "lr": 1e-3}], weight_decay=0) for name, model in arms.items()}
    dataset, validation = FeatureDataset(train_records, aug, seed), FeatureDataset(validation_records)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log = (destination / "train.jsonl").open("a")

    def emit(**row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(event="recipe", train=len(dataset), validation=len(validation), train_speakers=sorted(ts),
         validation_speakers=sorted(vs), seed=seed, bound_frames=BOUND, epochs=epochs, max_steps=max_steps,
         encoder_lr=1e-5, fusion_lr=1e-3, preservation_weight=5,
         parameters={n: sum(p.numel() for p in m.parameters()) for n, m in arms.items()})
    best = {name: float("inf") for name in arms}
    step = 0
    for epoch in range(epochs):
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset, num_workers=4, collate_fn=feature_batch,
            batch_sampler=batches([r["n_samples"] for r in train_records], batch_seconds, seed + epoch))
        for batch in loader:
            x, lens, mask = batch["features"].cuda(), batch["lens"].cuda(), batch["mask"].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                reference, ref_lens = anchor(x.masked_fill(mask, 0), lens)
                reference = reference.float()
            metrics = {}
            for name, model in arms.items():
                # Encoder dropout uses identical draws in both arms; augmentation
                # is evaluated once per batch, before either model sees it.
                torch.manual_seed(seed + step)
                model.train()
                model.acoustic.streaming_eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, out_lens = model(x, lens, mask)
                lp = lp.float()
                if not torch.equal(out_lens, ref_lens):
                    raise RuntimeError("Fusion changed the posterior clock")
                valid = torch.arange(lp.shape[1], device="cuda")[None] < out_lens[:, None]
                ctc = F.ctc_loss(lp.transpose(0, 1), batch["targets"].cuda(), out_lens, batch["target_lens"].cuda(),
                                 reduction="sum", zero_infinity=True) / out_lens.sum()
                preserve = (F.kl_div(lp, reference, log_target=True, reduction="none").sum(-1) * valid).sum() / valid.sum()
                loss = ctc + 5 * preserve
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training objective")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizers[name].step()
                metrics[name] = dict(ctc=float(ctc.detach()), preservation=float(preserve.detach()))
            step += 1
            if step % 20 == 0:
                emit(event="train", epoch=epoch, step=step, **metrics)
            if step >= max_steps:
                break
        for name, model in arms.items():
            model.eval()
            model.acoustic.streaming_eval()
            pairs, delays = [], []
            with torch.no_grad():
                for index in range(len(validation)):
                    batch = feature_batch([validation[index]])
                    x, lens = batch["features"].cuda(), batch["lens"].cuda()
                    lp, out_lens = model(x, lens)
                    hyp = ids_phonemes(ctc_collapse(lp[0, :out_lens[0]].argmax(-1).tolist()))
                    pairs.append((hyp, validation_records[index]["target"]))
                    raw = torch.cat((model.fusion.initial_state(1, x.device) + LOG_EPS, x), 1)
                    _, offsets = model.fusion._sample(raw, model.fusion.history)
                    delays.append(float(offsets.mean()))
            per = corpus_per(pairs)
            emit(event="validation", arm=name, epoch=epoch, step=step, per=per, mean_delay_frames=sum(delays) / len(delays))
            if per < best[name]:
                best[name] = per
                selected = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                cfg=checkpoint["cfg"], state=dict(step=step, epoch=epoch), per=per, bound=BOUND, arm=name)
                torch.save(selected, destination / f"{name}.pt")
        if step >= max_steps:
            break
    emit(event="selected", best=best, steps=step)
    log.close()
    return best


def copy_auxiliary_assets(source, destination):
    """Keep the folded insertion verifier exactly, and validate its bundle hashes."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["pnpm", "--filter", "@tilawa/sdk", "exec", "tsx", "-e",
        "import { MODEL_REGISTRY } from './src/registry.ts'; console.log(JSON.stringify(MODEL_REGISTRY.auxiliary))"],
        cwd=root, capture_output=True, text=True, check=True)
    assets = [a for a in json.loads(result.stdout) if a["purpose"] == "seam-insertion"]
    if len(assets) != 1:
        raise ValueError("Expected one unchanged folded insertion teacher")
    for asset in [assets[0], *assets[0]["externalData"]]:
        path = Path(source) / asset["file"]
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Missing or incomplete insertion asset: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as data:
            for block in iter(lambda: data.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"Insertion asset checksum mismatch: {path}")
        target = Path(destination) / asset["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main():
    import argparse
    from pathlib import Path
    import numpy as np
    from student_train import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "export"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--original", default="/student/s1")
    parser.add_argument("--learners", default="/lab/data/help-train")
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--auxiliary-source")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    original = original_records(args.original, root / "docs/benchmarks/gold-per-v1-exclude.txt", args.hours)
    learners = learner_records(args.learners, root / "packages/core/assets/ref-v2.json")
    ot, ov = partition_records(original)
    ht, hv = partition_records(learners)
    if not ht or not hv:
        raise ValueError("None-labelled learners require both training-speaker partitions")
    validation = ov[:128] + hv
    if args.action == "train":
        if not args.checkpoint:
            parser.error("train requires --checkpoint")
        mixed = ot + [ht[i % len(ht)] for i in range(max(len(ht), len(ot) // 4))]
        aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
        train_pair(args.checkpoint, mixed, validation, args.output, aug)
    else:
        if not args.auxiliary_source:
            parser.error("export requires the complete unchanged --auxiliary-source bundle")
        parity = FeatureDataset(validation[:3])
        for name in ("synchronous", "asynchronous"):
            ck = torch.load(Path(args.output) / f"{name}.pt", map_location="cpu", weights_only=False)
            model = FusedStudent(build_model(ck["cfg"]["size"]), name == "asynchronous", ck["bound"])
            model.load_state_dict(ck["model"])
            bundle = Path(args.output) / name
            export_bundle(model, ck, bundle, f"band-fusion-{name}", [parity[i]["features"] for i in range(len(parity))])
            copy_auxiliary_assets(args.auxiliary_source, bundle)


if __name__ == "__main__":
    main()
