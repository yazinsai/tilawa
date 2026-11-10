"""Learn frequency weights before discarding FFT powers; no change to the frame clock.

Each filter is a positive distribution on its original contiguous mel support.
Its fixed mass restores the shipped mel energy scale, including at initialization.
The streaming ONNX encoder still consumes 80 log energies; export_contract() is
installed by the SDK's Fbank, for both primary streaming and independent crops.
"""

import torch
from torch import nn
from torchaudio.compliance import kaldi

SR, WIN, HOP, NFFT, MELS = 16000, 400, 160, 512, 80
EPS = torch.finfo(torch.float32).eps


def fft_powers(wave):
    """Kaldi float-wave fbank analysis, with no aggregation or spectral floor."""
    if wave.numel() < WIN:
        return wave.new_empty((0, NFFT // 2 + 1))
    frames = wave.unfold(0, WIN, HOP)
    frames = frames - frames.mean(-1, keepdim=True)
    prev = torch.cat((frames[:, :1], frames[:, :-1]), -1)
    window = torch.hann_window(WIN, periodic=False, device=wave.device).pow(0.85)
    spectrum = torch.fft.rfft((frames - 0.97 * prev) * window, n=NFFT)
    return spectrum.abs().square()


class FrequencyPooling(nn.Module):
    def __init__(self, learned=True):
        super().__init__()
        bank, _ = kaldi.get_mel_banks(MELS, NFFT, SR, 20.0, -400.0, 100.0, -500.0, 1.0)
        bank = torch.nn.functional.pad(bank, (0, 1))
        self.register_buffer("support", bank > 0)
        self.register_buffer("mass", bank.sum(-1, keepdim=True))
        self.register_buffer("initial_logits", bank.clamp_min(EPS).log())
        # Fixed and learned arms have identical allocated parameters and encoder.
        self.delta = nn.Parameter(torch.zeros_like(bank), requires_grad=learned)

    def weights(self):
        logits = (self.initial_logits + self.delta).masked_fill(~self.support, -torch.inf)
        return logits.softmax(-1) * self.mass

    def forward(self, powers):
        # Keep the weak-energy floor AFTER summing the original FFT powers.
        return (powers.float() @ self.weights().t()).clamp_min(EPS).log()

    def export_contract(self):
        weights = self.weights().detach().cpu()
        filters = []
        for w, mass in zip(weights, self.mass.detach().cpu().flatten()):
            bins = w.nonzero().flatten()
            start, end = int(bins[0]), int(bins[-1]) + 1
            filters.append(dict(start=start, weights=(w[start:end] / mass).tolist(), scale=float(mass)))
        return dict(schema=1, fftSize=NFFT, filters=filters)


class PooledStudent(nn.Module):
    """Training adapter: the exported encoder itself has unchanged dimensions."""
    def __init__(self, acoustic, learned=True):
        super().__init__()
        self.acoustic = acoustic
        self.pooling = FrequencyPooling(learned)

    def forward(self, powers, lens, mask=None):
        features = self.pooling(powers)
        valid = torch.arange(features.shape[1], device=features.device)[None] < lens[:, None]
        features = features.masked_fill(~valid[..., None], -23.0259)
        if mask is not None:
            features = features.masked_fill(mask, 0)
        return self.acoustic(features, lens)


def export_bundle(model, checkpoint, destination, name, chunk=16, left=128):
    """Export encoder plus its trained input contract as one loadable int8 bundle."""
    import hashlib
    import json
    from pathlib import Path
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import export_onnx, manifest

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    fp32 = destination / "encoder-fp32.onnx"
    path = destination / "model.onnx"
    states = export_onnx(model.acoustic, str(fp32), chunk, left)
    quantize_dynamic(str(fp32), str(path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    exported = manifest(name, states, chunk, left, str(path), checkpoint)
    exported["format"] = "onnx-int8"
    exported["fbank"]["frequencyPooling"] = model.pooling.export_contract()
    exported["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(exported, ensure_ascii=False, indent=1))
    return exported


class PowerDataset:
    """Exactly the shipped waveform augmentation and feature-mask distribution."""
    def __init__(self, records, aug=None, seed=0):
        self.records, self.aug, self.seed, self.epoch = records, aug, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import io
        import numpy as np
        import soundfile as sf
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        r = self.records[i]
        wave, sr = sf.read(io.BytesIO(r["audio"]), dtype="float32")
        if sr != SR or wave.ndim != 1:
            raise ValueError("Training input must be mono 16 kHz PCM")
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + i)
        if self.aug is not None:
            if rng.random() < 0.2:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < 0.5:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(5, 25), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        powers = fft_powers(torch.from_numpy(np.ascontiguousarray(wave)))
        # Apply SpecAugment after learned aggregation, using a shared boolean mask.
        mask = spec_augment(torch.ones(len(powers), MELS), rng) == 0 if self.aug is not None else torch.zeros(len(powers), MELS, dtype=torch.bool)
        return dict(powers=powers, mask=mask, target=torch.tensor(phoneme_ids(r["target"])), key=r["key"])


def power_batch(items):
    from torch.nn.utils.rnn import pad_sequence
    return dict(powers=pad_sequence([x["powers"] for x in items], batch_first=True),
                mask=pad_sequence([x["mask"] for x in items], batch_first=True),
                lens=torch.tensor([len(x["powers"]) for x in items]),
                targets=torch.cat([x["target"] for x in items]),
                target_lens=torch.tensor([len(x["target"]) for x in items]))


def train_pair(checkpoint, train_records, validation_records, destination, aug,
               epochs=4, batch_seconds=120, seed=41, max_minutes=50):
    """Matched joint fine-tune; select each arm only on speaker-held-out training data.

    Records are authorized performed transcripts with explicit speaker partitions.
    The caller admits none-labelled learner takes and transcribed original speech;
    practice audio and mistake scripts must never be passed to this function.
    """
    import copy
    import json
    import time
    from pathlib import Path
    import torch.nn.functional as F
    from student import corpus_per, ctc_collapse, ids_phonemes
    from student_train import batches, build_model

    train_speakers = {r["speaker"] for r in train_records}
    validation_speakers = {r["speaker"] for r in validation_records}
    if not train_records or not validation_records or train_speakers & validation_speakers:
        raise ValueError("Checkpoint selection requires nonempty disjoint speaker partitions")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    acoustic = build_model(ck["cfg"]["size"])
    acoustic.load_state_dict(ck["model"])
    anchor = copy.deepcopy(acoustic).cuda().eval()
    anchor.streaming_eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    arms = {name: PooledStudent(copy.deepcopy(acoustic), name == "learned").cuda()
            for name in ("fixed", "learned")}
    opts = {name: torch.optim.AdamW([
        {"params": model.acoustic.parameters(), "lr": 1e-5},
        {"params": [model.pooling.delta], "lr": 1e-3}], weight_decay=0)
        for name, model in arms.items()}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log = (destination / "train.jsonl").open("a")
    started = time.monotonic()

    def emit(**row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(event="partitions", train=len(train_records), validation=len(validation_records),
         train_speakers=len(train_speakers), validation_speakers=len(validation_speakers),
         epochs=epochs, seed=seed, anchor_weight=5, encoder_lr=1e-5, pooling_lr=1e-3)
    ds = PowerDataset(train_records, aug, seed)
    val = PowerDataset(validation_records)
    best = {name: float("inf") for name in arms}
    step = 0
    for epoch in range(epochs):
        ds.epoch = epoch
        sizes = [r["n_samples"] for r in train_records]
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches(sizes, batch_seconds, seed + epoch),
                                             collate_fn=power_batch, num_workers=4)
        for b in loader:
            powers, lens, mask = b["powers"].cuda(), b["lens"].cuda(), b["mask"].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                reference_features = arms["fixed"].pooling(powers)
                valid_features = torch.arange(powers.shape[1], device="cuda")[None] < lens[:, None]
                reference_features = reference_features.masked_fill(~valid_features[..., None], -23.0259).masked_fill(mask, 0)
                ref, ref_lens = anchor(reference_features, lens)
                ref = ref.float()
            metrics = {}
            for name, model in arms.items():
                torch.manual_seed(seed + step)
                model.train()
                model.acoustic.streaming_eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, out_lens = model(powers, lens, mask)
                lp = lp.float()
                if not torch.equal(out_lens, ref_lens):
                    raise RuntimeError("Pooling changed the output clock")
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
                emit(event="train", epoch=epoch, step=step, **metrics)
            if time.monotonic() - started > max_minutes * 60:
                break
        for name, model in arms.items():
            model.eval()
            model.acoustic.streaming_eval()
            pairs = []
            with torch.no_grad():
                for i in range(len(val)):
                    b = power_batch([val[i]])
                    lp, lens = model(b["powers"].cuda(), b["lens"].cuda())
                    hyp = ids_phonemes(ctc_collapse(lp[0, :lens[0]].argmax(-1).tolist()))
                    pairs.append((hyp, validation_records[i]["target"]))
            per = corpus_per(pairs)
            emit(event="validation", arm=name, epoch=epoch, step=step, per=per)
            if per < best[name]:
                best[name] = per
                state = dict(model=model.cpu().state_dict(), cfg=ck["cfg"], state=dict(step=step, epoch=epoch), per=per)
                torch.save(state, destination / f"{name}.pt")
                model.cuda()
        if time.monotonic() - started > max_minutes * 60:
            break
    emit(event="selected", best=best, steps=step)
    log.close()
    for name, model in arms.items():
        selected = torch.load(destination / f"{name}.pt", map_location="cpu", weights_only=False)
        model.cpu().load_state_dict(selected["model"])
        export_bundle(model, selected, destination / name, f"frequency-pooling-{name}")
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
    parser.add_argument("--original", default="/student/s1")
    parser.add_argument("--learners", default="/lab/data/help-train")
    parser.add_argument("--hours", type=float, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = original_records(args.original, root / "docs/benchmarks/gold-per-v1-exclude.txt", args.hours)
    learners = learner_records(args.learners, root / "packages/core/assets/ref-v2.json")
    ot, ov = partition_records(original)
    ht, hv = partition_records(learners)
    if not ht or not hv:
        raise ValueError("None-labelled learners need both training-speaker partitions")
    # Retain every original item, and mix learner takes at 1:4 by item count.
    mixed = ot + [ht[i % len(ht)] for i in range(max(len(ht), len(ot) // 4))]
    # Selection is training-only: a fixed subset of disjoint original/learner speakers.
    validation = ov[:128] + hv
    aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
    train_pair(args.checkpoint, mixed, validation, args.output, aug)


if __name__ == "__main__":
    main()
