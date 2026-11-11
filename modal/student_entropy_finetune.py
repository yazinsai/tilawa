"""Matched continuation/entropy fine-tuning on authorized training speakers.

Run directly in the supervisor's training image, with student_train and tools
on PYTHONPATH. No practice inputs enter this program. Both arms receive the
same examples, augmentations, dropout draws, initialization and preservation
loss. Selection uses only speaker partitions inside the training population.
"""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from student_alignment_entropy import alignment_entropy


def partition(speaker):
    return "validation" if int(hashlib.sha256(str(speaker).encode()).hexdigest()[:8], 16) % 5 == 0 else "fit"


def help_none_rows(rows):
    """Admit audited ordinary takes only, excluding entire held-out speakers.

    A none prompt alone is insufficient: the importer marks takes with extra
    mistakes unscored, and unsupported mistakes may have no core event.
    """
    heldout = {row["speaker"] for row in rows if row.get("split") != "train"}
    return [row for row in rows
            if row.get("split") == "train" and row["speaker"] not in heldout
            and row.get("kind") == "none" and row.get("scored") is True
            and not any(row.get(key) for key in ("excluded", "exclude", "events", "non_errors", "unsupported"))]


def load_recordings(s1, help_root, reference, exclusion_file, max_hours, seed):
    import pyarrow.parquet as pq
    from per_breakdown import collapse
    from student import exclusions

    excluded = exclusions(Path(exclusion_file).read_text())
    files = sorted(Path(s1).glob("*.parquet"))
    metadata, denied = [], set()
    for file in files:
        rows = pq.read_table(file, columns=["shard", "row", "cluster", "bucket", "split", "n_samples"]).to_pylist()
        for index, row in enumerate(rows):
            speaker = str(row["cluster"])
            if row["split"] != "train" or (row["shard"], row["row"]) in excluded:
                denied.add(speaker)
            metadata.append((file, index, row))
    eligible = [(file, index, row) for file, index, row in metadata
                if str(row["cluster"]) not in denied and row["split"] == "train"
                and row["bucket"] == "clean-match" and 1 <= row["n_samples"] / 16000 <= 20]
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    selected, seconds = {}, 0
    for file, index, row in eligible:
        if seconds >= max_hours * 3600:
            break
        selected.setdefault(file, []).append(index)
        seconds += row["n_samples"] / 16000
    recordings = []
    for file, indices in selected.items():
        rows = pq.read_table(file).take(indices).to_pylist()
        for row in rows:
            recordings.append(dict(key=f"{row['shard']}:{row['row']}", speaker=f"quran:{row['cluster']}",
                                   split=partition(f"quran:{row['cluster']}"), transcript=collapse(row["ref"]),
                                   audio=row["flac"], samples=row["n_samples"],
                                   ids=row["top5_ids"], probs=row["top5_p"], teacher_frames=row["n_frames"]))
    ayahs = {(a["s"], a["a"]): a for a in json.loads(Path(reference).read_text())["ayahs"]}
    help_rows = [json.loads(line) for line in (Path(help_root) / "labels.jsonl").read_text().splitlines() if line.strip()]
    for row in help_none_rows(help_rows):
        phones = "".join(ayahs[tuple(map(int, key.split(":")))]["p"] for key in row["ayahs"])
        speaker = f"help:{row['speaker']}"
        recordings.append(dict(key=f"help:{row['id']}", speaker=speaker, split=partition(speaker),
                               transcript=collapse(phones), audio=(Path(help_root) / row["wav"]).read_bytes(),
                               samples=None, ids=None, probs=None, teacher_frames=0))
    if {r["split"] for r in recordings} != {"fit", "validation"}:
        raise ValueError("Need authorized recordings in both training-speaker partitions")
    return recordings, dict(eligible_quran=len(eligible), denied_quran_speakers=len(denied),
                            help_none=sum(r["key"].startswith("help:") for r in recordings),
                            recordings=[{k: r[k] for k in ("key", "speaker", "split", "transcript")} for r in recordings])


class Recordings:
    def __init__(self, rows, aug=None, epoch=0, seed=0, augmentation=None):
        self.rows, self.aug, self.epoch, self.seed = rows, aug, epoch, seed
        self.augmentation = augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        import torchaudio
        import student_train as T
        from student import mix_noise, phoneme_ids

        row = self.rows[i]
        wave, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(-1)
        if rate != T.SR:
            wave = torchaudio.functional.resample(torch.from_numpy(wave), rate, T.SR).numpy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        if self.aug:
            cfg = self.augmentation
            if cfg is None:
                raise ValueError("Augmentation must inherit the initial checkpoint's recipe")
            if rng.random() < cfg["rir_p"]:
                wave = T.reverb(wave, T.pick(self.aug["rir"], rng))
            if rng.random() < cfg["noise_p"]:
                wave = mix_noise(wave, T.pick(self.aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
                    num_mel_bins=80, sample_frequency=T.SR, dither=0., energy_floor=1e-10, high_freq=-400)
        if self.aug and self.augmentation["spec_aug"]:
            feats = T.spec_augment(feats, rng)
        ids = np.frombuffer(row["ids"], np.uint8).reshape(-1, 5).astype(np.int64) if row["ids"] else np.zeros((1, 5), np.int64)
        probs = np.frombuffer(row["probs"], np.float16).reshape(-1, 5).astype(np.float32) if row["probs"] else np.zeros((1, 5), np.float32)
        return dict(i=i, feats=feats, target=torch.tensor(phoneme_ids(row["transcript"]), dtype=torch.long),
                    top_ids=torch.from_numpy(ids), top_p=torch.from_numpy(probs))


def objective(lp, lengths, batch, frozen, cfg, regularize):
    from student import soft_target_loss

    targets, sizes = batch["targets"].to(lp.device), batch["target_lens"].to(lp.device)
    frames = lengths.sum()
    ctc = F.ctc_loss(lp.transpose(0, 1), targets, lengths, sizes, reduction="sum", zero_infinity=False) / frames
    time = lp.shape[1]
    mask = torch.arange(time, device=lp.device)[None] < lengths[:, None]
    anchor = F.kl_div(lp, frozen.exp(), reduction="none").sum(-1)
    anchor = (anchor * mask).sum() / frames
    # Cached top-5 KD is applied only where supplied. Help none clips retain
    # CTC and shipped preservation; no mistake script or pseudo-label is used.
    ids, probs = batch["top_ids"].to(lp.device), batch["top_p"].to(lp.device)
    teacher_lengths = batch["n_teacher"].to(lp.device)
    indices = torch.minimum(torch.arange(time, device=lp.device)[None], (teacher_lengths - 1)[:, None])
    ids = ids.gather(1, indices[:, :, None].expand(-1, -1, 5))
    probs = probs.gather(1, indices[:, :, None].expand(-1, -1, 5))
    kd = soft_target_loss(lp, ids, probs, mask & (probs.sum(-1) > 0))
    h, _, neutral = alignment_entropy(lp, targets, lengths, sizes)
    deficit = (neutral - h).sum() / frames
    loss = ctc + cfg["kd_weight"] * kd + cfg["preservation_weight"] * anchor
    if regularize:
        loss = loss + cfg["entropy_weight"] * deficit
    return loss, dict(ctc=ctc.item(), kd=kd.item(), preservation=anchor.item(),
                      entropy=h.sum().item() / frames.item(), neutral_entropy=neutral.sum().item() / frames.item(),
                      entropy_deficit=deficit.item())


def validate(model, frozen, rows, workers):
    import student_train as T
    from student import ctc_collapse, edit_distance, phoneme_ids

    model.eval()
    total_edits = total_phones = 0
    total_kl = total_frames = 0
    loader = torch.utils.data.DataLoader(Recordings(rows), batch_size=8, collate_fn=T.collate, num_workers=workers)
    with torch.no_grad():
        for batch in loader:
            lp, lengths = model(batch["feats"].cuda(), batch["feat_lens"].cuda())
            ref, _ = frozen(batch["feats"].cuda(), batch["feat_lens"].cuda())
            mask = torch.arange(lp.shape[1], device=lp.device)[None] < lengths[:, None]
            total_kl += (F.kl_div(lp.float(), ref.float().exp(), reduction="none").sum(-1) * mask).sum().item()
            total_frames += lengths.sum().item()
            best = lp.argmax(-1).cpu()
            for k, i in enumerate(batch["i"].tolist()):
                target = phoneme_ids(rows[i]["transcript"])
                total_edits += edit_distance(ctc_collapse(best[k, :lengths[k]].tolist()), target)
                total_phones += len(target)
    model.train()
    return dict(per=total_edits / max(1, total_phones), preservation_kl=total_kl / total_frames,
                selection_score=total_edits / max(1, total_phones) + 0.1 * total_kl / total_frames)


def export_bundle(checkpoint, output, base_bundle):
    import shutil
    import student_train as T
    import student_export as E
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = T.build_model(ck["cfg"]["size"])
    model.load_state_dict(ck["model"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fp32 = output / "fp32.onnx"
    states = E.export_onnx(model, str(fp32), 16, 128)
    quantize_dynamic(str(fp32), str(output / "model.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    manifest = E.manifest(output.name, states, 16, 128, str(output / "model.onnx"), ck)
    manifest["format"] = "onnx-int8"
    base_bundle = Path(base_bundle)
    base = json.loads((base_bundle / "manifest.json").read_text())
    for field in ("inventory", "tokens", "blank", "sampleRate", "frameMs", "fbank", "window", "shift", "chunk", "leftContext", "states"):
        if manifest[field] != base[field]:
            raise ValueError(f"Export changes the primary listener contract: {field}")
    # Retain the installed listener's auxiliary assets, including its folded
    # insertion teacher. Fine-tuning the primary must not disable that path.
    for item in base_bundle.iterdir():
        if item.name not in ("manifest.json", base["file"]):
            if item.is_dir():
                shutil.copytree(item, output / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output / item.name)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    fp32.unlink()


def main():
    import student_train as T

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--s1", default="/student/s1")
    parser.add_argument("--help", default="/lab/data/help-train")
    parser.add_argument("--reference", default="/app/packages/core/assets/ref-v2.json")
    parser.add_argument("--exclusions", default="/app/docs/benchmarks/gold-per-v1-exclude.txt")
    parser.add_argument("--hours", type=float, default=4.)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-seconds", type=float, default=80.)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--entropy-weight", type=float, default=.05)
    parser.add_argument("--preservation-weight", type=float, default=1.)
    parser.add_argument("--kd-weight", type=float, default=.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--base-bundle", help="Installed primary and auxiliary assets to preserve during export")
    args = parser.parse_args()
    if args.export and not args.base_bundle:
        parser.error("--export requires --base-bundle")
    if min(args.hours, args.epochs, args.batch_seconds, args.lr) <= 0 or args.workers < 0:
        parser.error("Hours, epochs, batch duration and learning rate must be positive; workers must be nonnegative")
    if min(args.entropy_weight, args.preservation_weight, args.kd_weight) < 0:
        parser.error("Loss weights must be nonnegative")
    cfg = vars(args)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_recordings(args.s1, args.help, args.reference, args.exclusions, args.hours, args.seed)
    (out / "partitions.json").write_text(json.dumps(provenance, ensure_ascii=False))
    fit, validation = [r for r in rows if r["split"] == "fit"], [r for r in rows if r["split"] == "validation"]
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg["size"] = ck["cfg"]["size"]
    cfg["augmentation"] = {key: ck["cfg"].get(key, T.DEFAULTS[key])
                           for key in ("rir_p", "noise_p", "snr_db", "spec_aug")}
    models = {}
    for name in ("control", "entropy", "frozen"):
        model = T.build_model(cfg["size"]).cuda()
        model.load_state_dict(ck["model"], strict=True)
        model.streaming_eval(16, 128)
        for module in model.modules():
            module.batch_count = ck["state"]["step"] * ck["cfg"].get("max_batch_s", 600) / 600
        models[name] = model
    frozen = models.pop("frozen").eval().requires_grad_(False)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.) for name, model in models.items()}
    aug = {key: dict(np.load(f"/student/aug/{key}.npz")) for key in ("noise", "rir")}
    best = {name: float("inf") for name in models}
    log = (out / "log.jsonl").open("a")

    def emit(**values):
        line = json.dumps(values)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(event="config", fit=len(fit), validation=len(validation), cfg=cfg)
    step = 0
    for epoch in range(args.epochs):
        dataset = Recordings(fit, aug, epoch, args.seed, cfg["augmentation"])
        # Determine missing help durations once; no audio-dependent selection.
        for i, row in enumerate(fit):
            if row["samples"] is None:
                import soundfile as sf
                info = sf.info(io.BytesIO(row["audio"]))
                row["samples"] = round(info.duration * 16000)
        plan = T.batches(np.array([row["samples"] for row in fit]), args.batch_seconds, args.seed + epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=plan, collate_fn=T.collate, num_workers=args.workers)
        for model in models.values():
            model.train()
        for batch in loader:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                anchor, _ = frozen(batch["feats"].cuda(), batch["feat_lens"].cuda())
            cpu_rng, cuda_rng, python_rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
            metrics = {}
            for name, model in models.items():
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                random.setstate(python_rng)
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp, lengths = model(batch["feats"].cuda(), batch["feat_lens"].cuda())
                loss, metrics[name] = objective(lp.float(), lengths, batch, anchor.float(), cfg, name == "entropy")
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite matched-training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
            step += 1
            if step % 20 == 0:
                emit(event="train", epoch=epoch, step=step, arms=metrics)
        for name, model in models.items():
            values = validate(model, frozen, validation, args.workers)
            emit(event="validation", epoch=epoch, step=step, arm=name, **values)
            if values["selection_score"] < best[name]:
                best[name] = values["selection_score"]
                torch.save(dict(model=model.state_dict(), cfg=cfg, state=dict(step=step, epoch=epoch),
                                validation=values), out / f"{name}.pt")
    emit(event="selected", scores=best)
    if args.export:
        for name in models:
            export_bundle(out / f"{name}.pt", out / name, args.base_bundle)


if __name__ == "__main__":
    main()
