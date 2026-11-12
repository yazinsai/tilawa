"""Matched upper-encoder continuations with transcript-conditioned decorrelation.

Run directly inside the supervisor's train image. No Modal client is used here.
Selection uses new speaker-grouped partitions of training material only; this
module neither opens practice sessions nor knows any target incident.

    python modal/student_decorrelation.py audit --hours 6
    python modal/student_decorrelation.py train --checkpoint /student/prior-runs/spliced-clean/ckpt.pt --out /student/runs/decorrelation
"""

import argparse
import bisect
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np

from phonetic_memory import balanced_indices, covariance_losses, occurrence_occupancy, pool_occurrences


def partition(speaker):
    """Stable 80/10/10 fit/development/confirmation split of training speakers."""
    bucket = int(hashlib.sha256(str(speaker).encode()).hexdigest()[:8], 16) % 10
    return "development" if bucket == 0 else "confirmation" if bucket == 1 else "fit"


class Pronunciations:
    """Recover explicit word boundaries from the canonical pronunciation index.

    The reference spells sustained sounds as repeated labels within a word.
    Collapse those runs, but preserve equal phones across distinct words.
    Ambiguous boundary layouts are excluded rather than guessed.
    """

    def __init__(self, asset):
        self.ayahs = {f"{a['s']}:{a['a']}": a for a in asset["ayahs"]}
        self.text, self.boundaries = "", set()
        for a in asset["ayahs"]:
            offset = len(self.text)
            self.boundaries.update(offset + i for i in a["o"])
            self.text += a["p"]
            self.boundaries.add(len(self.text))
        self.boundaries = sorted(self.boundaries)

    @staticmethod
    def articulate(words):
        return "".join(c for word in words for i, c in enumerate(word) if i == 0 or c != word[i - 1])

    def span(self, text):
        if not text:
            return None
        layouts, start = set(), 0
        while (start := self.text.find(text, start)) >= 0:
            lo = bisect.bisect_right(self.boundaries, start)
            hi = bisect.bisect_left(self.boundaries, start + len(text))
            cuts = [0] + [b - start for b in self.boundaries[lo:hi]] + [len(text)]
            layouts.add(self.articulate([text[a:b] for a, b in zip(cuts, cuts[1:])]))
            start += 1
        return next(iter(layouts)) if len(layouts) == 1 else None

    def passage(self, ayahs):
        words = []
        for key in ayahs:
            a = self.ayahs[key]
            cuts = list(a["o"]) + [len(a["p"])]
            words.extend(a["p"][lo:hi] for lo, hi in zip(cuts, cuts[1:]))
        return self.articulate(words)


def load_records(hours, seed, max_clip_s=30):
    """Clean transcribed s1 training clips and help training speakers' none takes.

    Blocks whole voice clusters containing a gold-excluded recording or any
    non-training row. Only metadata is scanned before selecting audio rows.
    No release-held-out help recordings are present in the authorized mount;
    nevertheless reject any non-training split found there.
    """
    import pyarrow.parquet as pq
    from student import exclusions, spliced_target

    reference = Pronunciations(json.loads(Path("packages/core/assets/ref-v2.json").read_text()))
    gold = exclusions(Path("/root/gold-per-v1-exclude.txt").read_text())
    files, metadata, blocked = sorted(Path("/student/s1").glob("*.parquet")), [], set()
    for path in files:
        rows = pq.read_table(path, columns=["shard", "row", "cluster", "split", "bucket", "n_samples"]).to_pylist()
        for index, row in enumerate(rows):
            speaker = f"tlog:{row['cluster']}"
            if row["cluster"] is None:
                raise ValueError("Training data needs explicit voice-cluster provenance")
            if row["split"] != "train" or (row["shard"], row["row"]) in gold:
                blocked.add(speaker)
            if row["split"] == "train" and row["bucket"] == "clean-match" and row["n_samples"] <= 16000 * max_clip_s:
                metadata.append((path, index, row, speaker))
    metadata = [r for r in metadata if r[3] not in blocked]
    rng = np.random.default_rng(seed)
    rng.shuffle(metadata)
    if hours:
        total, selected = 0, []
        for r in metadata:
            if total >= hours * 3600 * 16000:
                break
            selected.append(r)
            total += r[2]["n_samples"]
        metadata = selected
    by_file = {}
    for path, index, row, speaker in metadata:
        by_file.setdefault(path, []).append((index, speaker))
    records, ambiguous = [], 0
    for path, selected in by_file.items():
        table = pq.read_table(path).take([i for i, _ in selected]).to_pylist()
        for r, (_, speaker) in zip(table, selected):
            occurrence = reference.span(r["ref"])
            if not occurrence:
                ambiguous += 1
                continue
            records.append(dict(key=f"{r['shard']}:{r['row']}", speaker=speaker,
                                audio=r["flac"], samples=r["n_samples"],
                                target=spliced_target(r["ref"], r["hyp"]), occurrence=occurrence,
                                teacher_ids=r["top5_ids"], teacher_p=r["top5_p"]))
    help_root = Path("/lab/data/help-train")
    help_rows = [json.loads(line) for line in (help_root / "labels.jsonl").read_text().splitlines() if line.strip()]
    if any(r.get("split") != "train" or not r.get("speaker") for r in help_rows):
        raise ValueError("Help data contains missing provenance or a non-training speaker")
    for r in help_rows:
        if r["kind"] != "none" or r.get("exclude") or r.get("events") or r.get("extra_mistake"):
            continue
        ayahs = r["ayahs"]
        occurrence = reference.passage(ayahs)
        path = (help_root / r["wav"]).resolve()
        if not path.is_relative_to(help_root.resolve() / "wav"):
            raise ValueError("Help recording outside authorized wav directory")
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("Expected prepared mono 16 kHz training audio")
        if info.frames > 16000 * max_clip_s:
            continue
        # Help has no cached KD teacher targets. The frozen shipped output
        # provides its ordinary-reading preservation loss in both arms.
        records.append(dict(key=r["id"], speaker=f"help:{r['speaker']}", audio=path,
                            samples=info.frames, target=occurrence, occurrence=occurrence,
                            teacher_ids=None, teacher_p=None))
    groups = {name: [r for r in records if partition(r["speaker"]) == name]
              for name in ("fit", "development", "confirmation")}
    if any(not rows for rows in groups.values()):
        raise ValueError("Every training-speaker partition must contain recordings")
    if not any(r["speaker"].startswith("help:") for r in groups["fit"]):
        raise ValueError("No eligible none-labelled help training takes")
    audit = {name: dict(clips=len(rows), speakers=len({r["speaker"] for r in rows}),
                       hours=sum(r["samples"] for r in rows) / 16000 / 3600,
                       help_clips=sum(r["speaker"].startswith("help:") for r in rows)) for name, rows in groups.items()}
    audit.update(blocked_clusters=len(blocked), ambiguous_transcripts=ambiguous)
    return groups, audit


class RecordDataset:
    def __init__(self, records, cfg, training, augmentation, epoch=0):
        self.records, self.cfg, self.training, self.augmentation, self.epoch = records, cfg, training, augmentation, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import soundfile as sf
        import torch
        import torchaudio
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        r, cfg = self.records[i], self.cfg
        data = io.BytesIO(r["audio"]) if isinstance(r["audio"], bytes) else r["audio"]
        wave, sr = sf.read(data, dtype="float32")
        if sr != 16000 or wave.ndim != 1:
            raise ValueError("Expected mono 16 kHz prepared recording")
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], self.epoch, i]))
        if self.training:
            if self.augmentation:
                if rng.random() < cfg["rir_p"]:
                    wave = reverb(wave, pick(self.augmentation["rir"], rng))
                if rng.random() < cfg["noise_p"]:
                    wave = mix_noise(wave, pick(self.augmentation["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
                num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
        if self.training and cfg["spec_aug"]:
            feats = spec_augment(feats, rng)
        return dict(i=i, feats=feats, target=torch.tensor(phoneme_ids(r["target"])),
                    occurrence=phoneme_ids(r["occurrence"]),
                    top_ids=torch.from_numpy(np.frombuffer(r["teacher_ids"], np.uint8).copy().reshape(-1, 5).astype(np.int64)) if r["teacher_ids"] is not None else torch.empty(0, 5, dtype=torch.long),
                    top_p=torch.from_numpy(np.frombuffer(r["teacher_p"], np.float16).copy().reshape(-1, 5).astype(np.float32)) if r["teacher_p"] is not None else torch.empty(0, 5),
                    )


def collate(items):
    from student_train import collate as base_collate
    b = base_collate(items)
    b["occurrence"] = [x["occurrence"] for x in items]
    return b


def loader(records, cfg, training, aug, epoch=0):
    import torch
    from student_train import batches
    ds = RecordDataset(records, cfg, training, aug, epoch)
    return torch.utils.data.DataLoader(ds, batch_sampler=batches(np.array([r["samples"] for r in records]),
           cfg["max_batch_s"], cfg["seed"] + epoch, shuffle=training), collate_fn=collate, num_workers=cfg["workers"])


def alignments(lp, lens, targets):
    return [occurrence_occupancy(lp[k, :int(n)].detach().cpu().numpy(), y) for k, (n, y) in enumerate(zip(lens, targets))]


def embeddings(features, lens, occupancies, targets, seed, limit):
    import torch
    xs, ys = [], []
    for k, (n, occupancy, y) in enumerate(zip(lens, occupancies, targets)):
        if occupancy is not None:
            x, labels = pool_occurrences(features[k, :int(n)], occupancy, y)
            xs.append(x)
            ys.append(labels)
    if not xs:
        raise ValueError("Batch has no transcript-conditioned occurrences")
    x, y = torch.cat(xs), torch.cat(ys)
    indices = balanced_indices(y.cpu().numpy(), limit, seed)
    return x[indices], y[indices]


def objective(lp, lens, b, anchor_lp, cfg):
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss
    device = lp.device
    valid = torch.arange(lp.shape[1], device=device)[None] < lens[:, None]
    ctc = F.ctc_loss(lp.transpose(0, 1), b["targets"].to(device), lens,
                    b["target_lens"].to(device), blank=0, reduction="sum", zero_infinity=False) / lens.sum()
    preservation = (anchor_lp.exp() * (anchor_lp - lp)).sum(-1)[valid].mean()
    kd = lp.sum() * 0
    has_kd = b["n_teacher"] > 0
    if has_kd.any() and cfg["kd_weight"]:
        n = b["n_teacher"][has_kd].to(device)
        idx = (torch.arange(lp.shape[1], device=device)[None] - cfg["kd_delay"]).clamp_min(0)
        idx = torch.minimum(idx, (n - 1)[:, None])
        ids = b["top_ids"][has_kd].to(device).gather(1, idx[..., None].expand(-1, -1, 5))
        p = b["top_p"][has_kd].to(device).gather(1, idx[..., None].expand(-1, -1, 5))
        kd = soft_target_loss(lp[has_kd.to(device)], ids, p, valid[has_kd.to(device)])
    return cfg["ctc_weight"] * ctc + cfg["kd_weight"] * kd + cfg["preservation_weight"] * preservation


def evaluate(model, records, cfg):
    import torch
    from student import ctc_collapse, edit_distance, phoneme_ids
    model.eval()
    model.streaming_eval()
    edits, phones, divergences, frames = 0, 0, 0., 0
    with torch.no_grad():
        for b in loader(records, cfg, False, None):
            lp, lens = model(b["feats"].cuda(), b["feat_lens"].cuda())
            for k, n in enumerate(lens):
                reference = phoneme_ids(records[int(b["i"][k])]["occurrence"])
                edits += edit_distance(ctc_collapse(lp[k, :int(n)].argmax(-1).tolist()), reference)
                phones += len(reference)
            divergences += float(torch.nn.functional.ctc_loss(lp.transpose(0, 1), b["targets"].cuda(), lens,
                       b["target_lens"].cuda(), reduction="sum", zero_infinity=False))
            frames += int(lens.sum())
    return dict(per=edits / max(1, phones), ctc=divergences / max(1, frames), phones=phones)


def export_bundle(args):
    """Export the selected primary model, preserving the base SDK geometry/assets.

    Checks the actual streaming PyTorch wrapper against fp32 ONNX, then records
    int8 agreement on training-only confirmation recordings. Every declared
    auxiliary file is copied before the bundle can be marked complete.
    """
    import shutil
    import torch
    from student_train import build_model, DEFAULTS
    from student_export import export_onnx, manifest, stream_onnx, wrap, PAD, LOG_EPS
    from quantize_onnx import int8

    base, destination = Path(args.base_bundle), Path(args.out)
    original = json.loads((base / "manifest.json").read_text())
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]["size"])
    model.load_state_dict(ck["model"])
    model.eval()
    chunk, left = original["chunk"], original["leftContext"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").unlink(missing_ok=True)
    fp32 = destination / "model-fp32.onnx"
    states = export_onnx(model, str(fp32), chunk, left)
    generated = manifest(destination.name, states, chunk, left, str(fp32), ck)
    for key in ("inventory", "tokens", "blank", "sampleRate", "frameMs", "fbank", "window", "shift", "chunk", "leftContext", "states"):
        if generated[key] != original[key]:
            raise ValueError(f"Export changed the SDK interface: {key}")
    int8(fp32, destination / "model.onnx")
    groups, _ = load_records(args.hours, args.seed)
    cfg = {**DEFAULTS, **ck["cfg"]}
    dataset = RecordDataset(groups["confirmation"][:6], cfg, False, None)
    streaming = wrap(model, chunk, left)
    parity = []
    with torch.no_grad():
        for item in dataset:
            f = item["feats"]
            T, shift = chunk * 2 + PAD, chunk * 2
            need = ((len(f) - 7) // 2 + 1) // 2
            windows = max(1, -(-need // (chunk // 2)))
            total = T + (windows - 1) * shift
            padded = torch.cat([f, torch.full((max(0, total - len(f)), 80), LOG_EPS)])
            state, outputs = streaming.init_states(), []
            for a in range(0, total - T + 1, shift):
                lp, state = streaming(padded[None, a:a + T], state)
                outputs.append(lp[0])
            reference = torch.cat(outputs).numpy()
            actual = stream_onnx(str(fp32), f, states, chunk)
            quantized = stream_onnx(str(destination / "model.onnx"), f, states, chunk)
            error = float(np.max(np.abs(np.exp(reference) - np.exp(actual))))
            if error > 1e-4:
                raise ValueError(f"Streaming ONNX parity failed: {error}")
            parity.append(dict(key=dataset.records[item["i"]]["key"], frames=len(reference),
                               fp32_max_probability_error=error,
                               int8_top1_agreement=float((actual.argmax(-1) == quantized.argmax(-1)).mean()),
                               int8_max_probability_error=float(np.max(np.abs(np.exp(actual) - np.exp(quantized))))))
    # The folded insertion verifier is a separate, unchanged model. Refuse to
    # export a silently verifier-free replacement of the primary listener.
    registry = Path("packages/sdk/src/registry.ts").read_text()
    import re
    assets = re.findall(r'file: "([^\"]+)", bytes: (\d+),\s*sha256: "([a-f0-9]+)"', registry)
    if not assets:
        raise ValueError("Cannot verify the shipped insertion-teacher asset declaration")
    for filename, size, digest in assets:
        source = base / filename
        if source.stat().st_size != int(size):
            raise ValueError(f"Missing or incomplete auxiliary asset: {filename}")
        checksum = hashlib.sha256()
        with source.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                checksum.update(block)
        if checksum.hexdigest() != digest:
            raise ValueError(f"Auxiliary checksum mismatch: {filename}")
        shutil.copyfile(source, destination / filename)
    data = (destination / "model.onnx").read_bytes()
    generated.update(format="onnx-int8", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                     quantizedFrom=generated["sha256"])
    fp32.unlink()
    (destination / "parity.json").write_text(json.dumps(parity, indent=2))
    # Write the manifest last: an interrupted bundle has no completion marker.
    (destination / "manifest.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2))
    print(json.dumps(dict(bundle=str(destination), sha256=generated["sha256"], parity=parity)), flush=True)


def train(args):
    import torch
    from student_train import DEFAULTS, build_model
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import ScaledAdam

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = {**DEFAULTS, **ck["cfg"], "seed": args.seed, "workers": 4, "max_batch_s": args.batch_seconds,
           "preservation_weight": .1, "variance_weight": .1, "occurrence_limit": 32}
    groups, audit = load_records(args.hours, args.seed)
    print(json.dumps(dict(event="data", audit=audit, cfg=cfg)), flush=True)
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    def fresh():
        model = build_model(cfg["size"]).cuda()
        model.load_state_dict(ck["model"])
        return model

    anchor = fresh().eval()
    anchor.requires_grad_(False)
    anchor.streaming_eval()
    # Calibrate dimension scales on fit speakers only, without augmentation.
    sample_x, sample_y = [], []
    with torch.no_grad():
        for b in loader(groups["fit"][:args.calibration_clips], cfg, False, None):
            features, lens = anchor.forward_features(b["feats"].cuda(), b["feat_lens"].cuda())
            lp = anchor.ctc_output(features).log_softmax(-1)
            x, y = embeddings(features, lens, alignments(lp, lens, b["occurrence"]), b["occurrence"], args.seed, cfg["occurrence_limit"])
            sample_x.append(x)
            sample_y.append(y)
    x, y = torch.cat(sample_x), torch.cat(sample_y)
    _, inverse, counts = torch.unique(y, return_inverse=True, return_counts=True)
    w = counts[inverse].float().reciprocal() / len(counts)
    scale = ((x - (w[:, None] * x).sum(0)).square() * w[:, None]).sum(0).sqrt().clamp_min(.05).detach()
    torch.save(scale.cpu(), out / "scale.pt")
    del sample_x, sample_y, x, y
    weights = [0.] + [float(w) for w in args.covariance_weights.split(",")]
    if any(w <= 0 for w in weights[1:]):
        raise ValueError("Treatment weights must be positive")
    models, optimizers = [], []
    for weight in weights:
        model = fresh()
        model.encoder_embed.requires_grad_(False)
        for block in model.encoder.encoders[:3]:
            block.requires_grad_(False)
        optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=args.lr, include_names=True), lr=args.lr, clipping_scale=2.)
        models.append(model)
        optimizers.append(optimizer)
    best, epoch, step = None, 0, 0
    while step < args.steps:
        for b in loader(groups["fit"], cfg, True, aug, epoch):
            feats, feat_lens = b["feats"].cuda(), b["feat_lens"].cuda()
            with torch.no_grad():
                anchor_lp, lens = anchor(feats, feat_lens)
                occupancies = alignments(anchor_lp, lens, b["occurrence"])
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            metrics = []
            for weight, model, optimizer in zip(weights, models, optimizers):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                model.train()
                model.training_mode()
                for module in model.modules():
                    module.batch_count = ck["state"]["step"] * ck["cfg"].get("max_batch_s", 600) / 600 + step * cfg["max_batch_s"] / 600
                features, actual_lens = model.forward_features(feats, feat_lens)
                if not torch.equal(actual_lens, lens):
                    raise ValueError("Training and preservation clocks differ")
                lp = model.ctc_output(features).log_softmax(-1).float()
                x, y = embeddings(features, lens, occupancies, b["occurrence"], args.seed + step, cfg["occurrence_limit"])
                variance, covariance = covariance_losses(x, y, scale)
                ordinary = objective(lp, lens, b, anchor_lp.detach(), cfg)
                loss = ordinary + cfg["variance_weight"] * variance + weight * covariance
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite matched-arm loss; do not silently drop an update")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.append(dict(weight=weight, loss=float(loss.detach()), ordinary=float(ordinary.detach()),
                                    variance=float(variance.detach()), covariance=float(covariance.detach()), occurrences=len(x)))
            step += 1
            if step % 10 == 0:
                print(json.dumps(dict(event="train", step=step, arms=metrics)), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                results = [evaluate(model, groups["development"], cfg) for model in models]
                print(json.dumps(dict(event="development", step=step, weights=weights, arms=results)), flush=True)
                for arm, model in enumerate(models):
                    torch.save(dict(model=model.state_dict(), cfg=cfg, state=dict(step=ck["state"]["step"] + step),
                                    covariance_weight=weights[arm], continuation_steps=step), out / f"arm-{arm}-step-{step}.pt")
                for arm in range(1, len(models)):
                    rank = (results[arm]["per"], results[arm]["ctc"], weights[arm], step)
                    if best is None or rank < best["rank"]:
                        best = dict(rank=rank, arm=arm, steps=step, weight=weights[arm], development=results[arm], control_development=results[0])
            if step == args.steps:
                break
        epoch += 1
    # Confirmation is read only once after selection; it cannot change the model.
    confirmation = {}
    for name, arm in (("regularized", best["arm"]), ("control", 0)):
        selected = torch.load(out / f"arm-{arm}-step-{best['steps']}.pt", map_location="cpu", weights_only=False)
        model = models[arm]
        model.load_state_dict(selected["model"])
        confirmation[name] = evaluate(model, groups["confirmation"], cfg)
        destination = out / name
        destination.mkdir(exist_ok=True)
        torch.save(selected, destination / "ckpt.pt")
    result = dict(selected=best, confirmation=confirmation, audit=audit, cfg=cfg,
                  comparison="identical batches, augmentations, dropout draws, occurrence samples, variance floor and updates; covariance weight differs")
    (out / "selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(event="selection", **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "train", "export"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    parser.add_argument("--base-bundle")
    parser.add_argument("--hours", type=float, default=6.)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-seconds", type=float, default=160.)
    parser.add_argument("--calibration-clips", type=int, default=100)
    parser.add_argument("--covariance-weights", default="0.01,0.1")
    parser.add_argument("--lr", type=float, default=.00045)
    args = parser.parse_args()
    if args.action == "audit":
        _, audit = load_records(args.hours, args.seed)
        print(json.dumps(audit, indent=2))
    elif args.action == "train":
        if not args.checkpoint or not args.out:
            parser.error("train requires --checkpoint and --out")
        train(args)
    else:
        if not args.checkpoint or not args.out or not args.base_bundle:
            parser.error("export requires --checkpoint, --out and --base-bundle")
        export_bundle(args)


if __name__ == "__main__":
    main()
