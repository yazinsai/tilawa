"""Matched routed/dense continuation; call directly inside the training image.

python modal/student_specialist_train.py --checkpoint /student/prior-runs/spliced-clean/ckpt.pt \
    --arm routed --out /scratch/<experiment>/routed

No practice audio or labels enter training or checkpoint selection. Run the two
arms with the same seed, data limits and schedule. Only none-labelled help takes
provide transcript supervision. All saved files stay under --out.
"""

import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import student_train as T
from student import phoneme_ids, spliced_target
from per_breakdown import collapse
from student_specialists import ctc_regime_occupancy, regime_map, router_loss


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def copy_verification_assets(source, destination):
    # These are the assets loaded by the unchanged SDK registry. Missing them
    # would silently disable insertion verification in loadNodeModel.
    for name in ("teacher.onnx", "teacher.onnx.data"):
        src, dst = source / name, destination / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing shipped verification asset: {src}")
        shutil.copy2(src, dst)
        if file_hash(src) != file_hash(dst):
            raise RuntimeError(f"Verification asset copy differs: {name}")


def streaming_parity(model, path, features, states, chunk=16, left=128):
    """Check every fp32 output and cache against the same PyTorch call loop."""
    import onnxruntime as ort
    from student_export import LOG_EPS, PAD, wrap

    streaming = wrap(model, chunk, left)
    cached = streaming.init_states()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {s["name"]: t.numpy() for s, t in zip(states, cached)}
    size, shift = chunk * 2 + PAD, chunk * 2
    need = ((features.shape[0] - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (chunk // 2)))
    total = size + (windows - 1) * shift
    padded = F.pad(features, (0, 0, 0, max(0, total - len(features))), value=LOG_EPS)
    difference = 0.
    with torch.no_grad():
        for at in range(0, total - size + 1, shift):
            x = padded[None, at:at + size]
            expected, cached = streaming(x, cached)
            got = session.run(None, {"x": x.numpy(), **feeds})
            if len(got) != len(cached) + 1 or not np.isfinite(got[0]).all():
                raise RuntimeError("Fp32 export changed the cache interface or produced nonfinite posteriors")
            np.testing.assert_allclose(got[0], expected.numpy(), rtol=1e-3, atol=1e-4)
            difference = max(difference, float(abs(np.exp(got[0]) - expected.exp().numpy()).max()))
            for actual, ref in zip(got[1:], cached):
                np.testing.assert_allclose(actual, ref.numpy(), rtol=1e-3, atol=1e-4)
            feeds = {s["name"]: actual for s, actual in zip(states, got[1:])}
    return difference


def none_takes(help_dir, reference):
    labels = [json.loads(line) for line in (help_dir / "labels.jsonl").read_text().splitlines() if line]
    held = {r["speaker"] for r in labels if r.get("split") != "train"}
    if held & {r["speaker"] for r in labels if r.get("split") == "train"}:
        raise ValueError("A help speaker occurs in both training and held-out splits")
    by = {(a["s"], a["a"]): a["p"] for a in reference["ayahs"]}
    result = []
    for row in labels:
        if (row.get("split") != "train" or row.get("kind") != "none" or
                row.get("excluded") or row.get("scored") is False or row.get("events") or row.get("non_errors") or
                row.get("unsupported") or row.get("relabel")):
            continue
        wav = (help_dir / row["wav"]).resolve()
        if not wav.is_relative_to((help_dir / "wav").resolve()):
            raise ValueError("Help audio must be inside its supplied wav directory")
        wave, rate = sf.read(wav, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(1)
        if rate != T.SR:
            import torchaudio
            wave = torchaudio.functional.resample(torch.from_numpy(wave), rate, T.SR).numpy()
        target = "".join(by[tuple(map(int, ayah.split(":")))] for ayah in row["ayahs"])
        result.append((row["speaker"], wave, collapse(target)))
    if not result:
        raise ValueError("No eligible none-labelled training-speaker help takes")
    return result


class PairedData:
    def __init__(self, clips, help_takes, cfg, aug, seed):
        self.clips, self.help, self.cfg, self.aug, self.seed = clips, help_takes, cfg, aug, seed
        self.n_samples = np.concatenate((clips["n_samples"], [len(x[1]) for x in help_takes]))
        self.epoch = 0

    def __len__(self):
        return len(self.n_samples)

    def __getitem__(self, i):
        return self.item(i, self.epoch)

    def item(self, i, epoch):
        import torchaudio
        c = self.clips
        if i < len(c["ref"]):
            wave, _ = sf.read(io.BytesIO(c["flac"][c["flac_off"][i]:c["flac_off"][i + 1]].tobytes()), dtype="float32")
            target = spliced_target(c["ref"][i], c["hyp"][i])
            reference = collapse(c["ref"][i])
            lo, hi = c["top_off"][i], c["top_off"][i + 1]
            ids = torch.from_numpy(c["ids"][lo:hi].reshape(-1, 5).astype(np.int64))
            probs = torch.from_numpy(c["probs"][lo:hi].reshape(-1, 5).astype(np.float32))
        else:
            _, wave, target = self.help[i - len(c["ref"])]
            reference = target
            ids, probs = torch.zeros((1, 5), dtype=torch.long), torch.zeros((1, 5))
        # Per-item draws make channel/spec augmentation identical in both arms,
        # independently of their different model dropout and optimizer draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, i]))
        if rng.random() < self.cfg["rir_p"]:
            wave = T.reverb(wave, T.pick(self.aug["rir"], rng))
        if rng.random() < self.cfg["noise_p"]:
            from student import mix_noise
            wave = mix_noise(wave, T.pick(self.aug["noise"], rng), rng.uniform(*self.cfg["snr_db"]), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
            num_mel_bins=80, sample_frequency=T.SR, dither=0., energy_floor=1e-10, high_freq=-400)
        if self.cfg["spec_aug"]:
            feats = T.spec_augment(feats, rng)
        return dict(i=i, feats=feats, target=torch.tensor(phoneme_ids(target), dtype=torch.long),
                    reference=torch.tensor(phoneme_ids(reference), dtype=torch.long), top_ids=ids, top_p=probs)


def continuation(args):
    from student import TOKENS, soft_target_loss
    from student_export import export_onnx, manifest, stream_onnx
    from quantize_onnx import int8

    if args.epochs < 1 or args.router_steps < 0:
        raise ValueError("Continuation requires at least one epoch and nonnegative router steps")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = {**T.DEFAULTS, **ck["cfg"], "specialist_arm": args.arm}
    if cfg["size"] != "lite":
        raise ValueError("This continuation requires the shipped lite architecture")
    clips = T.load_clips("clean-match", "train", args.hours, 30., args.seed,
                         exclude_path=str(ROOT / "docs/benchmarks/gold-per-v1-exclude.txt"))
    help_takes = none_takes(args.help, json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text()))
    # Speaker-disjoint help validation and a disjoint subset of authorized s1
    # clips. These remain training-only data, never release or practice scoring.
    speakers = sorted({speaker for speaker, _, _ in help_takes})
    held_speakers = set(speakers[::5])
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    ds = PairedData(clips, help_takes, cfg, aug, args.seed)
    n_quran = len(clips["ref"])
    valid_indices = set(range(0, n_quran, 10)) | {
        n_quran + i for i, row in enumerate(help_takes) if row[0] in held_speakers}
    train_indices = [i for i in range(len(ds)) if i not in valid_indices]
    if not train_indices or not valid_indices:
        raise ValueError("Continuation requires disjoint nonempty training and validation data")
    base = T.build_model(cfg["size"])
    base.load_state_dict(ck["model"])
    model = T.build_model(cfg["size"])
    model.load_state_dict(ck["model"])
    from student_specialists import replace_upper_block
    block = replace_upper_block(model, args.arm)
    base, model = base.cuda().eval(), model.cuda().eval()
    base.streaming_eval(16, 128)
    model.streaming_eval(16, 128)
    base.requires_grad_(False)
    model.requires_grad_(False)
    # Keep every shared lower block and the single CTC head fixed. Jointly
    # update all expert/router parameters in the upper residual block.
    block.requires_grad_(True)
    optimizer = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=.01)
    classes = regime_map(TOKENS)
    log = (args.out / "train.jsonl").open("w")

    def emit(**row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    emit(event="recipe", args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
         cfg=cfg, quran_clips=n_quran, help_takes=len(help_takes), help_speakers=len(speakers),
         training_items=len(train_indices), validation_items=len(valid_indices),
         parameters=sum(p.numel() for p in model.parameters()),
         base_checkpoint_sha256=file_hash(args.checkpoint))

    def batch_loss(indices, training, warm=False):
        # Checkpoint selection always sees the same validation augmentation.
        items = [ds.item(i, ds.epoch if training else 0) for i in indices]
        b = T.collate(items)
        feats, lens = b["feats"].cuda(), b["feat_lens"].cuda()
        with torch.no_grad():
            anchor, out_lens = base(feats, lens)
        block.train(training)
        if warm and args.arm == "routed":
            block.experts.eval()
        lp, lengths = model(feats, lens)
        if not torch.equal(out_lens, lengths):
            raise RuntimeError("Specialization changed the ordinary CTC frame clock")
        frames = torch.arange(lp.shape[1], device=lp.device)[None] < lengths[:, None]
        ctc = F.ctc_loss(lp.transpose(0, 1), b["targets"].cuda(), lengths,
                         b["target_lens"].cuda(), reduction="sum", zero_infinity=True) / frames.sum()
        preservation = F.kl_div(lp, anchor.exp(), reduction="none").sum(-1)[frames].mean()
        kd_frames = frames & (b["i"].cuda() < n_quran)[:, None]
        ix = (torch.arange(lp.shape[1])[None] - cfg["kd_delay"]).clamp_min(0)
        ix = ix.minimum((b["n_teacher"] - 1)[:, None])
        kd = soft_target_loss(lp, b["top_ids"].gather(1, ix[..., None].expand(-1, -1, 5)).cuda(),
                             b["top_p"].gather(1, ix[..., None].expand(-1, -1, 5)).cuda(), kd_frames)
        route = lp.new_zeros(())
        if args.arm == "routed" and training:
            occupancy = torch.zeros(lp.shape[0], lp.shape[1], 3)
            usable_lengths = lengths.clone()
            anchor_cpu = anchor.cpu()
            for i, item in enumerate(items):
                # Supervise routing from authorized text, never teacher-decoded
                # edge phones. The acoustic CTC target retains shipped edges.
                q = ctc_regime_occupancy(anchor_cpu[i, :int(lengths[i])], item["reference"], classes)
                if q is None:
                    usable_lengths[i] = 0
                else:
                    occupancy[i, :len(q)] = q
            route = router_loss(block.route_logits, occupancy.cuda(), usable_lengths)
        total = cfg["ctc_weight"] * ctc + cfg["kd_weight"] * kd + args.anchor * preservation
        if args.arm == "routed" and training:
            total = args.route_weight * route if warm else total + args.route_weight * route
        return total, dict(ctc=float(ctc), kd=float(kd), anchor=float(preservation), route=float(route))

    plan = T.batches(ds.n_samples[train_indices], args.batch_seconds, args.seed)
    for step in range(args.router_steps):
        indices = [train_indices[i] for i in plan[step % len(plan)]]
        if args.arm == "routed":
            block.experts.requires_grad_(False)
            loss, stats = batch_loss(indices, True, warm=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            emit(event="router", step=step, **stats)
    block.requires_grad_(True)
    best, step = float("inf"), 0
    for epoch in range(args.epochs):
        ds.epoch = epoch
        plan = T.batches(ds.n_samples[train_indices], args.batch_seconds, args.seed + epoch)
        for part in plan:
            loss, stats = batch_loss([train_indices[i] for i in part], True)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite continuation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 2.)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                emit(event="joint", epoch=epoch, step=step, loss=float(loss), **stats)
        with torch.no_grad():
            scores = [float(batch_loss([i], False)[0]) for i in sorted(valid_indices)]
        score = sum(scores) / len(scores)
        emit(event="validation", epoch=epoch, objective=score)
        if score < best:
            best = score
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            cfg=cfg, state=dict(step=step, epoch=epoch)),
                       args.out / "ckpt.pt")
    saved = torch.load(args.out / "ckpt.pt", map_location="cpu", weights_only=False)
    model = T.build_model(cfg["size"], args.arm)
    model.load_state_dict(saved["model"])
    model.eval()
    fp32 = args.out / "fp32"
    fp32.mkdir(exist_ok=True)
    states = export_onnx(model, str(fp32 / "model.onnx"), 16, 128)
    m = manifest("specialist-" + args.arm, states, 16, 128, str(fp32 / "model.onnx"), saved)
    (fp32 / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    bundle = args.out / "int8"
    bundle.mkdir(exist_ok=True)
    int8(fp32 / "model.onnx", bundle / "model.onnx")
    data = (bundle / "model.onnx").read_bytes()
    m.update(format="onnx-int8", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
             quantizedFrom=m["sha256"])
    (bundle / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    # Retain the complete installed teacher bundle used by unchanged verification.
    copy_verification_assets(args.shipped, bundle)
    # Full deployed cache loop over held-back training audio, both precisions.
    ds.epoch = 0
    parity = []
    for i in sorted(valid_indices)[:8]:
        features = ds[i]["feats"]
        torch_difference = streaming_parity(model, fp32 / "model.onnx", features, states)
        a = stream_onnx(str(fp32 / "model.onnx"), features, states, 16)
        b = stream_onnx(str(bundle / "model.onnx"), features, states, 16)
        if a.shape != b.shape or not np.isfinite(b).all():
            raise RuntimeError("Int8 export changed the frame interface or produced nonfinite posteriors")
        parity.append(dict(item=i, frames=len(a), torch_max_probability_difference=torch_difference,
                           agreement=float((a.argmax(-1) == b.argmax(-1)).mean()),
                           max_probability_difference=float(abs(np.exp(a) - np.exp(b)).max())))
    (args.out / "parity.json").write_text(json.dumps(parity, indent=1))
    emit(event="export", bundle=str(bundle), validation_objective=best, parity=parity)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=["routed", "dense"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--help-data", dest="help", type=Path, default=Path("/lab/data/help-train"))
    p.add_argument("--shipped", type=Path, default=Path("/lab/data/models/spliced-clean-int8"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--hours", type=float, default=8.)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--router-steps", type=int, default=100)
    p.add_argument("--batch-seconds", type=float, default=200.)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=4.)
    p.add_argument("--route-weight", type=float, default=.25)
    continuation(p.parse_args())


if __name__ == "__main__":
    main()
