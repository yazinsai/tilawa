"""Whole-recording acoustic Group DRO with a matched mean-loss adapter control.

Runs directly in the training image; does not invoke Modal. Speaker partitions,
acoustic clusters and checkpoint selection use training inputs exclusively.
Adapters merge into ordinary Linear weights before the existing streaming export.

  PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_group_dro.py inspect --out /scratch/.../data
  PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_group_dro.py train --data /scratch/.../data --parent /student/prior-runs/spliced-clean/ckpt.pt --out /scratch/.../fit
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


def partition(speaker: str, folds: int = 5) -> str:
    if folds < 2:
        raise ValueError("Need at least two speaker partitions")
    if not speaker:
        raise ValueError("Speaker or speaker-cluster identity is required")
    return "validation" if int(hashlib.sha256(speaker.encode()).hexdigest()[:16], 16) % folds == 0 else "train"


def clean_help(row: dict) -> bool:
    # Never reinterpret a scripted deviation, repeat or unreviewed exclusion as clean.
    return (row.get("split") == "train" and row.get("kind") == "none"
            and row.get("scored") is True and not row.get("excluded")
            and not any(row.get(k) for k in ("events", "non_errors", "unsupported")))


def descriptors(wave: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Text-free descriptors of the complete recording, including its pauses."""
    w = np.asarray(wave, dtype=np.float64)
    if w.ndim != 1 or len(w) < sr // 10 or not np.isfinite(w).all():
        raise ValueError("Need finite mono audio of at least 100 ms")
    n = 400
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    frames = np.lib.stride_tricks.sliding_window_view(w, n)[::160]
    energy = np.mean(frames ** 2, axis=1)
    q = np.quantile(energy, [.1, .5, .9])
    logq = np.log(np.maximum(q, 1e-12))
    power = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    low = power[:, (f >= 100) & (f < 1000)].sum(1)
    high = power[:, (f >= 2500) & (f < 7000)].sum(1)
    zcr = (frames[:, 1:] * frames[:, :-1] < 0).mean()
    flatness = np.exp(np.log(power).mean(1)) / power.mean(1)
    return np.array([logq[1], logq[2] - logq[0],
                     (energy < max(q[2] * .01, 1e-12)).mean(), zcr,
                     np.median(np.log(high / (low + 1e-12))),
                     np.median(flatness), np.quantile(flatness, .9)], dtype=np.float64)


class AcousticGroups:
    """Fit robust scaling and k-means only on training-speaker recordings."""
    def fit(self, x: np.ndarray, groups: int = 6):
        if groups < 2 or x.ndim != 2 or len(x) < groups or not np.isfinite(x).all():
            raise ValueError("Insufficient finite training descriptors")
        self.center = np.median(x, axis=0)
        self.scale = np.maximum(np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0), 1e-6)
        z = (x - self.center) / self.scale
        centers = [z[np.argmin(np.sum(z * z, axis=1))]]
        for _ in range(1, groups):
            d = np.min(((z[:, None] - np.array(centers)) ** 2).sum(-1), axis=1)
            if d.max() <= 0:
                raise ValueError("Too few distinct acoustic regimes")
            centers.append(z[np.argmax(d)])
        self.centers = np.array(centers)
        for _ in range(100):
            labels = self.predict(x)
            if len(set(labels)) != groups:
                raise ValueError("An acoustic cluster is empty")
            c = np.array([z[labels == g].mean(0) for g in range(groups)])
            if np.allclose(c, self.centers):
                break
            self.centers = c
        return self

    def predict(self, x):
        z = (np.asarray(x) - self.center) / self.scale
        return ((z[:, None] - self.centers) ** 2).sum(-1).argmin(-1)

    def save(self, path):
        np.savez(path, center=self.center, scale=self.scale, centers=self.centers)

    @classmethod
    def load(cls, path):
        obj = cls()
        for k, v in dict(np.load(path)).items():
            setattr(obj, k, v)
        return obj


class GroupObjective:
    """Exponentiated-gradient dual ascent on whole-recording group means."""
    def __init__(self, counts, robust: bool, eta: float = .01):
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 1 or len(counts) < 2 or (counts <= 0).any() or not np.isfinite(counts).all() or not math.isfinite(eta) or eta < 0:
            raise ValueError("All groups need positive counts and a valid ascent rate")
        self.robust, self.eta = robust, eta
        self.log_weights = np.log(np.ones(len(counts)) / len(counts) if robust else counts / counts.sum())

    @property
    def weights(self):
        q = np.exp(self.log_weights - self.log_weights.max())
        return q / q.sum()

    def update(self, losses):
        losses = np.asarray(losses, dtype=float)
        if losses.shape != self.log_weights.shape or not np.isfinite(losses).all():
            raise ValueError("Every group must have a finite loss")
        if self.robust:
            self.log_weights += self.eta * (losses - losses.mean())
            self.log_weights -= np.logaddexp.reduce(self.log_weights)
        return self.weights.copy()


def group_plan(labels, steps: int, per_group: int, seed: int):
    """Uniform within each group; both arms use the same records and batch order.

    Group prevalence weights recover the original empirical recording mixture in
    the mean-loss arm. DRO changes only those weights during optimization.
    """
    if steps <= 0 or per_group <= 0:
        raise ValueError("Positive optimization steps and group batch sizes are required")
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(np.asarray(labels) == g) for g in range(max(labels) + 1)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("Empty acoustic group")
    for _ in range(steps):
        yield [int(i) for p in pools for i in rng.choice(p, per_group, replace=len(p) < per_group)]


def install_adapters(model, rank: int, seed: int):
    import torch
    from torch import nn
    if rank <= 0:
        raise ValueError("Adapter rank must be positive")
    for p in model.parameters():
        p.requires_grad_(False)

    class LowRank(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.down = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
            self.up = nn.Parameter(torch.zeros(base.out_features, rank))
            self.enabled = True

        def forward(self, x):
            out = self.base(x)
            return out + (x @ self.down.t()) @ self.up.t() / rank if self.enabled else out

    torch.manual_seed(seed)
    names = []
    # Leave the frontend and phone projection frozen; encoder Linear corrections
    # have the same receptive fields and streaming state as their base layers.
    for name, base in list(model.encoder.named_modules()):
        if type(base) is nn.Linear:
            parent_name, _, leaf = name.rpartition(".")
            parent = model.encoder.get_submodule(parent_name) if parent_name else model.encoder
            setattr(parent, leaf, LowRank(base))
            names.append(name)
    if not names:
        raise ValueError("No encoder Linear layers found")
    return names


@contextmanager
def base_only(model):
    adapters = [m for m in model.modules() if hasattr(m, "enabled") and hasattr(m, "base")]
    prior = [m.enabled for m in adapters]
    try:
        for m in adapters:
            m.enabled = False
        yield
    finally:
        for m, value in zip(adapters, prior):
            m.enabled = value


def merge_adapters(model):
    import torch
    for name, m in list(model.encoder.named_modules()):
        if hasattr(m, "base") and hasattr(m, "down") and hasattr(m, "up"):
            with torch.no_grad():
                m.base.weight.add_(m.up @ m.down / m.down.shape[0])
            parent_name, _, leaf = name.rpartition(".")
            parent = model.encoder.get_submodule(parent_name) if parent_name else model.encoder
            setattr(parent, leaf, m.base)
    return model


def read_wave(record):
    import soundfile as sf
    if record.get("audio_cache"):
        w, sr = sf.read(record["audio_cache"], dtype="float32")
    elif record["source"] == "original":
        data = original_column(record["path"], "flac")[record["index"]]
        w, sr = sf.read(io.BytesIO(data), dtype="float32")
    else:
        w, sr = sf.read(record["path"], dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        from scipy.signal import resample_poly
        d = math.gcd(sr, 16000)
        w = resample_poly(w, 16000 // d, sr // d).astype(np.float32)
    return w


def inspect(out: Path, max_hours: float, groups: int):
    if max_hours < 0:
        raise ValueError("Training hours must be nonnegative")
    import pyarrow.parquet as pq
    from student import exclusions, spliced_target
    root = Path(__file__).resolve().parents[1]
    excluded = exclusions((root / "docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    files = sorted(Path("/student/s1").glob("*.parquet"))
    if not files:
        raise ValueError("Authorized original training recordings are missing")
    candidates, excluded_clusters = [], set()
    for p in files:
        table = pq.read_table(p, columns=["shard", "row", "bucket", "split", "cluster", "ref", "hyp", "n_samples"]).to_pylist()
        for i, row in enumerate(table):
            speaker = f"tlog/{row['cluster']}"
            if (row["shard"], row["row"]) in excluded:
                excluded_clusters.add(speaker)
            if row["bucket"] == "clean-match" and row["split"] == "train" and row["n_samples"] <= 30 * 16000:
                candidates.append(dict(source="original", path=str(p), index=i, key=f"{row['shard']}:{row['row']}",
                                       speaker=speaker, n_samples=row["n_samples"], target=spliced_target(row["ref"], row["hyp"])))
    candidates = [r for r in candidates if r["speaker"] not in excluded_clusters]
    rng = np.random.default_rng(0)
    rng.shuffle(candidates)
    if not candidates:
        raise ValueError("No eligible original training recordings after speaker exclusions")
    if max_hours:
        cumulative = np.cumsum([r["n_samples"] for r in candidates]) / 16000
        candidates = candidates[:int(np.searchsorted(cumulative, max_hours * 3600)) + 1]
    candidates.sort(key=lambda r: (r["path"], r["index"]))
    rows = []
    cache = out / "original-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for r in candidates:
        # Extract only selected originals once; random group batches must not
        # reread entire parquet audio columns for every example at every step.
        key = hashlib.sha256(r["key"].encode()).hexdigest()
        audio_path, teacher_path = cache / f"{key}.flac", cache / f"{key}.npz"
        audio_path.write_bytes(original_column(r["path"], "flac")[r["index"]])
        np.savez(teacher_path,
                 ids=np.frombuffer(original_column(r["path"], "top5_ids")[r["index"]], np.uint8).reshape(-1, 5),
                 probs=np.frombuffer(original_column(r["path"], "top5_p")[r["index"]], np.float16).reshape(-1, 5))
        r["audio_cache"], r["teacher_cache"] = str(audio_path), str(teacher_path)
        w = read_wave(r)
        rows.append({**r, "descriptor": descriptors(w).tolist(), "seconds": len(w) / 16000})
    help_root = Path("/lab/data/help-train")
    label_path = help_root / "labels.jsonl"
    if not label_path.exists():
        raise ValueError("Authorized help-train labels are missing")
    labels = [json.loads(l) for l in label_path.read_text().splitlines() if l.strip()]
    help_heldout = {r["speaker"] for r in labels if r.get("split") != "train"}
    ref = json.loads((root / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in ref["ayahs"]}
    from per_breakdown import collapse
    for r in labels:
        if not clean_help(r) or r["speaker"] in help_heldout:
            continue
        path = (help_root / r["wav"]).resolve()
        if not path.is_relative_to(help_root.resolve()):
            raise ValueError("Help audio must be inside the authorized training directory")
        record = dict(source="help", path=str(path), key=r["id"], speaker=f"help/{r['speaker']}",
                      target=collapse("".join(ayahs[a] for a in r["ayahs"])))
        w = read_wave(record)
        # Whole recordings are never trimmed, even when they include long pauses.
        rows.append({**record, "descriptor": descriptors(w).tolist(), "seconds": len(w) / 16000})
    if not any(r["source"] == "help" for r in rows):
        raise ValueError("No eligible none-labelled help-train recordings")
    for r in rows:
        if not r["target"]:
            raise ValueError("An eligible recording has no sequence transcript")
        r["partition"] = partition(r["speaker"])
    train = [r for r in rows if r["partition"] == "train"]
    clustering = AcousticGroups().fit(np.array([r["descriptor"] for r in train]), groups)
    for r, g in zip(rows, clustering.predict(np.array([r["descriptor"] for r in rows]))):
        r["group"] = int(g)
    for split in ("train", "validation"):
        selected = [r for r in rows if r["partition"] == split]
        if set(r["group"] for r in selected) != set(range(groups)):
            raise ValueError(f"{split} does not cover every acoustic group")
        if not any(r["source"] == "help" for r in selected):
            raise ValueError(f"{split} has no eligible help recordings")
    out.mkdir(parents=True, exist_ok=True)
    clustering.save(out / "groups.npz")
    (out / "recordings.json").write_text(json.dumps(rows, ensure_ascii=False))
    report = {s: {"recordings": sum(r["partition"] == s for r in rows),
                  "speakers": len({r["speaker"] for r in rows if r["partition"] == s}),
                  "groups": [sum(r["partition"] == s and r["group"] == g for r in rows) for g in range(groups)]}
              for s in ("train", "validation")}
    (out / "coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def recording_losses(log_probs, parent_probs, out_lens, targets, target_lens, teacher=None, kd_weight=2.0, components=False):
    """Length-normalized sequence loss plus KD and preservation, including blank frames."""
    import torch
    import torch.nn.functional as F
    lengths = out_lens.to(log_probs.device)
    valid = torch.arange(log_probs.shape[1], device=log_probs.device)[None] < lengths[:, None]
    ctc = F.ctc_loss(log_probs.transpose(0, 1), targets, out_lens, target_lens,
                     reduction="none", zero_infinity=False) / lengths
    kl = (parent_probs * (parent_probs.clamp_min(1e-10).log() - log_probs)).sum(-1)
    if teacher is None:
        kd = -(parent_probs * log_probs).sum(-1)
    else:
        ids, probs, n_teacher = teacher
        t = torch.arange(log_probs.shape[1], device=log_probs.device)[None]
        idx = torch.minimum(t.expand(len(lengths), -1), n_teacher[:, None] - 1)
        ids = ids.gather(1, idx[..., None].expand(-1, -1, 5))
        probs = probs.gather(1, idx[..., None].expand(-1, -1, 5))
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
        kd = -(probs * log_probs.gather(-1, ids)).sum(-1)
    kd = (kd * valid).sum(1) / lengths
    kl = (kl * valid).sum(1) / lengths
    return {"ctc": ctc, "kd": kd, "kl": kl} if components else ctc + kd_weight * kd + kl


def training_batch(records, step, augmentation, cfg, device):
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from student import phoneme_ids, mix_noise
    from student_train import reverb, pick, spec_augment
    features, targets, teachers = [], [], []
    for r in records:
        # The seed depends on the item and optimization step, never on the arm.
        seed = int(hashlib.sha256(f"{r['key']}/{step}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        wave = read_wave(r)
        if rng.random() < cfg["rir_p"]:
            wave = reverb(wave, pick(augmentation["rir"], rng))
        if rng.random() < cfg["noise_p"]:
            wave = mix_noise(wave, pick(augmentation["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
                num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
        if cfg["spec_aug"]:
            f = spec_augment(f, rng)
        features.append(f)
        targets.append(torch.tensor(phoneme_ids(r["target"]), dtype=torch.long))
        if r["source"] == "original":
            with np.load(r["teacher_cache"]) as cached:
                teachers.append((cached["ids"].copy(), cached["probs"].astype(np.float32)))
        else:
            teachers.append(None)
    batch = dict(feats=pad_sequence(features, batch_first=True, padding_value=-23.0259).to(device),
                 feat_lens=torch.tensor([len(f) for f in features], device=device),
                 targets=torch.cat(targets).to(device), target_lens=torch.tensor([len(t) for t in targets], device=device),
                 teachers=teachers)
    return batch


from functools import lru_cache


@lru_cache(maxsize=24)
def original_column(path, column):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[column]).column(0).to_pylist()


def batch_loss(model, b):
    import torch
    # eval() fixes dropout and inference chunk masks in both parent and adapter;
    # gradients still flow through the adapters. Audio augmentation is unchanged.
    with torch.no_grad(), base_only(model):
        parent, lengths = model(b["feats"], b["feat_lens"])
        parent = parent.float().exp()
    lp, out_lens = model(b["feats"], b["feat_lens"])
    if not torch.equal(lengths, out_lens):
        raise RuntimeError("Adapters changed the output clock")
    lp = lp.float()
    out = {k: [] for k in ("ctc", "kd", "kl")}
    offset = 0
    for j, top in enumerate(b["teachers"]):
        n = int(b["target_lens"][j])
        teacher = None
        if top is not None:
            ids, probs = top
            teacher = (torch.tensor(ids, device=lp.device).long()[None],
                       torch.tensor(probs, device=lp.device)[None],
                       torch.tensor([len(ids)], device=lp.device))
        loss = recording_losses(lp[j:j+1], parent[j:j+1], out_lens[j:j+1],
                                b["targets"][offset:offset+n], b["target_lens"][j:j+1], teacher, components=True)
        for key, value in loss.items():
            out[key].append(value[0])
        offset += n
    return {key: torch.stack(value) for key, value in out.items()}


def fit(data: Path, parent_path: Path, out: Path, steps: int, rank: int, per_group: int, eval_every: int):
    if steps <= 0 or eval_every <= 0:
        raise ValueError("Optimization and validation intervals must be positive")
    import torch
    from student_train import build_model
    torch.set_num_threads(8)
    rows = json.loads((data / "recordings.json").read_text())
    train = [r for r in rows if r["partition"] == "train"]
    val = [r for r in rows if r["partition"] == "validation"]
    if {r["speaker"] for r in train} & {r["speaker"] for r in val}:
        raise ValueError("Speaker partition leak")
    ck = torch.load(parent_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    # Missing shipped augmentation is an error rather than an ablation.
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    labels = [r["group"] for r in train]
    groups = max(labels) + 1
    counts = np.bincount(labels, minlength=groups)
    plan = list(group_plan(labels, steps, per_group, 0))
    # Fixed training-speaker validation subset chosen without model or text scores.
    val_indices = []
    for g in range(groups):
        pool = [i for i, r in enumerate(val) if r["group"] == g]
        pool.sort(key=lambda i: hashlib.sha256(val[i]["key"].encode()).digest())
        val_indices.extend(pool[:32])
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.json").write_text(json.dumps(dict(parent=str(parent_path),
        parent_sha256=file_digest(parent_path),
        data_sha256=hashlib.sha256((data / "recordings.json").read_bytes()).hexdigest(),
        steps=steps, rank=rank, per_group=per_group, lr=1e-4, eta=.01,
        selection="minimum maximum validation-group recording loss", cfg=cfg), indent=2))
    for arm in ("mean", "dro"):
        torch.manual_seed(0)
        model = build_model(cfg["size"])
        model.load_state_dict(ck["model"], strict=True)
        names = install_adapters(model, rank, 0)
        model = model.cuda().eval()
        model.streaming_eval(16, 128)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        objective = GroupObjective(counts, robust=arm == "dro")
        best, best_step = math.inf, None
        log = (out / f"{arm}.jsonl").open("w")
        for step, indices in enumerate(plan, 1):
            b = training_batch([train[i] for i in indices], step, aug, cfg, "cuda")
            loss = batch_loss(model, b)
            group_means = loss["ctc"].reshape(groups, per_group).mean(1)
            # Dual ascent sees detached current losses; model descent sees fixed q.
            q = objective.update(group_means.detach().cpu().numpy())
            q_tensor = torch.tensor(q, device=group_means.device, dtype=group_means.dtype)
            prevalence = torch.tensor(counts / counts.sum(), device=group_means.device, dtype=group_means.dtype)
            # KD and parent preservation retain the empirical source mixture;
            # only normalized CTC sequence risk is adversarially reweighted.
            preservation = (2 * loss["kd"] + loss["kl"]).reshape(groups, per_group).mean(1)
            total = (q_tensor * group_means).sum() + (prevalence * preservation).sum()
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite whole-recording loss; checkpoint is invalid")
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            opt.step()
            row = dict(step=step, group_loss=group_means.detach().cpu().tolist(), weights=q.tolist())
            if step % eval_every == 0 or step == steps:
                sums, n = np.zeros(groups), np.zeros(groups)
                with torch.no_grad():
                    for i in val_indices:
                        # Identical augmentation draws for every checkpoint and arm.
                        loss_v = float(batch_loss(model, training_batch([val[i]], -1, aug, cfg, "cuda"))["ctc"][0])
                        sums[val[i]["group"]] += loss_v
                        n[val[i]["group"]] += 1
                if (n == 0).any() or not np.isfinite(sums).all():
                    raise RuntimeError("Invalid validation-group coverage or losses")
                score = float((sums / n).max())
                row["validation"] = (sums / n).tolist()
                if score < best:
                    best, best_step = score, step
                    torch.save(dict(adapters={k: v.cpu() for k, v in model.state_dict().items()
                                               if k.endswith((".up", ".down"))},
                                    step=step, score=score, weights=q.tolist(), layers=names), out / f"{arm}-best.pt")
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 20 == 0:
                print(json.dumps(dict(arm=arm, **row)), flush=True)
        log.close()
        selected = torch.load(out / f"{arm}-best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["adapters"], strict=False)
        model = merge_adapters(model.cpu())
        # A normal student checkpoint loads without adapter classes or custom states.
        export_dir = out / arm
        export_dir.mkdir(exist_ok=True)
        torch.save(dict(model=model.state_dict(), cfg=cfg,
                        state={**ck["state"], "step": best_step}, validation=best,
                        objective=arm), export_dir / "ckpt.pt")
        del model, opt
        torch.cuda.empty_cache()


def export_bundle(ckpt: Path, out: Path, auxiliary: Path):
    """Existing SDK graph and dynamic-int8 recipe; no supplied manifest is edited."""
    import shutil
    import torch
    from student_train import build_model
    from student_export import export_onnx, manifest, onnx_deps
    source = json.loads((auxiliary / "manifest.json").read_text())
    required = ("teacher.onnx", "teacher.onnx.data")
    for name in required:
        if not (auxiliary / name).is_file():
            raise ValueError(f"Missing declared shipped auxiliary asset: {name}")
    onnx_deps()
    from onnxruntime.quantization import quantize_dynamic, QuantType
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(saved["cfg"]["size"])
    model.load_state_dict(saved["model"], strict=True)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "float.onnx"
    states = export_onnx(model, str(fp), 16, 128)
    path = out / "model.onnx"
    quantize_dynamic(str(fp), str(path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], per_channel=True)
    m = manifest(out.name, states, 16, 128, str(path), saved)
    m["format"] = "onnx-int8"
    for key in ("inventory", "tokens", "blank", "sampleRate", "frameMs", "window", "shift", "states"):
        if m[key] != source[key]:
            raise RuntimeError(f"Export changed the shipped streaming interface: {key}")
    # Require a real ONNX session with the entire streaming interface before declaring it.
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in states}
    feeds["x"] = np.zeros((1, m["window"], 80), dtype=np.float32)
    outputs = session.run(None, feeds)
    if outputs[0].shape != (1, 8, len(m["tokens"]) + 1) or not np.isfinite(outputs[0]).all():
        raise RuntimeError("Export has invalid streaming posteriors")
    for name in required:
        shutil.copyfile(auxiliary / name, out / name)
        if file_digest(auxiliary / name) != file_digest(out / name):
            raise RuntimeError(f"Copied auxiliary asset differs: {name}")
    fp.unlink()
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(json.dumps(dict(export=str(out), sha256=m["sha256"], bytes=m["bytes"])), flush=True)


def file_digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=20)
    p.add_argument("--groups", type=int, default=6)
    p = sub.add_parser("train")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=200)
    p = sub.add_parser("export")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--auxiliary", type=Path, required=True, help="Installed shipped bundle whose teacher assets must be preserved")
    args = vars(ap.parse_args())
    command = args.pop("command")
    if command == "inspect":
        inspect(**args)
    elif command == "train":
        args["parent_path"] = args.pop("parent")
        fit(**args)
    else:
        export_bundle(**args)


if __name__ == "__main__":
    main()
