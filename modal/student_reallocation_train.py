"""Bounded, resumable three-arm upper-block recycling experiment.

Run only in the supplied supervisor's train image (CPU):
  python modal/student_reallocation_train.py --checkpoint <prior ckpt.pt> \
    --source-model "$LAB_DATA/models/spliced-clean-int8" --out <scratch directory>

Reads only training s1 clips and help-train none takes. Practice audio is not
opened. All checkpoint selection happens before any exported target replay.
An output manifest describes a newly generated model; supplied assets are never
edited. Exported bundles retain every source auxiliary asset.
"""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, ctc_collapse, edit_distance, exclusions, phoneme_ids
from student_reallocation import (admitted_none, channel_statistics, dormant_channels,
    freeze_stem, preservation, recycle, speaker_partition, transcript_path, upper_feedforwards)

SEED = 1729
ARMS = ("continuation", "dormant", "random")


class InvocationBoundary(Exception):
    """Preparation completed a durable unit before this sandbox's deadline."""


def boundary(deadline):
    if time.monotonic() >= deadline:
        raise InvocationBoundary


def save_torch(path, value):
    import torch
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio path leaves its authorized corpus")
    return path


def bundle_identity(source):
    """Require the same insertion teacher and all ONNX external tensors."""
    import onnx
    for name in ("teacher.onnx", "teacher.onnx.data"):
        if not (source / name).is_file():
            raise ValueError("Incomplete source insertion-verification bundle: " + name)
    for graph in source.glob("*.onnx"):
        model = onnx.load(str(graph), load_external_data=False)
        for tensor in model.graph.initializer:
            for item in tensor.external_data:
                if item.key == "location" and not contained(source, item.value).is_file():
                    raise ValueError("Missing ONNX external tensor: " + item.value)
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("A source bundle may not contain symlinks")
    return {str(p.relative_to(source)): digest(p) for p in paths}


def verify_checkpoint_source(model, source, out, manifest):
    """Match all exported initializer values, not merely a checkpoint step.

    Both exports use the existing pinned exporter and MatMul-only quantization.
    A different export topology is an unresolved source identity, not permission
    to silently refit a different listener.
    """
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import export_onnx
    fp, quantized = out / "source-check-fp32.onnx", out / "source-check-int8.onnx"
    try:
        states = export_onnx(model, str(fp), manifest["chunk"], manifest["leftContext"])
        if states != manifest["states"]:
            raise ValueError("Checkpoint geometry does not match the installed listener")
        quantize_dynamic(str(fp), str(quantized), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        def tensors(path):
            graph = onnx.load(str(path))
            return {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
        expected, actual = tensors(source / manifest["file"]), tensors(quantized)
        if (set(expected) != set(actual) or any(not np.array_equal(expected[k], actual[k]) for k in expected)):
            raise ValueError("Checkpoint export does not reproduce installed listener weights")
        write_json(out / "source-parity.json", dict(initializers_equal=True, initializers=len(expected),
                    source_sha256=manifest["sha256"]))
    finally:
        fp.unlink(missing_ok=True)
        quantized.unlink(missing_ok=True)


def enroll(args):
    """Discover all excluded speaker clusters before sampling any training rows."""
    import pyarrow.parquet as pq
    import soundfile as sf

    gold = exclusions((Path(__file__).resolve().parents[1] /
                       "docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    files = sorted(Path(args.student_data, "s1").glob("*.parquet"))
    meta = []
    forbidden = set()
    for f in files:
        for i, r in enumerate(pq.read_table(f, columns=["shard", "row", "split", "bucket", "cluster", "n_samples"]).to_pylist()):
            if r.get("cluster") in (None, ""):
                raise ValueError("Training clip lacks speaker-cluster provenance")
            speaker = "quran:" + str(r["cluster"])
            if r["split"] != "train" or (r["shard"], r["row"]) in gold:
                forbidden.add(speaker)
            meta.append((f, i, r, speaker))
    allowed = [r for r in meta if r[2]["split"] == "train" and r[2]["bucket"] == "clean-match"
               and r[3] not in forbidden and 0 < r[2]["n_samples"] <= args.max_clip_s * 16000]
    rng = np.random.default_rng(SEED)
    rng.shuffle(allowed)
    rows = []
    by_file = {}
    for f, i, _, speaker in allowed[:args.recordings]:
        by_file.setdefault(f, []).append((i, speaker))
    for f, picks in by_file.items():
        t = pq.read_table(f, columns=["ref", "flac", "top5_ids", "top5_p"]).take([i for i, _ in picks]).to_pylist()
        for r, (_, speaker) in zip(t, picks):
            rows.append(dict(speaker=speaker, source="quran", audio=r["flac"], phones=r["ref"],
                top_ids=np.frombuffer(r["top5_ids"], np.uint8).reshape(-1, 5).astype(np.int64),
                top_p=np.frombuffer(r["top5_p"], np.float16).reshape(-1, 5).astype(np.float32)))
    labels = [json.loads(line) for line in Path(args.help_labels).read_text().splitlines() if line.strip()]
    clean = []
    for r in admitted_none(labels, []):
        path = contained(Path(args.help_labels).parent, r["wav"])
        info = sf.info(path)
        if 0 < info.frames <= args.max_clip_s * info.samplerate:
            clean.append(r)
    rng.shuffle(clean)
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    for r in clean[:args.recordings]:
        rows.append(dict(speaker="help:" + r["speaker"], source="help-none",
            audio=contained(Path(args.help_labels).parent, r["wav"]).read_bytes(),
            phones="".join(ayahs[a] for a in r["ayahs"])))
    if not rows:
        raise ValueError("No authorized training recordings")
    identities = [hashlib.sha256(r["audio"]).hexdigest() for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate recording identity would duplicate weights or leak partitions")
    groups = speaker_partition([r["speaker"] for r in rows], SEED)
    for group in ("fit", "selection"):
        if {r["source"] for r in rows if groups[r["speaker"]] == group} != {"quran", "help-none"}:
            raise ValueError("Both speaker partitions must include Quran and help-none recordings")
    return rows, groups, sorted(forbidden)


def model_forward(model, f):
    import torch
    lp, lens = model(f[None], torch.tensor([len(f)]))
    return lp[0, :int(lens[0])]


def cache_views(rows, baseline, args, out, deadline):
    import torch
    import torchaudio
    import soundfile as sf
    from student_train import pick, reverb, spec_augment
    from student import mix_noise

    banks = {k: dict(np.load(Path(args.student_data, "aug", k + ".npz"), allow_pickle=False))
             for k in ("rir", "noise")}
    cached = []
    for i, r in enumerate(rows):
        boundary(deadline)
        file = out / f"training-recording-{i:05d}.pt"
        identity = dict(audio=hashlib.sha256(r["audio"]).hexdigest(), phones=r["phones"], speaker=r["speaker"])
        if file.exists():
            item = torch.load(file, map_location="cpu", weights_only=False)
            if item["identity"] != identity:
                raise ValueError("Resumed training recording/transcript identity changed")
            cached.append(item["row"])
            continue
        wave, sr = sf.read(io.BytesIO(r["audio"]), dtype="float32", always_2d=True)
        wave = torch.from_numpy(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        if not 0 < len(wave) <= args.max_clip_s * 16000 or not torch.isfinite(wave).all():
            raise ValueError("Enrolled recording outside the bounded audio budget")
        # The shipped recipe collapses duration-expanded transcript runs.
        target = torch.tensor(ctc_collapse(phoneme_ids(r["phones"])), dtype=torch.long)
        views = []
        for v in range(2):
            rng = np.random.default_rng(SEED + 2 * i + v)
            audio = wave.numpy().copy()
            if v:
                if rng.random() < .2:
                    audio = reverb(audio, pick(banks["rir"], rng))
                if rng.random() < .5:
                    audio = mix_noise(audio, pick(banks["noise"], rng), rng.uniform(5., 25.), rng)
                audio *= 10 ** (rng.uniform(-10., 6.) / 20)
            f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(audio)[None], num_mel_bins=80,
                sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            if v:
                f = spec_augment(f, rng)
            with torch.no_grad():
                shipped = model_forward(baseline, f).clone()
            if not len(target) or len(shipped) < len(target) + int((target[1:] == target[:-1]).sum()):
                raise ValueError("Performed transcript cannot fit the unchanged acoustic clock")
            views.append(dict(feats=f, shipped=shipped, target=target,
                top_ids=torch.from_numpy(r["top_ids"]) if "top_ids" in r else None,
                top_p=torch.from_numpy(r["top_p"]) if "top_p" in r else None))
        item = dict(speaker=r["speaker"], source=r["source"], views=views, audio_sha256=identity["audio"])
        save_torch(file, dict(identity=identity, row=item))
        cached.append(item)
    return cached


def calibration(model, rows, out, deadline):
    """Unpadded upper hidden observations and training-consonant templates."""
    import torch

    sites = upper_feedforwards(model)
    file = out / "channel-calibration.pt"
    if file.exists():
        cached = torch.load(file, map_location="cpu", weights_only=False)
        observations, poor, completed = cached["observations"], cached["poor"], cached["completed"]
    else:
        observations, poor, completed = {k: [] for k in sites}, {k: [] for k in sites}, 0
    for i, row in enumerate(rows):
        if i < completed:
            continue
        boundary(deadline)
        for view in row["views"]:
            f = view["feats"]
            captures, handles = {}, []
            with torch.no_grad():
                _, embedded_lens = model.encoder_embed(f[None], torch.tensor([len(f)]))
            for name, (module, ds) in sites.items():
                limit = (int(embedded_lens[0]) + ds - 1) // ds
                def incoming(m, args, name=name, limit=limit):
                    captures.setdefault(name, {})["input"] = args[0][:limit, 0].detach().numpy().copy()
                def outgoing(m, args, name=name, limit=limit):
                    captures.setdefault(name, {})["pre"] = args[0][:limit, 0].detach().numpy().copy()
                handles.extend([module.in_proj.register_forward_pre_hook(incoming),
                                module.out_proj.register_forward_pre_hook(outgoing)])
            try:
                with torch.no_grad():
                    lp = model_forward(model, f).numpy()
            finally:
                for h in handles:
                    h.remove()
            aligned = transcript_path(lp, view["target"].numpy())
            # Inventory's first 28 symbols are consonants (not vowel/tajweed expansions).
            weak = np.flatnonzero((aligned > 0) & (aligned <= 28))
            weak = sorted(weak, key=lambda t: -lp[t, aligned[t]], reverse=True)[:16]
            for name, (module, ds) in sites.items():
                c = captures[name]
                observations[name].append(channel_statistics(c["pre"], module.out_proj.weight.detach().numpy()))
                for t in weak:
                    index = min(len(c["input"]) - 1, (2 * int(t)) // ds)
                    x = c["input"][index]
                    if np.linalg.norm(x) > 0:
                        poor[name].append((float(-lp[t, aligned[t]]), x.copy()))
                poor[name] = sorted(poor[name], key=lambda p: p[0], reverse=True)[:32]
        save_torch(file, dict(observations=observations, poor=poor, completed=i + 1))
    plan = {name: dormant_channels(r) for name, r in observations.items()}
    plan = {k: v for k, v in plan.items() if v}
    if not plan:
        return None, observations
    if any(not poor[k] for k in plan):
        raise ValueError("Dormant site lacks poorly represented training consonant examples")
    rng = np.random.default_rng(SEED)
    random_plan = {k: rng.choice(sites[k][0].in_proj.weight.shape[0], len(v), replace=False).tolist()
                   for k, v in plan.items()}
    return dict(dormant=plan, random=random_plan,
                examples={k: np.stack([p[1] for p in poor[k]]) for k in plan}), observations


def objective(model, view):
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss

    lp = model_forward(model, view["feats"])
    n = len(lp)
    ctc = F.ctc_loss(lp[:, None], view["target"], torch.tensor([n]),
                     torch.tensor([len(view["target"])]), reduction="sum", zero_infinity=False) / n
    # Retain the original complete distribution on speech AND real pauses.
    preserve = F.kl_div(lp, view["shipped"].exp(), reduction="sum") / n
    kd = lp.sum() * 0
    if view["top_ids"] is not None:
        index = torch.arange(n).clamp(max=len(view["top_ids"]) - 1)
        kd = soft_target_loss(lp[None], view["top_ids"][index][None], view["top_p"][index][None],
                              torch.ones((1, n), dtype=torch.bool))
    return ctc + 2 * kd + preserve


def validate(model, rows):
    import torch
    model.eval()
    scores = {}
    with torch.no_grad():
        for row in rows:
            value = []
            for view in row["views"]:
                lp = model_forward(model, view["feats"])
                hyp = ctc_collapse(lp.argmax(-1).tolist())
                value.append([float(objective(model, view)), edit_distance(hyp, view["target"].tolist()) / len(view["target"])])
            scores.setdefault(row["speaker"], []).append(np.mean(value, axis=0))
    score = np.mean([np.mean(r, axis=0) for r in scores.values()], axis=0)
    return dict(objective=float(score[0]), per=float(score[1]), speakers=len(scores))


def refit_state(model):
    """Keep trainable tensors only; three full frozen-stem copies waste CPU RAM."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_refit(model, state):
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Resumed upper-block state differs from the fixed refit geometry")
    model.load_state_dict(state, strict=False)


def stream_torch(model, f, m):
    import torch
    from student_export import wrap, PAD, LOG_EPS
    s = wrap(model, m["chunk"], m["leftContext"])
    states, outs = s.init_states(), []
    need = ((len(f) - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (m["chunk"] // 2)))
    total = m["window"] + (windows - 1) * m["shift"]
    f = torch.cat([f, torch.full((max(0, total - len(f)), 80), LOG_EPS)])
    with torch.no_grad():
        for a in range(0, total - m["window"] + 1, m["shift"]):
            lp, states = s(f[None, a:a + m["window"]], states)
            outs.append(lp[0].numpy())
    return np.concatenate(outs)


def export_bundles(models, best, rows, source, out, best_step, source_identity, deadline):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import export_onnx, manifest, stream_onnx

    if bundle_identity(source) != source_identity:
        raise ValueError("Source primary or insertion-verification assets changed during fitting")
    m = json.loads((source / "manifest.json").read_text())
    outputs = {}
    for arm in ARMS:
        boundary(deadline)
        model = models[arm]
        restore_refit(model, best[arm])
        directory = out / (arm + "-int8")
        receipt = directory / "reallocation-export.json"
        if receipt.exists():
            done = json.loads(receipt.read_text())
            if done["source_sha256"] != m["sha256"] or done["best_step"] != best_step:
                raise ValueError("Resumed export belongs to a different comparison checkpoint")
            saved = json.loads((directory / "manifest.json").read_text())
            if digest(directory / saved["file"]) != saved["sha256"]:
                raise ValueError("Completed exported model changed during resume")
            for name, checksum in source_identity.items():
                if name not in (m["file"], "manifest.json") and digest(directory / name) != checksum:
                    raise ValueError("Completed auxiliary export changed during resume")
            bundle_identity(directory)
            outputs[arm] = dict(path=str(directory), sha256=saved["sha256"])
            continue
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(source, directory)
        fp = directory / "reallocation-fp32.onnx"
        states = export_onnx(model.eval(), str(fp), m["chunk"], m["leftContext"])
        if states != m["states"]:
            raise ValueError("Export changed streaming state geometry")
        quantize_dynamic(str(fp), str(directory / "model.onnx"), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
        parity = []
        for row in rows[:4]:
            f = row["views"][0]["feats"]
            expected = stream_torch(model, f, m)
            got = stream_onnx(str(fp), f, states, m["chunk"])
            result = preservation(expected, got, tolerance=1e-5)
            if not result["passes"]:
                raise ValueError("FP32 export parity failed")
            quantized = stream_onnx(str(directory / "model.onnx"), f, states, m["chunk"])
            if quantized.shape != got.shape or not np.isfinite(quantized).all():
                raise ValueError("Incomplete quantized streaming export")
            result["int8_frame_agreement"] = float((quantized.argmax(-1) == got.argmax(-1)).mean())
            parity.append(result)
        generated = manifest("upper-reallocation-" + arm, states, m["chunk"], m["leftContext"],
                             str(directory / "model.onnx"), None)
        # Preserve inventory/support/auxiliary declarations byte for byte as values.
        updated = dict(m, **{k: generated[k] for k in ("id", "bytes", "sha256")}, format="onnx-int8", file="model.onnx",
                       trainStep=(m.get("trainStep") or 0) + best_step, quantizedFrom=digest(fp))
        write_json(directory / "manifest.json", updated)
        fp.unlink()
        write_json(directory / "reallocation-parity.json", parity)
        for name, checksum in source_identity.items():
            if name not in (m["file"], "manifest.json") and digest(directory / name) != checksum:
                raise ValueError("Export changed a retained source asset: " + name)
        bundle_identity(directory)
        write_json(receipt, dict(source_sha256=m["sha256"], best_step=best_step))
        outputs[arm] = dict(path=str(directory), sha256=updated["sha256"])
    return outputs


def run_comparison(args):
    import torch
    from student_train import build_model
    from optim import ScaledAdam
    from icefall.utils import get_parameter_groups_with_lrs

    invocation_start = time.monotonic()
    deadline = invocation_start + args.budget_seconds
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    random.seed(SEED)
    out, source = Path(args.out).resolve(), Path(args.source_model).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Scratch outputs must be separate from the supplied source bundle")
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((source / "manifest.json").read_text())
    if (m["tokens"] != TOKENS or m["frameMs"] != 40 or m["blank"] != 0
            or m["sha256"] != digest(source / m["file"])):
        raise ValueError("Source listener identity/inventory/clock differs")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if m.get("trainStep") is not None and ck["state"]["step"] != m["trainStep"]:
        raise ValueError("Checkpoint step does not match the shipped listener")
    baseline = build_model(ck["cfg"]["size"]).eval()
    baseline.load_state_dict(ck["model"], strict=True)
    # Drop the source optimizer and duplicate source-weight tensors from RAM.
    ck = dict(cfg=ck["cfg"], state=ck["state"])
    baseline.streaming_eval(m["chunk"], m["leftContext"])
    checkpoint = out / "matched-state.pt"
    identity = dict(checkpoint=digest(args.checkpoint), source=m["sha256"], assets=bundle_identity(source), updates=args.updates,
                    recordings=args.recordings, seed=SEED, max_clip_s=args.max_clip_s,
                    implementation={name: digest(Path(__file__).parent / name)
                                    for name in ("student_reallocation.py", "student_reallocation_train.py")})
    preparation = out / "preparation.json"
    if preparation.exists():
        if json.loads(preparation.read_text()) != identity:
            raise ValueError("Resumed preparation changed its source or fixed experiment plan")
    else:
        write_json(preparation, identity)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Cannot change the data, schedule or source of a resumed comparison")
        rows, groups = state["rows"], state["groups"]
    else:
        verify_checkpoint_source(baseline, source, out, m)
        enrolled, groups, forbidden = enroll(args)
        rows = cache_views(enrolled, baseline, args, out, deadline)
        del enrolled
        fit = [r for r in rows if groups[r["speaker"]] == "fit"]
        plan, observations = calibration(baseline, fit, out, deadline)
        write_json(out / "calibration.json", {k: [{q: v.tolist() if isinstance(v, np.ndarray) else v
                                                   for q, v in r.items()} for r in rs] for k, rs in observations.items()})
        del observations
        if plan is None:
            write_json(out / "result.json", dict(status="premise-absent", reason="No persistently dormant low-contribution upper channels"))
            return
        state = dict(identity=identity, rows=rows, groups=groups, forbidden=forbidden,
                     plan=plan, step=0, elapsed=0., best_score=float("inf"), best_step=None,
                     best={}, models={}, optimizers={}, history=[])
        save_torch(checkpoint, state)
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    models, optimizers = {}, {}
    parity_file = out / "initial-parity.json"
    initial = json.loads(parity_file.read_text()) if parity_file.exists() else {}
    for arm in ARMS:
        model = copy.deepcopy(baseline)
        if not state.get("initialized") and arm != "continuation":
            sites = upper_feedforwards(model)
            for name, channels in state["plan"][arm].items():
                recycle(sites[name][0], channels, state["plan"]["examples"][name])
            initial.setdefault(arm, [])
            # Full streaming state, including real pauses, in both partitions.
            parity_views = [v for row in rows for v in row["views"]]
            for i, view in enumerate(parity_views):
                if i < len(initial[arm]):
                    continue
                boundary(deadline)
                f = view["feats"]
                p = preservation(stream_torch(baseline, f, m), stream_torch(model, f, m))
                initial[arm].append(p)
                write_json(parity_file, initial)
            if arm == "dormant" and not all(p["passes"] for p in initial[arm]):
                write_json(out / "result.json", dict(status="premise-absent",
                    reason="Removing calibrated contributions does not preserve initial streaming probabilities", initial=initial))
                return
        # These weights are immutable and every arm evaluates them in the same
        # mode. Sharing them keeps the CPU experiment within the small sandbox.
        model.encoder_embed = baseline.encoder_embed
        for i in range(len(model.encoder.encoders) - 2):
            model.encoder.encoders[i] = baseline.encoder.encoders[i]
        freeze_stem(model)
        if state.get("initialized"):
            restore_refit(model, state["models"][arm])
        model.streaming_eval(m["chunk"], m["leftContext"])
        opt = ScaledAdam(get_parameter_groups_with_lrs(model, lr=.002, include_names=True), lr=.002, clipping_scale=2.)
        if state.get("initialized"):
            opt.load_state_dict(state["optimizers"][arm])
        models[arm], optimizers[arm] = model, opt
    if initial:
        write_json(out / "initial-parity.json", initial)
    started = invocation_start

    def save_state():
        nonlocal started
        state["models"] = {arm: refit_state(model) for arm, model in models.items()}
        state["optimizers"] = {arm: opt.state_dict() for arm, opt in optimizers.items()}
        state["initialized"] = True
        state["elapsed"] += time.monotonic() - started
        started = time.monotonic()
        save_torch(checkpoint, state)

    save_state()
    if time.monotonic() - invocation_start >= args.budget_seconds and state["step"] < args.updates:
        print("Preparation saved; resume with the identical command", flush=True)
        return
    for step in range(state["step"], args.updates):
        index = int(np.random.default_rng(SEED + step).integers(len(fit)))
        view = fit[index]["views"][step % 2]
        for arm in ARMS:
            # Same cached waveform/features and stochastic draws in every arm.
            torch.manual_seed(SEED + step)
            random.seed(SEED + step)
            model, opt = models[arm], optimizers[arm]
            model.train()
            model.encoder_embed.eval()
            for stack in list(model.encoder.encoders)[:-2]:
                stack.eval()
            for module in model.modules():
                module.batch_count = ck["state"]["step"] + step
            loss = objective(model, view)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite matched loss; no arm may silently skip a recording")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite matched update")
        state["step"] = step + 1
        if (step + 1) % 10 == 0 or step + 1 == args.updates:
            scores = {arm: validate(model, selection) for arm, model in models.items()}
            # One shared checkpoint count selected symmetrically across all arms.
            score = float(np.mean([s["objective"] for s in scores.values()]))
            state["history"].append(dict(step=step + 1, scores=scores))
            if score < state["best_score"]:
                state["best_score"], state["best_step"] = score, step + 1
                state["best"] = {arm: refit_state(model) for arm, model in models.items()}
            print(json.dumps(state["history"][-1]), flush=True)
        save_state()
        if state["elapsed"] >= 24 * 3600:
            raise ValueError("One-day matched compute budget exhausted")
        # Wall budget is checked at a completed three-arm update boundary.
        if time.monotonic() - invocation_start >= args.budget_seconds and step + 1 < args.updates:
            print("Matched state saved; resume with the identical command", flush=True)
            return
    if state["elapsed"] >= 24 * 3600:
        raise ValueError("One-day compute budget exhausted before export")
    outputs = export_bundles(models, state["best"], selection, source, out, state["best_step"], identity["assets"], deadline)
    state["elapsed"] += time.monotonic() - started
    if state["elapsed"] > 24 * 3600:
        raise ValueError("One-day compute budget exhausted during export")
    write_json(out / "result.json", dict(status="exported", identity=identity,
        best_step=state["best_step"], selection_history=state["history"], outputs=outputs,
        channels=state["plan"]["dormant"], groups=groups, elapsed=state["elapsed"]))


def run(args):
    """Count setup, resumed preparation, refitting AND exports in the day budget."""
    out, source = Path(args.out).resolve(), Path(args.source_model).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Scratch outputs must be separate from the supplied source bundle")
    budget = out / "compute-budget.json"
    used = json.loads(budget.read_text())["seconds"] if budget.exists() else 0.
    if used >= 24 * 3600:
        raise ValueError("One-day compute budget already exhausted")
    args.budget_seconds = min(args.budget_seconds, 24 * 3600 - used)
    started = time.monotonic()
    try:
        run_comparison(args)
    finally:
        if out.is_dir():
            write_json(budget, dict(seconds=used + time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--student-data", default="/student")
    parser.add_argument("--help-labels", default=os.environ.get("LAB_DATA", "/lab/data") + "/help-train/labels.jsonl")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--recordings", type=int, default=120)
    parser.add_argument("--max-clip-s", type=float, default=30.)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=3000.)
    args = parser.parse_args()
    if args.updates < 10 or args.recordings < 3 or args.budget_seconds <= 0 or args.threads < 1:
        parser.error("Invalid bounded comparison budget")
    try:
        run(args)
    except InvocationBoundary:
        print("Preparation saved at a recording/arm boundary; resume the identical command", flush=True)
