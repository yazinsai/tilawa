"""Assistant -> compact student and compute-matched direct distillation.

Run directly in the authorized train sandbox (no Modal client calls):
  python modal/student_bridge_train.py prepare --run /scratch/<experiment>/bridge \
    --initial /student/prior-runs/spliced-clean/ckpt.pt \
    --base-bundle /lab/data/models/spliced-clean-int8
  python modal/student_bridge_train.py assistant --run ... --gpu-seconds 1800
  python modal/student_bridge_train.py targets --run ...
  python modal/student_bridge_train.py bridge --run ... --gpu-seconds 1800
  python modal/student_bridge_train.py direct --run ...
  python modal/student_bridge_train.py export --run ...

Training commands resume atomically and stop before the sandbox wall limit. Repeat
until each stage reports complete. Student arms share the same initial weights,
corpus, deterministic augmentation stream, losses and geometry. The direct arm
gets the bridge's *total measured device time*, including assistant training,
selection and target generation, rounded up by at most one optimizer update.
There are no practice sessions, labels or checkpoint-selection metrics here.
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "modal")]
from student_bridge_data import (assistant_config, budget_remaining, confined_file,
                                 draw_seed, help_none, release_speakers, require_matched_compute, speaker_partition)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.with_suffix(path.suffix + ".tmp").write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(path.with_suffix(path.suffix + ".tmp"), path)


def atomic_torch(path, obj):
    import torch
    path = Path(path)
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def emit(run, **event):
    event["time"] = time.time()
    print(json.dumps(event, ensure_ascii=False), flush=True)
    with (run / "log.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def auxiliary_assets(base):
    """Use the SDK's actual declarations, including external tensors, unchanged."""
    registry = (ROOT / "packages/sdk/src/registry.ts").read_text()
    assets = re.findall(r'file: "([^"]+)", bytes: (\d+),\s*sha256: "([0-9a-f]+)"', registry)
    if len(assets) < 2:
        raise ValueError("Cannot read complete SDK insertion-teacher declarations")
    for name, size, sha in assets:
        p = confined_file(base, name)
        if p.stat().st_size != int(size) or digest(p) != sha:
            raise ValueError(f"Insertion teacher differs from SDK declaration: {name}")
    return [name for name, _, _ in assets]


class OriginalTeacher:
    """The unchanged exported Muaalem teacher; help-none targets are shared once."""
    def __init__(self, base):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(base / "teacher.onnx"), options,
                                           providers=["CPUExecutionProvider"])

    def __call__(self, wave):
        import numpy as np
        import torch
        import torchaudio
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave * 32768).unsqueeze(0),
                    num_mel_bins=80, sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=8000)
        if len(feats) < 2:
            raise ValueError("Teacher input too short")
        feats = (feats - feats.mean(0)) / (feats.var(0, unbiased=True) + 1e-7).sqrt()
        odd = len(feats) % 2
        if odd:
            feats = torch.cat([feats, torch.ones(1, 80)])
        mask = np.ones((1, len(feats) // 2), dtype=np.int64)
        if odd:
            mask[0, -1] = 0
        lp = self.session.run(["log_probs"], {"input_features": feats.reshape(1, -1, 160).numpy(),
                                              "attention_mask": mask})[0][0]
        if lp.shape[1] != 43:
            raise ValueError("Original teacher inventory changed")
        p = torch.from_numpy(lp).exp()
        probs, ids = p.topk(5, -1)
        return ids.numpy().astype(np.uint8), probs.numpy().astype(np.float16)


def prepare(args):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from student import TOKENS, exclusions
    import student_train as T

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "recipe.json").exists():
        raise ValueError("Prepared recipe is immutable; resume its stages instead")
    initial, base = Path(args.initial), Path(args.base_bundle)
    ck = torch.load(initial, map_location="cpu", weights_only=False)
    if ck["cfg"]["size"] != "full" or ck["cfg"].get("model_config"):
        raise ValueError("This experiment requires the shipped full student checkpoint")
    m = json.loads((base / "manifest.json").read_text())
    if (m["tokens"], m["frameMs"], m["chunk"], m["leftContext"]) != (TOKENS, 40, 16, 128):
        raise ValueError("Base bundle differs from shipped inventory or geometry")
    if digest(base / m["file"]) != m["sha256"]:
        raise ValueError("Base listener checksum differs from manifest")
    if m.get("trainStep") != ck["state"]["step"]:
        raise ValueError("Starting checkpoint step differs from shipped bundle provenance")
    auxiliary_assets(base)
    gold = exclusions((ROOT / "docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    records, files = [], sorted(Path(args.student_data, "s1").glob("*.parquet"))
    if not files:
        raise ValueError("No authorized S1 training data")
    for file in files:
        meta = pq.read_table(file, columns=["shard", "row", "cluster", "bucket", "split", "n_samples"]).to_pylist()
        for index, r in enumerate(meta):
            speaker = r["cluster"]
            if speaker is None or not str(speaker).strip():
                raise ValueError(f"Unknown S1 speaker in {file.name}:{index}")
            records.append(dict(key=f"s1:{r['shard']}:{r['row']}", speaker=f"s1:{speaker}", split=r["split"],
                 gold=(r["shard"], r["row"]) in gold,
                 supervised=r["bucket"] == "clean-match" and 0 < r["n_samples"] <= args.max_clip_s * 16000,
                 samples=r["n_samples"], source=str(file), index=index))
    # Include all metadata when deriving denials, not just the chosen bucket.
    manifest = Path(args.student_data, "manifest-s1.jsonl")
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            if r.get("cluster") is None:
                raise ValueError("Unknown speaker in S1 source manifest")
            records.append(dict(key=f"metadata:{r['shard']}:{r['row']}", speaker=f"s1:{r['cluster']}",
                                split=r["split"], gold=(r["shard"], r["row"]) in gold, supervised=False))
    help_root = Path(args.help_data)
    labels = [json.loads(s) for s in (help_root / "labels.jsonl").read_text().splitlines()]
    for label in labels:
        speaker = label.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("Unknown help-train speaker")
        supervised = help_none(label)
        if supervised:
            info = sf.info(confined_file(help_root, label["wav"]))
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError("Help-none audio must be mono 16kHz")
            supervised = 0 < info.frames <= args.max_clip_s * 16000
        records.append(dict(key=f"help:{label['id']}", speaker=f"help:{speaker}", split=label.get("split", "unknown"),
                            supervised=supervised, label=label))
    excluded = release_speakers(json.loads((ROOT / "docs/benchmarks/gold-per-v1.json").read_text()))
    train, val = speaker_partition(records, excluded, args.seed)
    # Cap material only after deriving complete speaker denials/partitions. Always retain help-none takes.
    def cap(rows):
        help_rows = [r for r in rows if r["key"].startswith("help:")]
        s1 = sorted((r for r in rows if r["key"].startswith("s1:")), key=lambda r: draw_seed(args.seed, r["key"], 0))
        chosen, samples = [], 0
        for r in s1:
            if samples + r["samples"] <= args.max_hours * 3600 * 16000:
                chosen.append(r)
                samples += r["samples"]
        return sorted(help_rows + chosen, key=lambda r: r["key"])
    train, val = cap(train), cap(val)
    if (not train or not val or not any(r["key"].startswith("help:") for r in train)
            or not any(r["key"].startswith("s1:") for r in train)):
        raise ValueError("Training needs authorized Quran and help-none audio plus a selection partition")
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in ref["ayahs"]}
    teacher = None
    loaded = {}
    corpus = []
    for part, rows in (("train", train), ("selection", val)):
        for r in rows:
            if r["key"].startswith("s1:"):
                if r["source"] not in loaded:
                    loaded = {r["source"]: pq.read_table(r["source"], columns=["flac", "ref", "top5_ids", "top5_p"]).to_pylist()}
                row = loaded[r["source"]][r["index"]]
                audio, transcript = row["flac"], row["ref"]
                ids = np.frombuffer(row["top5_ids"], np.uint8).reshape(-1, 5).copy()
                probs = np.frombuffer(row["top5_p"], np.float16).reshape(-1, 5).copy()
            else:
                label = r["label"]
                audio = confined_file(help_root, label["wav"]).read_bytes()
                transcript = "".join(ayahs[a] for a in label["ayahs"])
                wave, rate = sf.read(io.BytesIO(audio), dtype="float32")
                if rate != 16000 or wave.ndim != 1 or len(wave) > args.max_clip_s * 16000:
                    raise ValueError(f"Help-none audio must be mono 16kHz and within clip limit: {r['key']}")
                teacher = teacher or OriginalTeacher(base)
                ids, probs = teacher(wave)
            wave, rate = sf.read(io.BytesIO(audio), dtype="float32")
            if (rate != 16000 or wave.ndim != 1 or not transcript or not len(ids) or ids.max() >= 43
                    or ids.shape != probs.shape or not np.isfinite(probs).all()
                    or (probs < 0).any() or (probs.sum(-1) <= 0).any()):
                raise ValueError(f"Invalid authorized transcript/targets/audio: {r['key']}")
            corpus.append(dict(key=r["key"], speaker=r["speaker"], partition=part, audio=audio, ref=transcript,
                               n_samples=len(wave), ids=ids, probs=probs))
    for bank in ("noise", "rir"):
        if not Path(args.student_data, "aug", f"{bank}.npz").is_file():
            raise ValueError(f"Missing shipped augmentation bank: {bank}")
    # Persist the starting weights and material so subsequent runs cannot silently change data.
    shutil.copyfile(initial, run / "initial.pt")
    atomic_torch(run / "corpus.pt", corpus)
    config = assistant_config(T.SIZES["full"])
    recipe = dict(schema=1, seed=args.seed, initialSha256=digest(run / "initial.pt"), corpusSha256=digest(run / "corpus.pt"),
         baseBundle=str(base), studentData=args.student_data, assistantConfig=config,
         losses=dict(ctc=1.0, kd=1.0, preservation=0.5), batchSeconds=args.batch_seconds, lr=args.lr,
         augmentation={k: T.DEFAULTS[k] for k in ("rir_p", "noise_p", "snr_db", "spec_aug")},
         augmentationSha256={k: digest(Path(args.student_data, "aug", f"{k}.npz")) for k in ("noise", "rir")})
    atomic_json(run / "recipe.json", recipe)
    atomic_json(run / "partition.json", [{k: r[k] for k in ("key", "speaker", "partition")} for r in corpus])
    emit(run, event="prepared", train=sum(r["partition"] == "train" for r in corpus),
         selection=sum(r["partition"] == "selection" for r in corpus), recipe=recipe)


def load_material(run):
    import torch
    recipe = json.loads((run / "recipe.json").read_text())
    if digest(run / "initial.pt") != recipe["initialSha256"] or digest(run / "corpus.pt") != recipe["corpusSha256"]:
        raise ValueError("Prepared material changed")
    for k, sha in recipe["augmentationSha256"].items():
        if digest(Path(recipe["studentData"], "aug", f"{k}.npz")) != sha:
            raise ValueError("Augmentation bank changed")
    corpus = torch.load(run / "corpus.pt", weights_only=False)
    return recipe, corpus


def features(record, recipe, visit, aug=None):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from student import mix_noise
    import student_train as T
    wave, _ = sf.read(io.BytesIO(record["audio"]), dtype="float32")
    rng = np.random.default_rng(draw_seed(recipe["seed"], record["key"], visit))
    cfg = recipe["augmentation"]
    if aug is not None:
        if rng.random() < cfg["rir_p"]:
            wave = T.reverb(wave, T.pick(aug["rir"], rng))
        if rng.random() < cfg["noise_p"]:
            wave = mix_noise(wave, T.pick(aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
    f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
                 num_mel_bins=80, sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=-400)
    return T.spec_augment(f, rng) if aug is not None and cfg["spec_aug"] else f


def batch(rows, recipe, visit, aug=None, targets=None):
    import torch
    from student import phoneme_ids, ctc_collapse
    from torch.nn.utils.rnn import pad_sequence
    fs = [features(r, recipe, visit, aug) for r in rows]
    soft = [targets[r["key"]] if targets is not None else r for r in rows]
    # Reference vowel/gemination runs use the same run-collapse convention as shipped spliced targets.
    transcripts = [ctc_collapse(phoneme_ids(r["ref"])) for r in rows]
    return dict(feats=pad_sequence(fs, batch_first=True, padding_value=-23.0259).cuda(),
                lens=torch.tensor([len(f) for f in fs], device="cuda"),
                transcript=torch.tensor([p for t in transcripts for p in t], device="cuda"),
                transcript_lens=torch.tensor([len(t) for t in transcripts], device="cuda"),
                ids=pad_sequence([torch.tensor(r["ids"].astype("int64")) for r in soft], batch_first=True).cuda(),
                probs=pad_sequence([torch.tensor(r["probs"].astype("float32")) for r in soft], batch_first=True).cuda(),
                teacher_lens=torch.tensor([len(r["ids"]) for r in soft], device="cuda"))


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    model.streaming_eval(16, 128)
    return model


def objective(model, anchor, b, weights):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, lens = model(b["feats"], b["lens"])
        with torch.no_grad():
            old, old_lens = anchor(b["feats"], b["lens"])
    if not torch.equal(lens, old_lens):
        raise ValueError("Assistant/student posterior clock changed")
    return distillation_loss(lp.float(), lens, old.float(), b, weights)


def distillation_loss(lp, lens, old, b, weights):
    """Mask padding for both supervisors and the ordinary-reading anchor."""
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss
    if (b["teacher_lens"] <= 0).any():
        raise ValueError("Empty supervisory posterior sequence")
    lp, old = lp.float(), old.detach().float()
    valid = torch.arange(lp.shape[1], device=lp.device)[None] < lens[:, None]
    idx = torch.minimum(torch.arange(lp.shape[1], device=lp.device)[None], (b["teacher_lens"] - 1)[:, None])
    ids = b["ids"].gather(1, idx[..., None].expand(-1, -1, 5))
    probs = b["probs"].gather(1, idx[..., None].expand(-1, -1, 5))
    ctc = F.ctc_loss(lp.transpose(0, 1), b["transcript"], lens, b["transcript_lens"],
                     blank=0, reduction="sum", zero_infinity=False) / lens.sum()
    kd = soft_target_loss(lp, ids, probs, valid)
    preserve = F.kl_div(lp, old.exp(), reduction="none").sum(-1)[valid].mean()
    return weights["ctc"] * ctc + weights["kd"] * kd + weights["preservation"] * preserve


def gpu_time(fn):
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end) / 1000


def select(model, corpus, recipe):
    """Greedy PER only on held-out *training* speakers; never on release/practice."""
    import torch
    import student_train as T
    from student import ctc_collapse, phoneme_ids, ids_phonemes, edit_distance
    model.eval()
    model.streaming_eval(16, 128)
    edits, phones = 0, 0
    with torch.no_grad():
        for r in corpus:
            if r["partition"] != "selection":
                continue
            f = features(r, recipe, 0).cuda()
            lp, lens = model(f[None], torch.tensor([len(f)], device="cuda"))
            hyp = ids_phonemes(ctc_collapse(lp[0, :lens[0]].argmax(-1).tolist()))
            ref = ids_phonemes(ctc_collapse(phoneme_ids(r["ref"])))
            edits += edit_distance(hyp, ref)
            phones += len(ref)
    model.train()
    model.training_mode()
    return edits / phones


def make_model(initial, recipe, assistant=False):
    import student_train as T
    model = T.build_model("full", recipe["assistantConfig"] if assistant else None)
    if assistant:
        # Transfer geometry-compatible weights; widened FF projections start fresh.
        compatible = {k: v for k, v in initial["model"].items() if model.state_dict()[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    else:
        model.load_state_dict(initial["model"], strict=True)
    return model


def train_stage(args):
    import numpy as np
    import torch
    import student_train as T
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import Eden, ScaledAdam

    run, stage = Path(args.run), args.command
    recipe, corpus = load_material(run)
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    target_file = run / "assistant-targets.pt"
    targets = None
    if stage == "bridge":
        cert = json.loads((run / "targets.json").read_text())
        if (not cert["complete"] or digest(target_file) != cert["sha256"]
                or digest(run / "assistant-best.pt") != cert["assistantSha256"]):
            raise ValueError("Frozen assistant targets changed")
        targets = torch.load(target_file, weights_only=False)
    if stage == "direct":
        bridge = torch.load(run / "bridge.pt", weights_only=False)
        assistant = torch.load(run / "assistant.pt", weights_only=False)
        if not bridge["complete"] or not assistant["complete"]:
            raise ValueError("Complete bridge arm before allocating the matched direct budget")
        cert = json.loads((run / "targets.json").read_text())
        if not cert["complete"]:
            raise ValueError("Assistant targets are incomplete")
        goal = bridge["gpuSeconds"] + assistant["gpuSeconds"] + cert["gpuSeconds"]
    else:
        goal = args.gpu_seconds
    if not budget_remaining(0, goal):
        raise ValueError("Empty training budget")
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    model = make_model(initial, recipe, stage == "assistant").cuda()
    anchor = freeze(make_model(initial, recipe).cuda())
    params = sum(p.numel() for p in model.parameters())
    if stage == "assistant" and not sum(p.numel() for p in anchor.parameters()) < params < 600_000_000:
        raise ValueError("Assistant capacity must lie strictly between student and original teacher")
    optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=recipe["lr"], include_names=True),
                           lr=recipe["lr"], clipping_scale=2.0)
    scheduler = Eden(optimizer, 7500, 3.5)
    path = run / f"{stage}.pt"
    state = dict(epoch=0, batch=0, step=0, gpuSeconds=0.0, bestPer=None, lastUpdate=0.0, selectionSeconds=0.0, complete=False,
                 goal=goal, params=params, cfg=dict(size="full", model_config=recipe["assistantConfig"] if stage == "assistant" else None))
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["goal"] != goal:
            raise ValueError("Cannot change a stage's compute budget after starting")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        torch.set_rng_state(state["torchRng"])
        torch.cuda.set_rng_state(state["cudaRng"])
        random.setstate(state["pythonRng"])
    if state["complete"]:
        emit(run, event="already-complete", stage=stage, gpuSeconds=state["gpuSeconds"])
        return
    aug = {k: dict(np.load(Path(recipe["studentData"], "aug", f"{k}.npz"))) for k in ("noise", "rir")}
    rows = [r for r in corpus if r["partition"] == "train"]
    started = time.monotonic()

    def save():
        state.update(model=model.state_dict(), opt=optimizer.state_dict(), sched=scheduler.state_dict(),
                     torchRng=torch.get_rng_state(), cudaRng=torch.cuda.get_rng_state(), pythonRng=random.getstate())
        atomic_torch(path, state)

    def selection():
        metric, seconds = gpu_time(lambda: select(model, corpus, recipe))
        state["gpuSeconds"] += seconds
        state["selectionSeconds"] = seconds
        if state["gpuSeconds"] >= goal:
            raise ValueError("Selection exhausted the stage budget; no matched experiment can be exported")
        if state["bestPer"] is None or metric < state["bestPer"]:
            state["bestPer"] = metric
            atomic_torch(run / f"{stage}-best.pt", dict(model=model.state_dict(), cfg=state["cfg"],
                    state=dict(step=state["step"]), selectionPer=metric, recipeSha256=digest(run / "recipe.json")))
        emit(run, event="training-speaker-selection", stage=stage, per=metric, step=state["step"], gpuSeconds=state["gpuSeconds"])

    model.train()
    model.training_mode()
    if state["bestPer"] is None:
        selection()
        save()
    while budget_remaining(state["gpuSeconds"], goal) and time.monotonic() - started < args.wall_seconds:
        plan = T.batches([r["n_samples"] for r in rows], recipe["batchSeconds"], recipe["seed"] + state["epoch"])
        scheduler.step_epoch(state["epoch"])
        if state["batch"] >= len(plan):
            state["epoch"], state["batch"] = state["epoch"] + 1, 0
            continue
        b = batch([rows[i] for i in plan[state["batch"]]], recipe, state["epoch"], aug, targets)
        for m in model.modules():
            m.batch_count = initial.get("state", {}).get("step", 0) + state["step"] * recipe["batchSeconds"] / 600

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, anchor, b, recipe["losses"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; stop without silently dropping a recording")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite gradient; training budget is invalid")
            optimizer.step()
            scheduler.step_batch(state["step"])
            return float(loss.detach())

        loss, seconds = gpu_time(update)
        state["gpuSeconds"] += seconds
        state["lastUpdate"] = seconds
        state["step"], state["batch"] = state["step"] + 1, state["batch"] + 1
        # Select on fixed update intervals; the direct arm can use all its extra compute.
        # Reserve enough budget that a selection pass cannot cause final overshoot.
        if (state["step"] % 100 == 0
                and goal - state["gpuSeconds"] > 2 * state["selectionSeconds"]):
            selection()
        if state["step"] % 20 == 0:
            emit(run, event="train", stage=stage, step=state["step"], loss=loss, gpuSeconds=state["gpuSeconds"], goal=goal)
            save()
    state["complete"] = not budget_remaining(state["gpuSeconds"], goal)
    save()
    emit(run, event="stage-end", stage=stage, complete=state["complete"], step=state["step"], gpuSeconds=state["gpuSeconds"], goal=goal)


def cache_targets(args):
    import numpy as np
    import torch
    run = Path(args.run)
    recipe, corpus = load_material(run)
    terminal = torch.load(run / "assistant.pt", map_location="cpu", weights_only=False)
    if not terminal["complete"]:
        raise ValueError("Assistant training is incomplete")
    best_path = run / "assistant-best.pt"
    sha = digest(best_path)
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    model = freeze(make_model(initial, recipe, True).cuda())
    model.load_state_dict(ck["model"])
    path = run / "assistant-targets.pt"
    meta_path = run / "targets.json"
    cache, seconds = {}, 0.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["assistantSha256"] != sha or digest(path) != meta["sha256"]:
            raise ValueError("Assistant target cache identity changed")
        cache, seconds = torch.load(path, weights_only=False), meta["gpuSeconds"]
    started = time.monotonic()
    with torch.no_grad():
        for r in corpus:
            if r["partition"] != "train" or r["key"] in cache:
                continue
            if time.monotonic() - started > args.wall_seconds:
                break
            f = features(r, recipe, 0).cuda()
            def generate():
                lp, lens = model(f[None], torch.tensor([len(f)], device="cuda"))
                probs, ids = lp[0, :lens[0]].float().exp().topk(5, -1)
                return dict(ids=ids.cpu().numpy().astype(np.uint8), probs=probs.cpu().numpy().astype(np.float16))
            cache[r["key"]], elapsed = gpu_time(generate)
            seconds += elapsed
    complete = all(r["key"] in cache for r in corpus if r["partition"] == "train")
    atomic_torch(path, cache)
    atomic_json(meta_path, dict(assistantSha256=sha, sha256=digest(path), gpuSeconds=seconds, complete=complete, recordings=len(cache)))
    emit(run, event="assistant-targets", complete=complete, recordings=len(cache), gpuSeconds=seconds)


def stream_torch(model, feats):
    """Parity oracle uses real recurrent state, not a whole-clip chunk mask."""
    import numpy as np
    import torch
    import student_export as E
    stream = E.wrap(model, 16, 128)
    state = stream.init_states()
    n = len(feats)
    need = ((n - 7) // 2 + 1) // 2
    windows = max(1, -(-need // 8))
    total = 45 + (windows - 1) * 32
    f = torch.cat([feats, torch.full((max(0, total - n), 80), E.LOG_EPS)])
    outputs = []
    with torch.no_grad():
        for start in range(0, total - 45 + 1, 32):
            lp, state = stream(f[None, start:start + 45], state)
            outputs.append(lp[0].numpy())
    return np.concatenate(outputs)


def verify_bundle(path, best_sha, recipe_sha):
    manifest = json.loads((path / "manifest.json").read_text())
    validation = json.loads((path / "validation.json").read_text())
    if (digest(path / manifest["file"]) != manifest["sha256"]
            or validation["bestCheckpointSha256"] != best_sha or validation["recipeSha256"] != recipe_sha):
        raise ValueError("Previously exported bundle changed or belongs to a different training run")
    auxiliary_assets(path)
    return validation


def export(args):
    import numpy as np
    import torch
    import student_export as E
    from quantize_onnx import int8
    from student import ctc_collapse, edit_distance
    run = Path(args.run)
    recipe, corpus = load_material(run)
    bridge, direct, assistant = [torch.load(run / f"{s}.pt", map_location="cpu", weights_only=False) for s in ("bridge", "direct", "assistant")]
    cert = json.loads((run / "targets.json").read_text())
    if not all(c["complete"] for c in (bridge, direct, assistant)) or not cert["complete"]:
        raise ValueError("Both matched training arms and assistant targets must be complete")
    matched = require_matched_compute(dict(assistant=assistant["gpuSeconds"], assistant_targets=cert["gpuSeconds"], student=bridge["gpuSeconds"]),
                                       dict(student=direct["gpuSeconds"], last_update=direct["lastUpdate"]))
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    base = Path(recipe["baseBundle"])
    assets = auxiliary_assets(base)
    base_manifest = json.loads((base / "manifest.json").read_text())
    reports = {}
    for stage in ("bridge", "direct"):
        best_path = run / f"{stage}-best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model = make_model(initial, recipe)
        model.load_state_dict(best["model"], strict=True)
        out = run / f"{stage}-int8"
        if out.exists():
            validated = verify_bundle(out, digest(best_path), digest(run / "recipe.json"))
            if validated["matchedCompute"] != matched:
                raise ValueError("Exported control's compute accounting changed")
            reports[stage] = dict(path=str(out), manifestSha256=digest(out / "manifest.json"), validation=validated["recordings"])
            continue
        pending = run / f"{stage}-int8.pending"
        pending.mkdir(exist_ok=True)
        fp = pending / "model-fp32.onnx"
        cls, original_forward = model.encoder.__class__, model.encoder.__class__.forward
        try:
            states = E.export_onnx(model, str(fp), 16, 128)
        finally:
            cls.forward = original_forward
        int8(fp, pending / "model.onnx")
        manifest = E.manifest(f"assistant-bridge-{stage}", states, 16, 128, str(pending / "model.onnx"), best)
        manifest.update(format="onnx-int8", quantizedFrom=digest(fp))
        for key in ("tokens", "blank", "sampleRate", "frameMs", "fbank", "window", "shift", "chunk", "leftContext", "states"):
            if manifest[key] != base_manifest[key]:
                raise ValueError(f"Compact export changed shipped interface: {key}")
        rows = []
        # This is export validation on training-only selection audio, not checkpoint selection.
        for r in [r for r in corpus if r["partition"] == "selection"][:6]:
            f = features(r, recipe, 0)
            a = E.stream_onnx(str(fp), f, states, 16)
            b = E.stream_onnx(str(pending / "model.onnx"), f, states, 16)
            recurrent = stream_torch(model, f)
            np.testing.assert_allclose(a, recurrent, rtol=1e-3, atol=1e-3)
            collapsed = ctc_collapse(a.argmax(-1))
            edits = edit_distance(collapsed, ctc_collapse(b.argmax(-1)))
            rows.append(dict(key=r["key"], frames=len(b), top1Agreement=float((a.argmax(-1) == b.argmax(-1)).mean()),
                      edits=edits, refPhones=len(collapsed), maxTorchLogProbDifference=float(np.abs(a - recurrent).max())))
            if not np.isfinite(b).all() or a.shape != b.shape or edits / max(1, len(collapsed)) > 0.02:
                raise ValueError("Invalid or materially different quantized recurrent decode")
        for name in assets:
            shutil.copyfile(base / name, pending / name)
        auxiliary_assets(pending)
        atomic_json(pending / "manifest.json", manifest)
        atomic_json(pending / "validation.json", dict(recordings=rows, matchedCompute=matched, selectionPer=best["selectionPer"],
                   bestCheckpointSha256=digest(best_path), recipeSha256=digest(run / "recipe.json")))
        fp.unlink()
        os.replace(pending, out)
        reports[stage] = dict(path=str(out), manifestSha256=digest(out / "manifest.json"), validation=rows)
    atomic_json(run / "export.json", dict(bundles=reports, matchedCompute=matched))
    emit(run, event="export-complete", bundles=reports, matchedCompute=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "assistant", "targets", "bridge", "direct", "export"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--initial")
    parser.add_argument("--base-bundle")
    parser.add_argument("--student-data", default="/student")
    parser.add_argument("--help-data", default="/lab/data/help-train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--max-clip-s", type=float, default=30)
    parser.add_argument("--batch-seconds", type=float, default=120)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--gpu-seconds", type=float, default=1800)
    parser.add_argument("--wall-seconds", type=float, default=2700)
    args = parser.parse_args()
    if args.command == "prepare" and (not args.initial or not args.base_bundle):
        parser.error("prepare requires --initial and --base-bundle")
    if args.max_hours <= 0 or args.batch_seconds <= 0 or args.lr <= 0 or args.wall_seconds <= 0:
        parser.error("Training limits must be positive")
    {"prepare": prepare, "targets": cache_targets, "export": export}.get(args.command, train_stage)(args)


if __name__ == "__main__":
    main()
