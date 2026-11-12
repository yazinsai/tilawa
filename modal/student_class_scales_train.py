"""Matched class-local/tensor-grouped phone-head refits on authorized audio.

Preparation enrolls performed transcripts and help-train training speakers' none
recordings, excluding all held-out speakers before reading audio. Fit sees only
the common frozen cache. All outputs belong under this experiment's scratch.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, mix_noise
from student_class_scales import admit_recordings, speaker_partition, matched_refit
from student_class_scales_export import digest, load_source, expose, session, stream, insertion_assets, graft, head_parameters

SEED, EPOCHS = 1729, 12
MAX_SECONDS = 24 * 60 * 60
OPTIMIZER_SHA256 = "7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Training audio leaves its enrolled corpus")
    return path


def recordings(help_labels, transcribed, excluded, reference):
    labels = [json.loads(s) for s in Path(help_labels).read_text().splitlines() if s.strip()]
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    additional = [json.loads(s) for s in Path(transcribed).read_text().splitlines() if s.strip()]
    # All splits participate in exclusion discovery before any enrollment.
    excluded |= {r.get("speaker") for r in additional if r.get("split") != "train"}
    rows = []
    for r in labels:
        if (r.get("split") != "train" or r.get("speaker") in excluded or
                r.get("kind") != "none" or r.get("scored") is not True or
                any(r.get(k) != [] for k in ("events", "non_errors", "unsupported")) or
                r.get("exclude") or r.get("excluded")):
            continue
        if not r.get("speaker") or not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks audio, transcript or training-speaker identity")
        path = contained(Path(help_labels).parent, r["wav"])
        rows.append(dict(r, source="help-none", wav=path, audioSha256=digest(path),
                         phones="".join(ayahs[a] for a in r["ayahs"])))
    for r in additional:
        if r.get("speaker") in excluded or r.get("split") != "train":
            continue
        if (not r.get("speaker") or r.get("authorized") is not True or
                r.get("verified") is not True or not r.get("phones") or not r.get("wav")):
            raise ValueError("Additional recording is not an authorized performed transcript")
        path = contained(Path(transcribed).parent, r["wav"])
        if not r.get("sha256") or digest(path) != r["sha256"]:
            raise ValueError("Enrolled audio identity differs from its performed transcript")
        rows.append(dict(r, source="transcribed", wav=path, audioSha256=digest(path)))
    rows = admit_recordings(rows, excluded)
    for r in rows:
        phoneme_ids(r["phones"])
    return rows, sorted(s for s in excluded if s)


def augmented(wave, banks, seed):
    if banks is None:
        return wave.copy()
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    def pick(bank):
        i = int(rng.integers(len(bank["offsets"]) - 1))
        return bank["data"][bank["offsets"][i]:bank["offsets"][i + 1]].astype("float32")
    if rng.random() < .2:
        rir = pick(banks["rir"])
        altered = fftconvolve(wave, rir[int(np.argmax(np.abs(rir))):])[:len(wave)]
        wave = (altered * (np.sqrt(np.mean(wave ** 2)) /
                          max(1e-8, np.sqrt(np.mean(altered ** 2))))).astype("float32")
    if rng.random() < .5:
        wave = mix_noise(wave, pick(banks["noise"]), rng.uniform(5., 25.), rng)
    return (wave * 10 ** (rng.uniform(-10., 6.) / 20)).astype("float32")


def fbank(wave, spec_seed=None):
    import torchaudio
    f = torchaudio.compliance.kaldi.fbank(torch.tensor(wave)[None], num_mel_bins=80,
        sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
    if spec_seed is not None:
        # Use the shipped masks without importing the Modal app or changing it.
        rng = np.random.default_rng(spec_seed)
        for _ in range(2):
            width = int(rng.integers(0, 28))
            start = int(rng.integers(0, 80 - width + 1))
            f[:, start:start + width] = 0
        for _ in range(int(rng.integers(0, 11))):
            width = int(rng.integers(0, max(1, int(len(f) * .05)) + 1))
            start = int(rng.integers(0, max(1, len(f) - width)))
            f[start:start + width] = 0
    return f.numpy()


def teacher_rows(runtime, wave):
    """The installed SDK teacher's exact frontend; no teacher phone relabels."""
    import torchaudio
    f = torchaudio.compliance.kaldi.fbank(torch.tensor(wave * 32768)[None],
        num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=8000)
    if len(f) < 2:
        raise ValueError("Insufficient teacher feature frames")
    f = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=True) + 1e-7)
    odd = len(f) % 2
    if odd:
        f = torch.cat([f, torch.ones_like(f[:1])])
    mask = np.ones((1, len(f) // 2), dtype=np.int64)
    if odd:
        mask[0, -1] = 0
    lp = runtime.run(["log_probs"], {"input_features": f.numpy().reshape(1, -1, 160),
                                     "attention_mask": mask})[0][0]
    if lp.ndim != 2 or lp.shape[1] != len(TOKENS) + 1 or not np.isfinite(lp).all():
        raise ValueError("Teacher alphabet or full-distribution evidence differs")
    return lp


def prepare(args):
    import soundfile as sf
    import torchaudio
    torch.set_num_threads(1)
    out, source = Path(args.out), Path(args.model).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation directory must be empty")
    deadline = time.time() + MAX_SECONDS
    m, native, _, _ = load_source(source)
    head_parameters(native)  # Reject unsupported native projections before caching audio.
    if m["tokens"] != TOKENS:
        raise ValueError("Ordered alphabet differs from the installed listener")
    auxiliary = insertion_assets(source)
    cfg = json.loads(Path(args.config).read_text())
    for field in ("base_lr", "lr_batches", "lr_epochs", "ctc_weight", "kd_weight", "kd_delay"):
        if field not in cfg:
            raise ValueError("Need explicit source training optimizer/objective configuration")
    excluded = json.loads(Path(args.exclusions).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Need an enrolled excluded-speaker list")
    ref = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    rows, excluded = recordings(args.help_train, args.transcribed, excluded, ref)
    groups = speaker_partition([r["speaker"] for r in rows])
    for group in ("fit", "selection"):
        if {r["source"] for r in rows if groups[r["speaker"]] == group} != {"help-none", "transcribed"}:
            raise ValueError("Both training-only partitions need both enrolled corpora")
    banks = {k: dict(np.load(Path(args.augmentation) / (k + ".npz"), allow_pickle=False))
             for k in ("noise", "rir")}
    for b in banks.values():
        if (b["data"].ndim != 1 or not np.isfinite(b["data"]).all() or
                not np.issubdtype(b["offsets"].dtype, np.integer) or
                b["offsets"].ndim != 1 or len(b["offsets"]) < 2 or
                b["offsets"][0] != 0 or b["offsets"][-1] != len(b["data"]) or
                np.any(np.diff(b["offsets"]) <= 0)):
            raise ValueError("Invalid shipped augmentation bank")
    m, feature, width, path = expose(source, out)
    original, frozen = session(source / m["file"]), session(path)
    teacher = session(source / "teacher.onnx") if cfg["kd_weight"] else None
    feeds = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in m["states"]}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(SEED)
    for _ in range(8):
        feeds["x"] = rng.normal(-10., 2., (1, m["window"], 80)).astype("float32")
        a, b = original.run(names, feeds), frozen.run(names, feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg="Frozen extraction changed " + name)
        feeds.update({s["name"]: b[names.index("new_" + s["name"])] for s in m["states"]})
    cached, edge_rows = [], 0
    for i, r in enumerate(rows):
        if time.time() >= deadline:
            raise TimeoutError("Bounded preparation deadline exceeded")
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or non-finite enrolled audio")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        # ClipDataset augments student inputs but reuses the original tlog
        # targets. Keep that supervision contract in this cached head refit.
        teacher_lp = teacher_rows(teacher, wave.numpy()) if teacher else None
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=r["audioSha256"],
                   transcriptSha256=hashlib.sha256(r["phones"].encode()).hexdigest(), views=[])
        for v in range(3):
            seed = SEED + i * 3 + v
            audio = augmented(wave.numpy(), banks if v else None, seed)
            features = fbank(audio, seed if v else None)
            lp, x = stream(frozen, m, features, feature)
            t = teacher_lp if teacher else lp
            edge_rows += abs(len(t) - len(lp))
            target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
            if not len(target) or len(lp) < len(target) + int((target[1:] == target[:-1]).sum()):
                raise ValueError("Complete enrolled transcript cannot align")
            file = f"recording-{i:05d}-view-{v}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, target=target,
                                teacher=t, fbank=features)
            row["views"].append(file)
        row["viewHashes"] = [digest(out / file) for file in row["views"]]
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, source=str(source), sourceSha256=m["sha256"],
        tokens=TOKENS, width=width, featureOutput=feature, rows=cached, groups=groups,
        excludedSpeakers=excluded, auxiliaryHashes=auxiliary, unpairedEdgeRows=edge_rows,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks},
        config=cfg, deadline=deadline))


def load_prepared(path):
    info = json.loads((path / "prepared.json").read_text())
    m, _, _, width = load_source(info["source"])
    if (m["sha256"] != info["sourceSha256"] or width != info["width"] or
            info["tokens"] != TOKENS or insertion_assets(info["source"]) != info["auxiliaryHashes"]):
        raise ValueError("Installed listener/teacher changed since preparation")
    groups = speaker_partition([r["speaker"] for r in info["rows"]])
    if groups != info["groups"] or set(groups) & set(info["excludedSpeakers"]):
        raise ValueError("Speaker provenance or selection partition changed")
    rows = []
    for r in info["rows"]:
        if len(r["views"]) != 3 or len(r["viewHashes"]) != 3:
            raise ValueError("Missing matched recording views")
        views = []
        for file, checksum in zip(r["views"], r["viewHashes"]):
            file = contained(path, file)
            if digest(file) != checksum:
                raise ValueError("Cached training evidence changed")
            with np.load(file, allow_pickle=False) as d:
                views.append({k: torch.tensor(d[k]) for k in ("features", "shipped", "target", "teacher")})
            v = views[-1]
            if v["features"].shape != (len(v["shipped"]), width):
                raise ValueError("Cached encoder feature clock differs")
            if any(not torch.isfinite(v[k]).all() for k in v):
                raise ValueError("Non-finite cached training evidence")
        if any(not torch.equal(v["target"], views[0]["target"]) for v in views[1:]):
            raise ValueError("Augmentation changed the transcript")
        rows.append(dict(r, cacheFiles=r["views"], views=views))
    return info, rows


def fit(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    if time.time() >= info["deadline"]:
        raise TimeoutError("Bounded preparation/refit deadline exceeded")
    _, graph, _, _ = load_source(info["source"])
    w, b = head_parameters(graph)
    initial = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        initial.weight.copy_(torch.tensor(w))
        initial.bias.copy_(torch.tensor(b))
    import importlib.util
    if digest(args.optimizer) != OPTIMIZER_SHA256:
        raise ValueError("Optimizer differs from the trainer's pinned icefall source")
    spec = importlib.util.spec_from_file_location("class_scales_pinned_optim", args.optimizer)
    optim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optim)
    cfg = info["config"]
    heads, report = matched_refit(initial, rows,
        lambda params: optim.ScaledAdam(params, lr=cfg["base_lr"], clipping_scale=2.),
        lambda optimizer: optim.Eden(optimizer, cfg["lr_batches"], cfg["lr_epochs"]),
        epochs=EPOCHS, seed=SEED, ctc_weight=cfg["ctc_weight"], kd_weight=cfg["kd_weight"],
        kd_delay=cfg["kd_delay"], deadline=info["deadline"],
        emit=lambda r: print(json.dumps(r), flush=True))
    report.update(seed=SEED, epochs=EPOCHS, optimizerSha256=OPTIMIZER_SHA256,
        config=cfg, dropout=.1,
        objective="equal-recording-mean-views-shipped-frame-normalized-CTC-plus-top-five-KD",
        head="independent-class-weight-row-and-scalar-bias-units-in-one-clipping-group",
        groups=info["groups"], excludedSpeakers=info["excludedSpeakers"],
        preparedSha256=digest(prepared / "prepared.json"), arms={})
    out.mkdir(parents=True, exist_ok=True)
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    for arm, head in heads.items():
        torch.save(dict(state=head.state_dict(), epoch=report["selectedEpoch"],
                        sourceSha256=info["sourceSha256"]), out / (arm + ".pt"))
        model_path = graft(info["source"], head, out / arm, arm)
        runtime = session(model_path, optimize=True)
        manifest = json.loads((out / arm / "manifest.json").read_text())
        worst = 0.
        for r in selection:
            for file, v in zip(r["cacheFiles"], r["views"]):
                with np.load(contained(prepared, file), allow_pickle=False) as data:
                    got, _ = stream(runtime, manifest, data["fbank"])
                with torch.no_grad():
                    want = head(v["features"]).log_softmax(-1).numpy()
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(sha256=digest(model_path), parityMaxProbabilityError=worst,
            parityViews=3 * len(selection), selectedUpdates=report["selectedUpdates"],
            gradientEvaluations=report["gradientEvaluations"][arm], auxiliaryHashes=insertion_assets(out / arm))
    # This report exists only after both complete bundles pass native inference.
    write_json(out / "training-report.json", report)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help-train", "transcribed", "exclusions", "augmentation", "config", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "optimizer", "out"):
        p.add_argument("--" + name, required=True)
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    (prepare if args.command == "prepare" else fit)(args)
