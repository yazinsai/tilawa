"""Matched affine-head refits, using only cached authorized training evidence.

Run through the supplied supervisor. `prepare` reads help-train none takes and
an enrolled JSONL of authorized performed transcripts (speaker, split=train,
authorized=true, verified=true, phones, wav, sha256). `fit` reads no audio or
practice labels and selects a shared update count within training speakers.
Both arms use the same encoder, head initialization, views, order and weights.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance, mix_noise
from student_curvature import admitted_none, speaker_partition, phone_head, log_probs, objectives, matched_refit
from student_curvature_export import digest, load_source, expose, session, stream, insertion_assets, graft, head_parameters

SEED, EPOCHS, LR = 1729, 12, .02
DAMPING_RATIOS = (.1, 1., 10.)


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
    rows = [dict(speaker=r["speaker"], source="help-none", phones="".join(ayahs[a] for a in r["ayahs"]),
                 wav=contained(Path(help_labels).parent, r["wav"]))
            for r in admitted_none(labels, excluded)]
    additional = [json.loads(s) for s in Path(transcribed).read_text().splitlines() if s.strip()]
    # All splits participate in exclusion discovery before any enrollment.
    excluded |= {r.get("speaker") for r in additional if r.get("split") != "train"}
    rows = [r for r in rows if r["speaker"] not in excluded]
    for r in additional:
        if r.get("speaker") in excluded or r.get("split") != "train":
            continue
        if (not r.get("speaker") or r.get("authorized") is not True or
                r.get("verified") is not True or not r.get("phones") or not r.get("wav")):
            raise ValueError("Additional recording is not an authorized performed transcript")
        path = contained(Path(transcribed).parent, r["wav"])
        if not r.get("sha256") or digest(path) != r["sha256"]:
            raise ValueError("Enrolled audio identity differs from its performed transcript")
        rows.append(dict(speaker=r["speaker"], source="transcribed", phones=r["phones"], wav=path))
    if {r["source"] for r in rows} != {"help-none", "transcribed"}:
        raise ValueError("Both authorized transcripts and training-speaker none takes are required")
    seen = set()
    for r in rows:
        r["audioSha256"] = digest(r["wav"])
        if r["audioSha256"] in seen:
            raise ValueError("Repeated audio would leak across partitions or duplicate gradients")
        seen.add(r["audioSha256"])
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
    m, native, _, _ = load_source(source)
    head_parameters(native)  # Reject unsupported native projections before caching audio.
    if m["tokens"] != TOKENS:
        raise ValueError("Ordered alphabet differs from the installed listener")
    auxiliary = insertion_assets(source)
    excluded = json.loads(Path(args.exclusions).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Need an enrolled excluded-speaker list")
    ref = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    rows, excluded = recordings(args.help, args.transcribed, excluded, ref)
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
    original, frozen, teacher = session(source / m["file"]), session(path), session(source / "teacher.onnx")
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
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or non-finite enrolled audio")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=r["audioSha256"],
                   transcriptSha256=hashlib.sha256(r["phones"].encode()).hexdigest(), views=[])
        for v in range(3):
            seed = SEED + i * 3 + v
            audio = augmented(wave.numpy(), banks if v else None, seed)
            features = fbank(audio, seed if v else None)
            lp, x = stream(frozen, m, features, feature)
            t = teacher_rows(teacher, audio)
            edge_rows += abs(len(t) - len(lp))
            target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
            if not len(target) or len(lp) < len(target) + int((target[1:] == target[:-1]).sum()):
                raise ValueError("Complete enrolled transcript cannot align")
            file = f"recording-{i:05d}-view-{v}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, target=target,
                                teacher=t[:len(lp)], fbank=features)
            row["views"].append(file)
        row["viewHashes"] = [digest(out / file) for file in row["views"]]
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, source=str(source), sourceSha256=m["sha256"],
        tokens=TOKENS, width=width, featureOutput=feature, rows=cached, groups=groups,
        excludedSpeakers=excluded, auxiliaryHashes=auxiliary, unpairedEdgeRows=edge_rows,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks}))


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
        rows.append(dict(r, views=views))
    return info, rows


def validate(head, rows):
    if not rows:
        raise ValueError("Checkpoint selection needs training-speaker recordings")
    speakers = {}
    with torch.no_grad():
        for r in rows:
            values = []
            for v in r["views"]:
                lp = log_probs(head, v["features"])
                ctc, kd, preserve = objectives(lp, v["target"], v["teacher"], v["shipped"])
                hyp = ctc_collapse(lp.argmax(-1).tolist())
                values.append([float(ctc + kd + preserve), edit_distance(hyp, v["target"].tolist()) / len(v["target"])])
            speakers.setdefault(r["speaker"], []).append(np.mean(values, axis=0))
    result = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(result[0]), per=float(result[1]), speakers=len(speakers))


def fit(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    _, graph, _, _ = load_source(info["source"])
    initial = phone_head(*head_parameters(graph))
    heads, report, factors = matched_refit(initial, rows, info["groups"], validate,
        DAMPING_RATIOS, EPOCHS, LR, SEED, emit=lambda r: print(json.dumps(r), flush=True))
    report.update(seed=SEED, epochs=EPOCHS, learningRate=LR, dampingCandidates=DAMPING_RATIOS,
        objective="equal-recording-mean-views-CTC-plus-full-KD-plus-shipped-KL",
        curvature="frozen-fit-only-framewise-model-Fisher-Kronecker-mean-scale-normalized",
        head="existing-native-affine-weights-dequantized-in-both-arms",
        groups=info["groups"], excludedSpeakers=info["excludedSpeakers"],
        preparedSha256=digest(prepared / "prepared.json"), arms={})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "curvature-factors.npz", features=factors[0], outputs=factors[1])
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
                    want = log_probs(head, v["features"]).numpy()
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(sha256=digest(model_path), parityMaxProbabilityError=worst,
            parityViews=3 * len(selection), selectedUpdates=report["selectedUpdates"],
            trainingUpdates=report["trainingUpdates"], auxiliaryHashes=insertion_assets(out / arm))
    # This report exists only after BOTH complete bundles pass native inference.
    write_json(out / "training-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help", "transcribed", "exclusions", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else fit)(args)
