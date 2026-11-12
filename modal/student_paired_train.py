"""Prepare, fit and export matched paired-view phone-head refits.

Run via the authorized supervisor, with help-train labels and shipped noise/RIR
banks. Optional additional JSONL transcripts require speaker, split=train,
authorized=true, verified=true, phones, wav and audio sha256. Scripted mistakes
are never taken as performed transcripts. Preparation and fitting read no
practice sessions. Selection is confined to training speakers.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance, mix_noise
from student_paired import admitted_none, speaker_partition, phone_head, log_probs, matched_refit, paired_paths
from student_head_export import digest, load_source, expose, session, stream, insertion_assets, graft, head_parameters

SEED = 1729


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Training file leaves its enrolled corpus")
    return path


def recordings(help_labels, reference, transcribed=None, excluded=()):
    labels = [json.loads(s) for s in Path(help_labels).read_text().splitlines() if s.strip()]
    additional = ([json.loads(s) for s in Path(transcribed).read_text().splitlines() if s.strip()]
                  if transcribed else [])
    # Discover exclusions across BOTH corpora before admitting any takes.
    excluded = set(excluded) | {r.get("speaker") for r in labels + additional if r.get("split") != "train"}
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    def articulation(text):
        # Repeated reference symbols encode held duration in the shipped recipe.
        return "".join(c for i, c in enumerate(text) if not i or c != text[i - 1])
    rows = [dict(speaker=r["speaker"], source="help-none",
                 phones="".join(articulation(ayahs[a]) for a in r["ayahs"]),
                 wav=contained(Path(help_labels).parent, r["wav"])) for r in admitted_none(labels, excluded)]
    for r in additional:
        if r.get("speaker") in excluded or r.get("split") != "train":
            continue
        if (not r.get("speaker") or r.get("authorized") is not True or
                r.get("verified") is not True or not r.get("phones") or not r.get("wav")):
            raise ValueError("Need authorized verified performed-phone transcripts")
        path = contained(Path(transcribed).parent, r["wav"])
        if not r.get("sha256") or digest(path) != r["sha256"]:
            raise ValueError("Performed transcript and enrolled audio identity differ")
        rows.append(dict(speaker=r["speaker"], source="transcribed", phones=r["phones"], wav=path))
    seen = set()
    for r in rows:
        r["audioSha256"] = digest(r["wav"])
        if r["audioSha256"] in seen:
            raise ValueError("Duplicate recording could leak between partitions")
        seen.add(r["audioSha256"])
        if not phoneme_ids(r["phones"]):
            raise ValueError("Empty performed transcript")
    return rows, sorted(s for s in excluded if s)


def phone_view(wave, banks, seed):
    """Room/direct-path trim, phone band and shipped noise; exact zero delay.

    The direct-path prefix is removed before convolution; the bandpass is zero
    phase. No time stretching or sample insertion occurs. Return measured delay
    explicitly, rather than silently assuming arbitrary augmentation is aligned.
    """
    from scipy.signal import butter, fftconvolve, sosfiltfilt
    rng = np.random.default_rng(seed)
    def pick(bank):
        i = int(rng.integers(len(bank["offsets"]) - 1))
        return bank["data"][bank["offsets"][i]:bank["offsets"][i + 1]].astype("float64")
    rir = pick(banks["rir"])
    rir = rir[int(np.argmax(np.abs(rir))):]
    norm = np.sqrt(np.sum(rir ** 2))
    if norm == 0:
        raise ValueError("Silent room impulse response")
    altered = fftconvolve(wave, rir / norm)[:len(wave)]
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    altered *= rms(wave) / max(rms(altered), 1e-9)
    altered = sosfiltfilt(butter(4, [120., 7000.], btype="band", fs=16000, output="sos"), altered)
    altered = mix_noise(altered, pick(banks["noise"]), rng.uniform(5., 25.), rng)
    altered *= 10 ** (rng.uniform(-10., 6.) / 20)
    peak = float(np.max(np.abs(altered)))
    if peak > .89:
        altered *= .89 / peak
    return altered.astype("float32"), 0


def sample_times(frames, sample_rate, frame_ms, delay_samples):
    stride = sample_rate * frame_ms // 1000
    if (sample_rate * frame_ms % 1000 or frames < 1 or
            not isinstance(delay_samples, (int, np.integer)) or delay_samples < 0):
        raise ValueError("Invalid sample clock or unknown augmentation delay")
    return np.arange(frames, dtype=np.int64) * stride - delay_samples


def fbank(wave):
    import torch
    import torchaudio
    return torchaudio.compliance.kaldi.fbank(torch.tensor(wave)[None], num_mel_bins=80,
        sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()


def prepare(args):
    import soundfile as sf
    import torch
    import torchaudio
    torch.set_num_threads(1)
    out, source = Path(args.out), Path(args.model).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation directory must be empty")
    manifest, graph, _, _ = load_source(source)
    head_parameters(graph)
    if manifest["tokens"] != TOKENS:
        raise ValueError("Ordered alphabet differs from the shipped listener")
    auxiliary = insertion_assets(source)
    excluded = json.loads(Path(args.exclusions).read_text()) if args.exclusions else []
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Invalid excluded-speaker enrollment")
    ref = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    rows, excluded = recordings(args.help, ref, args.transcribed, excluded)
    groups = speaker_partition([r["speaker"] for r in rows])
    banks = {k: dict(np.load(Path(args.augmentation) / (k + ".npz"), allow_pickle=False)) for k in ("noise", "rir")}
    for b in banks.values():
        if (b["data"].ndim != 1 or not np.isfinite(b["data"]).all() or
                b["offsets"].ndim != 1 or not np.issubdtype(b["offsets"].dtype, np.integer) or
                len(b["offsets"]) < 2 or b["offsets"][0] != 0 or
                b["offsets"][-1] != len(b["data"]) or np.any(np.diff(b["offsets"]) <= 0)):
            raise ValueError("Invalid shipped noise/RIR bank")
    manifest, feature, width, path = expose(source, out)
    original, frozen = session(source / manifest["file"]), session(path)
    feeds = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in manifest["states"]}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(SEED)
    for _ in range(8):
        feeds["x"] = rng.normal(-10., 2., (1, manifest["window"], 80)).astype("float32")
        a, b = original.run(names, feeds), frozen.run(names, feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg="Feature extraction changed " + name)
        feeds.update({s["name"]: b[names.index("new_" + s["name"])] for s in manifest["states"]})
    cached = []
    for i, r in enumerate(rows):
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or non-finite enrolled audio")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        degraded, delay = phone_view(wave.numpy(), banks, SEED + i)
        if len(degraded) != len(wave):
            raise ValueError("Phone augmentation changed sample-time correspondence")
        target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=r["audioSha256"],
            transcriptSha256=hashlib.sha256(r["phones"].encode()).hexdigest(), target=target.tolist(),
            views=[], delaySamples=[0, delay])
        evidence, times = [], []
        for v, (audio, offset) in enumerate(((wave.numpy(), 0), (degraded, delay))):
            features = fbank(audio)
            lp, x = stream(frozen, manifest, features, feature)
            t = sample_times(len(lp), manifest["sampleRate"], manifest["frameMs"], offset)
            evidence.append(lp)
            times.append(t)
            file = f"recording-{i:05d}-view-{v}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, times=t, fbank=features)
            row["views"].append(file)
        paired_paths(evidence, target, times)  # Reject impossible transcripts before fitting.
        row["viewHashes"] = [digest(out / file) for file in row["views"]]
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, source=str(source), sourceSha256=manifest["sha256"],
        sourceManifestSha256=digest(source / "manifest.json"), tokens=TOKENS, width=width,
        rows=cached, groups=groups, excludedSpeakers=excluded, auxiliaryHashes=auxiliary,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks},
        correspondence="integer original-sample timestamps; direct-path RIR trimming and zero-phase filter",
        supervisedSources=sorted({r["source"] for r in rows})))


def load_prepared(path):
    info = json.loads((path / "prepared.json").read_text())
    m, _, _, width = load_source(info["source"])
    if (m["sha256"] != info["sourceSha256"] or width != info["width"] or info["tokens"] != TOKENS or
            digest(Path(info["source"]) / "manifest.json") != info["sourceManifestSha256"] or
            insertion_assets(info["source"]) != info["auxiliaryHashes"]):
        raise ValueError("Source bundle identity changed since preparation")
    groups = speaker_partition([r["speaker"] for r in info["rows"]])
    if groups != info["groups"] or set(groups) & set(info["excludedSpeakers"]):
        raise ValueError("Excluded speaker or changed selection partition")
    rows = []
    for r in info["rows"]:
        if len(r["views"]) != 2 or len(r["viewHashes"]) != 2 or len(r["delaySamples"]) != 2:
            raise ValueError("Missing paired views or sample correspondence")
        views = []
        for file, checksum, delay in zip(r["views"], r["viewHashes"], r["delaySamples"]):
            file = contained(path, file)
            if digest(file) != checksum:
                raise ValueError("Cached view identity changed")
            with np.load(file, allow_pickle=False) as d:
                view = {k: d[k].copy() for k in ("features", "shipped", "times")}
            if (view["features"].shape != (len(view["shipped"]), width) or
                    not all(np.isfinite(v).all() for v in view.values()) or
                    not np.array_equal(view["times"], sample_times(len(view["shipped"]), m["sampleRate"], m["frameMs"], delay))):
                raise ValueError("Cached view frame clock or correspondence changed")
            views.append(view)
        target = np.asarray(r["target"], dtype=np.int64)
        if hashlib.sha256("".join(TOKENS[p - 1] for p in target).encode()).hexdigest() != r["transcriptSha256"]:
            raise ValueError("Cached performed transcript identity changed")
        rows.append(dict(r, cacheFiles=r["views"], views=views, target=target))
    return info, rows


def validate(head, rows):
    speakers = {}
    for r in rows:
        errors = [edit_distance(ctc_collapse(log_probs(head, v["features"]).argmax(-1).tolist()), r["target"].tolist())
                  / len(r["target"]) for v in r["views"]]
        speakers.setdefault(r["speaker"], []).append(float(np.mean(errors)))
    return dict(per=float(np.mean([np.mean(v) for v in speakers.values()])), speakers=len(speakers))


def fit(args):
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    _, graph, _, _ = load_source(info["source"])
    initial = phone_head(*head_parameters(graph))
    heads, report = matched_refit(initial, rows, info["groups"], validate,
        emit=lambda r: print(json.dumps(r), flush=True))
    report.update(seed=SEED, preparedSha256=digest(prepared / "prepared.json"), groups=info["groups"],
        excludedSpeakers=info["excludedSpeakers"], objective="paired CTC / (2 * transcript length) + mean-view shipped KL + head L2",
        head="existing native affine head, dequantized equally in both arms; frozen int8 encoder", arms={})
    out.mkdir(parents=True, exist_ok=True)
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    for arm, head in heads.items():
        np.savez(out / (arm + ".npz"), weight=head.weight, bias=head.bias)
        model_path = graft(info["source"], head, out / arm, "paired-path-" + arm)
        runtime = session(model_path, optimize=True)
        manifest = json.loads((out / arm / "manifest.json").read_text())
        worst = 0.
        for r in selection:
            for file, v in zip(r["cacheFiles"], r["views"]):
                with np.load(contained(prepared, file), allow_pickle=False) as d:
                    got, _ = stream(runtime, manifest, d["fbank"])
                want = log_probs(head, v["features"])
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(sha256=digest(model_path), parityMaxProbabilityError=worst,
            parityViews=2 * len(selection), auxiliaryHashes=insertion_assets(out / arm))
    # No successful report or model declaration before BOTH bundles are verified.
    write_json(out / "training-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    for name in ("transcribed", "exclusions"):
        p.add_argument("--" + name)
    p = commands.add_parser("fit")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else fit)(args)
