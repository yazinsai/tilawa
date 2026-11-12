"""Bounded paired upper-block grounding refit, via the supplied supervisor.

Requires the shipped native checkpoint, authorized performed-transcript JSONL,
help-train labels, release speaker exclusions and the shipped augmentation banks.
Selection and complete bundle verification finish before any practice analysis.
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids
from student_locality import (UpperRefit, directional_response, occurrence_masks,
    occurrence_posteriors, paired_refit, probe_directions, speaker_partition)

SEED = 1729
FIT_SECONDS = 45 * 60  # Leave export time within the supervisor's 60-minute job.
RUN_SECONDS = 60 * 60


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio leaves its enrolled training corpus")
    return path


def enroll(help_rows, transcript_rows, excluded, references):
    """Exclude speakers globally before interpreting any scripted labels."""
    excluded = set(excluded) | {r.get("speaker") for r in help_rows + transcript_rows
                                if r.get("split") != "train"}
    rows = []
    for r in help_rows:
        if r.get("split") != "train" or r.get("speaker") in excluded:
            continue
        if (r.get("kind") != "none" or r.get("scored") is not True or r.get("excluded")
                or r.get("events") != [] or r.get("non_errors") != []
                or r.get("unsupported") != []):
            continue
        if not r.get("speaker") or not r.get("ayahs") or not r.get("wav"):
            raise ValueError("None take lacks speaker/audio/transcript provenance")
        rows.append(dict(speaker=r["speaker"], source="help-none", wav=r["wav"],
                         phones="".join(references[a] for a in r["ayahs"])))
    for r in transcript_rows:
        if r.get("split") != "train" or r.get("speaker") in excluded:
            continue
        if (r.get("authorized") is not True or r.get("verified") is not True
                or not r.get("speaker") or not r.get("phones") or not r.get("wav")
                or not r.get("sha256")):
            raise ValueError("Need an authorized verified performed transcript and audio hash")
        rows.append(dict(speaker=r["speaker"], source="transcribed", wav=r["wav"],
                         phones=r["phones"], sha256=r["sha256"]))
    if {r["source"] for r in rows} != {"help-none", "transcribed"}:
        raise ValueError("Both training corpora are required")
    return rows, sorted(s for s in excluded if s)


def bundle_assets(source):
    """Verify the unchanged SDK insertion teacher and every external tensor file."""
    import onnx
    source = Path(source).resolve()
    if not (source / "teacher.onnx").is_file() or not (source / "teacher.onnx.data").is_file():
        raise ValueError("Source lacks the unchanged SDK insertion-verification bundle")
    for graph_file in source.rglob("*.onnx"):
        model = onnx.load(str(graph_file), load_external_data=False)
        for tensor in onnx.external_data_helper._get_all_tensors(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            fields = {v.key: v.value for v in tensor.external_data}
            member = contained(graph_file.parent, fields["location"])
            offset, length = int(fields.get("offset", 0)), fields.get("length")
            if (not member.is_file() or offset < 0 or member.stat().st_size <= offset
                    or (length is not None and (int(length) < 0
                        or offset + int(length) > member.stat().st_size))):
                raise ValueError("Missing or truncated external acoustic asset")
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}


def runtime(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def export_bundle(model, source, destination, name):
    """Reuse the shipped streaming exporter; copy and verify auxiliary assets."""
    from student_export import export_onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source, destination = Path(source), Path(destination)
    assets = bundle_assets(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Export directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    native = copy.deepcopy(model).cpu().eval()
    cls, saved = native.encoder.__class__, native.encoder.__class__.forward
    try:
        states = export_onnx(native, str(destination / "native.onnx"), manifest["chunk"], manifest["leftContext"])
    finally:
        cls.forward = saved
    if states != manifest["states"]:
        raise ValueError("Refit changed the SDK streaming state contract")
    quantize_dynamic(str(destination / "native.onnx"), str(destination / "model.onnx"),
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    for member, checksum in assets.items():
        if member in (manifest["file"], "manifest.json", "native.onnx"):
            continue
        output = destination / member
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / member, output)
        if digest(output) != checksum:
            raise ValueError("Export changed an auxiliary asset")
    output = destination / "model.onnx"
    updated = copy.deepcopy(manifest)
    updated.update(id=manifest["id"] + "-" + name, format="onnx-int8", file="model.onnx",
                   bytes=output.stat().st_size, sha256=digest(output))
    write_json(destination / "manifest.json", updated)
    bundle_assets(destination)
    return updated


def random_stream_parity(a, b, manifest, exact=False):
    """Compare real recurrent state transitions, rather than masked full forward."""
    names = ["log_probs"] + ["new_" + s["name"] for s in manifest["states"]]
    state = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in manifest["states"]}
    other = copy.deepcopy(state)
    rng = np.random.default_rng(SEED)
    worst = 0.
    for _ in range(8):
        x = rng.normal(-10., 2., (1, manifest["window"], 80)).astype("float32")
        left, right = a.run(names, dict(x=x, **state)), b.run(names, dict(x=x, **other))
        for name, l, r in zip(names, left, right):
            if exact:
                np.testing.assert_allclose(l, r, atol=2e-5, rtol=2e-5, err_msg=name)
            elif name == "log_probs":
                np.testing.assert_allclose(np.exp(l), np.exp(r), atol=.02, rtol=.02, err_msg=name)
            if not np.isfinite(l).all() or not np.isfinite(r).all():
                raise ValueError("Nonfinite recurrent export output")
        worst = max(worst, float(np.abs(np.exp(left[0]) - np.exp(right[0])).max()))
        state = {s["name"]: v for s, v in zip(manifest["states"], left[1:])}
        other = {s["name"]: v for s, v in zip(manifest["states"], right[1:])}
    return worst


def main(args):
    import soundfile as sf
    import torchaudio
    from student_train import build_model, reverb, spec_augment
    from student import mix_noise
    from student_export import wrap

    started = time.monotonic()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Experiment output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.model) / "manifest.json").read_text())
    if (manifest["tokens"] != TOKENS or manifest["blank"] != 0 or manifest["frameMs"] != 40
            or manifest["sampleRate"] != 16000 or manifest["fbank"]["mels"] != 80
            or digest(Path(args.model) / manifest["file"]) != manifest["sha256"]):
        raise ValueError("Source primary model inventory, frontend or checksum differs")
    assets = bundle_assets(args.model)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]["size"])
    model.load_state_dict(ck["model"], strict=True)
    baseline_manifest = export_bundle(model, args.model, out / "initial", "locality-initial")
    # This comparison must pass before any enrollment: the checkpoint must
    # reproduce the installed quantized graph, including carried state values.
    random_stream_parity(runtime(Path(args.model) / manifest["file"]),
                         runtime(out / "initial/model.onnx"), manifest, exact=True)
    excluded = json.loads(Path(args.exclusions).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Need the enrolled release-held-out speaker list")
    help_rows = [json.loads(s) for s in Path(args.help).read_text().splitlines() if s.strip()]
    transcript_rows = [json.loads(s) for s in Path(args.transcribed).read_text().splitlines() if s.strip()]
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    references = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    enrolled, excluded = enroll(help_rows, transcript_rows, excluded, references)
    groups = speaker_partition([r["speaker"] for r in enrolled], excluded)
    if any({r["source"] for r in enrolled if groups[r["speaker"]] == group}
           != {"help-none", "transcribed"} for group in ("fit", "selection")):
        raise ValueError("Both training-speaker partitions need both source corpora")
    banks = {k: dict(np.load(Path(args.augmentation) / (k + ".npz"), allow_pickle=False)) for k in ("rir", "noise")}
    for bank in banks.values():
        offsets, data = bank["offsets"], bank["data"]
        if (data.ndim != 1 or not np.isfinite(data).all() or offsets.ndim != 1
                or not np.issubdtype(offsets.dtype, np.integer) or len(offsets) < 2
                or offsets[0] != 0 or offsets[-1] != len(data) or np.any(np.diff(offsets) <= 0)):
            raise ValueError("Invalid shipped augmentation bank")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    refit = UpperRefit(model.to(device), manifest["chunk"], manifest["leftContext"])
    rows, provenance, seen = [], [], set()
    for i, row in enumerate(enrolled):
        if time.monotonic() - started > FIT_SECONDS:
            raise TimeoutError("Preparation exceeded the bounded supervisor job")
        base = Path(args.help if row["source"] == "help-none" else args.transcribed).parent
        audio = contained(base, row["wav"])
        checksum = digest(audio)
        if row.get("sha256", checksum) != checksum or checksum in seen:
            raise ValueError("Audio provenance mismatch or duplicate across training partitions")
        seen.add(checksum)
        wave, sr = sf.read(audio, dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Invalid enrolled recording")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        views = []
        target = torch.tensor(phoneme_ids(row["phones"]), dtype=torch.long, device=device)
        for v in range(3):
            rng = np.random.default_rng(SEED + i * 3 + v)
            w = wave.numpy().copy()
            def pick(bank):
                k = int(rng.integers(len(bank["offsets"]) - 1))
                return bank["data"][bank["offsets"][k]:bank["offsets"][k + 1]].astype("float32")
            if v:
                if rng.random() < .2:
                    w = reverb(w, pick(banks["rir"]))
                if rng.random() < .5:
                    w = mix_noise(w, pick(banks["noise"]), rng.uniform(5., 25.), rng)
                w *= 10 ** (rng.uniform(-10., 6.) / 20.)
            features = torchaudio.compliance.kaldi.fbank(torch.tensor(w)[None],
                num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            if v:
                features = spec_augment(features, rng)
            cache = refit.cache(features.to(device))
            with torch.no_grad():
                lp = refit(cache)
                original, lens = model(features[None].to(device), torch.tensor([len(features)], device=device))
                torch.testing.assert_close(lp, original[0, :lens[0]], atol=1e-6, rtol=1e-6)
                _, gamma = occurrence_posteriors(lp, target)
                probes = []
                # Fixed, evenly distributed occurrences; no incident/word identity selection.
                occurrences = np.unique(np.linspace(0, len(target) - 1, min(4, len(target)), dtype=int))
                for j in occurrences:
                    own, local = occurrence_masks(gamma, int(j), len(cache.frames))
                    a, b = probe_directions(cache, own, local, SEED + i * 100 + v * 10 + int(j))
                    anchor = directional_response(refit, cache, target, int(j), a)
                    probes.append(dict(occurrence=int(j), own=a, remote=b, anchor=anchor))
            views.append(dict(cache=cache, target=target, shipped=lp.detach(), probes=probes,
                              fbank=features))
        rows.append(dict(speaker=row["speaker"], views=views))
        provenance.append(dict(speaker=row["speaker"], source=row["source"], audioSha256=checksum,
                               transcriptSha256=hashlib.sha256(row["phones"].encode()).hexdigest()))
        print(json.dumps(dict(prepared=i + 1, total=len(enrolled))), flush=True)
    remaining = FIT_SECONDS - (time.monotonic() - started)
    arms, report = paired_refit(refit, rows, groups, max_seconds=remaining,
                               emit=lambda r: print(json.dumps(r), flush=True))
    report.update(groups=groups, excludedSpeakers=excluded, recordings=provenance,
        checkpointSha256=digest(args.checkpoint), sourceSha256=manifest["sha256"],
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks},
        epochs=4, learningRate=1e-5, groundingWeight=.01, probeRadius=.01,
        seed=SEED, fitBudgetSeconds=FIT_SECONDS, budgetSeconds=RUN_SECONDS,
        baseline=baseline_manifest["sha256"], arms={})
    for name, arm in arms.items():
        torch.save(arm.model.state_dict(), out / (name + ".pt"))
        exported = export_bundle(arm.model, args.model, out / name, "locality-" + name)
        fp32, int8 = runtime(out / name / "native.onnx"), runtime(out / name / "model.onnx")
        quant_error = random_stream_parity(fp32, int8, exported)
        # Real training-selection views: exact native recurrent PyTorch vs ONNX.
        stream = wrap(copy.deepcopy(arm.model).cpu().eval(), manifest["chunk"], manifest["leftContext"])
        worst, count = 0., 0
        for row in rows:
            if groups[row["speaker"]] != "selection":
                continue
            for view in row["views"]:
                features = view["fbank"].numpy()
                states = stream.init_states()
                ort_states = {s["name"]: t.numpy() for s, t in zip(manifest["states"], states)}
                for start in range(0, len(features), manifest["shift"]):
                    x = np.full((1, manifest["window"], 80), manifest["fbank"]["padValue"], dtype=np.float32)
                    part = features[start:start + manifest["window"]]
                    x[0, :len(part)] = part
                    with torch.no_grad():
                        want, states = stream(torch.tensor(x), states)
                    got = fp32.run(None, dict(x=x, **ort_states))
                    np.testing.assert_allclose(got[0], want.numpy(), atol=2e-4, rtol=2e-4)
                    for spec, a, b in zip(manifest["states"], got[1:], states):
                        np.testing.assert_allclose(a, b.numpy(), atol=2e-4, rtol=2e-4)
                        ort_states[spec["name"]] = a
                    worst = max(worst, float(np.abs(np.exp(got[0]) - want.exp().numpy()).max()))
                count += 1
        report["arms"][name] = dict(sha256=exported["sha256"], parityViews=count,
            nativeStreamingMaxProbabilityError=worst, quantizationMaxProbabilityError=quant_error,
            selectedUpdates=report["selectedUpdates"], auxiliaryHashes={k: v for k, v in assets.items()
                if k not in (manifest["file"], "manifest.json", "native.onnx")})
    # Written only after BOTH complete exports pass. Never declare an incomplete asset.
    report["elapsedSeconds"] = time.monotonic() - started
    if report["elapsedSeconds"] > RUN_SECONDS:
        raise TimeoutError("Complete experiment exceeded its supervisor job budget")
    write_json(out / "training-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "help", "transcribed", "exclusions", "augmentation", "out"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
