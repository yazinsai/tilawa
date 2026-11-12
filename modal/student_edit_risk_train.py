"""Matched affine-head refits, using only cached authorized training evidence.

Run through the supplied supervisor. `prepare` reads help-train none takes and
an enrolled JSONL of authorized performed transcripts (speaker, split=train,
authorized=true, verified=true, phones, wav, sha256). `fit` reads no audio or
practice labels. `limits` fixes candidate bounds before premise audits; `fit`
selects a shared update count within training speakers.
Both arms use the same encoder, head initialization, views, order and weights.
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
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance, mix_noise
from student_edit_risk import (admitted_none, speaker_groups, phone_head, head_objective,
                              candidate_union, sequence_log_probabilities, edit_risk, matched_refit)
from student_edit_risk_export import digest, load_source, expose, session, stream, insertion_assets, graft, head_parameters

SEED, EPOCHS, LR = 1729, 12, .02
CANDIDATE_LIMITS = ((8, 8), (16, 16), (32, 32))


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


def prepare(args):
    import soundfile as sf
    import torchaudio
    torch.set_num_threads(1)
    started, wall_started = time.process_time(), time.monotonic()
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
    groups = speaker_groups([r["speaker"] for r in rows], excluded)
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
    cached = []
    for i, r in enumerate(rows):
        if max(time.process_time() - started, time.monotonic() - wall_started) >= 86400:
            raise TimeoutError("One-day CPU budget exhausted during cache preparation")
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
            target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
            if not len(target) or len(lp) < len(target) + int((target[1:] == target[:-1]).sum()):
                raise ValueError("Complete enrolled transcript cannot align")
            file = f"recording-{i:05d}-view-{v}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, target=target,
                                fbank=features)
            row["views"].append(file)
        row["cacheFiles"] = list(row["views"])
        row["viewHashes"] = [digest(out / file) for file in row["views"]]
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, source=str(source), sourceSha256=m["sha256"],
        tokens=TOKENS, width=width, featureOutput=feature, rows=cached, groups=groups,
        excludedSpeakers=excluded, auxiliaryHashes=auxiliary,
        cpuSeconds=time.process_time() - started, wallSeconds=time.monotonic() - wall_started,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks}))


def load_prepared(path):
    info = json.loads((path / "prepared.json").read_text())
    m, _, _, width = load_source(info["source"])
    if (m["sha256"] != info["sourceSha256"] or width != info["width"] or
            info["tokens"] != TOKENS or insertion_assets(info["source"]) != info["auxiliaryHashes"]):
        raise ValueError("Installed listener/teacher changed since preparation")
    groups = speaker_groups([r["speaker"] for r in info["rows"]], info["excludedSpeakers"])
    if groups != info["groups"] or set(groups) & set(info["excludedSpeakers"]):
        raise ValueError("Speaker provenance or selection partition changed")
    rows = []
    for r in info["rows"]:
        if len(r["views"]) != 3 or len(r["viewHashes"]) != 3:
            raise ValueError("Missing matched recording views")
        if r["cacheFiles"] != r["views"]:
            raise ValueError("Native parity views differ from the training cache")
        views = []
        for file, checksum in zip(r["views"], r["viewHashes"]):
            file = contained(path, file)
            if digest(file) != checksum:
                raise ValueError("Cached training evidence changed")
            with np.load(file, allow_pickle=False) as d:
                views.append({k: torch.tensor(d[k]) for k in ("features", "shipped", "target")})
            v = views[-1]
            if v["features"].shape != (len(v["shipped"]), width):
                raise ValueError("Cached encoder feature clock differs")
            if any(not torch.isfinite(v[k]).all() for k in v):
                raise ValueError("Non-finite cached training evidence")
        if any(not torch.equal(v["target"], views[0]["target"]) for v in views[1:]):
            raise ValueError("Augmentation changed the transcript")
        rows.append(dict(r, views=views))
    return info, rows


def choose_limits(rows, budget_seconds=86400):
    """Smallest registered beam within 95% of the largest beam's geometric
    retained mass on selection training speakers. This uses acoustic posteriors
    alone: neither performed transcripts nor practice labels affect bounds.
    Absolute log retained mass is logged; relative coverage is not a
    claim that the bounded lattice captures 95% of the full distribution.
    """
    curves = []
    started = time.monotonic()
    with torch.no_grad():
        for beam, retain in CANDIDATE_LIMITS:
            speakers = {}
            for r in rows:
                masses = []
                for v in r["views"]:
                    if time.monotonic() - started >= budget_seconds:
                        raise TimeoutError("One-day budget exhausted during lattice selection")
                    candidates = candidate_union(v["shipped"].tolist(), v["shipped"].tolist(), beam, retain)
                    masses.append(float(sequence_log_probabilities(v["shipped"], candidates).logsumexp(0)))
                speakers.setdefault(r["speaker"], []).append(float(np.mean(masses)))
            curves.append(dict(beam=beam, retain=retain,
                logMass=float(np.mean([np.mean(v) for v in speakers.values()]))))
    if not curves or not np.isfinite([c["logMass"] for c in curves]).all():
        raise ValueError("Bounded candidates have no measurable acoustic mass")
    chosen = next(c for c in curves if c["logMass"] >= curves[-1]["logMass"] + np.log(.95))
    return chosen["beam"], chosen["retain"], curves


def limits(args):
    torch.set_num_threads(1)
    started, wall_started = time.process_time(), time.monotonic()
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists():
        raise ValueError("Candidate limits must be recorded once before premise audits")
    info, rows = load_prepared(prepared)
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    beam, retain, curves = choose_limits(selection, 86400 - max(info["cpuSeconds"], info["wallSeconds"]))
    write_json(out, dict(beam=beam, retain=retain, curves=curves,
        cpuSeconds=time.process_time() - started, wallSeconds=time.monotonic() - wall_started,
        sourceSha256=info["sourceSha256"], preparedSha256=digest(prepared / "prepared.json"),
        groups=info["groups"], excludedSpeakers=info["excludedSpeakers"],
        rule="95%-of-largest-geometric-shipped-mass-on-selection-training-speakers"))


def validate(head, rows, beam, retain):
    if not rows:
        raise ValueError("Checkpoint selection needs training-speaker recordings")
    speakers = {}
    with torch.no_grad():
        for r in rows:
            values = []
            for v in r["views"]:
                lp = head(v["features"].detach()).log_softmax(-1)
                candidates = candidate_union(v["shipped"].tolist(), lp.tolist(), beam, retain)
                loss = head_objective(head, v, "ctc")
                risk = edit_risk(lp, candidates, tuple(v["target"].tolist()))
                hyp = ctc_collapse(lp.argmax(-1).tolist())
                values.append([float(loss), edit_distance(hyp, v["target"].tolist()) / len(v["target"]),
                    float(risk["conditional"]), float(risk["retainedMass"]),
                    float(risk["otherMass"]), float(risk["lower"]), float(risk["upper"])])
            speakers.setdefault(r["speaker"], []).append(np.mean(values, axis=0))
    result = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(result[0]), per=float(result[1]), speakers=len(speakers),
        conditionalRisk=float(result[2]), retainedMass=float(result[3]), otherMass=float(result[4]),
        fullRiskLower=float(result[5]), fullRiskUpper=float(result[6]))


def fit(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, wall_started = time.process_time(), time.monotonic()
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    _, graph, _, _ = load_source(info["source"])
    initial = phone_head(*head_parameters(graph))
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    bounds = json.loads(Path(args.limits).read_text())
    if (bounds["preparedSha256"] != digest(prepared / "prepared.json")
            or bounds["sourceSha256"] != info["sourceSha256"]
            or bounds["groups"] != info["groups"]
            or bounds["excludedSpeakers"] != info["excludedSpeakers"]
            or (bounds["beam"], bounds["retain"]) not in CANDIDATE_LIMITS):
        raise ValueError("Candidate limits differ from the registered training partition")
    beam, retain, curves = bounds["beam"], bounds["retain"], bounds["curves"]
    prior_seconds = max(info["cpuSeconds"], info["wallSeconds"]) + max(bounds["cpuSeconds"], bounds["wallSeconds"])
    heads, report = matched_refit(initial, rows, info["groups"], validate,
        beam, retain, EPOCHS, LR, SEED, emit=lambda r: print(json.dumps(r), flush=True),
        cpu_seconds=86400 - prior_seconds - max(time.process_time() - started, time.monotonic() - wall_started))
    report.update(seed=SEED, epochs=EPOCHS, learningRate=LR, candidateLimits=CANDIDATE_LIMITS,
        candidateSelection=curves, candidatePolicy="95%-of-largest-geometric-shipped-mass-on-selection-speakers",
        objective="equal-recording-mean-views-transcript-CTC-per-frame-plus-shipped-KL-per-frame",
        risk="conditional-edit-distance-per-reference-phone-plus-negative-log-retained-mass-per-frame",
        omitted="explicit-complement-mass-with-full-risk-lower-upper-bounds",
        checkpointSelection="paired-speaker-PER-then-shared-CTC-preservation",
        head="existing-native-affine-weights-dequantized-in-both-arms",
        groups=info["groups"], excludedSpeakers=info["excludedSpeakers"],
        preparedSha256=digest(prepared / "prepared.json"), limitsSha256=digest(args.limits), arms={})
    out.mkdir(parents=True, exist_ok=True)
    for arm, head in heads.items():
        if prior_seconds + max(time.process_time() - started, time.monotonic() - wall_started) >= 86400:
            raise TimeoutError("One-day CPU budget exhausted before complete export")
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
                    want = head(v["features"].detach()).log_softmax(-1).numpy()
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(sha256=digest(model_path), parityMaxProbabilityError=worst,
            parityViews=3 * len(selection), selectedUpdates=report["selectedUpdates"],
            trainingUpdates=report["trainingUpdates"], auxiliaryHashes=insertion_assets(out / arm))
    # Present only when BOTH complete SDK bundles pass native inference.
    report["cpuSeconds"] = info["cpuSeconds"] + bounds["cpuSeconds"] + time.process_time() - started
    report["wallSeconds"] = info["wallSeconds"] + bounds["wallSeconds"] + time.monotonic() - wall_started
    report["budgetSeconds"] = prior_seconds + max(time.process_time() - started, time.monotonic() - wall_started)
    if report["budgetSeconds"] >= 86400:
        raise TimeoutError("One-day CPU budget exhausted during native export validation")
    write_json(out / "training-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help", "transcribed", "exclusions", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "limits", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("limits")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "limits": limits, "fit": fit}[args.command](args)
