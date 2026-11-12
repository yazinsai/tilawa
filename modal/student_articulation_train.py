"""Matched duration-view head refits on explicitly enrolled training recordings.

prepare: authorize and speaker-split recordings, derive uncertain CTC boundaries,
resynthesize matched views, and cache the installed encoder. fit: read only that
cache and select distributions/checkpoints on ordinary training-speaker audio.
No practice audio, labels, inference rules or encoder parameters enter selection.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance, mix_noise
from student_articulation import (admitted_none, speaker_partition, safe_intervals,
    speech_islands, duration_view, global_view)
from student_head_export import (digest, load_source, expose, session, stream,
    insertion_assets, graft)

SEED, EPOCHS, LR = 1729, 12, .02
# Fixed before any practice decode; selection uses only training speakers.
DISTRIBUTIONS = {"mild": np.log(1.25), "wide": np.log(1.5)}
ARMS = ("identity", "global", "relative")
DAY = 24 * 60 * 60


def within_budget(start):
    if time.time() - start > DAY:
        raise RuntimeError("One-day preparation/refit budget exhausted")


def phone_head(width, vocab):
    head = torch.nn.Linear(width, vocab)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def log_probs(head, view):
    return (view["shipped"].detach() + head(view["features"].detach())).log_softmax(-1)


def objectives(head, view):
    lp, target = log_probs(head, view), view["target"]
    if (lp.ndim != 2 or not torch.isfinite(lp).all() or target.ndim != 1
            or target.dtype != torch.int64 or not len(target) or (target <= 0).any()
            or (target >= lp.shape[1]).any()
            or len(lp) < len(target) + int((target[1:] == target[:-1]).sum())):
        raise ValueError("Invalid complete recording transcript or phone clock")
    ctc = torch.nn.functional.ctc_loss(lp[:, None], target,
        torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=0,
        reduction="sum", zero_infinity=False) / len(target)
    q = view["shipped"].detach().log_softmax(-1)
    preserve = (q.exp() * (q - lp)).sum(-1).mean()
    return ctc, preserve


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
    started = time.time()
    out, source = Path(args.out), Path(args.model).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation directory must be empty")
    m, _, _, _ = load_source(source)
    if m["tokens"] != TOKENS:
        raise ValueError("Ordered alphabet differs from the installed listener")
    auxiliary = insertion_assets(source)
    excluded = json.loads(Path(args.exclusions).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Need an explicit excluded-speaker list")
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    rows, excluded = recordings(args.help, args.transcribed, excluded, reference)
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
        within_budget(started)
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or nonfinite training recording")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        wave = wave.numpy()
        target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
        ordinary_fbank = fbank(wave)
        lp, x = stream(frozen, m, ordinary_fbank, feature)
        intervals, z = safe_intervals(lp, r["phones"], target, len(wave))
        islands = speech_islands(lp, len(wave))
        # Do not alter any consonant core that overlaps a protected blank run.
        intervals = [s for s in intervals if any(a["start"] <= s["start"] and
                     s["end"] <= a["end"] for a in islands)]
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=r["audioSha256"],
                   transcriptSha256=hashlib.sha256(r["phones"].encode()).hexdigest(),
                   ctcLogZ=z, cores=intervals, speechIslands=islands, common=[], arms={})
        def cache(name, audio, seed=None, preset=None):
            within_budget(started)
            if preset is None:
                audio = augmented(audio, banks if seed is not None else None, seed)
                f = fbank(audio, seed)
                p, features = stream(frozen, m, f, feature)
            else:
                f, p, features = preset
            if len(p) < len(target) + int((target[1:] == target[:-1]).sum()):
                raise ValueError("Duration view has no complete transcript path")
            file = f"recording-{i:05d}-{name}.npz"
            np.savez_compressed(out / file, features=features, shipped=p, target=target, fbank=f)
            return dict(file=file, sha256=digest(out / file), samples=len(audio))
        for v in range(3):
            row["common"].append(cache(f"common-{v}", wave, SEED + i * 3 + v if v else None,
                                       (ordinary_fbank, lp, x) if not v else None))
        for distribution, log_width in DISTRIBUTIONS.items():
            for arm in ARMS:
                key = distribution + "-" + arm
                row["arms"][key] = [row["common"][0]]
                for v in (1, 2):
                    seed = SEED + i * 3 + v
                    # Duration and channel draws use separate random streams.
                    factors = np.exp(np.random.default_rng(seed + 1000003).uniform(
                        -log_width, log_width, len(intervals)))
                    if arm == "global":
                        audio, mapping = global_view(wave, islands, intervals, factors)
                    else:
                        audio, mapping = duration_view(wave, intervals,
                            factors if arm == "relative" else np.ones(len(intervals)))
                    view = cache(key + f"-{v}", audio, seed)
                    view["durationMapping"] = mapping
                    row["arms"][key].append(view)
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(rows), cores=len(intervals))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, startedAt=started, source=str(source),
        sourceSha256=m["sha256"], tokens=TOKENS, width=width, rows=cached, groups=groups,
        distributions={k: float(v) for k, v in DISTRIBUTIONS.items()}, excludedSpeakers=excluded,
        auxiliaryHashes=auxiliary,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks}))


def load_prepared(path):
    info = json.loads((path / "prepared.json").read_text())
    m, _, _, width = load_source(info["source"])
    if (m["sha256"] != info["sourceSha256"] or width != info["width"] or
            info["tokens"] != TOKENS or insertion_assets(info["source"]) != info["auxiliaryHashes"]
            or info["distributions"] != DISTRIBUTIONS):
        raise ValueError("Source bundle or frozen recipe changed since preparation")
    groups = speaker_partition([r["speaker"] for r in info["rows"]])
    if groups != info["groups"] or set(groups) & set(info["excludedSpeakers"]):
        raise ValueError("Speaker provenance or selection partition changed")
    rows = []
    for r in info["rows"]:
        loaded = {}
        def load(views):
            if len(views) != 3:
                raise ValueError("Missing matched recording views")
            result = []
            for v in views:
                if v["file"] not in loaded:
                    file = contained(path, v["file"])
                    if digest(file) != v["sha256"]:
                        raise ValueError("Cached evidence changed")
                    with np.load(file, allow_pickle=False) as d:
                        data = {k: torch.tensor(d[k]) for k in ("features", "shipped", "target")}
                    if (data["features"].shape != (len(data["shipped"]), width) or
                            data["shipped"].shape[1] != len(TOKENS) + 1 or
                            any(not torch.isfinite(value).all() for value in data.values())):
                        raise ValueError("Invalid cached feature alphabet/clock")
                    loaded[v["file"]] = (v["sha256"], data)
                if loaded[v["file"]][0] != v["sha256"]:
                    raise ValueError("Inconsistent identity for shared cached view")
                result.append(loaded[v["file"]][1])
            return result
        common = load(r["common"])
        keys = {d + "-" + a for d in DISTRIBUTIONS for a in ARMS}
        if set(r["arms"]) != keys:
            raise ValueError("Incomplete matched controls")
        arms = {k: load(v) for k, v in r["arms"].items()}
        if any(not torch.equal(v["target"], common[0]["target"])
               for v in common + [v for views in arms.values() for v in views]):
            raise ValueError("Resynthesis changed transcript order")
        rows.append(dict(r, common=common, arms=arms, commonFiles=[v["file"] for v in r["common"]]))
    return info, rows


def validate(head, rows):
    if not rows:
        raise ValueError("Checkpoint selection needs training-speaker recordings")
    speakers = {}
    with torch.no_grad():
        for r in rows:
            values = []
            # All arms/distributions are assessed on identical ordinary views.
            for v in r["common"]:
                ctc, preserve = objectives(head, v)
                hyp = ctc_collapse(log_probs(head, v).argmax(-1).tolist())
                values.append([float(ctc + preserve), edit_distance(hyp, v["target"].tolist()) / len(v["target"])])
            speakers.setdefault(r["speaker"], []).append(np.mean(values, axis=0))
    result = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(result[0]), per=float(result[1]), speakers=len(speakers))


def recording_loss(head, row, key):
    ctc = torch.stack([objectives(head, v)[0] for v in row["arms"][key]]).mean()
    # Ordinary outputs/real pauses are preserved identically in every arm.
    anchor = torch.stack([objectives(head, v)[1] for v in row["common"]]).mean()
    return ctc + anchor


def fit(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    within_budget(info["startedAt"])
    out.mkdir(parents=True, exist_ok=True)
    fit_rows = [r for r in rows if info["groups"][r["speaker"]] == "fit"]
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    if not fit_rows or not selection:
        raise ValueError("Empty speaker partition")
    report = dict(seed=SEED, epochs=EPOCHS, learningRate=LR, optimizer="SGD-no-momentum",
        preservationWeight=1., groups=info["groups"],
        distributions=info["distributions"], preparedSha256=digest(prepared / "prepared.json"), arms={})
    for distribution in DISTRIBUTIONS:
        for arm in ARMS:
            key = distribution + "-" + arm
            head = phone_head(info["width"], len(TOKENS) + 1)
            optimizer = torch.optim.SGD(head.parameters(), lr=LR)
            history = [dict(epoch=0, **validate(head, selection))]
            best, chosen, best_score = copy.deepcopy(head.state_dict()), 0, history[0]["objective"]
            for epoch in range(1, EPOCHS + 1):
                within_budget(info["startedAt"])
                order = list(range(len(fit_rows)))
                random.Random(SEED + epoch).shuffle(order)
                for i in order:
                    within_budget(info["startedAt"])
                    optimizer.zero_grad(set_to_none=True)
                    loss = recording_loss(head, fit_rows[i], key)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite recording objective")
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
                        raise ValueError("Nonfinite head gradient")
                    optimizer.step()
                metrics = dict(epoch=epoch, **validate(head, selection))
                history.append(metrics)
                if metrics["objective"] < best_score:
                    best, chosen, best_score = copy.deepcopy(head.state_dict()), epoch, metrics["objective"]
                print(json.dumps(dict(arm=key, **metrics)), flush=True)
            head.load_state_dict(best)
            head.eval()
            torch.save(dict(state=best, epoch=chosen, sourceSha256=info["sourceSha256"]), out / (key + ".pt"))
            model_path = graft(info["source"], head, out / key, key)
            runtime = session(model_path, optimize=True)
            manifest = json.loads((out / key / "manifest.json").read_text())
            worst = 0.
            for r in selection:
                within_budget(info["startedAt"])
                data = np.load(prepared / r["commonFiles"][0], allow_pickle=False)
                got, _ = stream(runtime, manifest, data["fbank"])
                with torch.no_grad():
                    want = log_probs(head, r["common"][0]).numpy()
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
            report["arms"][key] = dict(history=history, selectedEpoch=chosen, selectedObjective=best_score,
                sha256=digest(model_path), parityMaxProbabilityError=worst, parityRecordings=len(selection))
            write_json(out / "training-report.json", report)
    # Freeze the relative distribution before any full-session target decoding.
    selected = min((d + "-relative" for d in DISTRIBUTIONS),
                   key=lambda k: report["arms"][k]["selectedObjective"])
    report["selectedRelative"] = selected
    report["matchedControls"] = [selected.replace("relative", a) for a in ("identity", "global")]
    report["elapsedSeconds"] = time.time() - info["startedAt"]
    write_json(out / "training-report.json", report)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--help-labels", dest="help", required=True)
    for name in ("model", "transcribed", "exclusions", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    (prepare if args.command == "prepare" else fit)(args)
