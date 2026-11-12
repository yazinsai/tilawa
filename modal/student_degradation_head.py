"""Frozen-encoder acoustic-degradation continuation versus shuffled exposure.

Run on the authorized worker, with training-only audio and augmentation banks:
  python modal/student_degradation_head.py cache --model MODEL --help-train HELP \
      --corpus-index TRAINING_JSONL --noise NOISE_NPZ --rir RIR_NPZ --out CACHE
  python modal/student_degradation_head.py train --cache CACHE --out FIT
  python modal/student_degradation_head.py export --model MODEL --fit FIT --out BUNDLE

Five fixed waveform views interpolate from original to the existing phone
degradation. Each complete course has the same example multiset in both arms.
Only the first course's order differs; later balanced courses are shared. Select
one common checkpoint on speaker-grouped training validation, never practice.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids
from student_frozen_head import Encoder, digest, export

LEVELS = (0., .25, .5, .75, 1.)


def speaker_partition(speakers, seed=0):
    speakers = sorted(set(speakers), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if len(speakers) < 2:
        raise ValueError("need two authorized training speakers for grouped selection")
    valid = set(speakers[:min(len(speakers) - 1, max(1, math.ceil(len(speakers) / 5)))])
    return {s: "validation" if s in valid else "train" for s in speakers}


def authorized_none(row):
    return (row.get("split") == "train" and row.get("kind") == "none"
            and not any(row.get(k) for k in ("excluded", "events", "non_errors", "unsupported"))
            and bool(row.get("speaker")) and bool(row.get("ayahs")))


def training_rows(help_root, references, corpus_index=None):
    """Never promote mistake/repeat prompts or held-out speakers to transcripts."""
    help_root = Path(help_root).resolve()
    labels = [json.loads(s) for s in (help_root / "labels.jsonl").read_text().splitlines() if s]
    corpus_index = Path(corpus_index).resolve() if corpus_index else None
    corpus = [json.loads(s) for s in corpus_index.read_text().splitlines() if s] if corpus_index else []
    forbidden = {r["speaker"] for r in labels + corpus if r.get("split") != "train"}
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in references["ayahs"]}
    rows = []

    def add(row, root, audio, phones, source):
        path = (root / audio).resolve()
        if not path.is_relative_to(root):
            raise ValueError("recording escapes the authorized training root")
        target = phoneme_ids(phones)
        if not target:
            raise ValueError("empty performed transcript")
        rows.append(dict(id=f'{source}:{row["id"]}', speaker=row["speaker"], path=path, target=target))

    for row in labels:
        if authorized_none(row) and row["speaker"] not in forbidden:
            # Collapse held sounds within each ayah, retaining equal occurrences
            # separated by a real inter-ayah pause.
            phones = "".join("".join(c for i, c in enumerate(ayahs[a]) if not i or c != ayahs[a][i - 1])
                             for a in row["ayahs"])
            add(row, help_root, row["wav"], phones, "help-none")
    for row in corpus:
        if (row.get("split") == "train" and row["speaker"] not in forbidden and not row.get("excluded")
                and row.get("authorization") == "transcribed-training" and row.get("transcript")):
            add(row, corpus_index.parent, row["audio"], row["transcript"], "transcribed")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate authorized training catalog")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("duplicate recording across authorized catalogs")
    return rows


def course_orders(recordings, levels=LEVELS, passes=1, seed=0, course=0):
    """Matched checkpoints occur only after complete, equally weighted courses."""
    if not recordings or passes < 1 or tuple(sorted(levels)) != tuple(levels) or len(set(levels)) != len(levels):
        raise ValueError("need recordings, positive passes and unique ordered levels")
    rng = random.Random(f"{seed}:course:{course}")
    progressive = []
    for level in range(len(levels)):
        for _ in range(passes):
            order = list(range(recordings))
            rng.shuffle(order)
            progressive.extend((r, level) for r in order)
    shuffled = progressive.copy()
    rng.shuffle(shuffled)
    if Counter(progressive) != Counter(shuffled):
        raise AssertionError("unmatched example exposure")
    # Once alignment paths have been established, use identical balanced exposure.
    return {"continuation": progressive if course == 0 else shuffled.copy(), "shuffled": shuffled}


def rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.


def phone_degrade(wave, rir, noise_clips, snr_db):
    """The existing 16 kHz phone chain, with caller-owned inputs left intact.

    Trim the RIR to its direct path, match RMS, fourth-order 120–7000 Hz
    zero-phase filter, fade noise edges for 10 ms, scale to active-speech SNR,
    then limit the peak to 0.89. No recording- or benchmark-dependent parameters.
    """
    import numpy as np
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    x, r = np.asarray(wave, np.float64), np.asarray(rir, np.float64)
    if len(x) < 400 or not len(r) or not rms(r) or not noise_clips:
        raise ValueError("invalid full-recording degradation inputs")
    r = r[int(np.argmax(np.abs(r))):]
    y = fftconvolve(x, r / np.sqrt(np.sum(r * r)))[:len(x)]
    y *= rms(x) / max(rms(y), 1e-9)
    y = sosfiltfilt(butter(4, [120., 7000.], btype="band", fs=16000, output="sos"), y)
    fade, clips = np.sin(np.linspace(0, np.pi / 2, 160)) ** 2, []
    for c in noise_clips:
        if len(c) < 320:
            raise ValueError("noise clip too short for edge fades")
        c = np.asarray(c, np.float64) - np.mean(c)
        c[:160] *= fade
        c[-160:] *= fade[::-1]
        clips.append(c)
    noise = np.concatenate(clips)
    noise = noise[:len(y)] if len(noise) >= len(y) else np.resize(noise, len(y))
    n = len(y) // 160
    energy = np.mean(np.square(y[:n * 160].reshape(n, 160)), axis=1)
    active = float(np.sqrt(np.mean(np.sort(energy)[n // 2:])))
    noise *= active / max(rms(noise), 1e-9) / 10 ** (snr_db / 20)
    out = y + noise
    peak = np.max(np.abs(out))
    if peak > .89:
        out *= .89 / peak
    return out.astype(np.float32)


def waveform_views(original, degraded, levels=LEVELS):
    import numpy as np
    original, degraded = np.asarray(original, np.float32), np.asarray(degraded, np.float32)
    if original.shape != degraded.shape or original.ndim != 1 or not np.isfinite([original, degraded]).all():
        raise ValueError("views must contain the same finite sample clock")
    if not levels or levels[0] != 0 or levels[-1] != 1 or tuple(sorted(set(levels))) != tuple(levels):
        raise ValueError("progression must run from original to degraded")
    return [original.copy() if s == 0 else degraded.copy() if s == 1 else
            ((1 - s) * original + s * degraded).astype(np.float32) for s in levels]


def acoustic_loss(head, view, target):
    """Identical recording-normalized CTC + full-frame shipped-output KL."""
    import torch
    lp = head(view["features"]).log_softmax(-1)
    target = torch.as_tensor(target, dtype=torch.long)
    if not len(target) or bool(((target <= 0) | (target >= lp.shape[-1])).any()):
        raise ValueError("target must contain nonblank inventory IDs")
    ctc = torch.nn.functional.ctc_loss(lp[:, None], target, [len(lp)], [len(target)],
                                     blank=0, reduction="sum", zero_infinity=False) / len(target)
    # Real pauses and ordinary speech retain preservation supervision at every
    # output frame; no VAD mask, relabelled positives or consonant weighting.
    kl = torch.nn.functional.kl_div(lp, view["log_probs"].exp(), reduction="batchmean")
    return ctc + kl


def load_wave(path):
    import subprocess
    import numpy as np
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def fbank(wave):
    import torch
    import torchaudio
    return torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave)[None], num_mel_bins=80,
        sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()


def pick(bank, rng):
    import numpy as np
    k = int(rng.integers(len(bank["offsets"]) - 1))
    return bank["data"][bank["offsets"][k]:bank["offsets"][k + 1]].astype(np.float32)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def selected_fit(directory, arm):
    """An interrupted pair save must never expose mismatched selected arms."""
    import torch
    root = Path(directory)
    selection = json.loads((root / "selection.json").read_text())
    fits = {}
    for name in ("continuation", "shuffled"):
        entry = selection["files"][name]
        if digest(root / entry["file"]) != entry["sha256"]:
            raise ValueError("selected checkpoint checksum mismatch")
        fits[name] = torch.load(root / entry["file"], weights_only=False)
        if fits[name]["arm"] != name or any(fits[name][k] != selection[k] for k in
                ("source_sha256", "cache_sha256", "selected_updates", "selected_course")):
            raise ValueError("selected arms have unmatched exposure or provenance")
    return root / selection["files"][arm]["file"]


def cache(args):
    import numpy as np
    import torch
    if not 0 < args.hours <= 24:
        raise ValueError("cache/refit budget must be within one day")
    torch.set_num_threads(args.threads)
    encoder = Encoder(args.model, args.threads)
    m = encoder.manifest
    if (m["tokens"] != TOKENS or m["blank"] != 0 or m["sampleRate"] != 16000
            or m["fbank"].get("frameShiftMs") != 10 or m["fbank"].get("highFreq") != -400):
        raise ValueError("cache must use the shipped inventory and feature clock")
    rows = training_rows(args.help_train, json.loads(Path(args.reference).read_text()), args.corpus_index)
    if not args.corpus_index:
        raise ValueError("the experiment requires an authorized transcribed-training catalog as well as help none takes")
    if not any(r["id"].startswith("transcribed:") for r in rows) or not any(r["id"].startswith("help-none:") for r in rows):
        raise ValueError("both authorized transcribed and help-none sources are required")
    split = speaker_partition([r["speaker"] for r in rows], args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(schema=1, source_sha256=encoder.manifest["sha256"], seed=args.seed, levels=LEVELS,
                      frame_ms=40, speakers=split, hours=args.hours,
                      labels_sha256=digest(Path(args.help_train) / "labels.jsonl"), reference_sha256=digest(args.reference),
                      corpus_index_sha256=digest(args.corpus_index), noise_sha256=digest(args.noise), rir_sha256=digest(args.rir),
                      catalog=[dict(id=r["id"], speaker=r["speaker"], target=r["target"], audio_sha256=digest(r["path"])) for r in rows])
    index_path = out / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if any(index[k] != json.loads(json.dumps(v)) for k, v in provenance.items()):
            raise ValueError("cache resume provenance changed")
    else:
        index = {**provenance, "started_at": time.time(), "recordings": [], "complete": False}
        torch.save(dict(weight=encoder.weight, bias=encoder.bias), out / "initial.pt")
        index["initial_sha256"] = digest(out / "initial.pt")
        write_json(index_path, index)
    deadline = index["started_at"] + args.hours * 3600
    with np.load(args.noise) as noises, np.load(args.rir) as rirs:
        for i, row in enumerate(rows):
            if i < len(index["recordings"]):
                if digest(out / index["recordings"][i]["file"]) != index["recordings"][i]["cache_sha256"]:
                    raise ValueError("cached recording checksum mismatch")
                continue
            if time.time() >= deadline:
                raise TimeoutError("one-day cache/refit budget exhausted")
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{args.seed}:{row["id"]}'.encode()).digest()[:8], "little"))
            wave = load_wave(row["path"])
            snr = float(rng.uniform(8., 22.))
            degraded = phone_degrade(wave, pick(rirs, rng), [pick(noises, rng)], snr)
            views = [encoder.stream(fbank(w)) for w in waveform_views(wave, degraded)]
            for view in views:
                np.testing.assert_array_equal(view["times_ms"], views[0]["times_ms"])
                minimum = len(row["target"]) + sum(a == b for a, b in zip(row["target"], row["target"][1:]))
                if len(view["features"]) < minimum:
                    raise ValueError("transcript has no legal CTC path")
            name = f"record-{i:05d}.pt"
            torch.save(dict(target=row["target"], views=views), out / name)
            index["recordings"].append(dict(id=row["id"], speaker=row["speaker"], partition=split[row["speaker"]],
                                           file=name, cache_sha256=digest(out / name), samples=len(wave), snr_db=snr))
            write_json(index_path, index)
            print(json.dumps(dict(cached=i + 1, total=len(rows))), flush=True)
    index["complete"] = True
    write_json(index_path, index)


def train(args):
    import copy
    from functools import lru_cache
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root, out = Path(args.cache), Path(args.out)
    index = json.loads((root / "index.json").read_text())
    if (not index["complete"] or digest(root / "initial.pt") != index["initial_sha256"]
            or not 0 < index["hours"] <= 24 or tuple(index["levels"]) != LEVELS):
        raise ValueError("incomplete or changed frozen cache")
    deadline = index["started_at"] + index["hours"] * 3600
    if time.time() >= deadline:
        raise TimeoutError("one-day cache/refit budget exhausted")
    initial = torch.load(root / "initial.pt", weights_only=False)
    head = torch.nn.Linear(initial["weight"].shape[1], len(initial["bias"]))
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(initial["weight"]))
        head.bias.copy_(torch.from_numpy(initial["bias"]))
    heads = {a: copy.deepcopy(head) for a in ("continuation", "shuffled")}
    opts = {a: torch.optim.Adam(h.parameters(), lr=args.lr) for a, h in heads.items()}
    groups = {p: [] for p in ("train", "validation")}
    for row in index["recordings"]:
        if index["speakers"][row["speaker"]] != row["partition"] or digest(root / row["file"]) != row["cache_sha256"]:
            raise ValueError("changed partition or cached recording")
        groups[row["partition"]].append(row)

    @lru_cache(maxsize=8)
    def recording(filename):
        # Keep memory bounded by recordings in flight rather than corpus hours.
        record = torch.load(root / filename, weights_only=False)
        if len(record["views"]) != len(index["levels"]):
            raise ValueError("cached progression differs")
        for view in record["views"]:
            view["features"], view["log_probs"] = torch.from_numpy(view["features"]), torch.from_numpy(view["log_probs"])
        return record

    if not all(groups.values()):
        raise ValueError("both speaker partitions require recordings")
    out.mkdir(parents=True, exist_ok=False)
    history, best, updates = [], float("inf"), 0
    for course in range(args.courses):
        orders = course_orders(len(groups["train"]), index["levels"], args.stage_passes, args.seed, course)
        for pair in zip(*orders.values()):
            if time.time() >= deadline:
                # Do not select a partly completed course: its example multisets differ.
                write_json(out / "training.json", dict(history=history, aborted="one-day budget", args=vars(args)))
                if history:
                    return
                raise TimeoutError("no complete matched course within one-day budget")
            for (arm, h), (number, level) in zip(heads.items(), pair):
                record = recording(groups["train"][number]["file"])
                opts[arm].zero_grad()
                loss = acoustic_loss(h, record["views"][level], record["target"])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite loss; matched experiment aborted")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 5., error_if_nonfinite=True)
                opts[arm].step()
            updates += 1
        values = {}
        with torch.no_grad():
            for arm, h in heads.items():
                total = 0.
                for row in groups["validation"]:
                    if time.time() >= deadline:
                        write_json(out / "training.json", dict(history=history, aborted="one-day budget during selection", args=vars(args)))
                        if history:
                            return
                        raise TimeoutError("no complete validation within one-day budget")
                    record = recording(row["file"])
                    total += sum(acoustic_loss(h, v, record["target"]).item() for v in record["views"])
                values[arm] = total / (len(groups["validation"]) * len(index["levels"]))
        mean = sum(values.values()) / len(values)
        if not math.isfinite(mean):
            raise ValueError("nonfinite training-speaker validation")
        history.append(dict(course=course + 1, updates=updates, validation=values, mean=mean,
                            exposure_sha256=digest(root / "index.json")))
        print(json.dumps(history[-1]), flush=True)
        if mean < best:
            best = mean
            selection = dict(source_sha256=index["source_sha256"], selected_updates=updates,
                             selected_course=course + 1, cache_sha256=digest(root / "index.json"), files={})
            for arm, h in heads.items():
                filename = f"{arm}-course-{course + 1}.pt"
                torch.save(dict(head=h.state_dict(), source_sha256=index["source_sha256"], selected_updates=updates,
                                selected_course=course + 1, arm=arm, cache_sha256=selection["cache_sha256"]), out / filename)
                selection["files"][arm] = dict(file=filename, sha256=digest(out / filename))
            write_json(out / "selection.json", selection)
        write_json(out / "training.json", dict(history=history, args=vars(args),
            selection="common complete course: mean training-speaker validation CTC + full-frame shipped KL"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("cache")
    for name in ("model", "help-train", "corpus-index", "noise", "rir", "out"):
        c.add_argument(f"--{name}", required=True)
    c.add_argument("--reference", default="packages/core/assets/ref-v2.json")
    c.add_argument("--hours", type=float, default=24)
    t = commands.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--courses", type=int, default=4)
    t.add_argument("--stage-passes", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-4)
    e = commands.add_parser("export")
    for name in ("model", "fit", "out"):
        e.add_argument(f"--{name}", required=True)
    e.add_argument("--arm", choices=("continuation", "shuffled"), default="continuation")
    for cmd in (c, t, e):
        cmd.add_argument("--threads", type=int, default=2)
        cmd.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.threads < 1 or (args.command == "cache" and not 0 < args.hours <= 24):
        p.error("positive threads and a cache/refit budget of at most one day required")
    if args.command == "train" and (min(args.courses, args.stage_passes) < 1 or args.lr <= 0):
        p.error("courses, stage passes and learning rate must be positive")
    if args.command == "export":
        export(args.model, selected_fit(args.fit, args.arm), args.out, args.threads)
    else:
        {"cache": cache, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
