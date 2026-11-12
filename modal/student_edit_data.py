"""Enrollment and shipped augmentation for acoustic edit-transducer recordings.

Run through the supplied supervisor. `prepare` reads help-train none takes and
an enrolled JSONL of authorized performed transcripts (speaker, split=train,
authorized=true, verified=true, transcriptFormat=ctc-articulations, phones,
wav, sha256). Include verified real silence rows with kind=silence, phones="".
Fitting reads cached training audio features only. Selection and corruption
estimation keep whole speakers separate. No practice labels enter this module.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import phoneme_ids, mix_noise
from student_edit_native import digest

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Training audio leaves its enrolled corpus")
    return path


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for row in labels:
        if row.get("split") != "train" or row.get("kind") != "none":
            continue
        if not row.get("speaker") or row["speaker"] in excluded:
            continue
        if (row.get("scored") is not True or any(row.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or row.get("excluded") or row.get("exclude")):
            continue
        if not row.get("id") or not row.get("wav") or not row.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if row["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(row["id"])
        selected.append(row)
    return selected


def reference_articulations(phones):
    """Convert ref-v2 held notation using the SDK's collapseHeld semantics.

    Madd/ikhfa runs are one occurrence. Duration-expanded mushaddad noon/meem
    is two, as is a real doubled consonant. Collapsing ALL adjacent labels
    would erase gemination and give the auxiliary loss incorrect counts.
    Performed CTC transcripts already express occurrences; do not apply this
    conversion to them.
    """
    out, i = [], 0
    while i < len(phones):
        j = i + 1
        while j < len(phones) and phones[j] == phones[i]:
            j += 1
        run = j - i
        keep = 1 if phones[i] in "اۥۦں۾" else 2 if phones[i] in "نم" and run >= 3 else run
        out.append(phones[i] * keep)
        i = j
    return "".join(out)


def recordings(help_labels, transcribed, excluded, reference):
    labels = [json.loads(s) for s in Path(help_labels).read_text().splitlines() if s.strip()]
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    # ref-v2 held runs encode duration; other repetitions can be gemination.
    # Convert each ayah's notation before counting; preserve ayah boundaries.
    # Enrolled performed CTC transcripts below must already express occurrences.
    rows = [dict(speaker=r["speaker"], source="help-none", phones="".join(reference_articulations(ayahs[a]) for a in r["ayahs"]),
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
                r.get("verified") is not True or not isinstance(r.get("phones"), str) or
                r.get("transcriptFormat") != "ctc-articulations" or not r.get("wav") or
                (not r["phones"] and r.get("kind") != "silence") or
                (r["phones"] and r.get("kind") == "silence")):
            raise ValueError("Additional recording is not an authorized performed transcript")
        path = contained(Path(transcribed).parent, r["wav"])
        if not r.get("sha256") or digest(path) != r["sha256"]:
            raise ValueError("Enrolled audio identity differs from its performed transcript")
        rows.append(dict(speaker=r["speaker"], source="silence" if not r["phones"] else "transcribed",
                         phones=r["phones"], wav=path))
    if not {"help-none", "transcribed", "silence"} <= {r["source"] for r in rows}:
        raise ValueError("Both authorized transcripts and training-speaker none takes, plus real silence, are required")
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
