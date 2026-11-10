"""Admit only training-speaker none-labelled help takes for transcript CTC.

Mistake/repeat scripts are deliberately not converted into performed transcripts.
This is a training-input adapter, independent of practice scoring and evidence.
"""

import json
from pathlib import Path


def none_records(directory, reference):
    root = Path(directory)
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in reference["ayahs"]}
    labels = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines() if line]
    # Refuse a speaker appearing in any non-training split, even if one take is
    # inconsistently marked train. Supplied release-held-out speakers are absent.
    held = {r["speaker"] for r in labels if r.get("split") != "train"}
    records = []
    for row in labels:
        if row["speaker"] in held or row.get("split") != "train" or row.get("kind") != "none":
            continue
        if not row.get("scored", False) or row.get("excluded"):
            continue
        if row.get("events") or row.get("non_errors") or row.get("unsupported"):
            continue
        if not row.get("ayahs"):
            raise ValueError("none take is missing its transcript reference")
        path = (root / row["wav"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("training audio must be inside the supplied help directory")
        records.append({"key": f"help:{row['speaker']}:{row['id']}", "path": path,
                        "target": "".join(ayahs[a] for a in row["ayahs"])})
    if not records:
        raise ValueError("no eligible none-labelled training-speaker recordings")
    return records


def append_none(clips, directory, reference_path):
    import numpy as np
    import soundfile as sf

    records = none_records(directory, json.loads(Path(reference_path).read_text()))
    audio, sizes = [], []
    for row in records:
        info = sf.info(row["path"])
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("help training requires the supplied mono 16 kHz audio")
        audio.append(np.frombuffer(row["path"].read_bytes(), dtype=np.uint8))
        sizes.append(info.frames)
    n = len(records)
    for field in ("ref", "hyp", "target"):
        clips[field] += [r["target"] for r in records]
    clips["key"] += [r["key"] for r in records]
    clips["n_samples"] = np.concatenate([clips["n_samples"], sizes])
    # No teacher dump is supplied for these takes. A padded storage row permits
    # ordinary collation; n_frames=0 masks them out of the teacher-only KD term.
    clips["n_frames"] = np.concatenate([clips["n_frames"], np.zeros(n, dtype=np.int64)])
    clips["flac_off"] = np.concatenate([clips["flac_off"], clips["flac_off"][-1] + np.cumsum([len(a) for a in audio])])
    clips["flac"] = np.concatenate([clips["flac"], *audio])
    clips["top_off"] = np.concatenate([clips["top_off"], clips["top_off"][-1] + np.arange(1, n + 1) * 5])
    clips["ids"] = np.concatenate([clips["ids"], np.zeros(n * 5, dtype=clips["ids"].dtype)])
    clips["probs"] = np.concatenate([clips["probs"], np.zeros(n * 5, dtype=clips["probs"].dtype)])
    return n
