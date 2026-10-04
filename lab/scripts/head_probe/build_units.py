"""Pool frozen encoder frames onto tracker word spans. Writes /tmp only.

Prints aggregate counts. The npz has no clip ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HEAD_KINDS, IN_SCOPE, units_from_track, word_flags

SETS = ("help-acted", "help-clean", "ea", "tlog-train", "tlog-dev", "v1", "synth")
SET_ID = {name: i for i, name in enumerate(SETS)}
KIND_ID = {name: i for i, name in enumerate(
    ["skip_word", "substitution", "vowel", "skip_ayah", "repeat", "tajweed"])}


def load_enc(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(8), dtype="<u4")
        n, per = int(header[0]), int(header[1])
        raw = np.fromfile(handle, dtype="<f4")
    if per % 512 != 0:
        raise ValueError(f"encoder width {per} in {path}")
    frames = raw.reshape(-1, 512)
    if n and frames.shape[0] != n * (per // 512):
        raise ValueError(f"encoder shape {frames.shape} vs n={n} per={per}")
    return frames


def _label_for(rec: dict, labels: dict[str, dict]) -> dict | None:
    if rec["set"] == "synth":
        return {
            "mapped": True,
            "surah": int(rec["surah"]),
            "ayah": int(rec["ayah"]),
            "word_index": int(rec["word"]),
            "label_kind": rec["label_kind"],
            "span_s": rec.get("span_s"),
        }
    if rec["set"] != "help-acted":
        return None
    return labels.get(rec["id"])


def _hop(durations: list[float], nframes: list[int], tail_s: float = 2.0) -> float:
    ratios = []
    for dur, n in zip(durations, nframes):
        if dur > 3 and n > 20:
            ratios.append(dur / n)
            ratios.append((dur + tail_s) / n)
    if not ratios:
        raise SystemExit("cannot infer the encoder frame hop")
    med = float(np.median(ratios))
    # 25 Hz with or without the 2s tail both land near 0.04; accept 20–30 Hz.
    hz = 1.0 / med
    if not 20 <= hz <= 32:
        raise SystemExit(f"encoder hop looks wrong: {med:.4f}s ({hz:.1f} Hz)")
    return 0.04


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec", type=Path, required=True, help="directory of <set>.jsonl")
    ap.add_argument("--lp", type=Path, required=True, help="directory of <lpKey>.enc.f32")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="units npz path; frames written beside it")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--synth-manifest", type=Path,
                    help="synth.jsonl; the eval row does not keep the splice label")
    args = ap.parse_args()
    synth_meta: dict[str, dict] = {}
    if args.synth_manifest and args.synth_manifest.is_file():
        for line in args.synth_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                synth_meta[row["id"]] = row
    labels = {}
    if args.labels.is_file():
        for line in args.labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row
    want = [s.strip() for s in args.sets.split(",") if s.strip()]

    groups: dict[str, int] = {}
    clips = []
    words = []
    issues: list[tuple[int, int, int, int]] = []
    frame_path = args.out.with_suffix(".frames.f16")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hop_dur: list[float] = []
    hop_n: list[int] = []
    n_missing = n_err = n_clips = n_drop_synth = 0
    train_speakers: set[str] = set()
    test_speakers: set[str] = set()

    with frame_path.open("wb") as frames_out:
        cursor = 0
        for set_name in want:
            path = args.rec / f"{set_name}.jsonl"
            if not path.is_file():
                continue
            seen: dict[str, dict] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                seen[rec["id"]] = rec  # last write wins (resume retries)
            for rec in seen.values():
                if set_name == "synth":
                    meta = synth_meta.get(rec["id"])
                    if not meta:
                        n_missing += 1
                        continue
                    rec = {
                        **rec,
                        "surah": meta["surah"],
                        "ayah": meta["ayah"],
                        "word": meta["word"],
                        "label_kind": meta["label_kind"],
                        "span_s": meta.get("span_s"),
                        "speaker": meta.get("speaker") or rec.get("speaker"),
                    }
                n_clips += 1
                if rec.get("error"):
                    n_err += 1
                    continue
                key = rec.get("lpKey")
                enc_path = args.lp / f"{key}.enc.f32" if key else None
                if enc_path is None or not enc_path.is_file():
                    n_missing += 1
                    continue
                enc = load_enc(enc_path)
                dur = float(rec.get("duration_s") or 0)
                hop_dur.append(dur)
                hop_n.append(len(enc))
                speaker = str(rec.get("speaker") or rec["id"])
                split = rec.get("split") or "dev"
                if set_name == "help-acted" or set_name == "help-clean":
                    (test_speakers if split == "test" else train_speakers).add(speaker)
                if set_name in ("ea", "tlog-train", "synth") or (
                    set_name in ("help-acted", "help-clean") and split == "dev"
                ):
                    train_speakers.add(speaker)
                gid = groups.setdefault(speaker, len(groups))
                lab = _label_for(rec, labels)
                kind = -1
                mapped = 0
                in_scope = 0
                lab_s = lab_a = lab_w = -1
                span_s = None
                if lab and lab.get("mapped") and lab.get("label_kind"):
                    mapped = 1
                    kind = KIND_ID.get(lab["label_kind"], -1)
                    in_scope = int(lab["label_kind"] in IN_SCOPE)
                    lab_s = int(lab.get("surah") or -1)
                    lab_a = int(lab.get("ayah") or -1)
                    lab_w = -1 if lab.get("word_index") is None else int(lab["word_index"])
                    span_s = lab.get("span_s")
                rule_keys, extra = word_flags(rec.get("issues"))
                pending = []
                for unit in units_from_track(rec.get("trackPost")):
                    a, b = int(unit["a"]), int(unit["b"])
                    if b < a:
                        a, b = b, a
                    if b - a > 80:
                        mid = (a + b) // 2
                        a, b = mid - 40, mid + 40
                    lo = max(0, a - 1)
                    hi = min(len(enc), max(lo + 1, b + 1))
                    sl = np.asarray(enc[lo:hi], dtype=np.float32)
                    y = 0
                    ign = 0
                    if lab and mapped and unit["ayah"] == lab_a and (lab_s < 0 or unit["surah"] == lab_s):
                        if lab.get("label_kind") == "skip_ayah":
                            ign = 1
                        elif lab_w >= 0:
                            dist = abs(int(unit["word"]) - lab_w)
                            # A synth splice only counts when the tracker still
                            # emits that word. A gap-filled span is not the splice.
                            landed = dist == 0 and not (set_name == "synth" and unit["skipped"])
                            if lab["label_kind"] in HEAD_KINDS:
                                y = int(landed)
                                ign = int(0 < dist <= 1)
                            else:
                                ign = int(dist <= 1)
                    pending.append((
                        y, ign, int((unit["surah"], unit["ayah"], unit["word"]) in rule_keys),
                        unit["surah"], unit["ayah"], unit["word"], lo, a, b, sl,
                    ))
                # Dev splices whose word the tracker lost are not labels. Drop the clip.
                if set_name == "synth" and not any(row[0] for row in pending):
                    n_drop_synth += 1
                    continue
                clip_i = len(clips)
                for surah_i, ayah_i, word_i in rule_keys:
                    issues.append((clip_i, surah_i, ayah_i, word_i))
                clips.append((
                    SET_ID[set_name], 1 if split == "test" else 0, gid, dur,
                    in_scope, kind, lab_s, lab_a, lab_w, mapped, extra,
                    span_s if (lab and mapped) else None,
                ))
                for y, ign, ruled, surah_i, ayah_i, word_i, lo, a, b, sl in pending:
                    raw = sl.astype(np.float16)
                    raw.tofile(frames_out)
                    words.append((
                        y, ign, ruled, surah_i, ayah_i, word_i, clip_i,
                        cursor, cursor + len(sl), lo, a, b,
                    ))
                    cursor += len(sl)

    crossed = test_speakers & train_speakers
    if crossed:
        raise SystemExit(f"speaker overlap train/test: {len(crossed)}")

    hop = _hop(hop_dur, hop_n)
    # Patch span frame bounds. words row layout is fixed; span lives in two extra columns
    # computed from the clip's span_s and hop. words were stored without span, clip has span_s.
    span0 = np.full(len(words), -1, np.int32)
    span1 = np.full(len(words), -1, np.int32)
    for i, wrow in enumerate(words):
        if not wrow[0]:
            continue
        clip = clips[wrow[6]]
        span = clip[-1]
        if not span or len(span) != 2:
            continue
        span0[i] = int(np.floor(float(span[0]) / hop))
        span1[i] = int(np.ceil(float(span[1]) / hop))

    def col(j, dtype):
        return np.array([row[j] for row in words], dtype=dtype)

    clip_arr = [clips[i][:-1] for i in range(len(clips))]  # drop span_s
    def ccol(j, dtype):
        return np.array([row[j] for row in clip_arr], dtype=dtype)

    # mean/max recomputed would need enc again. They were not stored.
    # Store them: I appended only counts. Rebuild mean/max from the frames file
    # which is exactly the padded slice — mean/max of stored frames matches pool_span.
    frames = np.memmap(frame_path, dtype=np.float16, mode="r", shape=(cursor, 512))
    mean = np.zeros((len(words), 512), np.float16)
    mx = np.zeros((len(words), 512), np.float16)
    f0 = col(7, np.int64)
    f1 = col(8, np.int64)
    for i in range(len(words)):
        sl = np.asarray(frames[f0[i]:f1[i]], dtype=np.float32)
        if len(sl) == 0:
            continue
        mean[i] = sl.mean(0).astype(np.float16)
        mx[i] = sl.max(0).astype(np.float16)
    del frames

    np.savez(
        args.out,
        y=col(0, np.int8), ign=col(1, np.int8), rule=col(2, np.int8),
        surah=col(3, np.int16), ayah=col(4, np.int16), word=col(5, np.int16),
        clip=col(6, np.int32), f0=f0, f1=f1,
        abs_a=col(9, np.int32),
        mean=mean, amax=mx, span0=span0, span1=span1,
        clip_set=ccol(0, np.int8), clip_test=ccol(1, np.int8), clip_group=ccol(2, np.int32),
        clip_dur=ccol(3, np.float32), clip_in_scope=ccol(4, np.int8), clip_kind=ccol(5, np.int8),
        clip_lab_s=ccol(6, np.int16), clip_lab_a=ccol(7, np.int16), clip_lab_w=ccol(8, np.int16),
        clip_mapped=ccol(9, np.int8), clip_extra=ccol(10, np.int16),
        issue_clip=np.array([r[0] for r in issues], np.int32),
        issue_s=np.array([r[1] for r in issues], np.int16),
        issue_a=np.array([r[2] for r in issues], np.int16),
        issue_w=np.array([r[3] for r in issues], np.int16),
        hop=np.array([hop], np.float32),
    )
    y = col(0, np.int8)
    print(
        f"clips={len(clips)} err={n_err} missing_enc={n_missing} words={len(words)} "
        f"pos={int(y.sum())} ign={int(col(1, np.int8).sum())} frames={cursor} hop={hop:.3f} "
        f"train_speakers={len(train_speakers)} test_speakers={len(test_speakers)} "
        f"synth_dropped={n_drop_synth}"
    )


if __name__ == "__main__":
    main()
