"""Correction-mode eval on real recordings.

Scores the shipped recitation engine (`recognize(..., mode="correction")`)
for false flags on clean recitation and recall on real slips. Deliberately
incorrect recitation is not created: no splicing, no deleted or repeated
audio, no altered-text TTS, no acted mistakes. Audio and per-clip rows stay
under ``/tmp/correction_eval/`` and are not committed.

Metrics
-------
False flags per clean minute
    Issue count on clean clips divided by clean minutes. Also the fraction
    of clean clips with any issue.
Recall
    A slip is caught when any issue is on the same surah:ayah and ``word``
    is within ±1 of the located word index. Kind-correct recall restricts
    the catching issue: omitted → ``possible_omission``, substituted →
    ``possible_substitution``, repeated/restarted → any word kind
    (``possible_omission`` / ``possible_substitution`` / ``possible_vowel``)
    or ``unclear_ayah``. Exact-word is the same window with distance 0.
    Latency is the earliest catching issue's ``atSeconds`` minus the slip
    span start. The median is over caught slips that have a span.

Help slips are located first with ``locate_slips.py`` (v3 and a0w must agree
on the word, review-grade edits only). TLOG candidate rows are an unverified
recall set. Substituted TLOG slips are a seeded sample of 400 rows
(seed 0); omitted, repeated, and restarted rows are all kept.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

WORD_KINDS = frozenset({"possible_omission", "possible_substitution", "possible_vowel", "possible_repetition"})
KIND_CORRECT = {
    "omitted": frozenset({"possible_omission"}),
    "substituted": frozenset({"possible_substitution"}),
    "repeated": WORD_KINDS | frozenset({"unclear_ayah"}),
    "restarted": WORD_KINDS | frozenset({"unclear_ayah"}),
}
# Tuning vs held-out partition. Help: the manifest's speaker-disjoint split.
# TLOG candidates: by hash of the clip id. v1 is held-out only; TLOG clean dev
# is the false-flag guard in both.
PARTS = ("all", "tune", "held")


def tlog_half(clip_id: str) -> str:
    return "tune" if int(hashlib.sha1(str(clip_id).encode()).hexdigest()[:8], 16) % 2 == 0 else "held"


def in_part(set_name: str, clip: dict, part: str) -> bool:
    if part == "all":
        return True
    if set_name in ("help-clean", "help-slip", "help-acted"):
        return clip.get("split") == ("dev" if part == "tune" else "test")
    if set_name == "tlog-candidates":
        return tlog_half(clip["id"]) == part
    if set_name == "v1":
        return part == "held"
    return True
WORD_TOLERANCE = 1
# A TLOG slip whose span starts or ends this close to the clip edge is most
# likely a segmentation cut (word clipped by the ayah segmenter), not a slip.
EDGE_S = 0.3
SUB_SAMPLE = 400
SUB_SEED = 0
SLICE_FIELDS = ("device", "gender", "level", "ayah_span", "split")
ISSUE_KEYS = ("kind", "surah", "ayah", "word", "wordIndex", "atSeconds", "words")

DEFAULT_OUT = Path("/tmp/correction_eval")
DEFAULT_CORPUS = Path(os.environ.get("ZIPFORMER_CORPUS", "/workspace/lab/data/zipformer/quran.json"))
DEFAULT_ORT = Path(
    os.environ.get(
        "ZIPFORMER_ORT_DIR",
        "/workspace/.worktrees/beyond-qlab-phase0/web/frontend/node_modules",
    )
)


def kind_matches(issue_kind: str, slip_kind: str) -> bool:
    allowed = KIND_CORRECT.get(slip_kind)
    return bool(allowed) and issue_kind in allowed


def catches(issue: dict, slip: dict, tol: int = WORD_TOLERANCE) -> bool:
    """Same surah:ayah and ``word`` within ``tol`` of the located word index."""
    try:
        if int(issue["surah"]) != int(slip["surah"]) or int(issue["ayah"]) != int(slip["ayah"]):
            return False
        return abs(int(issue["word"]) - int(slip["word_index"])) <= tol
    except (KeyError, TypeError, ValueError):
        return False


def match_slip(slip: dict, issues: list[dict], tol: int = WORD_TOLERANCE) -> dict:
    """One slip against one clip's issues.

    Latency uses the earliest catching issue. Kind-correct and exact-word
    are true when any catching issue satisfies them.
    """
    catching = [issue for issue in issues if catches(issue, slip, tol)]
    exact = [issue for issue in catching if abs(int(issue["word"]) - int(slip["word_index"])) == 0]
    kind_ok = [issue for issue in catching if kind_matches(str(issue.get("kind")), str(slip.get("kind")))]
    latency = None
    span = slip.get("span_s")
    if catching and span and len(span) == 2:
        times = []
        for issue in catching:
            if issue.get("atSeconds") is None:
                continue
            try:
                times.append(float(issue["atSeconds"]))
            except (TypeError, ValueError):
                continue
        if times:
            latency = min(times) - float(span[0])
    return {
        "caught": bool(catching),
        "kind_correct": bool(kind_ok),
        "exact_word": bool(exact),
        "latency_s": latency,
    }


def _pack(rows: list[dict]) -> dict:
    n = len(rows)
    caught = sum(1 for row in rows if row["caught"])
    kind_n = sum(1 for row in rows if row["kind_correct"])
    exact = sum(1 for row in rows if row["exact_word"])
    lats = [row["latency_s"] for row in rows if row["caught"] and row["latency_s"] is not None]
    return {
        "n": n,
        "caught": caught,
        "recall": (caught / n) if n else None,
        "kind_correct": kind_n,
        "kind_recall": (kind_n / n) if n else None,
        "exact_word": exact,
        "exact_rate": (exact / n) if n else None,
        "median_latency_s": statistics.median(lats) if lats else None,
        "latency_n": len(lats),
    }


def score_slips(slips: list[dict], issues_by_id: dict[str, list[dict]], tol: int = WORD_TOLERANCE) -> dict:
    """Kind-agnostic and per-kind recall. Slips whose clip is absent are skipped.

    ``issues_by_id`` values are the issues for that clip. A missing id means
    the clip was not scored (caller should pass ``{}`` for a scored clip with
    no issues). Slips whose id is missing are returned under ``unscored``.
    """
    scored: list[dict] = []
    unscored = 0
    for slip in slips:
        cid = str(slip.get("id"))
        if cid not in issues_by_id:
            unscored += 1
            continue
        scored.append({**match_slip(slip, issues_by_id[cid], tol), "kind": slip.get("kind")})
    by_kind = {}
    for kind in sorted({row["kind"] for row in scored}):
        by_kind[kind] = _pack([row for row in scored if row["kind"] == kind])
    return {"all": _pack(scored), "by_kind": by_kind, "unscored": unscored}


def false_flag_stats(clips: list[dict]) -> dict:
    """Issues on clean clips / minutes, and the fraction of clips with any issue."""
    n = len(clips)
    issues = sum(len(clip.get("issues") or []) for clip in clips)
    seconds = 0.0
    for clip in clips:
        seconds += float(clip.get("duration_s") or 0.0)
    minutes = seconds / 60.0
    flagged = sum(1 for clip in clips if clip.get("issues"))
    return {
        "clips": n,
        "issues": issues,
        "seconds": seconds,
        "minutes": minutes,
        "per_minute": (issues / minutes) if minutes else None,
        "flagged_clips": flagged,
        "clip_fraction": (flagged / n) if n else None,
    }


def _slice_value(clip: dict, field: str) -> str:
    raw = clip.get(field)
    if raw is None or raw == "":
        return "unknown"
    return str(raw)


def slice_false_flags(clips: list[dict], fields: tuple[str, ...] = SLICE_FIELDS) -> dict:
    out: dict[str, dict] = {}
    for field in fields:
        groups: dict[str, list[dict]] = defaultdict(list)
        for clip in clips:
            groups[_slice_value(clip, field)].append(clip)
        out[field] = {key: false_flag_stats(group) for key, group in sorted(groups.items())}
    return out


def position_class(word_index: int, n_words: int | None) -> str:
    """``edge`` words have no in-ayah neighbour on one side, so word rules cannot fire."""
    if n_words is None or n_words <= 0:
        return "unknown"
    if int(word_index) <= 0 or int(word_index) >= int(n_words) - 1:
        return "edge"
    return "middle"


def position_recall(
    slips: list[dict],
    issues_by_id: dict[str, list[dict]],
    n_words_of,
) -> dict:
    """Recall split by whether the located word can have both neighbours."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for slip in slips:
        cid = str(slip.get("id"))
        if cid not in issues_by_id:
            continue
        n_words = slip.get("n_words")
        if n_words is None and n_words_of is not None:
            n_words = n_words_of(int(slip["surah"]), int(slip["ayah"]))
        kind = position_class(int(slip["word_index"]), None if n_words is None else int(n_words))
        buckets[kind].append(match_slip(slip, issues_by_id[cid]))
    return {key: _pack(rows) for key, rows in sorted(buckets.items())}


def issue_kind_counts(clips: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        for issue in clip.get("issues") or []:
            counts[str(issue.get("kind"))] += 1
    return dict(sorted(counts.items()))


def interior_slips(slips: list[dict], durations: dict[str, float], edge_s: float = EDGE_S) -> list[dict]:
    """Slips whose span sits at least ``edge_s`` inside both clip edges."""
    out = []
    for slip in slips:
        span = slip.get("span_s")
        dur = durations.get(str(slip.get("id")))
        if not span or len(span) != 2 or not dur:
            continue
        if float(span[0]) >= edge_s and float(dur) - float(span[1]) >= edge_s:
            out.append(slip)
    return out


def select_tlog_slips(rows: list[dict], n_sub: int = SUB_SAMPLE, seed: int = SUB_SEED) -> list[dict]:
    """Every omitted, repeated, and restarted row, plus ``n_sub`` substituted rows.

    Substituted rows are sorted by (id, surah, ayah, word_index) before
    ``random.Random(seed).sample``, so the draw is stable.
    """
    keep = [row for row in rows if row.get("kind") in ("omitted", "repeated", "restarted")]
    subs = [row for row in rows if row.get("kind") == "substituted"]
    subs.sort(key=lambda row: (str(row.get("id")), int(row["surah"]), int(row["ayah"]), int(row["word_index"])))
    if n_sub < len(subs):
        subs = random.Random(seed).sample(subs, n_sub)
    return keep + subs


def slim_issue(issue: dict) -> dict:
    return {key: issue[key] for key in ISSUE_KEYS if key in issue}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _samples(manifest: Path) -> list[dict]:
    data = load_json(manifest)
    samples = data["samples"] if isinstance(data, dict) else data
    if isinstance(samples, dict):
        return list(samples.values())
    return list(samples)


def help_clips(manifest: Path, audio_dir: Path, use: str) -> list[dict]:
    clips = []
    for row in _samples(manifest):
        if row.get("use") != use:
            continue
        n_ayahs = int(row.get("n_ayahs") or len(row.get("expected_verses") or []) or 1)
        clips.append(
            {
                "id": row["id"],
                "audio": str(audio_dir / row["file"]),
                "duration_s": row.get("duration_s"),
                "surah": row.get("surah"),
                "ayah": row.get("ayah"),
                "expected_verses": row.get("expected_verses") or [],
                "n_ayahs": n_ayahs,
                "device": row.get("device"),
                "gender": row.get("gender"),
                "level": row.get("level"),
                "split": row.get("split"),
                "speaker": row.get("speaker"),
                "ayah_span": "single" if n_ayahs <= 1 else "multi",
                "use": use,
            }
        )
    clips.sort(key=lambda clip: clip["id"])
    return clips


def v1_clips(root: Path) -> list[dict]:
    clips = []
    for row in _samples(root / "manifest.json"):
        clips.append(
            {
                "id": row["id"],
                "audio": str(root / row["file"]),
                "duration_s": row.get("duration_s"),
                "expected_verses": row.get("expected_verses") or [],
                "n_ayahs": len(row.get("expected_verses") or []) or 1,
                "ayah_span": "single" if len(row.get("expected_verses") or []) <= 1 else "multi",
                "source": row.get("source"),
            }
        )
    clips.sort(key=lambda clip: clip["id"])
    return clips


def tlog_dev_clips(dev_ids: Path, audio_dir: Path) -> list[dict]:
    clips = []
    for cid in sorted({line.strip() for line in dev_ids.read_text(encoding="utf-8").splitlines() if line.strip()}):
        clips.append({"id": cid, "audio": str(audio_dir / f"{cid}.flac"), "duration_s": None})
    return clips


def tlog_candidate_clips(rows: list[dict], audio_dir: Path) -> tuple[list[dict], list[dict]]:
    slips = select_tlog_slips(rows)
    by_id: dict[str, dict] = {}
    for slip in slips:
        cid = str(slip["id"])
        by_id.setdefault(cid, {"id": cid, "audio": str(audio_dir / f"{cid}.flac"), "duration_s": None})
    return [by_id[cid] for cid in sorted(by_id)], slips


def audio_duration_s(path: str) -> float:
    import soundfile as sf

    try:
        info = sf.info(path)
    except Exception:
        import subprocess

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    if info.samplerate <= 0:
        raise ValueError(path)
    return float(info.frames) / float(info.samplerate)


def _recognize():
    path = LAB / "experiments" / "zipformer-ctc" / "run.py"
    spec = importlib.util.spec_from_file_location("zipformer_ctc_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.recognize


def _done_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in load_jsonl(path)}


def trace_clips(recognize, clips: list[dict], dest: Path) -> None:
    """Record correction-controller inputs (harness ZIPFORMER_TRACE=1) per clip."""
    done = _done_ids(dest)
    pending = [clip for clip in clips if clip["id"] not in done]
    for n, clip in enumerate(pending, start=1):
        t0 = time.time()
        row = {"id": clip["id"], "duration_s": clip.get("duration_s"), "split": clip.get("split")}
        if not Path(clip["audio"]).is_file():
            row["error"] = "missing audio"
        else:
            try:
                if not row["duration_s"]:
                    row["duration_s"] = round(audio_duration_s(clip["audio"]), 3)
                res = recognize(clip["audio"], mode="correction")
                row["trace"] = res.get("trace") or []
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        append_jsonl(dest, row)
        print(f"{dest.name} {n}/{len(pending)} {clip['id']} ops={len(row.get('trace') or [])} {time.time() - t0:.1f}s", flush=True)


def run_clips(recognize, clips: list[dict], dest: Path, limit: int = 0) -> None:
    done = _done_ids(dest)
    pending = [clip for clip in clips if clip["id"] not in done]
    if limit:
        pending = pending[:limit]
    total = len(pending)
    for n, clip in enumerate(pending, start=1):
        t0 = time.time()
        row = {
            "id": clip["id"],
            "duration_s": clip.get("duration_s"),
            "device": clip.get("device"),
            "gender": clip.get("gender"),
            "level": clip.get("level"),
            "split": clip.get("split"),
            "ayah_span": clip.get("ayah_span"),
            "issues": [],
        }
        audio = clip["audio"]
        if not Path(audio).is_file():
            row["error"] = "missing audio"
        else:
            try:
                if not row["duration_s"]:
                    try:
                        row["duration_s"] = round(audio_duration_s(audio), 3)
                    except Exception:
                        row["duration_s"] = None
                res = recognize(audio, mode="correction")
                row["issues"] = [slim_issue(issue) for issue in (res.get("corrections") or [])]
                if not row["duration_s"]:
                    # Harness already consumed the file; duration stays unknown only if both failed.
                    row["duration_s"] = row["duration_s"] or 0.0
                row["decode_ms"] = int(res.get("decodeMs") or round((time.time() - t0) * 1000))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        append_jsonl(dest, row)
        flag = "ERR" if row.get("error") else str(len(row["issues"]))
        print(
            f"{dest.name} {n}/{total} {clip['id']} issues={flag} {time.time() - t0:.1f}s",
            flush=True,
        )


def _locate_module():
    name = "locate_slips"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "locate_slips.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _decode_passage(session, audio: str, tokens, corpus, verses: list[dict], loc) -> list[dict]:
    from shared.audio import load_audio
    from shared.fbank import compute_fbank
    from shared.zipformer_score import ctc_viterbi_spans, greedy_ids

    wave = load_audio(audio)
    duration = len(wave) / 16000.0
    lp = session.log_probs(compute_fbank(wave, sr=16000))
    hyp = greedy_ids(lp)
    words: list[str] = []
    index_map: list[tuple[int, int, int]] = []
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for verse in verses:
        surah, ayah = int(verse["surah"]), int(verse["ayah"])
        phonemes = corpus.word_phonemes(surah, ayah)
        for i, phoneme in enumerate(phonemes):
            words.append(phoneme)
            index_map.append((surah, ayah, i))
            counts[(surah, ayah)] += 1
    if not words:
        return []
    raw = [slip for slip in loc.locate_clip(words, hyp, tokens) if loc.is_review_slip(slip)]
    hyp_frames = ctc_viterbi_spans(lp, hyp) if hyp else []
    try:
        ref_frames, ref_groups = loc.ref_frames_by_word(words, lp, tokens)
    except Exception:
        ref_frames, ref_groups = None, []
    out = []
    for slip in raw:
        mapped = loc.ayah_local_slip(slip, index_map)
        if mapped is None:
            continue
        key, local = mapped
        span = loc.clamp_span(loc.slip_span(slip, hyp_frames, ref_frames, ref_groups), duration)
        out.append(
            {
                "surah": key[0],
                "ayah": key[1],
                "span": span,
                "slip": local,
                "n_words": counts[key],
            }
        )
    del lp
    return out


def _agree_clip(clip_id: str, left: list[dict], right: list[dict], loc) -> list[dict]:
    left_g: dict[tuple[int, int], list] = defaultdict(list)
    right_g: dict[tuple[int, int], list] = defaultdict(list)
    left_span = {}
    right_span = {}
    n_words = {}
    for item in left:
        key = (item["surah"], item["ayah"])
        left_g[key].append(item["slip"])
        left_span[(key, item["slip"].word_index)] = item["span"]
        n_words[key] = item["n_words"]
    for item in right:
        key = (item["surah"], item["ayah"])
        right_g[key].append(item["slip"])
        right_span[(key, item["slip"].word_index)] = item["span"]
        n_words[key] = item["n_words"]
    rows = []
    for key in sorted(set(left_g) & set(right_g)):
        merged = loc.merge_agreed(
            left_g[key],
            right_g[key],
            n_words=n_words[key],
            left_name="v3",
            right_name="a0w-ep1-a0.5",
        )
        for row in merged:
            span = loc.union_spans(
                [
                    left_span.get((key, row["word_index"])),
                    right_span.get((key, row["word_index"])),
                ]
            )
            rows.append(
                {
                    "id": clip_id,
                    "surah": key[0],
                    "ayah": key[1],
                    "word_index": row["word_index"],
                    "kind": row["kind"],
                    "extent": row["extent"],
                    "span_s": span,
                    "confidence": row["confidence"],
                    "n_words": n_words[key],
                }
            )
    return rows


def locate_help(manifest: Path, audio_dir: Path, v3_model: Path, a0w_model: Path, out: Path, threads: int) -> dict:
    from shared.phoneme_labels import PhonemeCorpus, load_tokens
    from shared.zipformer_score import StreamingZipformer

    loc = _locate_module()
    tokens = load_tokens()
    corpus = PhonemeCorpus(DEFAULT_CORPUS)
    clips = help_clips(manifest, audio_dir, "slip")
    per_model: dict[str, dict[str, list]] = {"v3": {}, "a0w": {}}
    errors = []
    for name, model in (("v3", v3_model), ("a0w", a0w_model)):
        session = StreamingZipformer(str(model), threads=threads)
        for n, clip in enumerate(clips, start=1):
            try:
                per_model[name][clip["id"]] = _decode_passage(
                    session, clip["audio"], tokens, corpus, clip["expected_verses"], loc
                )
            except Exception as exc:
                per_model[name][clip["id"]] = []
                errors.append({"id": clip["id"], "model": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"locate {name} {n}/{len(clips)} {clip['id']} slips={len(per_model[name][clip['id']])}", flush=True)
        del session
    located = []
    per_clip = []
    for clip in clips:
        left = per_model["v3"].get(clip["id"]) or []
        right = per_model["a0w"].get(clip["id"]) or []
        agreed = _agree_clip(clip["id"], left, right, loc)
        located.extend(agreed)
        per_clip.append(
            {
                "id": clip["id"],
                "n_ayahs": clip["n_ayahs"],
                "v3": len(left),
                "a0w": len(right),
                "agreed": len(agreed),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in located:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "clips": len(clips),
        "located_clips": sum(1 for row in per_clip if row["agreed"]),
        "located_slips": len(located),
        "multi_ayah_clips": sum(1 for clip in clips if clip["n_ayahs"] > 1),
        "multi_located_clips": sum(1 for row in per_clip if row["n_ayahs"] > 1 and row["agreed"]),
        "by_kind": dict(sorted(
            {kind: sum(1 for row in located if row["kind"] == kind) for kind in {r["kind"] for r in located}}.items()
        )) if located else {},
        "errors": len(errors),
        "per_clip": per_clip,
    }
    summary_path = out.with_name("help_locate_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2), flush=True)
    return summary


ACTED_KIND = {
    "skip_word": "omitted",
    "substitution": "substituted",
    "vowel": "vowel",
    "repeat": "repeated",
    "skip_ayah": "skipped_ayah",
    "tajweed": "tajweed",
}
_HARAKAT = "".join(chr(c) for c in list(range(0x064B, 0x0653)) + [0x0670, 0x0640] + list(range(0x06D6, 0x06EE)))


def _bare(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFC", text).replace("ٱ", "ا")
    return "".join(ch for ch in text if ch not in _HARAKAT)


def acted_word_index(label_word: int, expected: str | None, text_words: list[str], n_phoneme_words: int) -> int | None:
    """0-based phoneme-corpus word index of a 1-based export label, or None.

    ``text_words`` is the ayah without basmala. The label must name the
    expected word (diacritics ignored) and the text and phoneme word counts
    must agree, else the word cannot be placed.
    """
    index = int(label_word) - 1
    if len(text_words) != n_phoneme_words:
        return None
    if expected:
        want = _bare(expected)
        if not (0 <= index < len(text_words)) or _bare(text_words[index]) != want:
            # Labels on basmala-prefixed text sit past the ayah start.
            hits = [i for i, word in enumerate(text_words) if _bare(word) == want]
            if not hits:
                return None
            index = min(hits, key=lambda i: abs(i - index))
    return index if 0 <= index < n_phoneme_words else None


def _acted_evidence(slips: list[tuple[tuple[int, int], object]], key: tuple[int, int], word: int | None, n_words: int) -> dict:
    """What the free decode vs reference alignment shows at the labelled place."""
    local = [slip for k, slip in slips if k == key]
    if word is None:
        covered: set[int] = set()
        for slip in local:
            if slip.kind == "omitted":
                covered.update(range(slip.word_index, min(slip.word_end, n_words)))
        frac = len(covered) / n_words if n_words else 0.0
        return {"near": frac >= 0.5, "exact": frac >= 0.5, "omitted_frac": round(frac, 3)}
    near = [slip for slip in local if slip.word_index - 1 <= word <= slip.word_end]
    exact = [slip for slip in local if slip.word_index <= word < slip.word_end]
    return {"near": bool(near), "exact": bool(exact), "kinds": sorted({slip.kind for slip in near})}


def locate_acted(manifest: Path, audio_dir: Path, models: dict[str, Path], out: Path, threads: int) -> dict:
    """Place each acted label on a phoneme-corpus word, with a time span.

    The label gives the word; a forced alignment of the expected passage
    gives its span (the slot where the word should be). Each model's free
    decode, aligned against the reference, says whether there is an edit at
    or next to that word (``confirmed``).
    """
    from shared.audio import load_audio
    from shared.fbank import compute_fbank
    from shared.phoneme_labels import PhonemeCorpus, load_tokens
    from shared.quran_db import QuranDB
    from shared.zipformer_score import StreamingZipformer, greedy_ids

    loc = _locate_module()
    tokens = load_tokens()
    corpus = PhonemeCorpus(DEFAULT_CORPUS)
    db = QuranDB()
    texts = {(int(v["surah"]), int(v["ayah"])): (v.get("text_clean_no_bsm") or v["text_clean"]).split() for v in db.verses}
    samples = [row for row in _samples(manifest) if row.get("use") == "acted"]
    samples.sort(key=lambda row: row["id"])
    sessions = {name: StreamingZipformer(str(path), threads=threads) for name, path in models.items()}
    rows = []
    for n, row in enumerate(samples, start=1):
        label = row["mistake"]
        key = (int(label["surah"]), int(label["ayah"]))
        n_words = len(corpus.word_phonemes(*key))
        word = None
        if label["word"] is not None:
            word = acted_word_index(label["word"], label.get("expected"), texts.get(key, []), n_words)
        out_row = {
            "id": row["id"],
            "split": row.get("split"),
            "label_kind": label["kind"],
            "kind": ACTED_KIND[label["kind"]],
            "surah": key[0],
            "ayah": key[1],
            "word_index": 0 if label["kind"] == "skip_ayah" else word,
            "n_words": n_words,
            "mapped": label["kind"] == "skip_ayah" or word is not None,
            "duration_s": row.get("duration_s"),
            "mechanism": row.get("mechanism"),
        }
        try:
            wave = load_audio(str(audio_dir / row["file"]))
            duration = len(wave) / 16000.0
            feats = compute_fbank(wave, sr=16000)
            words: list[str] = []
            index_map: list[tuple[int, int, int]] = []
            for verse in row["expected_verses"]:
                vkey = (int(verse["surah"]), int(verse["ayah"]))
                for i, phonemes in enumerate(corpus.word_phonemes(*vkey)):
                    words.append(phonemes)
                    index_map.append((vkey[0], vkey[1], i))
            target = [i for i, (s, a, w) in enumerate(index_map) if (s, a) == key
                      and (w == (word if word is not None else 0))]
            spans = []
            evidence = {}
            for name, session in sessions.items():
                lp = session.log_probs(feats)
                hyp = greedy_ids(lp)
                slips = []
                for slip in loc.locate_clip(words, hyp, tokens):
                    mapped = loc.ayah_local_slip(slip, index_map)
                    if mapped is not None:
                        slips.append(mapped)
                evidence[name] = _acted_evidence(slips, key, None if label["kind"] == "skip_ayah" else word, n_words)
                if target and out_row["mapped"]:
                    try:
                        frames, groups = loc.ref_frames_by_word(words, lp, tokens)
                        span = loc.clamp_span(loc.span_seconds(list(frames), groups[target[0]]), duration) if frames else None
                    except Exception:
                        span = None
                    if span:
                        spans.append(span)
                del lp
            out_row["span_s"] = [round(min(s[0] for s in spans), 3), round(max(s[1] for s in spans), 3)] if spans else None
            out_row["evidence"] = evidence
            out_row["confirmed_any"] = out_row["mapped"] and any(e["near"] for e in evidence.values())
            out_row["confirmed_all"] = out_row["mapped"] and all(e["near"] for e in evidence.values())
            out_row["exact_any"] = out_row["mapped"] and any(e["exact"] for e in evidence.values())
        except Exception as exc:
            out_row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(out_row)
        print(f"locate-acted {n}/{len(samples)} {out_row['kind']} mapped={out_row['mapped']} "
              f"confirmed={out_row.get('confirmed_any')}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = acted_locate_summary(rows)
    out.with_name("acted_locate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def acted_locate_summary(rows: list[dict]) -> dict:
    """Aggregate localisation rates by label kind and split. No ids."""
    out: dict[str, dict] = {}
    for kind in sorted({r["label_kind"] for r in rows}) + ["all"]:
        chosen = [r for r in rows if kind == "all" or r["label_kind"] == kind]
        block = {}
        for split in ("dev", "test", "all"):
            part = [r for r in chosen if split == "all" or r.get("split") == split]
            block[split] = {
                "n": len(part),
                "mapped": sum(1 for r in part if r.get("mapped")),
                "span": sum(1 for r in part if r.get("span_s")),
                "confirmed_any": sum(1 for r in part if r.get("confirmed_any")),
                "confirmed_all": sum(1 for r in part if r.get("confirmed_all")),
                "exact_any": sum(1 for r in part if r.get("exact_any")),
                "errors": sum(1 for r in part if r.get("error")),
            }
        out[kind] = block
    return out


def _ok(rows: list[dict]) -> tuple[list[dict], int]:
    good = [row for row in rows if not row.get("error")]
    return good, len(rows) - len(good)


def _issues_map(rows: list[dict]) -> dict[str, list[dict]]:
    return {str(row["id"]): list(row.get("issues") or []) for row in rows if not row.get("error")}


def _rtf(rows: list[dict]) -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        if row.get("error") or not row.get("decode_ms") or not row.get("duration_s"):
            continue
        num += float(row["decode_ms"]) / 1000.0
        den += float(row["duration_s"])
    if den <= 0:
        return None
    return num / den


def build_report(tag_dir: Path, located: list[dict], tlog_slips: list[dict], n_words_of, part: str = "all") -> dict:
    sets = {}
    raw = {}
    tlog_slips = [slip for slip in tlog_slips if in_part("tlog-candidates", slip, part)]
    for name in ("help-clean", "help-slip", "v1", "tlog-dev", "tlog-candidates"):
        rows = [row for row in load_jsonl(tag_dir / f"{name}.jsonl") if in_part(name, row, part)]
        good, errors = _ok(rows)
        raw[name] = good
        sets[name] = {
            "rows": len(rows),
            "errors": errors,
            "false_flags": false_flag_stats(good) if name in ("help-clean", "v1", "tlog-dev") else None,
            "issue_kinds": issue_kind_counts(good),
            "rtf": _rtf(rows),
            "minutes": false_flag_stats(good)["minutes"],
        }
    help_map = _issues_map(raw["help-slip"])
    tlog_map = _issues_map(raw["tlog-candidates"])
    return {
        "tag": tag_dir.name,
        "part": part,
        "sets": sets,
        "slices": slice_false_flags(raw["help-clean"]),
        "help_slips": score_slips(located, help_map),
        "help_position": position_recall(located, help_map, n_words_of),
        "tlog_candidates": score_slips(tlog_slips, tlog_map),
        "tlog_interior": score_slips(
            interior_slips(tlog_slips, {str(r["id"]): r.get("duration_s") for r in raw["tlog-candidates"]}), tlog_map
        ),
        "tlog_position": position_recall(tlog_slips, tlog_map, n_words_of),
    }


def _fmt_rate(stat: dict | None) -> str:
    if not stat or stat.get("per_minute") is None:
        return "—"
    pct = 100.0 * (stat.get("clip_fraction") or 0.0)
    return f"{stat['per_minute']:.3f} ({stat['issues']}/{stat['minutes']:.1f} min, {pct:.0f}% clips)"


def _fmt_recall(block: dict | None) -> str:
    if not block or not block.get("n"):
        return "—"
    recall = 100.0 * (block.get("recall") or 0.0)
    kind = 100.0 * (block.get("kind_recall") or 0.0)
    exact = 100.0 * (block.get("exact_rate") or 0.0)
    lat = block.get("median_latency_s")
    lat_s = "—" if lat is None else f"{lat:.2f}s"
    return (
        f"{block['caught']}/{block['n']} ({recall:.0f}%) kind {block['kind_correct']}/{block['n']} ({kind:.0f}%) "
        f"exact {block['exact_word']}/{block['n']} ({exact:.0f}%) lat {lat_s}"
    )


def render_markdown(reports: list[dict], locate_summary: dict) -> str:
    lines = ["# correction eval", ""]
    if locate_summary:
        lines.append(
            f"help located clips {locate_summary.get('located_clips')}/{locate_summary.get('clips')} "
            f"slips {locate_summary.get('located_slips')} "
            f"multi {locate_summary.get('multi_located_clips')}/{locate_summary.get('multi_ayah_clips')} "
            f"kinds {locate_summary.get('by_kind')}"
        )
        lines.append("")
    lines.append("| model | help clean | tlog dev | v1 |")
    lines.append("|---|---|---|---|")
    for report in reports:
        sets = report["sets"]
        lines.append(
            f"| {report['tag']} | {_fmt_rate(sets['help-clean']['false_flags'])} | "
            f"{_fmt_rate(sets['tlog-dev']['false_flags'])} | {_fmt_rate(sets['v1']['false_flags'])} |"
        )
    lines.append("")
    lines.append("| model | set | kind | recall |")
    lines.append("|---|---|---|---|")
    for report in reports:
        for label, key in (("help slips", "help_slips"), ("tlog", "tlog_candidates"), ("tlog interior", "tlog_interior")):
            block = report[key]
            lines.append(f"| {report['tag']} | {label} | all | {_fmt_recall(block['all'])} |")
            for kind, row in block["by_kind"].items():
                lines.append(f"| {report['tag']} | {label} | {kind} | {_fmt_recall(row)} |")
        pos = report.get("tlog_position") or {}
        help_pos = report.get("help_position") or {}
        lines.append(f"| {report['tag']} | tlog position | {json.dumps({k: _fmt_recall(v) for k, v in pos.items()})} | |")
        lines.append(f"| {report['tag']} | help position | {json.dumps({k: _fmt_recall(v) for k, v in help_pos.items()})} | |")
        kinds = {name: report["sets"][name]["issue_kinds"] for name in report["sets"]}
        lines.append(f"| {report['tag']} | issue kinds | {json.dumps(kinds)} | |")
    if reports:
        lines.append("")
        lines.append("| slice | value | " + " | ".join(r["tag"] for r in reports) + " |")
        lines.append("|---|---|" + "|".join("---" for _ in reports) + "|")
        fields = reports[0]["slices"]
        for field, groups in fields.items():
            keys = sorted(set().union(*[r["slices"].get(field, {}) for r in reports]))
            for key in keys:
                cells = []
                for report in reports:
                    stat = report["slices"].get(field, {}).get(key)
                    cells.append(_fmt_rate(stat) if stat else "—")
                lines.append(f"| {field} | {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _prepare_env(model: Path) -> None:
    os.environ["ZIPFORMER_MODEL"] = str(model.resolve())
    os.environ.setdefault("ZIPFORMER_CORPUS", str(DEFAULT_CORPUS))
    os.environ.setdefault("ZIPFORMER_ORT_DIR", str(DEFAULT_ORT))
    os.environ.setdefault("TILAWA_DATA_ROOT", os.environ.get("TILAWA_DATA_ROOT", "/workspace/lab/data"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def _fetch_missing(ids: list[str], audio_dir: Path, workers: int = 8) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import modal

    audio_dir.mkdir(parents=True, exist_ok=True)
    missing = [cid for cid in ids if not (audio_dir / f"{cid}.flac").is_file()]
    if not missing:
        return
    vol = modal.Volume.from_name("zipformer-ctc-training")
    print(f"fetch {len(missing)} flacs", flush=True)

    def one(cid: str) -> None:
        data = bytearray()
        for chunk in vol.read_file(f"audio/tlog/{cid}.flac"):
            data.extend(chunk)
        if not data:
            raise FileNotFoundError(cid)
        (audio_dir / f"{cid}.flac").write_bytes(data)

    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, cid): cid for cid in missing}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception:
                fail += 1
            if done % 100 == 0 or done == len(futures):
                print(f"fetch {done}/{len(futures)} fail {fail}", flush=True)


def _n_words_lookup():
    from shared.phoneme_labels import PhonemeCorpus

    corpus = PhonemeCorpus(DEFAULT_CORPUS)
    cache: dict[tuple[int, int], int | None] = {}

    def lookup(surah: int, ayah: int) -> int | None:
        key = (surah, ayah)
        if key not in cache:
            try:
                cache[key] = len(corpus.word_phonemes(surah, ayah))
            except (ValueError, KeyError):
                cache[key] = None
        return cache[key]

    return lookup


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    loc_p = sub.add_parser("locate-help")
    loc_p.add_argument("--manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    loc_p.add_argument("--audio-dir", type=Path, default=Path("/tmp/help"))
    loc_p.add_argument("--v3", type=Path, default=Path("/tmp/models/v3.onnx"))
    loc_p.add_argument("--a0w", type=Path, default=Path("/tmp/models/a0w-ep1-a0.5.onnx"))
    loc_p.add_argument("--out", type=Path, default=DEFAULT_OUT / "help_located.jsonl")
    loc_p.add_argument("--threads", type=int, default=4)

    act_p = sub.add_parser("locate-acted")
    act_p.add_argument("--manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    act_p.add_argument("--audio-dir", type=Path, default=Path("/tmp/help"))
    act_p.add_argument("--models", default="shipped=/tmp/models/shipped.onnx,a0w=/tmp/models/a0w-ep1-a0.5.onnx")
    act_p.add_argument("--out", type=Path, default=DEFAULT_OUT / "acted_located.jsonl")
    act_p.add_argument("--threads", type=int, default=4)

    run_p = sub.add_parser("run")
    run_p.add_argument("--model", type=Path, required=True)
    run_p.add_argument("--tag", required=True)
    run_p.add_argument("--sets", default="help-clean,help-slip,v1,tlog-dev,tlog-candidates")
    run_p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run_p.add_argument("--help-manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    run_p.add_argument("--help-audio", type=Path, default=Path("/tmp/help"))
    run_p.add_argument("--v1", type=Path, default=Path("/tmp/phase0/v1"))
    run_p.add_argument("--tlog-audio", type=Path, default=DEFAULT_OUT / "audio" / "tlog")
    run_p.add_argument("--dev-ids", type=Path, default=Path("/tmp/tlog_meta/dev_ids.txt"))
    run_p.add_argument("--candidates", type=Path, default=Path("/tmp/tlog_slips/tlog_candidates.jsonl"))
    run_p.add_argument("--listen-ids", type=Path, default=Path("/tmp/tlog_meta/listen_ids.txt"),
                       help="TLOG clip ids of the listening-check queues (set tlog-listen)")
    run_p.add_argument("--limit", type=int, default=0)
    run_p.add_argument("--fetch", action="store_true")
    run_p.add_argument("--part", choices=PARTS, default="all")
    run_p.add_argument("--trace", action="store_true", help="record controller inputs instead of issues")

    rep_p = sub.add_parser("report")
    rep_p.add_argument("--tags", default="shipped,v3,a0w")
    rep_p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    rep_p.add_argument("--located", type=Path, default=DEFAULT_OUT / "help_located.jsonl")
    rep_p.add_argument("--candidates", type=Path, default=Path("/tmp/tlog_slips/tlog_candidates.jsonl"))
    rep_p.add_argument("--locate-summary", type=Path, default=DEFAULT_OUT / "help_locate_summary.json")
    rep_p.add_argument("--part", choices=PARTS, default="all")
    rep_p.add_argument("--name", default="report", help="output basename under --out-dir")

    args = parser.parse_args(argv)
    if args.cmd == "locate-help":
        locate_help(args.manifest, args.audio_dir, args.v3, args.a0w, args.out, args.threads)
        return
    if args.cmd == "locate-acted":
        models = dict(part.split("=", 1) for part in args.models.split(",") if part)
        locate_acted(args.manifest, args.audio_dir, {k: Path(v) for k, v in models.items()}, args.out, args.threads)
        return
    if args.cmd == "run":
        if args.trace:
            os.environ["ZIPFORMER_TRACE"] = "1"
        _prepare_env(args.model)
        recognize = _recognize()
        want = [part.strip() for part in args.sets.split(",") if part.strip()]
        tag_dir = args.out_dir / args.tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        tlog_slips: list[dict] = []
        if "tlog-candidates" in want:
            tlog_slips = select_tlog_slips(load_jsonl(args.candidates))
        builders = {
            "help-clean": lambda: help_clips(args.help_manifest, args.help_audio, "clean"),
            "help-slip": lambda: help_clips(args.help_manifest, args.help_audio, "slip"),
            "help-acted": lambda: help_clips(args.help_manifest, args.help_audio, "acted"),
            "v1": lambda: v1_clips(args.v1),
            "tlog-dev": lambda: tlog_dev_clips(args.dev_ids, args.tlog_audio),
            "tlog-listen": lambda: tlog_dev_clips(args.listen_ids, args.tlog_audio),
            "tlog-candidates": lambda: tlog_candidate_clips(load_jsonl(args.candidates), args.tlog_audio)[0],
        }
        if args.fetch:
            ids = []
            for name in ("tlog-dev", "tlog-listen"):
                if name in want:
                    ids.extend(clip["id"] for clip in builders[name]())
            if "tlog-candidates" in want:
                ids.extend(clip["id"] for clip in builders["tlog-candidates"]())
            _fetch_missing(sorted(set(ids)), args.tlog_audio)
        for name in want:
            clips = [clip for clip in builders[name]() if in_part(name, clip, args.part)]
            print(f"run {args.tag} {name} part={args.part} clips={len(clips)}", flush=True)
            if args.trace:
                trace_clips(recognize, clips, tag_dir / f"{name}.jsonl")
            else:
                run_clips(recognize, clips, tag_dir / f"{name}.jsonl", args.limit)
        return
    tags = [part.strip() for part in args.tags.split(",") if part.strip()]
    located = load_jsonl(args.located)
    tlog_slips = select_tlog_slips(load_jsonl(args.candidates)) if args.candidates.is_file() else []
    lookup = _n_words_lookup()
    reports = [build_report(args.out_dir / tag, located, tlog_slips, lookup, args.part) for tag in tags]
    locate_summary = load_json(args.locate_summary) if args.locate_summary.is_file() else {}
    text = render_markdown(reports, locate_summary)
    (args.out_dir / f"{args.name}.md").write_text(text, encoding="utf-8")
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
