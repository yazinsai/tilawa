#!/usr/bin/env python3
"""
Benchmark: Streaming vs Batch ASR for Quran Verse Identification.

Compares multiple modes:
  1. Batch: Process entire audio at once
  2. Stream 3s: Independent 3-second chunks
  3. Stream 3s + overlap: 3s chunks with 1s overlap
  4. Stream 2s: Independent 2-second chunks
  5. Stream 1s: Independent 1-second chunks (fastest response)

Metrics:
  - Time to first verse identification
  - Final verse accuracy (correct surah:ayah?)
  - Word tracking accuracy
  - Total processing time
  - Chunk processing latency (per-chunk)
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from streaming_transcriber import StreamingTranscriber
from verse_position_tracker import VersePositionTracker
from word_tracker import WordTimestampExtractor

# ── Test cases ──────────────────────────────────────────────────────────────

TEST_AUDIO_DIR = PROJECT_ROOT / "data" / "test_audio"
REF_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_audio"

TEST_CASES = [
    {
        "file": str(REF_AUDIO_DIR / "001001.mp3"),
        "expected_surah": 1,
        "expected_ayah": 1,
        "description": "Al-Fatiha:1 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "001002.mp3"),
        "expected_surah": 1,
        "expected_ayah": 2,
        "description": "Al-Fatiha:2 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "002255.mp3"),
        "expected_surah": 2,
        "expected_ayah": 255,
        "description": "Ayat al-Kursi (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "112001.mp3"),
        "expected_surah": 112,
        "expected_ayah": 1,
        "description": "Al-Ikhlas:1 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "036001.mp3"),
        "expected_surah": 36,
        "expected_ayah": 1,
        "description": "Ya-Sin:1 (ref)",
    },
    {
        "file": str(TEST_AUDIO_DIR / "nisa_51.m4a"),
        "expected_surah": 4,
        "expected_ayah": 51,
        "description": "An-Nisa:51 (user)",
    },
    {
        "file": str(TEST_AUDIO_DIR / "ikhlas_2_3.m4a"),
        "expected_surah": 112,
        "expected_ayah": 2,
        "description": "Al-Ikhlas:2-3 (user)",
    },
]

# Filter to existing files
TEST_CASES = [tc for tc in TEST_CASES if Path(tc["file"]).exists()]

db = QuranDB()


# ── Streaming modes ─────────────────────────────────────────────────────────

MODES = {
    "batch": {"chunk_seconds": 9999, "overlap_seconds": 0.0},
    "stream_3s": {"chunk_seconds": 3.0, "overlap_seconds": 0.0},
    "stream_3s_overlap": {"chunk_seconds": 3.0, "overlap_seconds": 1.0},
    "stream_2s": {"chunk_seconds": 2.0, "overlap_seconds": 0.0},
    "stream_1s": {"chunk_seconds": 1.0, "overlap_seconds": 0.0},
}


# ── Benchmark functions ─────────────────────────────────────────────────────

def benchmark_batch(tc: dict) -> dict:
    """Benchmark batch mode (whole file at once)."""
    transcriber = StreamingTranscriber()
    tracker = VersePositionTracker(db)

    t0 = time.time()
    result = transcriber.transcribe_batch(tc["file"])
    proc_time = time.time() - t0

    match = tracker.update_from_text(result["text"])

    correct = (
        match is not None
        and match.surah == tc["expected_surah"]
        and match.ayah == tc["expected_ayah"]
    )

    return {
        "mode": "batch",
        "test": tc["description"],
        "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
        "transcription": result["text"][:200],
        "match_ref": match.reference if match else "NONE",
        "match_surah_ayah": f"{match.surah}:{match.ayah}" if match else "NONE",
        "confidence": round(match.confidence, 4) if match else 0.0,
        "correct": correct,
        "word_position": match.word_position if match else 0,
        "total_words": match.total_words if match else 0,
        "word_pct": round(match.progress_pct, 1) if match else 0.0,
        "time_to_first_match_s": round(proc_time, 3),  # same as total for batch
        "total_processing_time_s": round(proc_time, 3),
        "audio_duration_s": result["duration_s"],
        "num_chunks": 1,
        "avg_chunk_latency_s": round(proc_time, 3),
        "num_words_extracted": len(result["words"]),
        "rtf": round(proc_time / max(result["duration_s"], 0.01), 3),
    }


def benchmark_streaming(tc: dict, chunk_seconds: float, overlap_seconds: float, mode_name: str) -> dict:
    """Benchmark a streaming mode."""
    transcriber = StreamingTranscriber(
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    tracker = VersePositionTracker(db)

    total_start = time.time()
    first_match_time = None
    first_match_correct = False
    chunk_latencies = []
    final_match = None
    chunk_count = 0
    all_words_count = 0

    for event in transcriber.stream(tc["file"]):
        chunk_count += 1
        chunk_latencies.append(event["processing_time_s"])

        new_words = [w["word"] for w in event["words"]]
        if not new_words:
            continue

        all_words_count += len(new_words)
        match = tracker.update(new_words)

        if match:
            if first_match_time is None:
                first_match_time = time.time() - total_start
                first_match_correct = (
                    match.surah == tc["expected_surah"]
                    and match.ayah == tc["expected_ayah"]
                )
            final_match = match

    total_time = time.time() - total_start

    correct = (
        final_match is not None
        and final_match.surah == tc["expected_surah"]
        and final_match.ayah == tc["expected_ayah"]
    )

    audio_duration = event["duration_s"] if chunk_count > 0 else 0

    return {
        "mode": mode_name,
        "test": tc["description"],
        "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
        "transcription": (tracker.accumulated_text or "")[:200],
        "match_ref": final_match.reference if final_match else "NONE",
        "match_surah_ayah": f"{final_match.surah}:{final_match.ayah}" if final_match else "NONE",
        "confidence": round(final_match.confidence, 4) if final_match else 0.0,
        "correct": correct,
        "word_position": final_match.word_position if final_match else 0,
        "total_words": final_match.total_words if final_match else 0,
        "word_pct": round(final_match.progress_pct, 1) if final_match else 0.0,
        "time_to_first_match_s": round(first_match_time, 3) if first_match_time else None,
        "first_match_correct": first_match_correct,
        "total_processing_time_s": round(total_time, 3),
        "audio_duration_s": audio_duration,
        "num_chunks": chunk_count,
        "avg_chunk_latency_s": round(
            sum(chunk_latencies) / max(len(chunk_latencies), 1), 3
        ),
        "min_chunk_latency_s": round(min(chunk_latencies), 3) if chunk_latencies else 0,
        "max_chunk_latency_s": round(max(chunk_latencies), 3) if chunk_latencies else 0,
        "num_words_extracted": all_words_count,
        "rtf": round(total_time / max(audio_duration, 0.01), 3),
    }


def benchmark_word_timestamps(tc: dict) -> dict:
    """Benchmark word-level timestamp extraction quality."""
    extractor = WordTimestampExtractor()

    t0 = time.time()
    words = extractor.extract(tc["file"])
    elapsed = time.time() - t0

    # Get expected verse for alignment comparison
    verse = db.get_verse(tc["expected_surah"], tc["expected_ayah"])
    verse_words = verse["text_clean"].split() if verse else []

    extracted_text = " ".join(w.word for w in words)

    return {
        "test": tc["description"],
        "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
        "num_words": len(words),
        "expected_word_count": len(verse_words),
        "extracted_text": extracted_text[:200],
        "processing_time_s": round(elapsed, 3),
        "words": [
            {"word": w.word, "start": w.start, "end": w.end, "duration": round(w.duration, 3)}
            for w in words
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  STREAMING ASR BENCHMARK")
    print("  Paradigm D: Streaming vs Batch for Quran Verse Identification")
    print("=" * 80)
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  Modes: {', '.join(MODES.keys())}")
    print()

    all_results = []

    # Warm up model
    print("  Warming up model...")
    transcriber = StreamingTranscriber()
    transcriber._ensure_model()
    print("  Model ready.")
    print()

    # ── Run benchmarks per mode ─────────────────────────────────────────────

    for mode_name, mode_cfg in MODES.items():
        print(f"\n{'─' * 80}")
        print(f"  MODE: {mode_name}")
        if mode_name == "batch":
            print(f"  (whole file at once)")
        else:
            print(f"  (chunk={mode_cfg['chunk_seconds']}s, overlap={mode_cfg['overlap_seconds']}s)")
        print(f"{'─' * 80}")

        mode_results = []
        for tc in TEST_CASES:
            print(f"\n  Testing: {tc['description']}...")

            if mode_name == "batch":
                result = benchmark_batch(tc)
            else:
                result = benchmark_streaming(
                    tc,
                    chunk_seconds=mode_cfg["chunk_seconds"],
                    overlap_seconds=mode_cfg["overlap_seconds"],
                    mode_name=mode_name,
                )

            mode_results.append(result)

            # Print inline result
            status = "OK" if result["correct"] else "MISS"
            print(
                f"    [{status}] {result['match_ref']} "
                f"(conf={result['confidence']:.3f}) "
                f"proc={result['total_processing_time_s']:.2f}s "
                f"words={result.get('word_position', '?')}/{result.get('total_words', '?')}"
            )

        all_results.extend(mode_results)

        # Mode summary
        correct = sum(1 for r in mode_results if r["correct"])
        total = len(mode_results)
        avg_time = sum(r["total_processing_time_s"] for r in mode_results) / max(total, 1)
        avg_conf = sum(r["confidence"] for r in mode_results) / max(total, 1)
        first_match_times = [
            r["time_to_first_match_s"]
            for r in mode_results
            if r.get("time_to_first_match_s") is not None
        ]
        avg_first = sum(first_match_times) / max(len(first_match_times), 1)

        print(f"\n  {mode_name} Summary:")
        print(f"    Accuracy:        {correct}/{total} ({100 * correct / max(total, 1):.0f}%)")
        print(f"    Avg confidence:  {avg_conf:.3f}")
        print(f"    Avg total time:  {avg_time:.2f}s")
        print(f"    Avg first match: {avg_first:.2f}s")

    # ── Word timestamp benchmark ────────────────────────────────────────────
    # NOTE: word_timestamps=True in mlx-whisper is extremely slow (~200s for 6s audio)
    # due to DTW alignment. We only test on the 2 shortest files to characterize.

    print(f"\n{'─' * 80}")
    print(f"  WORD TIMESTAMP EXTRACTION")
    print(f"  (Limited to short files -- word_timestamps=True is ~160x slower)")
    print(f"{'─' * 80}")

    # Only test on the shortest reference files
    word_test_cases = [
        tc for tc in TEST_CASES
        if tc["description"] in ("Al-Ikhlas:1 (ref)", "Al-Fatiha:1 (ref)")
    ]

    word_results = []
    for tc in word_test_cases:
        print(f"\n  {tc['description']}...")
        result = benchmark_word_timestamps(tc)
        word_results.append(result)
        print(f"    Words: {result['num_words']} (expected ~{result['expected_word_count']})")
        print(f"    Time: {result['processing_time_s']:.2f}s")
        if result["words"]:
            print(f"    First: [{result['words'][0]['start']:.2f}s] {result['words'][0]['word']}")
            print(f"    Last:  [{result['words'][-1]['end']:.2f}s] {result['words'][-1]['word']}")

    # ── Grand summary ───────────────────────────────────────────────────────

    print(f"\n\n{'#' * 80}")
    print(f"  GRAND SUMMARY")
    print(f"{'#' * 80}")

    # Header
    print(f"\n  {'Mode':<20s} | {'Accuracy':>8s} | {'Avg Conf':>8s} | {'Avg Time':>8s} | {'First Match':>11s} | {'Avg RTF':>7s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*11}-+-{'-'*7}")

    for mode_name in MODES:
        mode_results = [r for r in all_results if r["mode"] == mode_name]
        correct = sum(1 for r in mode_results if r["correct"])
        total = len(mode_results)
        avg_conf = sum(r["confidence"] for r in mode_results) / max(total, 1)
        avg_time = sum(r["total_processing_time_s"] for r in mode_results) / max(total, 1)
        first_times = [
            r["time_to_first_match_s"]
            for r in mode_results
            if r.get("time_to_first_match_s") is not None
        ]
        avg_first = sum(first_times) / max(len(first_times), 1)
        avg_rtf = sum(r.get("rtf", 0) for r in mode_results) / max(total, 1)

        print(
            f"  {mode_name:<20s} | "
            f"{correct}/{total:<5d} | "
            f"{avg_conf:>8.3f} | "
            f"{avg_time:>7.2f}s | "
            f"{avg_first:>10.2f}s | "
            f"{avg_rtf:>7.3f}"
        )

    # ── Save results ────────────────────────────────────────────────────────

    out = {
        "benchmark_results": all_results,
        "word_timestamp_results": word_results,
        "metadata": {
            "model": "mlx-community/whisper-base-mlx",
            "test_cases": len(TEST_CASES),
            "modes": list(MODES.keys()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    print()


if __name__ == "__main__":
    main()
