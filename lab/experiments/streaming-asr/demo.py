#!/usr/bin/env python3
"""
Interactive demo: Real-time Quran verse tracking from audio.

Processes a test audio file and shows progressive word-by-word tracking
with progress bars, timing, and verse identification.

Usage:
    python demo.py [audio_file]
    python demo.py                          # defaults to reference Ayat al-Kursi
    python demo.py ../../data/test_audio/nisa_51.m4a
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from streaming_transcriber import StreamingTranscriber
from verse_position_tracker import VersePositionTracker


def make_progress_bar(pct: float, width: int = 30) -> str:
    """Create a text progress bar."""
    filled = int(width * pct / 100)
    empty = width - filled
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {pct:.0f}%"


def format_time(seconds: float) -> str:
    """Format seconds as m:ss.s"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:04.1f}"


def run_demo(audio_path: str, chunk_seconds: float = 3.0, overlap_seconds: float = 0.0):
    """Run the streaming demo on an audio file."""
    path = Path(audio_path)
    if not path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  STREAMING QURAN VERSE TRACKER")
    print("  Paradigm D: Real-Time Word Position Tracking")
    print("=" * 70)
    print(f"  Audio:   {path.name}")
    print(f"  Chunks:  {chunk_seconds}s" + (f" (overlap: {overlap_seconds}s)" if overlap_seconds else ""))
    print("=" * 70)
    print()

    # Initialize
    db = QuranDB()
    transcriber = StreamingTranscriber(
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    tracker = VersePositionTracker(db)

    total_start = time.time()
    first_match_time = None
    final_match = None
    chunk_count = 0

    print("  Time     | Chunk | Proc  | Words Heard          | Verse Match")
    print("  " + "-" * 66)

    for event in transcriber.stream(audio_path):
        chunk_count += 1
        ts = event["timestamp_s"]
        proc = event["processing_time_s"]

        # Extract new words from this chunk
        new_words = [w["word"] for w in event["words"]]
        if not new_words:
            continue

        # Update verse tracking
        match = tracker.update(new_words)

        # Format chunk words (truncate if too long)
        words_display = " ".join(new_words)
        if len(words_display) > 22:
            words_display = words_display[:19] + "..."

        # Format match info
        if match:
            if first_match_time is None:
                first_match_time = time.time() - total_start
            final_match = match
            match_display = (
                f"{match.surah}:{match.ayah} "
                f"{match.word_position}/{match.total_words} "
                f"{make_progress_bar(match.progress_pct, 15)}"
            )
        else:
            match_display = "..."

        print(
            f"  {format_time(ts):>7s}  | "
            f"#{event['chunk_idx']:<3d}  | "
            f"{proc:.2f}s | "
            f"{words_display:<22s} | "
            f"{match_display}"
        )

    total_time = time.time() - total_start

    # Final summary
    print()
    print("  " + "=" * 66)
    print("  RESULT")
    print("  " + "=" * 66)

    if final_match:
        print(f"  Verse:        {final_match.reference}")
        print(f"  Confidence:   {final_match.confidence:.3f}")
        print(f"  Words:        {final_match.word_position}/{final_match.total_words} "
              f"({final_match.progress_pct:.0f}%)")
        print(f"  First match:  {format_time(first_match_time) if first_match_time else 'N/A'}")
        print(f"  Total time:   {format_time(total_time)}")
        print(f"  Chunks:       {chunk_count}")

        # Show highlighted verse text
        highlighted = tracker.get_highlighted_text()
        if highlighted:
            print()
            print("  Verse text (matched words in brackets):")
            # Word wrap at 60 chars
            words = highlighted.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 68:
                    print(line)
                    line = "    "
                line += w + " "
            if line.strip():
                print(line)
    else:
        print("  No verse identified.")
        print(f"  Total time: {format_time(total_time)}")

    print()
    return final_match


def run_batch_comparison(audio_path: str):
    """Also run batch mode for comparison."""
    print()
    print("  " + "-" * 66)
    print("  BATCH MODE (for comparison)")
    print("  " + "-" * 66)

    db = QuranDB()
    transcriber = StreamingTranscriber()

    result = transcriber.transcribe_batch(audio_path)
    text = result["text"]

    # Match
    tracker = VersePositionTracker(db)
    match = tracker.update_from_text(text)

    print(f"  Transcription: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Processing:    {result['processing_time_s']:.2f}s")

    if match:
        print(f"  Match:         {match.reference} (conf: {match.confidence:.3f})")
        print(f"  Words:         {match.word_position}/{match.total_words}")
    else:
        print("  Match:         None")

    print()


def demo_all_test_files():
    """Run demo on all available test files."""
    test_files = [
        (PROJECT_ROOT / "data" / "reference_audio" / "001001.mp3", "Al-Fatiha:1"),
        (PROJECT_ROOT / "data" / "reference_audio" / "112001.mp3", "Al-Ikhlas:1"),
        (PROJECT_ROOT / "data" / "reference_audio" / "002255.mp3", "Ayat al-Kursi"),
        (PROJECT_ROOT / "data" / "test_audio" / "nisa_51.m4a", "An-Nisa:51 (user)"),
        (PROJECT_ROOT / "data" / "test_audio" / "ikhlas_2_3.m4a", "Al-Ikhlas:2-3 (user)"),
    ]

    print("\n" + "=" * 70)
    print("  RUNNING DEMO ON ALL TEST FILES")
    print("=" * 70)

    results = []
    for path, desc in test_files:
        if path.exists():
            print(f"\n{'~' * 70}")
            print(f"  {desc}: {path.name}")
            print(f"{'~' * 70}")
            match = run_demo(str(path))
            results.append((desc, match))
        else:
            print(f"  Skipping {desc}: file not found")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for desc, match in results:
        if match:
            print(f"  {desc:<25s} -> {match.reference:<25s} ({match.confidence:.3f})")
        else:
            print(f"  {desc:<25s} -> No match")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            demo_all_test_files()
        else:
            audio = sys.argv[1]
            run_demo(audio)
            run_batch_comparison(audio)
    else:
        # Default: run on Ayat al-Kursi reference
        default = str(PROJECT_ROOT / "data" / "reference_audio" / "002255.mp3")
        run_demo(default)
        run_batch_comparison(default)
