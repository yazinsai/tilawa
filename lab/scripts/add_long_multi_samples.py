"""
Add long and multi-ayah samples to the benchmark test corpus.

Downloads single long verses and concatenates consecutive verses
from EveryAyah (Alafasy reciter) for multi-ayah samples.

Usage:
    .venv/bin/python scripts/add_long_multi_samples.py
"""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "test_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"

BASE_URL = "https://everyayah.com/data/Alafasy_128kbps"

# Silence gap between concatenated ayahs (0.5s at 16kHz)
SILENCE_GAP_SAMPLES = 8000

# --- Long single-ayah samples (15+ words, diverse surahs) ---
LONG_SAMPLES = [
    # (surah, ayah, description)
    (24, 35, "Ayat an-Nur (Light verse) - 49 words"),
    (3, 191, "Reflection on creation - 21 words"),
    (59, 23, "Names of Allah pt2 - 19 words"),
    (59, 24, "Names of Allah pt3 - 17 words"),
    (48, 29, "Muhammad rasulullah - 54 words"),
    (2, 285, "End of Al-Baqarah pt1 - commonly recited"),
    (2, 286, "End of Al-Baqarah pt2 - commonly recited"),
    (23, 115, "Did you think We created you in vain"),
    (33, 56, "Salawat verse - 13 words"),
    (74, 31, "Al-Muddaththir - 57 words"),
]

# --- Multi-ayah samples (consecutive verses, concatenated) ---
MULTI_SAMPLES = [
    # (surah, ayah_start, ayah_end, description)
    (36, 1, 5, "Ya-Sin opening"),
    (113, 1, 5, "Al-Falaq complete"),
    (114, 1, 6, "An-Nas complete"),
    (59, 22, 24, "End of Al-Hashr - Names of Allah"),
    (55, 1, 4, "Ar-Rahman opening"),
    (67, 1, 4, "Al-Mulk opening"),
    (2, 285, 286, "End of Al-Baqarah"),
    (103, 1, 3, "Al-Asr complete"),
]


def load_quran_lengths():
    """Return word counts for categorization."""
    with open(QURAN_PATH) as f:
        verses = json.load(f)
    return {(v["surah"], v["ayah"]): len(v["text_clean"].split()) for v in verses}


def download_ayah(surah: int, ayah: int) -> np.ndarray:
    """Download a single ayah from EveryAyah, return as mono float32 numpy array."""
    code = f"{surah:03d}{ayah:03d}"
    url = f"{BASE_URL}/{code}.mp3"
    print(f"    Downloading {surah}:{ayah} from {url}")
    response = urllib.request.urlopen(url)
    audio_bytes = response.read()
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    arr = arr.astype(np.float32)
    # Convert stereo to mono if needed
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr, sr


def concatenate_ayahs(surah: int, ayah_start: int, ayah_end: int):
    """Download and concatenate consecutive ayahs with silence gaps."""
    segments = []
    sr = None
    for ayah in range(ayah_start, ayah_end + 1):
        arr, file_sr = download_ayah(surah, ayah)
        if sr is None:
            sr = file_sr
        if segments:
            gap = np.zeros(SILENCE_GAP_SAMPLES, dtype=np.float32)
            segments.append(gap)
        segments.append(arr)
    return np.concatenate(segments), sr


def categorize(word_count: int) -> str:
    if word_count <= 5:
        return "short"
    elif word_count <= 15:
        return "medium"
    else:
        return "long"


def main():
    lengths = load_quran_lengths()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    existing_ids = {s["id"] for s in manifest["samples"]}
    existing_verses = {
        (s["surah"], s["ayah"], s.get("ayah_end"))
        for s in manifest["samples"]
    }

    added = 0

    # --- Add long single-ayah samples ---
    print("=== Adding long single-ayah samples ===")
    for surah, ayah, desc in LONG_SAMPLES:
        sample_id = f"ref_{surah:03d}{ayah:03d}"
        if sample_id in existing_ids:
            print(f"  Skip {surah}:{ayah} - already exists")
            continue
        if (surah, ayah, None) in existing_verses:
            print(f"  Skip {surah}:{ayah} - verse already in corpus")
            continue

        wc = lengths.get((surah, ayah), 0)
        cat = categorize(wc)
        print(f"  Adding {surah}:{ayah} ({desc}) - {wc} words [{cat}]")

        try:
            arr, sr = download_ayah(surah, ayah)
            filename = f"{surah:03d}{ayah:03d}.mp3"
            # Save as WAV for consistency (even though source is MP3)
            wav_filename = f"long_{surah:03d}_{ayah:03d}.wav"
            filepath = CORPUS_DIR / wav_filename
            sf.write(str(filepath), arr, sr)

            manifest["samples"].append({
                "id": sample_id,
                "file": wav_filename,
                "surah": surah,
                "ayah": ayah,
                "ayah_end": None,
                "category": cat,
                "source": "everyayah",
            })
            added += 1
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- Add multi-ayah samples ---
    print("\n=== Adding multi-ayah samples ===")
    for surah, ayah_start, ayah_end, desc in MULTI_SAMPLES:
        sample_id = f"multi_{surah:03d}_{ayah_start:03d}_{ayah_end:03d}"
        if sample_id in existing_ids:
            print(f"  Skip {surah}:{ayah_start}-{ayah_end} - already exists")
            continue

        # Total word count across all ayahs
        total_wc = sum(
            lengths.get((surah, a), 0)
            for a in range(ayah_start, ayah_end + 1)
        )
        print(f"  Adding {surah}:{ayah_start}-{ayah_end} ({desc}) - {total_wc} words total")

        try:
            arr, sr = concatenate_ayahs(surah, ayah_start, ayah_end)
            wav_filename = f"multi_{surah:03d}_{ayah_start:03d}_{ayah_end:03d}.wav"
            filepath = CORPUS_DIR / wav_filename
            sf.write(str(filepath), arr, sr)

            manifest["samples"].append({
                "id": sample_id,
                "file": wav_filename,
                "surah": surah,
                "ayah": ayah_start,
                "ayah_end": ayah_end,
                "category": "multi",
                "source": "everyayah",
            })
            added += 1
        except Exception as e:
            print(f"    FAILED: {e}")

    # Save updated manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done! Added {added} samples. Manifest now has {len(manifest['samples'])} total. ===")

    # Summary
    cats = {}
    for s in manifest["samples"]:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print("Category breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
