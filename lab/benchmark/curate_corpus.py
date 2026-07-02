"""
One-time script to curate ~50 test samples from RetaSy/quranic_audio_dataset.

Usage:
    python benchmark/curate_corpus.py

Downloads samples, matches verse text to surah:ayah references,
categorizes by verse length, picks diverse speakers,
copies audio to benchmark/test_corpus/, and updates manifest.json.

Dataset schema (RetaSy/quranic_audio_dataset):
    - audio: Audio
    - Surah: str (surah name)
    - Aya: str (verse text in Arabic)
    - reciter_id: str
    - final_label: str ("correct", "incorrect", etc.)
    - duration_ms: int
"""

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from Levenshtein import ratio

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.normalizer import normalize_arabic

CORPUS_DIR = Path(__file__).parent / "test_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"

TARGETS = {
    "short": 15,
    "medium": 15,
    "long": 10,
}

random.seed(42)


def load_quran_db():
    with open(QURAN_PATH) as f:
        verses = json.load(f)
    by_text = {}
    lengths = {}
    for v in verses:
        clean = normalize_arabic(v["text_uthmani"])
        by_text[clean] = (v["surah"], v["ayah"])
        lengths[(v["surah"], v["ayah"])] = len(v["text_clean"].split())
    return verses, by_text, lengths


def match_aya_text(aya_text, verses, by_text):
    normalized = normalize_arabic(aya_text)
    if normalized in by_text:
        return by_text[normalized]
    best_score = 0
    best_ref = None
    for v in verses:
        score = ratio(normalized, v["text_clean"])
        if score > best_score:
            best_score = score
            best_ref = (v["surah"], v["ayah"])
    if best_score >= 0.7:
        return best_ref
    return None


def categorize(word_count):
    if word_count <= 5:
        return "short"
    elif word_count <= 15:
        return "medium"
    else:
        return "long"


def main():
    print("Loading Quran DB...")
    verses, by_text, lengths = load_quran_db()
    print(f"  {len(verses)} verses loaded")

    print("Loading RetaSy dataset (no audio decode)...")
    # Load with audio decoding disabled to avoid torchcodec errors on bad files
    ds = load_dataset(
        "RetaSy/quranic_audio_dataset",
        split="train",
        streaming=True,
    )
    # Cast audio to decode=False so we get raw bytes
    ds = ds.cast_column("audio", Audio(decode=False))

    candidates = {"short": [], "medium": [], "long": []}
    seen_reciters = {"short": set(), "medium": set(), "long": set()}
    errors = 0

    for i, example in enumerate(ds):
        try:
            label = example.get("final_label", "")
            if label != "correct":
                continue

            aya_text = example.get("Aya", "")
            if not aya_text:
                continue

            ref = match_aya_text(aya_text, verses, by_text)
            if ref is None:
                continue

            surah, ayah = ref
            wc = lengths.get((surah, ayah), 0)
            if wc == 0:
                continue

            cat = categorize(wc)
            reciter = example.get("reciter_id", str(i))

            if reciter in seen_reciters[cat] and len(candidates[cat]) >= TARGETS[cat]:
                continue

            # Keep raw audio bytes for later decoding
            audio_raw = example.get("audio", {})
            if not audio_raw or not audio_raw.get("bytes"):
                continue

            candidates[cat].append({
                "audio_bytes": audio_raw["bytes"],
                "surah": surah,
                "ayah": ayah,
                "word_count": wc,
                "reciter": reciter,
                "idx": i,
            })
            seen_reciters[cat].add(reciter)

            total = sum(len(v) for v in candidates.values())
            if i % 200 == 0:
                print(f"  Scanned {i}, collected {total} (errors: {errors})")

            if total >= 80:
                break

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Skip #{i}: {e}")
            continue

    total_candidates = sum(len(v) for v in candidates.values())
    print(f"\nTotal: scanned {i+1}, collected {total_candidates}, errors {errors}")

    # Select samples
    selected = []
    for cat, target in TARGETS.items():
        pool = candidates[cat]
        random.shuffle(pool)
        selected.extend(pool[:target])

    already = set(id(s) for s in selected)
    remaining_long = [c for c in candidates["long"] if id(c) not in already]
    remaining_long.sort(key=lambda x: x["word_count"], reverse=True)
    selected.extend(remaining_long[:10])

    print(f"\nSelected {len(selected)} samples:")
    for cat in ["short", "medium", "long"]:
        count = sum(1 for s in selected if categorize(s["word_count"]) == cat)
        print(f"  {cat}: {count}")

    # Load existing manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Decode audio and save
    saved = 0
    for i, sample in enumerate(selected):
        try:
            audio_bytes = sample["audio_bytes"]
            arr, sr = sf.read(io.BytesIO(audio_bytes))
            arr = arr.astype(np.float32)

            filename = f"retasy_{i:03d}.wav"
            filepath = CORPUS_DIR / filename
            sf.write(str(filepath), arr, sr)

            cat = categorize(sample["word_count"])
            manifest["samples"].append({
                "id": f"retasy_{i:03d}",
                "file": filename,
                "surah": sample["surah"],
                "ayah": sample["ayah"],
                "ayah_end": None,
                "category": cat,
                "source": "retasy",
            })
            saved += 1
        except Exception as e:
            print(f"  Failed to save sample {i} ({sample['surah']}:{sample['ayah']}): {e}")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {saved} samples. Manifest now has {len(manifest['samples'])} total entries.")


if __name__ == "__main__":
    main()
