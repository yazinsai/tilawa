"""
Generate a fresh validation corpus (v2) with a different random seed.
Uses the same RetaSy/quranic_audio_dataset source but different samples.

Usage:
    python benchmark/curate_corpus_v2.py
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

CORPUS_DIR = Path(__file__).parent / "test_corpus_v2"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"

TARGETS = {
    "short": 25,
    "medium": 15,
    "long": 5,
}

# Different seed from v1 (which used 42)
random.seed(2026)

# Track v1 samples to avoid overlap
V1_MANIFEST = Path(__file__).parent / "test_corpus" / "manifest.json"


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


def load_v1_refs():
    """Load v1 verse references to avoid overlap."""
    refs = set()
    if V1_MANIFEST.exists():
        with open(V1_MANIFEST) as f:
            m = json.load(f)
        for s in m["samples"]:
            if s["source"] == "retasy":
                refs.add((s["surah"], s["ayah"]))
    return refs


def main():
    print("Loading Quran DB...")
    verses, by_text, lengths = load_quran_db()
    print(f"  {len(verses)} verses loaded")

    v1_refs = load_v1_refs()
    print(f"  {len(v1_refs)} v1 retasy refs to avoid")

    print("Loading RetaSy dataset...")
    ds = load_dataset(
        "RetaSy/quranic_audio_dataset",
        split="train",
        streaming=True,
    )
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
            # Skip verses already in v1
            if (surah, ayah) in v1_refs:
                continue

            wc = lengths.get((surah, ayah), 0)
            if wc == 0:
                continue

            cat = categorize(wc)
            reciter = example.get("reciter_id", str(i))

            if reciter in seen_reciters[cat] and len(candidates[cat]) >= TARGETS[cat]:
                continue

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

            if total >= 120:
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

    print(f"\nSelected {len(selected)} samples:")
    for cat in ["short", "medium", "long"]:
        count = sum(1 for s in selected if categorize(s["word_count"]) == cat)
        print(f"  {cat}: {count}")

    # Build manifest
    manifest = {"samples": []}

    # Decode audio and save
    saved = 0
    for i, sample in enumerate(selected):
        try:
            audio_bytes = sample["audio_bytes"]
            arr, sr = sf.read(io.BytesIO(audio_bytes))
            arr = arr.astype(np.float32)

            filename = f"retasy_v2_{i:03d}.wav"
            filepath = CORPUS_DIR / filename
            sf.write(str(filepath), arr, sr)

            cat = categorize(sample["word_count"])
            manifest["samples"].append({
                "id": f"retasy_v2_{i:03d}",
                "file": filename,
                "surah": sample["surah"],
                "ayah": sample["ayah"],
                "ayah_end": None,
                "category": cat,
                "source": "retasy",
                "expected_verses": [
                    {"surah": sample["surah"], "ayah": sample["ayah"]}
                ],
            })
            saved += 1
        except Exception as e:
            print(f"  Failed to save sample {i} ({sample['surah']}:{sample['ayah']}): {e}")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {saved} samples to {CORPUS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
