"""
Build test_corpus_v3 — a larger (~250-sample) benchmark corpus so that per-sample
noise stops dominating the median metrics on v1 (53 samples).

Three sources, no overlap with v1 or v2:

1. RetaSy crowdsourced (unique speakers, fresh random seed). Targets:
   - short:  80
   - medium: 60
   - long:   20

2. EveryAyah multi-reciter single verses, spanning short / medium / long. Targets:
   - ~25 long single-ayah samples across 3 reciters (Alafasy, Husary, Shatri)

3. EveryAyah multi-ayah concatenations. Targets:
   - ~25 multi-ayah sequences, 3-6 ayahs each, varied surahs

Grand target: ~210 samples. Exactly balances statistical power with disk+download cost.

Usage:
    .venv/bin/python benchmark/build_v3_corpus.py
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.normalizer import normalize_arabic  # type: ignore  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "test_corpus_v3"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"
V1_MANIFEST = Path(__file__).parent / "test_corpus" / "manifest.json"
V2_MANIFEST = Path(__file__).parent / "test_corpus_v2" / "manifest.json"

RETASY_TARGETS = {"short": 80, "medium": 60, "long": 20}
RANDOM_SEED = 2027
SILENCE_GAP_SAMPLES = 8000  # 0.5s at 16kHz

RECITERS = [
    ("Alafasy_128kbps", "alafasy"),
    ("Husary_128kbps", "husary"),
    ("Shatri_128kbps", "shatri"),
]

# Long single-ayah picks (word_count >= 16) across 3 reciters. Not in v1 or v2.
LONG_SINGLES = [
    (2, 143), (2, 177), (2, 219), (2, 282),
    (3, 8), (3, 145), (3, 173),
    (5, 3), (5, 54),
    (9, 60),
    (17, 85),
    (22, 5),
    (24, 61),
    (33, 6), (33, 33),
    (43, 36),
    (57, 20),
    (64, 17),
    (65, 12),
    (76, 5),
]

# Multi-ayah picks, 3-6 ayahs, varied surahs. Not in v1 or v2.
MULTI_SEQUENCES = [
    (1, 1, 7),       # Al-Fatiha full (7 ayahs)
    (25, 63, 68),    # ibad ar-rahman opening
    (26, 78, 82),    # Ibrahim dua
    (27, 87, 90),
    (29, 45, 49),
    (39, 71, 75),    # End of Az-Zumar
    (44, 1, 5),
    (50, 1, 5),
    (51, 1, 6),      # Adh-Dhariyat opening
    (53, 1, 5),
    (54, 1, 5),
    (68, 1, 5),
    (69, 1, 6),
    (70, 1, 5),
    (71, 1, 4),
    (73, 1, 5),
    (77, 1, 7),      # Al-Mursalat opening
    (78, 1, 6),
    (79, 1, 5),
    (80, 1, 5),
    (87, 1, 5),
    (89, 1, 5),
    (90, 1, 6),
    (93, 1, 5),      # Ad-Duha complete (11 ayahs but take 5)
    (95, 1, 5),
    (105, 1, 5),     # Al-Fil complete
    (106, 1, 4),     # Quraysh complete
    (108, 1, 3),     # Al-Kawthar complete
    (109, 1, 6),     # Al-Kafirun complete
]


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


def load_existing_refs() -> set[tuple[int, int]]:
    refs: set[tuple[int, int]] = set()
    for m_path in (V1_MANIFEST, V2_MANIFEST):
        if not m_path.exists():
            continue
        with open(m_path) as f:
            m = json.load(f)
        for s in m["samples"]:
            if s["source"] != "retasy":
                # EveryAyah overlaps are allowed across different reciters only if
                # the reciter differs. For safety we exclude exact (surah,ayah) too.
                pass
            refs.add((s["surah"], s["ayah"]))
    return refs


def match_aya_text(aya_text, verses, by_text):
    from Levenshtein import ratio

    normalized = normalize_arabic(aya_text)
    if normalized in by_text:
        return by_text[normalized]
    best_score = 0.0
    best_ref = None
    for v in verses:
        score = ratio(normalized, v["text_clean"])
        if score > best_score:
            best_score = score
            best_ref = (v["surah"], v["ayah"])
    if best_score >= 0.7:
        return best_ref
    return None


def categorize(word_count: int) -> str:
    if word_count <= 5:
        return "short"
    elif word_count <= 15:
        return "medium"
    return "long"


def download_ea_verse(reciter_dir: str, surah: int, ayah: int, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    url = f"https://everyayah.com/data/{reciter_dir}/{surah:03d}{ayah:03d}.mp3"
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "30", url, "-o", str(out_path)],
        capture_output=True,
    )
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        return False
    return True


def make_multi_wav(reciter_dir: str, surah: int, start: int, end: int, out_wav: Path) -> bool:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return True
    tmp_dir = CORPUS_DIR / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    mp3s = []
    for ayah in range(start, end + 1):
        p = tmp_dir / f"{reciter_dir}_{surah:03d}{ayah:03d}.mp3"
        if not download_ea_verse(reciter_dir, surah, ayah, p):
            # cleanup
            for m in mp3s:
                m.unlink(missing_ok=True)
            return False
        mp3s.append(p)
    # Decode each MP3 through ffmpeg to 16kHz mono f32le, concat in-memory with silence gaps.
    segments = []
    for m in mp3s:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(m),
                "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1",
            ],
            capture_output=True,
        )
        if r.returncode != 0 or not r.stdout:
            for m2 in mp3s:
                m2.unlink(missing_ok=True)
            return False
        arr = np.frombuffer(r.stdout, dtype=np.float32)
        if segments:
            segments.append(np.zeros(SILENCE_GAP_SAMPLES, dtype=np.float32))
        segments.append(arr)
    combined = np.concatenate(segments)
    sf.write(str(out_wav), combined, 16000)
    for m in mp3s:
        m.unlink(missing_ok=True)
    return True


def curate_retasy(verses, by_text, lengths, existing_refs: set[tuple[int, int]]):
    print("=== RetaSy (streaming) ===")
    from datasets import Audio, load_dataset

    random.seed(RANDOM_SEED)
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    candidates: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    seen_reciters: dict[str, set[str]] = {"short": set(), "medium": set(), "long": set()}
    errors = 0
    max_total = sum(RETASY_TARGETS.values()) * 2  # oversample so random.shuffle has room

    scanned = 0
    for i, example in enumerate(ds):
        scanned = i
        try:
            if example.get("final_label", "") != "correct":
                continue
            aya_text = example.get("Aya", "")
            if not aya_text:
                continue
            ref = match_aya_text(aya_text, verses, by_text)
            if ref is None:
                continue
            surah, ayah = ref
            if (surah, ayah) in existing_refs:
                continue
            wc = lengths.get((surah, ayah), 0)
            if wc == 0:
                continue
            cat = categorize(wc)
            reciter = str(example.get("reciter_id", i))
            # Prefer distinct reciters per category
            if reciter in seen_reciters[cat] and len(candidates[cat]) >= RETASY_TARGETS[cat]:
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
            })
            seen_reciters[cat].add(reciter)

            total = sum(len(v) for v in candidates.values())
            if i % 500 == 0:
                print(f"  Scanned {i}, collected {total} (errors: {errors})")
            if total >= max_total:
                break
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Skip #{i}: {e}")
            continue

    total_candidates = sum(len(v) for v in candidates.values())
    print(f"  Total: scanned {scanned+1}, collected {total_candidates}, errors {errors}")

    selected: list[dict] = []
    for cat, target in RETASY_TARGETS.items():
        pool = candidates[cat]
        random.shuffle(pool)
        selected.extend(pool[:target])

    print(f"  Selected {len(selected)} RetaSy samples")
    # Decode + save WAVs
    out_samples: list[dict] = []
    for i, s in enumerate(selected):
        try:
            arr, sr = sf.read(io.BytesIO(s["audio_bytes"]))
            arr = arr.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            fn = f"retasy_v3_{i:03d}.wav"
            sf.write(str(CORPUS_DIR / fn), arr, sr)
            out_samples.append({
                "id": f"retasy_v3_{i:03d}",
                "file": fn,
                "surah": s["surah"],
                "ayah": s["ayah"],
                "ayah_end": None,
                "category": categorize(s["word_count"]),
                "source": "retasy",
                "expected_verses": [{"surah": s["surah"], "ayah": s["ayah"]}],
            })
        except Exception as e:
            print(f"  Failed sample {i}: {e}")
    return out_samples


def build_ea_singles(lengths):
    print("\n=== EveryAyah long single-ayah (multi-reciter) ===")
    out_samples: list[dict] = []
    for idx, (surah, ayah) in enumerate(LONG_SINGLES):
        reciter_dir, short_name = RECITERS[idx % len(RECITERS)]
        fn = f"ea_{short_name}_{surah:03d}{ayah:03d}.mp3"
        out_path = CORPUS_DIR / fn
        ok = download_ea_verse(reciter_dir, surah, ayah, out_path)
        if not ok:
            print(f"  skip {surah}:{ayah} ({reciter_dir})")
            continue
        wc = lengths.get((surah, ayah), 0)
        cat = categorize(wc) if wc else "long"
        out_samples.append({
            "id": f"ea_{short_name}_{surah:03d}{ayah:03d}",
            "file": fn,
            "surah": surah,
            "ayah": ayah,
            "ayah_end": None,
            "category": cat,
            "source": "everyayah",
            "reciter": short_name,
            "expected_verses": [{"surah": surah, "ayah": ayah}],
        })
        print(f"  + {short_name} {surah}:{ayah} ({cat}, {wc} words)")
    return out_samples


def build_ea_multi():
    print("\n=== EveryAyah multi-ayah sequences (multi-reciter) ===")
    out_samples: list[dict] = []
    for idx, (surah, start, end) in enumerate(MULTI_SEQUENCES):
        reciter_dir, short_name = RECITERS[idx % len(RECITERS)]
        fn = f"ea_{short_name}_multi_{surah:03d}_{start:03d}_{end:03d}.wav"
        out_path = CORPUS_DIR / fn
        ok = make_multi_wav(reciter_dir, surah, start, end, out_path)
        if not ok:
            print(f"  skip {surah}:{start}-{end} ({reciter_dir})")
            continue
        expected = [{"surah": surah, "ayah": a} for a in range(start, end + 1)]
        out_samples.append({
            "id": f"ea_{short_name}_multi_{surah:03d}_{start:03d}_{end:03d}",
            "file": fn,
            "surah": surah,
            "ayah": start,
            "ayah_end": end,
            "category": "multi",
            "source": "everyayah",
            "reciter": short_name,
            "expected_verses": expected,
        })
        print(f"  + {short_name} {surah}:{start}-{end}")
    return out_samples


def main():
    CORPUS_DIR.mkdir(exist_ok=True)

    print("Loading Quran DB...")
    verses, by_text, lengths = load_quran_db()
    print(f"  {len(verses)} verses loaded")

    existing_refs = load_existing_refs()
    print(f"  {len(existing_refs)} (surah,ayah) refs to avoid from v1+v2")

    all_samples: list[dict] = []
    all_samples.extend(build_ea_singles(lengths))
    all_samples.extend(build_ea_multi())
    all_samples.extend(curate_retasy(verses, by_text, lengths, existing_refs))

    # Cleanup .tmp
    tmp = CORPUS_DIR / ".tmp"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()

    manifest = {"samples": all_samples}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    cats = Counter(s["category"] for s in all_samples)
    srcs = Counter(s["source"] for s in all_samples)
    print(f"\n=== v3 corpus: {len(all_samples)} samples ===")
    print(f"  by category: {dict(cats)}")
    print(f"  by source:   {dict(srcs)}")
    print(f"  manifest:    {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
