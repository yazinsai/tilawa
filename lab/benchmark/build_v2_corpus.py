"""
Build v2 validation corpus from EveryAyah + RetaSy samples.
Downloads EveryAyah verses and combines with retasy shorts.

Usage:
    python benchmark/build_v2_corpus.py
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = Path(__file__).parent / "test_corpus_v2"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

RECITER = "Husary_128kbps"
BASE_URL = f"https://everyayah.com/data/{RECITER}"

# Load Quran DB to get word counts for categorization
with open(QURAN_PATH) as f:
    quran = json.load(f)
verse_wc = {}
verse_info = {}
for v in quran:
    key = (v["surah"], v["ayah"])
    verse_wc[key] = len(v.get("text_clean", "").split())
    verse_info[key] = v


def categorize(wc):
    if wc <= 5:
        return "short"
    elif wc <= 15:
        return "medium"
    return "long"


def download_verse(surah, ayah):
    """Download a single verse MP3 from EveryAyah."""
    filename = f"ea_{surah:03d}{ayah:03d}.mp3"
    filepath = CORPUS_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 0:
        return filepath
    url = f"{BASE_URL}/{surah:03d}{ayah:03d}.mp3"
    print(f"  Downloading {surah}:{ayah} from {url}")
    result = subprocess.run(
        ["curl", "-sL", url, "-o", str(filepath)],
        capture_output=True,
    )
    if result.returncode != 0 or not filepath.exists() or filepath.stat().st_size == 0:
        print(f"  FAILED: {url}")
        filepath.unlink(missing_ok=True)
        return None
    return filepath


def make_multi(surah, start, end):
    """Download and concatenate multiple verses into one WAV."""
    outfile = CORPUS_DIR / f"ea_multi_{surah:03d}_{start:03d}_{end:03d}.wav"
    if outfile.exists() and outfile.stat().st_size > 0:
        print(f"  Already exists: {outfile.name}")
        return outfile

    mp3s = []
    for ayah in range(start, end + 1):
        f = download_verse(surah, ayah)
        if f is None:
            print(f"  Skipping multi {surah}:{start}-{end} (missing {surah}:{ayah})")
            return None
        mp3s.append(f)

    # Concatenate with ffmpeg
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for f in mp3s:
        cmd.extend(["-i", str(f)])
    cmd.extend([
        "-filter_complex", f"concat=n={len(mp3s)}:v=0:a=1",
        "-f", "wav", "-ar", "16000", "-ac", "1", str(outfile),
    ])
    print(f"  Concatenating {len(mp3s)} files -> {outfile.name}")
    subprocess.run(cmd, capture_output=True)

    # Clean up individual mp3s
    for f in mp3s:
        f.unlink(missing_ok=True)

    return outfile if outfile.exists() else None


# --- Define v2 samples (no overlap with v1 surahs/ayahs) ---

SINGLE_VERSES = [
    # Medium (6-15 words)
    (16, 90),   # An-Nahl — justice/kindness verse
    (31, 13),   # Luqman — shirk warning
    (49, 13),   # Al-Hujurat — nations verse
    (39, 53),   # Az-Zumar — mercy verse
    (21, 87),   # Al-Anbiya — Yunus dua
    (3, 26),    # Aal-Imran — sovereignty
    (7, 23),    # Al-A'raf — Adam's dua
    (40, 60),   # Ghafir — call upon me
    # Long (16+ words)
    (4, 12),    # An-Nisa — inheritance
    (5, 6),     # Al-Ma'idah — wudu verse
    (6, 151),   # Al-An'am — prohibitions
    (17, 23),   # Al-Isra — parents verse
    (4, 34),    # An-Nisa — long verse
]

MULTI_VERSES = [
    (2, 1, 5),      # Al-Baqarah opening
    (19, 1, 5),     # Maryam opening (Kaf Ha Ya Ain Sad)
    (56, 1, 4),     # Al-Waqi'ah opening
    (91, 1, 5),     # Ash-Shams opening
]


def main():
    CORPUS_DIR.mkdir(exist_ok=True)

    # Load existing retasy manifest
    existing_manifest = MANIFEST_PATH
    if existing_manifest.exists():
        with open(existing_manifest) as f:
            manifest = json.load(f)
    else:
        manifest = {"samples": []}

    # Remove any previous everyayah entries
    manifest["samples"] = [s for s in manifest["samples"] if s["source"] != "everyayah"]

    print("=== Downloading single verses ===")
    for surah, ayah in SINGLE_VERSES:
        f = download_verse(surah, ayah)
        if f is None:
            continue
        wc = verse_wc.get((surah, ayah), 10)
        cat = categorize(wc)
        manifest["samples"].append({
            "id": f"ea_{surah:03d}{ayah:03d}",
            "file": f.name,
            "surah": surah,
            "ayah": ayah,
            "ayah_end": None,
            "category": cat,
            "source": "everyayah",
            "expected_verses": [{"surah": surah, "ayah": ayah}],
        })
        print(f"    {surah}:{ayah} -> {cat} ({wc} words)")

    print("\n=== Downloading multi-verse sequences ===")
    for surah, start, end in MULTI_VERSES:
        f = make_multi(surah, start, end)
        if f is None:
            continue
        expected = [{"surah": surah, "ayah": a} for a in range(start, end + 1)]
        manifest["samples"].append({
            "id": f"ea_multi_{surah:03d}_{start:03d}_{end:03d}",
            "file": f.name,
            "surah": surah,
            "ayah": start,
            "ayah_end": end,
            "category": "multi",
            "source": "everyayah",
            "expected_verses": expected,
        })
        print(f"    {surah}:{start}-{end} -> multi")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    cats = {}
    for s in manifest["samples"]:
        c = s["category"]
        cats[c] = cats.get(c, 0) + 1
    print(f"\n=== v2 corpus: {len(manifest['samples'])} samples ===")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
