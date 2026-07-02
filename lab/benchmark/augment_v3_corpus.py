"""
Augment test_corpus_v3 with additional EveryAyah verses to reach ~200 total
samples. Pulls across Alafasy + Husary reciters, picks short/medium/long
verses that don't overlap with existing v3 entries.

Usage:
    .venv/bin/python benchmark/augment_v3_corpus.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = Path(__file__).parent / "test_corpus_v3"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"
V1_MANIFEST = Path(__file__).parent / "test_corpus" / "manifest.json"
V2_MANIFEST = Path(__file__).parent / "test_corpus_v2" / "manifest.json"

RECITERS = [
    ("Alafasy_128kbps", "alafasy"),
    ("Husary_128kbps", "husary"),
]

# Aim for these additional counts
ADD_TARGETS = {"short": 60, "medium": 60, "long": 20}
RANDOM_SEED = 2028


def categorize(wc: int) -> str:
    if wc <= 5:
        return "short"
    elif wc <= 15:
        return "medium"
    return "long"


def download_ea_verse(reciter_dir: str, surah: int, ayah: int, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    url = f"https://everyayah.com/data/{reciter_dir}/{surah:03d}{ayah:03d}.mp3"
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "20", url, "-o", str(out_path)],
        capture_output=True,
    )
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 500:
        out_path.unlink(missing_ok=True)
        return False
    return True


def load_refs_to_avoid() -> set[tuple[int, int]]:
    """Load (surah, ayah) refs already in v1, v2, v3 to avoid duplicates."""
    refs: set[tuple[int, int]] = set()
    for m_path in (V1_MANIFEST, V2_MANIFEST, MANIFEST_PATH):
        if not m_path.exists():
            continue
        with open(m_path) as f:
            m = json.load(f)
        for s in m["samples"]:
            start = s["ayah"]
            end = s.get("ayah_end") or start
            for a in range(start, end + 1):
                refs.add((s["surah"], a))
    return refs


def main():
    with open(QURAN_PATH) as f:
        quran = json.load(f)

    # Index verses by category
    by_cat: dict[str, list[tuple[int, int, int]]] = {"short": [], "medium": [], "long": []}
    for v in quran:
        wc = len(v.get("text_clean", "").split())
        cat = categorize(wc)
        by_cat[cat].append((v["surah"], v["ayah"], wc))
    for k in by_cat:
        print(f"  Quran has {len(by_cat[k])} {k} verses")

    refs_to_avoid = load_refs_to_avoid()
    print(f"  {len(refs_to_avoid)} refs to avoid from v1+v2+existing v3")

    # Load existing manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Pick candidates: shuffle each category pool, take first N that download cleanly
    random.seed(RANDOM_SEED)
    added = 0
    per_cat_added: Counter[str] = Counter()

    for cat in ("short", "medium", "long"):
        target = ADD_TARGETS[cat]
        pool = [sa for sa in by_cat[cat] if (sa[0], sa[1]) not in refs_to_avoid]
        random.shuffle(pool)
        print(f"\n=== {cat}: target {target}, pool {len(pool)} ===")

        for i, (surah, ayah, wc) in enumerate(pool):
            if per_cat_added[cat] >= target:
                break
            # Alternate reciter
            reciter_dir, short_name = RECITERS[per_cat_added[cat] % len(RECITERS)]
            sample_id = f"ea_{short_name}_{surah:03d}{ayah:03d}"
            # Dedup by id in manifest
            if any(s["id"] == sample_id for s in manifest["samples"]):
                continue
            fn = f"{sample_id}.mp3"
            out_path = CORPUS_DIR / fn
            ok = download_ea_verse(reciter_dir, surah, ayah, out_path)
            if not ok:
                continue
            manifest["samples"].append({
                "id": sample_id,
                "file": fn,
                "surah": surah,
                "ayah": ayah,
                "ayah_end": None,
                "category": cat,
                "source": "everyayah",
                "reciter": short_name,
                "expected_verses": [{"surah": surah, "ayah": ayah}],
            })
            refs_to_avoid.add((surah, ayah))
            per_cat_added[cat] += 1
            added += 1
            if added % 10 == 0:
                print(f"  added {added} so far ({dict(per_cat_added)})")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cats = Counter(s["category"] for s in manifest["samples"])
    srcs = Counter(s["source"] for s in manifest["samples"])
    print(f"\n=== v3 corpus now: {len(manifest['samples'])} samples ===")
    print(f"  by category: {dict(cats)}")
    print(f"  by source:   {dict(srcs)}")


if __name__ == "__main__":
    main()
