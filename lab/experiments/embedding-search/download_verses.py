#!/usr/bin/env python3
"""Download verse-by-verse audio from EveryAyah.com for building the embedding index."""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reciters and their EveryAyah paths
RECITERS = {
    "alafasy": "Alafasy_128kbps",
    "husary": "Husary_128kbps",
}

BASE_URL = "https://everyayah.com/data"
QURAN_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "quran.json"
CORPUS_DIR = Path(__file__).resolve().parent / "audio_corpus"

MAX_WORKERS = 10
MAX_RETRIES = 2


def load_verse_list() -> list[dict]:
    """Load verse list from quran.json."""
    with open(QURAN_JSON) as f:
        return json.load(f)


def download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> tuple[bool, str]:
    """Download a file with retry logic. Returns (success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skipped (exists)"

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 0:
                return True, "downloaded"
            else:
                dest.unlink(missing_ok=True)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return False, f"failed: {e}"
    return False, "failed after retries"


def download_reciter(reciter_key: str, verses: list[dict], subset: list[int] | None = None):
    """Download all verse audio for a given reciter.

    Args:
        reciter_key: Key in RECITERS dict (e.g., 'alafasy')
        verses: List of verse dicts from quran.json
        subset: Optional list of surah numbers to limit downloads to
    """
    reciter_path = RECITERS[reciter_key]
    out_dir = CORPUS_DIR / reciter_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter verses if subset specified
    if subset:
        subset_set = set(subset)
        download_verses = [v for v in verses if v["surah"] in subset_set]
    else:
        download_verses = verses

    # Build download tasks
    tasks = []
    for verse in download_verses:
        filename = f"{verse['surah']:03d}{verse['ayah']:03d}.mp3"
        url = f"{BASE_URL}/{reciter_path}/{filename}"
        dest = out_dir / filename
        tasks.append((url, dest, filename))

    total = len(tasks)
    print(f"\n{'='*60}")
    print(f"Downloading {reciter_key}: {total} files")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {}
        for url, dest, filename in tasks:
            future = executor.submit(download_file, url, dest)
            future_to_file[future] = filename

        for i, future in enumerate(as_completed(future_to_file), 1):
            filename = future_to_file[future]
            success, msg = future.result()

            if msg == "skipped (exists)":
                skipped += 1
            elif success:
                downloaded += 1
            else:
                failed += 1
                failed_files.append((filename, msg))

            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} "
                      f"(downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    if failed_files:
        print(f"Failed files:")
        for fname, msg in failed_files[:20]:
            print(f"  {fname}: {msg}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")

    return downloaded, skipped, failed


def main():
    verses = load_verse_list()
    print(f"Loaded {len(verses)} verses from quran.json")

    # Parse command line args
    reciters_to_download = list(RECITERS.keys())
    subset = None

    if "--reciter" in sys.argv:
        idx = sys.argv.index("--reciter")
        if idx + 1 < len(sys.argv):
            reciter = sys.argv[idx + 1]
            if reciter in RECITERS:
                reciters_to_download = [reciter]
            else:
                print(f"Unknown reciter: {reciter}. Available: {list(RECITERS.keys())}")
                sys.exit(1)

    if "--subset" in sys.argv:
        # Download only specific surahs (for faster testing)
        # Default subset: surahs needed for test cases + some variety
        subset = [1, 2, 4, 36, 112]
        print(f"Subset mode: downloading surahs {subset} only")

    if "--subset-extended" in sys.argv:
        # First 10 surahs + test case surahs
        subset = list(range(1, 11)) + [36, 67, 112, 113, 114]
        subset = sorted(set(subset))
        print(f"Extended subset mode: downloading surahs {subset}")

    for reciter_key in reciters_to_download:
        download_reciter(reciter_key, verses, subset=subset)

    # Count total files
    for reciter_key in reciters_to_download:
        out_dir = CORPUS_DIR / reciter_key
        if out_dir.exists():
            count = len(list(out_dir.glob("*.mp3")))
            print(f"\n{reciter_key}: {count} files in {out_dir}")


if __name__ == "__main__":
    main()
