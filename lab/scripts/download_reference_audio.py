"""Download reference audio from EveryAyah for test verses."""
import urllib.request
from pathlib import Path

BASE_URL = "https://everyayah.com/data/Alafasy_128kbps"
REF_DIR = Path(__file__).parent.parent / "data" / "reference_audio"
REF_DIR.mkdir(parents=True, exist_ok=True)

# Download verses we need to test against, plus some neighbors
VERSES = [
    # An-Nisa around ayah 51
    (4, 49), (4, 50), (4, 51), (4, 52), (4, 53),
    # Al-Ikhlas (full surah)
    (112, 1), (112, 2), (112, 3), (112, 4),
    # Aal-i-Imraan 23 (the confusing verse)
    (3, 22), (3, 23), (3, 24),
    # Al-Fatiha (easy test case)
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    # Some other well-known verses
    (2, 255),  # Ayat al-Kursi
    (36, 1), (36, 2), (36, 3),  # Ya-Sin opening
]

for surah, ayah in VERSES:
    code = f"{surah:03d}{ayah:03d}"
    url = f"{BASE_URL}/{code}.mp3"
    out = REF_DIR / f"{code}.mp3"
    if out.exists():
        print(f"  Already: {out.name}")
        continue
    print(f"  Downloading {surah}:{ayah} -> {out.name}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:
        print(f"    FAILED: {e}")

print("Done!")
