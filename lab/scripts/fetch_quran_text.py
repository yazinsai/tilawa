"""Fetch Quran text from alquran.cloud API and save as JSON."""
import json
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.normalizer import normalize_arabic

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "quran.json"

URL = "https://api.alquran.cloud/v1/quran/quran-uthmani"

print("Fetching Quran text from alquran.cloud...")
with urllib.request.urlopen(URL) as resp:
    data = json.loads(resp.read())

surahs = data["data"]["surahs"]
verses = []

for surah in surahs:
    for ayah in surah["ayahs"]:
        verses.append({
            "surah": surah["number"],
            "ayah": ayah["numberInSurah"],
            "text_uthmani": ayah["text"],
            "text_clean": normalize_arabic(ayah["text"]),
            "surah_name": surah["name"],
            "surah_name_en": surah["englishName"],
        })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(verses, f, ensure_ascii=False, indent=2)

print(f"Saved {len(verses)} verses to {OUT_PATH}")
