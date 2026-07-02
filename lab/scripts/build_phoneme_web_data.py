"""Build quran_phonemes.json for the web frontend.

Reads data/quran_phonemes.json and data/quran.json, computes joined phoneme
representations, and outputs web/frontend/public/quran_phonemes.json.

For each verse, produces:
  - surah, ayah, text_uthmani, surah_name, surah_name_en
  - phonemes: raw space-separated phoneme string
  - phonemes_joined: phonemes within words concatenated, words space-separated
      e.g. "b i s m i | a l l a h i" -> "bismi allahi"
  - phoneme_words: array of joined phoneme words
      e.g. ["bismi", "allahi"]
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEB_PUBLIC = PROJECT_ROOT / "web" / "frontend" / "public"


def join_phonemes(raw: str) -> tuple[str, list[str]]:
    """Convert raw phoneme string to joined word representation.

    "b i s m i | a l l a h i" -> ("bismi allahi", ["bismi", "allahi"])
    """
    words = []
    current_word_tokens = []

    for token in raw.split():
        if token == "|":
            if current_word_tokens:
                words.append("".join(current_word_tokens))
            current_word_tokens = []
        else:
            current_word_tokens.append(token)

    if current_word_tokens:
        words.append("".join(current_word_tokens))

    return " ".join(words), words


def main():
    # Load source data
    with open(DATA_DIR / "quran.json", encoding="utf-8") as f:
        quran = json.load(f)
    with open(DATA_DIR / "quran_phonemes.json", encoding="utf-8") as f:
        phonemes = json.load(f)

    # Index phonemes by (surah, ayah)
    phoneme_map = {(p["surah"], p["ayah"]): p["phonemes"] for p in phonemes}

    # Build combined data
    result = []
    for v in quran:
        key = (v["surah"], v["ayah"])
        raw = phoneme_map.get(key, "")
        if not raw:
            print(f"WARNING: No phonemes for {key}")
            continue

        joined, words = join_phonemes(raw)

        result.append({
            "surah": v["surah"],
            "ayah": v["ayah"],
            "text_uthmani": v["text_uthmani"],
            "surah_name": v["surah_name"],
            "surah_name_en": v["surah_name_en"],
            "phonemes": raw,
            "phonemes_joined": joined,
            "phoneme_words": words,
        })

    # Write output
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    out_path = WEB_PUBLIC / "quran_phonemes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"Written {len(result)} verses to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Spot-check first verse
    v0 = result[0]
    print(f"\nSample (1:1):")
    print(f"  phonemes:        {v0['phonemes']}")
    print(f"  phonemes_joined: {v0['phonemes_joined']}")
    print(f"  phoneme_words:   {v0['phoneme_words']}")


if __name__ == "__main__":
    main()
