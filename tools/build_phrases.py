# /// script
# requires-python = ">=3.12"
# dependencies = ["quran-transcript==0.6.4"]
# ///
"""Build the non-Quran phrase asset consumed by @tilawa/core: what reciters say around a recitation.

    uv run --python cpython-3.12-macos-aarch64-none tools/build_phrases.py

Texts follow the reference text's encoding (hamzat wasl, sukun, dagger alif as in `ref-v2`); the phonetizer passes
other mark styles through unconverted.
"""

import json
import re
import sys
from pathlib import Path

from quran_transcript import MoshafAttributes, quran_phonetizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import INVENTORY, OUT as INDEX, PROFILE  # noqa: E402

VERSION = "phrases-v1"
OUT = INDEX.parent / f"{VERSION}.json"

# 16:98 holds the isti'adha's wording after its first word.
ISTIADHA_TAIL = 'بِٱللَّهِ مِنَ ٱلشَّيْطَـٰنِ ٱلرَّجِيمِ'
PHRASES = {
    "istiadha": f"أَعُوذُ {ISTIADHA_TAIL}",
    "takbir": "ٱللَّهُ أَكْبَرُ",
    "tasmi": "سَمِعَ ٱللَّهُ لِمَنْ حَمِدَهُۥ",
    "tahmid": "رَبَّنَا وَلَكَ ٱلْحَمْدُ",
    "ruku": "سُبْحَـٰنَ رَبِّىَ ٱلْعَظِيمِ",
    "sujud": "سُبْحَـٰنَ رَبِّىَ ٱلْأَعْلَىٰ",
    "taslim": "ٱلسَّلَـٰمُ عَلَيْكُمْ وَرَحْمَةُ ٱللَّهِ",
    "amin": "ءَامِينَ",
    "sadaq_allah": "صَدَقَ ٱللَّهُ",
    "sadaq_allah_al_azim": "صَدَقَ ٱللَّهُ ٱلْعَظِيمُ",
}


def normalise(text: str) -> str:
    """The reference writes shadda before its vowel; the phonetizer misreads the other order."""
    return re.sub("([\u064b-\u0650])\u0651", "\u0651\\1", text)


def main() -> None:
    moshaf = MoshafAttributes(**PROFILE)
    inventory = set(INVENTORY)
    phrases = []
    for name, raw in PHRASES.items():
        text = normalise(raw)
        p = quran_phonetizer(text, moshaf, remove_spaces=True).phonemes
        unknown = set(p) - inventory
        if unknown:
            sys.exit(f"{name} has phonemes outside the inventory: {unknown}")
        phrases.append({"name": name, "text": text, "p": p})
        print(f"{name}: {p}")
    OUT.write_text(json.dumps({"version": VERSION, "profile": PROFILE, "phrases": phrases}, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(phrases)} phrases -> {OUT}")


if __name__ == "__main__":
    main()
