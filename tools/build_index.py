# /// script
# requires-python = ">=3.12"
# dependencies = ["quran-transcript==0.6.4", "rapidfuzz==3.14.1"]
# ///
"""Build the versioned reference index consumed by @tilawa/core.

    uv run --python cpython-3.12-macos-aarch64-none tools/build_index.py
"""

import importlib.metadata
import json
import sys
from pathlib import Path

from quran_transcript import Aya, MoshafAttributes, quran_phonetizer
from rapidfuzz.distance import Levenshtein

INDEX_VERSION = "ref-v1"
INVENTORY_ID = "muaalem-v3_2-phonemes"
# Phoneme level of obadx/muaalem-model-v3_2 vocab.json in id order (1..42); id 0 is the CTC blank.
INVENTORY = [chr(c) for c in (
    0x621, 0x628, 0x62A, 0x62B, 0x62C, 0x62D, 0x62E, 0x62F, 0x630, 0x631, 0x632, 0x633, 0x634, 0x635,
    0x636, 0x637, 0x638, 0x639, 0x63A, 0x641, 0x642, 0x643, 0x644, 0x645, 0x646, 0x647, 0x648, 0x64A,
    0x627, 0x6E6, 0x6E5, 0x64E, 0x64F, 0x650, 0x6EA, 0x640, 0x672, 0x687, 0x6BA, 0x6FE, 0x6DC, 0x619,
)]
SHORT_VOWELS = {chr(0x64E), chr(0x64F), chr(0x650), chr(0x6EA)}
PROFILE = dict(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
)
OUT = Path(__file__).resolve().parent.parent / "packages/core/assets" / f"{INDEX_VERSION}.json"


def phonetize(text: str, moshaf: MoshafAttributes, sura: int) -> str:
    return quran_phonetizer(text, moshaf, remove_spaces=True, sura_idx=sura).phonemes


def word_offsets(continuous: str, isolated: list[str]) -> list[int]:
    """Start offset of each word in the continuous ayah phonemes.

    Cross-word assimilation means the continuous string can't be split on spaces, so each
    continuous character is assigned to a word by aligning against the isolated word forms.
    """
    joined = "".join(isolated)
    owner = []
    for i, w in enumerate(isolated):
        owner += [i] * len(w)
    char_word = [-1] * len(continuous)
    for op in Levenshtein.opcodes(joined, continuous):
        if op.tag in ("equal", "replace"):
            for k in range(op.dest_end - op.dest_start):
                src = min(op.src_start + k, op.src_end - 1)
                char_word[op.dest_start + k] = owner[src]
        elif op.tag == "insert":
            prev = owner[op.src_start - 1] if op.src_start > 0 else 0
            for k in range(op.dest_start, op.dest_end):
                char_word[k] = prev
    starts = []
    for i in range(len(isolated)):
        hits = [j for j, w in enumerate(char_word) if w == i]
        starts.append(hits[0] if hits else (starts[-1] if starts else 0))
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
        end = starts[i + 1] if i + 1 < len(starts) else len(continuous)
        # A short vowel belongs to the consonant before it, so a word never starts with one.
        while starts[i] < end and continuous[starts[i]] in SHORT_VOWELS:
            starts[i] += 1
    return starts


def main() -> None:
    assert len(INVENTORY) == 42 and len(set(INVENTORY)) == 42
    inventory = set(INVENTORY)
    moshaf = MoshafAttributes(**PROFILE)
    ayahs = []
    aya = Aya(1, 1)
    while True:
        a = aya.get()
        continuous = phonetize(a.uthmani, moshaf, a.sura_idx)
        isolated = [phonetize(w, moshaf, a.sura_idx) for w in a.uthmani_words]
        unknown = set(continuous).union(*map(set, isolated)) - inventory
        if unknown:
            sys.exit(f"{a.sura_idx}:{a.aya_idx} has phonemes outside the inventory: {unknown}")
        ayahs.append(
            {
                "s": a.sura_idx,
                "a": a.aya_idx,
                "w": a.uthmani_words,
                "p": continuous,
                "o": word_offsets(continuous, isolated),
                "iso": isolated,
            }
        )
        if (a.sura_idx, a.aya_idx) == (114, 6):
            break
        aya = aya.step(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "text": f"quran-transcript {importlib.metadata.version('quran-transcript')}",
                "profile": PROFILE,
                "inventory": {"id": INVENTORY_ID, "tokens": "".join(INVENTORY)},
                "ayahs": ayahs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    words = sum(len(a["w"]) for a in ayahs)
    print(f"{len(ayahs)} ayahs, {words} words -> {OUT}")


if __name__ == "__main__":
    main()
