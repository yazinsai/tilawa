#!/usr/bin/env python3
"""Precompute phoneme sequences for all Quran verses.

Reads data/quran.json (6,236 verses with fully diacritized Uthmani text)
and converts each verse to IqraEval's 68-phoneme MSA set using rule-based
Arabic-to-phoneme conversion.

Output: data/quran_phonemes.json

Usage:
    python scripts/precompute_quran_phonemes.py
"""

import json
import os
import sys
import unicodedata

# ---------------------------------------------------------------------------
# Consonant map: Arabic letter -> IqraEval phoneme
# ---------------------------------------------------------------------------
CONSONANT_MAP = {
    "\u0621": "<",   # ء hamza
    "\u0628": "b",   # ب
    "\u062A": "t",   # ت
    "\u062B": "^",   # ث tha
    "\u062C": "j",   # ج
    "\u062D": "H",   # ح
    "\u062E": "x",   # خ
    "\u062F": "d",   # د
    "\u0630": "*",   # ذ dhal
    "\u0631": "r",   # ر
    "\u0632": "z",   # ز
    "\u0633": "s",   # س
    "\u0634": "$",   # ش shin
    "\u0635": "S",   # ص
    "\u0636": "D",   # ض
    "\u0637": "T",   # ط
    "\u0638": "Z",   # ظ
    "\u0639": "E",   # ع
    "\u063A": "g",   # غ
    "\u0641": "f",   # ف
    "\u0642": "q",   # ق
    "\u0643": "k",   # ك
    "\u0644": "l",   # ل
    "\u0645": "m",   # م
    "\u0646": "n",   # ن
    "\u0647": "h",   # ه
    "\u0648": "w",   # و
    "\u064A": "y",   # ي
}

# Diacritics (combining marks)
FATHA = "\u064E"      # َ  short a
DAMMA = "\u064F"      # ُ  short u
KASRA = "\u0650"      # ِ  short i
FATHATAN = "\u064B"   # ً  tanween fatha
DAMMATAN = "\u064C"   # ٌ  tanween damma
KASRATAN = "\u064D"   # ٍ  tanween kasra
SHADDA = "\u0651"     # ّ  gemination
SUKUN = "\u0652"      # ْ  no vowel
MADDAH = "\u0653"     # ٓ  maddah above
HAMZA_ABOVE = "\u0654"  # ٔ  hamza above (combining)
SUPERSCRIPT_ALEF = "\u0670"  # ٰ  long aa marker

# Letters that need special handling
ALEF = "\u0627"         # ا
ALEF_WASLA = "\u0671"   # ٱ
ALEF_HAMZA_ABOVE = "\u0623"  # أ
ALEF_HAMZA_BELOW = "\u0625"  # إ
ALEF_MADDA = "\u0622"   # آ  (not in codepoints found, but handle anyway)
WAW = "\u0648"          # و
WAW_HAMZA = "\u0624"    # ؤ
YA = "\u064A"           # ي
YA_HAMZA = "\u0626"     # ئ
ALEF_MAQSURA = "\u0649" # ى
TAA_MARBUTA = "\u0629"  # ة
TATWEEL = "\u0640"       # ـ kashida

SHORT_VOWEL_DIACRITICS = {FATHA: "a", DAMMA: "u", KASRA: "i"}
TANWEEN_DIACRITICS = {FATHATAN: ("a", "n"), DAMMATAN: ("u", "n"), KASRATAN: ("i", "n")}
ALL_DIACRITICS = {FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SHADDA, SUKUN}

# Uthmani stop/pause marks and small letter annotations to skip
SKIP_CHARS = set()
for cp in range(0x06D6, 0x06EE):
    SKIP_CHARS.add(chr(cp))
# Also skip: rub el hizb, sajdah mark, etc.
SKIP_CHARS.add("\u06DE")  # ۞ rub el hizb
SKIP_CHARS.add("\u06E9")  # ۩ sajdah mark
SKIP_CHARS.add("\uFEFF")  # BOM


def _is_diacritic(ch: str) -> bool:
    """Check if character is an Arabic diacritic or combining mark."""
    return ch in ALL_DIACRITICS or ch == SUPERSCRIPT_ALEF or ch == MADDAH or ch == HAMZA_ABOVE


def _peek(text: str, pos: int) -> str | None:
    """Return character at pos if within bounds, else None."""
    if pos < len(text):
        return text[pos]
    return None


def arabic_to_phonemes(text: str) -> str:
    """Convert diacritized Arabic text to IqraEval phoneme sequence.

    Walks through the text character by character, looking ahead for
    diacritics and long-vowel letters. Produces a space-separated
    phoneme string.

    Args:
        text: Fully diacritized Arabic text (Uthmani script).

    Returns:
        Space-separated phoneme string using IqraEval's 68-phoneme set.
    """
    # Clean text
    text = text.replace("\uFEFF", "")  # remove BOM
    text = text.strip()

    phonemes: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # --- Skip Uthmani annotation marks ---
        if ch in SKIP_CHARS:
            i += 1
            continue

        # --- Tatweel (kashida) - skip ---
        if ch == TATWEEL:
            i += 1
            continue

        # --- Space → word boundary ---
        if ch == " ":
            # Don't add duplicate word boundaries
            if phonemes and phonemes[-1] != "|":
                phonemes.append("|")
            i += 1
            continue

        # --- Alef Madda آ → < aa ---
        if ch == ALEF_MADDA:
            phonemes.append("<")
            phonemes.append("aa")
            i += 1
            # Consume any trailing diacritics
            while i < n and _is_diacritic(text[i]):
                i += 1
            continue

        # --- Alef with hamza above أ ---
        if ch == ALEF_HAMZA_ABOVE:
            phonemes.append("<")
            i += 1
            # Check for shadda
            if i < n and text[i] == SHADDA:
                phonemes.append("<")  # geminate hamza
                i += 1
            # Check for vowel diacritic
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
            elif i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
            elif i < n and text[i] == SUKUN:
                i += 1
            else:
                # Default: fatha for hamza on alef
                phonemes.append("a")
            # Check for long vowel following
            if i < n and text[i] == ALEF:
                phonemes[-1] = "aa" if phonemes[-1] == "a" else phonemes[-1]
                if phonemes[-1] == "a":
                    phonemes[-1] = "aa"
                i += 1
            continue

        # --- Alef with hamza below إ ---
        if ch == ALEF_HAMZA_BELOW:
            phonemes.append("<")
            i += 1
            # Check for shadda
            if i < n and text[i] == SHADDA:
                phonemes.append("<")
                i += 1
            # Check for vowel
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
            elif i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
            elif i < n and text[i] == SUKUN:
                i += 1
            else:
                phonemes.append("i")
            continue

        # --- Alef Wasla ٱ ---
        if ch == ALEF_WASLA:
            # Alef wasla marks the definite article or verb/noun prefixes.
            # Emit a short vowel for the phoneme representation.
            i += 1
            # Collect any diacritics on the alef wasla itself
            vowel_emitted = False
            while i < n and (_is_diacritic(text[i]) or text[i] == SUKUN):
                if text[i] in SHORT_VOWEL_DIACRITICS:
                    phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                    vowel_emitted = True
                elif text[i] == SUPERSCRIPT_ALEF:
                    phonemes.append("aa")
                    vowel_emitted = True
                i += 1
            if not vowel_emitted:
                # Default: emit short 'a' for the wasla
                phonemes.append("a")
            # Sun letter assimilation: if this is ٱل + shadda'd consonant,
            # skip the lam (it assimilates into the following consonant).
            # Check for lam followed by a consonant with shadda.
            if i < n and text[i] == "\u0644":  # lam
                # Look ahead past the lam to check for sukun and then a
                # shadda'd consonant, or directly for a consonant with shadda
                j = i + 1
                # The lam may have sukun explicitly
                if j < n and text[j] == SUKUN:
                    j += 1
                # Now check if next char is a consonant followed by shadda
                if j < n and text[j] in CONSONANT_MAP:
                    # Scan the consonant's diacritics for shadda
                    k = j + 1
                    has_shadda_ahead = False
                    while k < n and (text[k] in ALL_DIACRITICS or text[k] == SUPERSCRIPT_ALEF or text[k] in SKIP_CHARS):
                        if text[k] == SHADDA:
                            has_shadda_ahead = True
                            break
                        k += 1
                    if has_shadda_ahead:
                        # Sun letter assimilation: skip the lam entirely
                        i = j  # advance past lam (+ sukun), land on the consonant
                        # Don't skip the consonant itself - it will be processed
                        # in the next loop iteration
            continue

        # --- Waw with hamza ؤ ---
        if ch == WAW_HAMZA:
            phonemes.append("<")
            i += 1
            # Process diacritics
            if i < n and text[i] == SHADDA:
                phonemes.append("<")
                i += 1
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
            elif i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
            elif i < n and text[i] == SUKUN:
                i += 1
            continue

        # --- Ya with hamza ئ ---
        if ch == YA_HAMZA:
            phonemes.append("<")
            i += 1
            if i < n and text[i] == SHADDA:
                phonemes.append("<")
                i += 1
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
            elif i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
            elif i < n and text[i] == SUKUN:
                i += 1
            continue

        # --- Standalone hamza ء ---
        if ch == "\u0621":
            phonemes.append("<")
            i += 1
            # Process trailing diacritics
            if i < n and text[i] == SHADDA:
                phonemes.append("<")
                i += 1
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
            elif i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
            elif i < n and text[i] == SUKUN:
                i += 1
            continue

        # --- Taa Marbuta ة ---
        if ch == TAA_MARBUTA:
            i += 1
            # At word end (followed by space, end of text, or pause mark) → 'h'
            # In idafa (followed by diacritics that indicate connection) → 't'
            # Check what follows (skip any diacritics first to check)
            has_tanween = False
            diac_phonemes: list[str] = []
            while i < n and (text[i] in ALL_DIACRITICS or text[i] == SUPERSCRIPT_ALEF or text[i] in SKIP_CHARS):
                if text[i] in TANWEEN_DIACRITICS:
                    has_tanween = True
                    v, nn = TANWEEN_DIACRITICS[text[i]]
                    diac_phonemes = [v, nn]
                elif text[i] in SHORT_VOWEL_DIACRITICS:
                    diac_phonemes = [SHORT_VOWEL_DIACRITICS[text[i]]]
                elif text[i] == SUKUN:
                    pass
                i += 1
            if has_tanween:
                # Tanween on taa marbuta → 't' + tanween vowel + n
                phonemes.append("t")
                phonemes.extend(diac_phonemes)
            elif diac_phonemes:
                # Has a vowel (idafa construct) → 't' + vowel
                phonemes.append("t")
                phonemes.extend(diac_phonemes)
            else:
                # End of word → 'h'
                phonemes.append("h")
            continue

        # --- Alef Maqsura ى → aa ---
        if ch == ALEF_MAQSURA:
            # Typically represents long 'aa' at end of word
            phonemes.append("aa")
            i += 1
            # Consume diacritics
            while i < n and (_is_diacritic(text[i]) or text[i] == SUKUN):
                i += 1
            continue

        # --- Plain Alef ا ---
        if ch == ALEF:
            i += 1
            # Alef after a consonant with fatha is usually part of a long vowel.
            # If preceded by fatha, it was already handled when emitting the consonant.
            # But if alef has its own diacritics (rare in Uthmani), handle:
            if i < n and text[i] == MADDAH:
                # Alef + maddah → < aa (rare, usually written as آ)
                phonemes.append("<")
                phonemes.append("aa")
                i += 1
                continue
            if i < n and text[i] == HAMZA_ABOVE:
                # Alef + combining hamza above
                phonemes.append("<")
                i += 1
                if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                    phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                    i += 1
                elif i < n and text[i] in TANWEEN_DIACRITICS:
                    v, nn = TANWEEN_DIACRITICS[text[i]]
                    phonemes.append(v)
                    phonemes.append(nn)
                    i += 1
                elif i < n and text[i] == SUKUN:
                    i += 1
                continue
            # Plain alef with own diacritics
            if i < n and text[i] in SHORT_VOWEL_DIACRITICS:
                phonemes.append(SHORT_VOWEL_DIACRITICS[text[i]])
                i += 1
                continue
            if i < n and text[i] in TANWEEN_DIACRITICS:
                v, nn = TANWEEN_DIACRITICS[text[i]]
                phonemes.append(v)
                phonemes.append(nn)
                i += 1
                continue
            # Alef as long vowel carrier - check if previous phoneme was 'a'
            # (long vowel already partially emitted by consonant handler)
            if phonemes and phonemes[-1] == "a":
                phonemes[-1] = "aa"
            elif phonemes and phonemes[-1] == "n" and len(phonemes) >= 2 and phonemes[-2] == "a":
                # tanween fathatan + alef: the 'a n' is already there, alef is silent
                pass
            else:
                # Bare alef (e.g., at word start) - typically silent carrier
                pass
            # Consume any remaining diacritics
            while i < n and (_is_diacritic(text[i]) or text[i] == SUKUN):
                i += 1
            continue

        # --- Superscript Alef ٰ (dagger alef) → aa ---
        if ch == SUPERSCRIPT_ALEF:
            # Replace previous short 'a' with long 'aa' if present
            if phonemes and phonemes[-1] == "a":
                phonemes[-1] = "aa"
            else:
                phonemes.append("aa")
            i += 1
            continue

        # --- Standard consonant ---
        if ch in CONSONANT_MAP:
            cons = CONSONANT_MAP[ch]
            i += 1

            # Collect all diacritics on this consonant (order varies in Unicode)
            has_shadda = False
            has_sukun = False
            has_superscript_alef = False
            vowel_diac = None       # short vowel char if present
            tanween_diac = None     # tanween char if present

            while i < n:
                c = text[i]
                if c == SHADDA:
                    has_shadda = True
                    i += 1
                elif c == SUKUN:
                    has_sukun = True
                    i += 1
                elif c == SUPERSCRIPT_ALEF:
                    has_superscript_alef = True
                    i += 1
                elif c in SHORT_VOWEL_DIACRITICS:
                    vowel_diac = c
                    i += 1
                elif c in TANWEEN_DIACRITICS:
                    tanween_diac = c
                    i += 1
                elif c in SKIP_CHARS:
                    i += 1
                elif c == MADDAH or c == HAMZA_ABOVE:
                    break  # these are handled separately
                else:
                    break

            # Emit consonant (geminated if shadda)
            if has_shadda:
                phonemes.append(cons)
                phonemes.append(cons)
            else:
                phonemes.append(cons)

            # Emit vowel
            if has_superscript_alef:
                phonemes.append("aa")
            elif vowel_diac is not None:
                vowel = SHORT_VOWEL_DIACRITICS[vowel_diac]
                phonemes.append(vowel)

                # Check for long vowel: fatha+alef, damma+waw, kasra+ya
                if vowel == "a" and i < n and text[i] in (ALEF, ALEF_MAQSURA):
                    phonemes[-1] = "aa"
                    i += 1
                    # Skip any diacritics on the alef
                    while i < n and (_is_diacritic(text[i]) or text[i] == SUKUN or text[i] in SKIP_CHARS):
                        i += 1
                elif vowel == "a" and i < n and text[i] == SUPERSCRIPT_ALEF:
                    phonemes[-1] = "aa"
                    i += 1
                elif vowel == "u" and i < n and text[i] == WAW:
                    # Check if waw is a long vowel
                    wi = i + 1
                    if wi < n and text[wi] == SUKUN:
                        phonemes[-1] = "uu"
                        i = wi + 1
                    elif wi < n and text[wi] == SHADDA:
                        pass  # waw is a consonant with shadda
                    elif wi < n and text[wi] in SHORT_VOWEL_DIACRITICS:
                        pass  # waw has its own vowel - it's a consonant
                    elif wi < n and text[wi] in TANWEEN_DIACRITICS:
                        pass
                    elif wi >= n or text[wi] == " " or text[wi] in SKIP_CHARS:
                        phonemes[-1] = "uu"
                        i = wi
                    else:
                        phonemes[-1] = "uu"
                        i = wi
                elif vowel == "i" and i < n and text[i] == YA:
                    yi = i + 1
                    if yi < n and text[yi] == SUKUN:
                        phonemes[-1] = "ii"
                        i = yi + 1
                    elif yi < n and text[yi] == SHADDA:
                        pass
                    elif yi < n and text[yi] in SHORT_VOWEL_DIACRITICS:
                        pass
                    elif yi < n and text[yi] in TANWEEN_DIACRITICS:
                        pass
                    elif yi >= n or text[yi] == " " or text[yi] in SKIP_CHARS:
                        phonemes[-1] = "ii"
                        i = yi
                    else:
                        phonemes[-1] = "ii"
                        i = yi

            elif tanween_diac is not None:
                v, nn = TANWEEN_DIACRITICS[tanween_diac]
                phonemes.append(v)
                phonemes.append(nn)
                # Tanween fathatan is often followed by alef
                if v == "a" and i < n and text[i] == ALEF:
                    i += 1  # silent alef after fathatan

            # elif has_sukun: no vowel emitted (already consumed)

            # Skip remaining annotation marks
            while i < n and text[i] in SKIP_CHARS:
                i += 1

            continue

        # --- Combining maddah above (when not on alef) ---
        if ch == MADDAH:
            # Maddah appearing as combining mark
            phonemes.append("aa")
            i += 1
            continue

        # --- Combining hamza above ---
        if ch == HAMZA_ABOVE:
            phonemes.append("<")
            i += 1
            continue

        # --- Unknown character: skip with warning tracking ---
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            # Combining mark we don't handle - skip silently
            i += 1
            continue

        # Non-combining unknown char
        i += 1

    # Clean up: remove trailing word boundary
    while phonemes and phonemes[-1] == "|":
        phonemes.pop()

    return " ".join(phonemes)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quran_path = os.path.join(project_root, "data", "quran.json")
    output_path = os.path.join(project_root, "data", "quran_phonemes.json")

    if not os.path.exists(quran_path):
        print(f"ERROR: {quran_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {quran_path}...")
    with open(quran_path, "r", encoding="utf-8") as f:
        quran = json.load(f)

    print(f"Processing {len(quran)} verses...")

    results = []
    warnings: dict[str, int] = {}  # track unhandled characters
    empty_count = 0

    for idx, verse in enumerate(quran):
        surah = verse["surah"]
        ayah = verse["ayah"]
        text = verse["text_uthmani"]

        phonemes = arabic_to_phonemes(text)

        if not phonemes.strip():
            empty_count += 1
            print(f"  WARNING: Empty phonemes for {surah}:{ayah}")

        results.append({
            "surah": surah,
            "ayah": ayah,
            "phonemes": phonemes,
        })

        # Progress every 500 verses
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(quran)} verses...")

    # Write output
    print(f"Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=None)

    # Stats
    total_phonemes = sum(len(r["phonemes"].split()) for r in results)
    avg_phonemes = total_phonemes / len(results) if results else 0

    print(f"\nDone!")
    print(f"  Total verses: {len(results)}")
    print(f"  Total phonemes: {total_phonemes}")
    print(f"  Average phonemes per verse: {avg_phonemes:.1f}")
    if empty_count:
        print(f"  Verses with empty phonemes: {empty_count}")

    # Show a few examples
    print(f"\nExamples:")
    for i in [0, 1, 5]:
        r = results[i]
        print(f"  {r['surah']}:{r['ayah']}: {r['phonemes'][:80]}...")


if __name__ == "__main__":
    main()
