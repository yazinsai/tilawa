"""Arabic text normalization for Quranic ASR matching.

Ported from https://github.com/yazinsai/arabic-text-normalizer (TypeScript).
Preserves hamza/alef-maqsura/teh-marbuta by default; opt into
`strip_hamza=True` for aggressive Uthmani <-> common Arabic matching.
"""

import re

# Diacritics (tashkeel/harakat, U+064B-U+065F)
_DIACRITICS = re.compile('[\u064B-\u065F]')
_ALIF_MADDA = '\u0622'           # آ
_ALIF_WASLA = '\u0671'           # ٱ
_ALIF_VARIANTS = re.compile('[\u0672\u0673]')  # ٲ ٳ
_ALEF_ALEF_KHANJARIYA = re.compile('\u0627\u0670')  # اٰ -> ا
_SUPERSCRIPT_ALIF = '\u0670'     # ٰ
_FARSI_YEH = re.compile('[\u06CC\u06D2]')  # ی ے
_FARSI_KAF = '\u06A9'            # ک

# Quranic markers + small letters (U+06D6-U+06ED) — sajdah, rub el hizb, etc.
_QURANIC_ANNOTATIONS = re.compile('[\u06D6-\u06ED]')

# Ornate parens + Arabic-Indic / Extended Arabic-Indic digits
_ORNATE_PARENS = re.compile('[\uFD3E\uFD3F]')
_ARABIC_DIGITS = re.compile('[\u0660-\u0669\u06F0-\u06F9]')

# Tatweel
_TATWEEL = '\u0640'

# Punctuation
_PUNCTUATION = re.compile('[.,;:!?\u2026\u060C\u061B\u061F]')

# stripHamza: aggressive Uthmani <-> common Arabic matching
_HAMZA_TO_STRIP = re.compile('[\u0621\u0623\u0625\u0626]')  # ء أ إ ئ
_ALEF_MAQSURA = '\u0649'  # ى
_UTHMANI_WAW_TA = re.compile('\u0648\u0627?\u0629')  # وة/واة -> اة
_DOUBLE_YA = re.compile('\u064A\u064A')  # يي -> ي
_SAD_SIN_VARIANTS = [
    (re.compile('\u0628\u0635\u0637'), '\u0628\u0633\u0637'),  # بصط -> بسط
    (re.compile('\u0635\u064A\u0637\u0631'), '\u0633\u064A\u0637\u0631'),  # صيطر -> يسيطر ->سيطر
]
_DEF_LAM_COLLAPSE = re.compile('\u0627\u0644\u0644')  # الال -> ال (after stripping hamza)


def normalize_arabic(
    text: str,
    diacritics: bool = True,
    markers: bool = True,
    verse_numbers: bool = True,
    tatweel: bool = True,
    small_letters: bool = True,
    punctuation: bool = True,
    collapse_whitespace: bool = True,
    strip_hamza: bool = False,
) -> str:
    """Normalize Arabic/Quranic text. See yazinsai/arabic-text-normalizer for spec."""
    text = str(text).replace('\uFEFF', '').replace('\u200F', '').replace('\u200E', '')

    if diacritics:
        text = _DIACRITICS.sub('', text)
        text = text.replace(_ALIF_MADDA, '\u0627')
        text = text.replace(_ALIF_WASLA, '\u0627')
        text = _ALIF_VARIANTS.sub('\u0627', text)
        text = _ALEF_ALEF_KHANJARIYA.sub('\u0627', text)
        text = text.replace(_SUPERSCRIPT_ALIF, '\u0627')
        text = _FARSI_YEH.sub('\u064A', text)
        text = text.replace(_FARSI_KAF, '\u0643')

    if markers or small_letters:
        text = _QURANIC_ANNOTATIONS.sub('', text)

    if verse_numbers:
        text = _ORNATE_PARENS.sub('', text)
        text = _ARABIC_DIGITS.sub('', text)

    if tatweel:
        text = text.replace(_TATWEEL, '')

    if punctuation:
        text = _PUNCTUATION.sub('', text)

    if strip_hamza:
        text = _HAMZA_TO_STRIP.sub('', text)
        text = text.replace(_ALEF_MAQSURA, '\u064A')
        text = _UTHMANI_WAW_TA.sub('\u0627\u0629', text)
        text = _DOUBLE_YA.sub('\u064A', text)
        for pat, rep in _SAD_SIN_VARIANTS:
            text = pat.sub(rep, text)
        text = _DEF_LAM_COLLAPSE.sub('\u0627\u0644', text)

    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    return text
