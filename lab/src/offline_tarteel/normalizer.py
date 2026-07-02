import re

# Arabic diacritic Unicode ranges
_DIACRITICS = re.compile(
    '[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
    '\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]'
)

# Character normalization map
_NORM_MAP = str.maketrans({
    '\u0623': '\u0627',  # أ -> ا
    '\u0625': '\u0627',  # إ -> ا
    '\u0622': '\u0627',  # آ -> ا
    '\u0671': '\u0627',  # ٱ -> ا
    '\u0629': '\u0647',  # ة -> ه
    '\u0649': '\u064A',  # ى -> ي
})


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub('', text)
    text = text.translate(_NORM_MAP)
    text = ' '.join(text.split())
    return text
