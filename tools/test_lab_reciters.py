# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with rapidfuzz==3.14.1 --with pytest pytest tools/test_lab_reciters.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_reciters import ar_tokens, en_tokens, same_person, split  # noqa: E402


def test_arabic_tokens_drop_kunya_article_and_join_abd():
    assert ar_tokens("علي بن عبدالرحمن الحذيفي") == ["علي", "عبدالرحمن", "حذيفي"]
    assert ar_tokens("عبد الباسط عبد الصمد") == ar_tokens("عبدالباسط عبدالصمد")
    assert ar_tokens("بندر بليلة") == ar_tokens("بندر بليله")
    assert ar_tokens("شيخ أبو بكر الشاطري") == ar_tokens("أبو بكر الشاطري")


def test_english_tokens_ignore_articles_spelling_and_labels():
    assert en_tokens("Hani Arrifai") == en_tokens("Hani Ar Rifai (Murattal)")
    assert en_tokens("Abdul Aziz Al-Ahmad") == en_tokens("AbdulAzeez al-Ahmad (Murattal)")
    assert en_tokens("Mohammed Ayyub") == en_tokens("Mohammed Ayoub")


@pytest.mark.parametrize("a,b", [
    ("علي بن عبدالرحمن الحذيفي", "علي الحذيفي"),
    ("أحمد عامر", "أحمد محمد عامر"),
    ("أحمد عيسى المعصراوي", "أحمد عيسى المعصرواوي"),
    ("مشاري العفاسي", "مشاري بن راشد العفاسي"),
])
def test_arabic_same_person(a, b):
    assert same_person(ar_tokens(a), ar_tokens(b))


@pytest.mark.parametrize("a,b", [
    ("محمد خليل القارئ", "محمود خليل الحصري"),
    ("هاني الرفاعي", "نبيل الرفاعي"),
    ("عبدالرحمن السويّد", "عبدالرحمن السديس"),
    ("ياسر الدوسري", "إبراهيم الدوسري"),
])
def test_arabic_different_people(a, b):
    assert same_person(ar_tokens(a), ar_tokens(b)) is None


@pytest.mark.parametrize("a,b,how", [
    ("Saud Al-Shuraim", "Saood bin Ibraaheem Ash-Shuraym", "first+last"),
    ("Ali Alhuthaifi", "Ali Al-Hudhaify", "first+last"),
    ("Abdullah Al-Johany", "Abdullaah 3awwaad Al-Juhaynee", "first+last"),
    ("Abdelbari Al-Toubayti", "AbdulBari ath-Thubaity (Murattal)", "first+last"),
    ("Mishary Alafasi", "Alafasy", "surname"),
    ("Saad Al-Ghamdi", "Ghamadi", "surname"),
])
def test_english_same_person(a, b, how):
    m = same_person(en_tokens(a), en_tokens(b))
    assert m and m[0] == how


@pytest.mark.parametrize("a,b", [
    ("Mohammad Khalil Al-Qari", "Mahmoud Khalil Al-Husari"),
    ("Sayed Ahmad Hashemi", "Ahmed Saud"),
    ("Alzain Mohammad Ahmad", "Mohamed Salem Ould Zine (Murattal)"),
    ("Nabil Al Rifay", "Hani Ar Rifai (Murattal)"),
    ("Abdulbari Mohammad", "AbdulBari ath-Thubaity (Murattal)"),
])
def test_english_different_people(a, b):
    assert same_person(en_tokens(a), en_tokens(b)) is None


def reciters(n):
    return [{"readId": i, "speedRatio": 0.8 + i / 50, "kbps": 64 if i % 4 == 0 else 128} for i in range(n)]


def test_split_is_seeded_and_balanced():
    a, b = reciters(13), reciters(13)
    split(a, 7)
    split(b, 7)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    assert sum(r["split"] == "practice" for r in a) == 7
    low = [r["split"] for r in a if r["kbps"] < 110]
    assert abs(low.count("practice") - low.count("exam")) <= 1
