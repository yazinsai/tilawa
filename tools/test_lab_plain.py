"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_plain.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_plain as P  # noqa: E402

LEFT_OUT = {
    "id": "lab-v1.5-e02-x5", "text": "أَعْلَمُ تَعْمَلُونَ فَكَذَّبُوهُ", "audio": {"path": "x.wav", "startS": 176.2, "endS": 182.0},
    "label": {"type": "omitted_word", "expected": "بِمَا", "said": ""},
    "context": "Nemotron: … · A word left out (joined mid-phrase, no pause) · 26:188, word 4 · session lab-v1.5-e02 at 178.9 s · Ibrahim Al-Asiri",
    "triage": {"reasons": ["disagree"], "class": "omitted_word", "alternative": "أَعْلَمُ بِمَا تَعْمَلُونَ فَكَذَّبُوهُ",
               "alternativeOp": "omitted word put back", "nemotron": {"decision": "mismatch", "diff": -2.7},
               "soniox": {"decision": "match", "diff": 2, "need": 2, "needAgainst": 2, "unit": "letters"}},
}


def test_a_left_out_word_asks_what_to_listen_for_and_marks_the_gap():
    p = P.plain(LEFT_OUT)
    assert p["ask"] == "Listen for: «بِمَا» should be missing between «أَعْلَمُ» and «تَعْمَلُونَ»."
    assert p["marks"] == [{"kind": "gap", "at": 1, "words": "بِمَا"}]
    assert p["tag"] == "The two computer checks disagree"
    assert p["where"] == "Ash-Shuara 26:188, 4th word · about 2.7 s into the clip · the edit is cut into flowing speech"
    assert p["footer"] == "Reciter: Ibrahim Al-Asiri · session lab-v1.5-e02 · item lab-v1.5-e02-x5"
    assert [c["text"] for c in p["checks"]] == [
        "Check 1 (matches both versions against the sound): hears it with «بِمَا» said (weak)",
        "Check 2 (writes down what it hears, without seeing our text): wrote down the version as described (just enough)"]
    assert not any(w in str(p) for w in ("nats", "label", "alternative", "Nemotron", "Soniox"))


def test_wrong_added_repeat_and_clean_items():
    wrong = {"text": "قَالَ يَـٰقَوْمِ كَلَّا", "label": {"type": "wrong_word", "expected": "فَيَكُونُ", "said": "كَلَّا"},
             "triage": {"reasons": ["seam"], "class": "mutashabihat", "alternative": "قَالَ يَـٰقَوْمِ فَيَكُونُ"}}
    p = P.plain(wrong)
    assert p["ask"] == "Listen for: «كَلَّا» said where the mushaf has «فَيَكُونُ». It's the wording of a look-alike ayah."
    assert p["marks"] == [{"kind": "mark", "from": 2, "to": 3, "words": "كَلَّا"}] and p["seam"] == P.SEAM
    added = {"text": "ٱلْمُسْرِفِينَ سَلِيمٍ ٱلَّذِينَ", "label": {"type": "added_word", "said": "سَلِيمٍ"}, "triage": {"reasons": ["spot"]}}
    assert P.plain(added)["ask"] == "Listen for: an extra «سَلِيمٍ» after «ٱلْمُسْرِفِينَ»."  # no alternative: the said word found in the text
    rep = {"text": "أَمَدَّكُم بِأَنْعَـٰمٍ أَمَدَّكُم بِأَنْعَـٰمٍ وَبَنِينَ", "label": {"type": "repeat", "said": "أَمَدَّكُم بِأَنْعَـٰمٍ"},
           "triage": {"reasons": ["unsure"], "alternative": "أَمَدَّكُم بِأَنْعَـٰمٍ وَبَنِينَ", "nemotron": {"decision": "mismatch", "diff": -30.0},
                      "soniox": {"decision": "unsure", "diff": 0, "need": 2}}}
    p = P.plain(rep)
    assert p["ask"].endswith("said twice. That's allowed, not a mistake.") and p["marks"][0]["kind"] == "mark"
    assert p["tag"] == "One computer check says this is wrong, the other can't tell"
    assert p["checks"][0]["text"].endswith("hears it said only once (strong)") and p["checks"][1]["text"].endswith("could be either version")
    clean = {"text": "بِهِۦ قَالُوا۟", "label": {"type": "clean"}, "context": "… · session lab-v1.5-p30 at 12.0 s · Someone",
             "triage": {"reasons": ["unsure"], "alternativeOp": "بِهِۦ claimed as بِحَرْبٍۢ", "nemotron": {"decision": "mismatch", "diff": -3.0}}}
    p = P.plain(clean)
    assert p["marks"] == [] and p["ask"].startswith("Listen for: every word said as written")
    assert p["checks"][0]["text"].endswith("hears it as a made-up test version with «بِحَرْبٍۢ» in place of «بِهِۦ» (weak)")
    assert p["footer"].startswith("Reciter: Someone")


def test_ordinals_and_surahs():
    assert [P.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st"]
    assert len(P.SURAHS) == 114 and P.surah(26) == "Ash-Shuara" and P.surah(200) == "Surah 200"
