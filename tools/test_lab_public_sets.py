# /// script
# requires-python = ">=3.12"
# dependencies = ["modal", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with modal --with pytest pytest tools/test_lab_public_sets.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modal"))
from lab_public_sets import dedup, error_class, ikhlas_cross_issues, ikhlas_file, norm, parse_ikhlas_row, phone_edits, repair_ikhlas_files, retasy_annotators, retasy_label, retasy_quranic, split_list  # noqa: E402


def row(i=1, verse=1, file="ID1V1F.wav", flag=1, typ="تجويد", loc="أَحَدٌ", expl="خطأ في قلقلة حرف الدال", code=5):
    return {"ID": i, "Surah name": "Surah Al-Ikhlas", "Verse number": verse, "Verse location": file, "True or False": flag, "Error type": typ, "Error location": loc, "Error explanation": expl, "Error number": code}


CORRECT = dict(flag=0, typ=0, loc=0, expl=0, code=0)


def test_norm_strips_marks_and_unifies_alef():
    assert norm("ٱللَّهُ") == norm("اللَّهُ") == "الله"
    assert norm("أَحَدٌۢ") == norm(" أَحَدٌ") == "احد"
    assert norm("لَّهُۥ") == "له"


def test_split_list_handles_both_commas_and_zero():
    assert split_list("كُفُوًا, أَحَدٌۢ ") == ["كُفُوًا", "أَحَدٌۢ"]
    assert split_list("نطق غير سليم في مخرج حرف الهاء، خطأ في قلقلة حرف الدال") == ["نطق غير سليم في مخرج حرف الهاء", "خطأ في قلقلة حرف الدال"]
    assert split_list(0) == split_list(None) == split_list("  ") == []


def test_error_class():
    assert error_class("تجويد ") == "tajweed"
    assert error_class("حركات وتجويد") == "harakat+tajweed"
    assert error_class("حفظ") == "other:حفظ"


def test_filename():
    assert ikhlas_file("ID105V2T.wav") == (105, 2, True)
    assert ikhlas_file("ID1V1F.wav") == (1, 1, False)
    assert ikhlas_file("x.wav") is None


def test_clean_error_and_correct_rows_have_no_issues():
    e = parse_ikhlas_row(row())
    assert e["label"] == "error" and e["error_classes"] == ["tajweed"] and e["error_words"] == ["أَحَدٌ"] and e["error_code"] == 5 and e["speaker"] == 1
    assert e["issues"] == []
    c = parse_ikhlas_row(row(file="ID3V1T.wav", **CORRECT))
    assert c["label"] == "correct" and c["error_classes"] == [] and c["error_code"] is None and c["issues"] == []


def test_flag_contradicting_filename_without_fields():
    it = parse_ikhlas_row(row(i=26, file="ID26V1T.wav", **{**CORRECT, "flag": 1}))
    assert "flag_vs_filename:error/T" in it["issues"]
    assert "error_missing:type,location,explanation" in it["issues"]


def test_correct_with_fields_and_verse_mismatch():
    it = parse_ikhlas_row(row(verse=2, file="ID1V1T.wav", flag=0))
    assert "verse_vs_filename:2/1" in it["issues"] and "correct_with_error_fields" in it["issues"]


def test_locations_checked_against_the_rows_verse():
    assert parse_ikhlas_row(row(verse=4, file="ID9V4F.wav", typ="تجويد, تجويد", loc="لَّهُ, كُفُوًا"))["issues"] == []
    it = parse_ikhlas_row(row(verse=1, typ="تجويد, تجويد", loc="ٱلصَّمَدُ, قلب"))
    assert "location_other_verse:ٱلصَّمَدُ" in it["issues"] and "location_not_in_surah:قلب" in it["issues"]


def test_types_vs_locations_count():
    assert "types_vs_locations:1/2" in parse_ikhlas_row(row(verse=3, file="ID1V3F.wav", loc="يَلِدْ, يُولَدْ"))["issues"]


def test_dedup_picks_first_sorted_id():
    assert dedup({"b": "h1", "a": "h1", "c": "h2", "d": "h1"}) == {"b": "a", "d": "a"}
    assert dedup({"a": "1", "b": "2"}) == {}


def test_cross_issues():
    a = parse_ikhlas_row(row(i=1, file="ID1V1F.wav"))
    b = parse_ikhlas_row(row(i=2, file="ID1V1F.wav", flag=1, typ="حركات"))
    c = parse_ikhlas_row(row(i=3, file="ID3V1T.wav", **CORRECT))
    d = parse_ikhlas_row(row(i=4, file="ID4V1F.wav"))
    out = ikhlas_cross_issues([a, b, c, d], {"ID1V1F.wav": "x", "ID3V1T.wav": "y", "ID4V1F.wav": "y", "ID9V1T.wav": "z"})
    kinds = {(o["kind"], o.get("file") or tuple(o.get("files", []))): o for o in out}
    assert kinds[("duplicate_filename_rows", "ID1V1F.wav")]["conflicting"] is True
    assert ("audio_without_row", "ID9V1T.wav") in kinds
    dup = kinds[("duplicate_audio", ("ID3V1T.wav", "ID4V1F.wav"))]
    assert dup["ids"] == ["ikhlas-0003", "ikhlas-0004"] and dup["conflicting"] is True


def test_repair_points_duplicate_rows_at_the_unreferenced_file_their_position_implies():
    items = [parse_ikhlas_row(row(i=k, verse=4, file=f, **CORRECT)) for k, f in enumerate(["ID65V4T.wav", "ID266V4T.wav", "ID67V4T.wav", "ID265V4T.wav", "ID266V4T.wav"], 1)]
    files = ["ID65V4T.wav", "ID66V4T.wav", "ID67V4T.wav", "ID265V4T.wav", "ID266V4T.wav"]
    assert repair_ikhlas_files(items, files) == [{"id": "ikhlas-0002", "sheet_file": "ID266V4T.wav", "file": "ID66V4T.wav"}]
    assert [i["file"] for i in items] == files and items[1]["speaker"] == 66 and items[4]["issues"] == []


def test_repair_skips_when_flag_or_verse_would_change():
    items = [parse_ikhlas_row(row(i=k, verse=2, file=f, **CORRECT)) for k, f in enumerate(["ID41V2T.wav", "ID41V2T.wav"], 1)]
    assert repair_ikhlas_files(items, ["ID41V2T.wav", "ID42V2F.wav"]) == []


def test_retasy_quranic():
    assert retasy_quranic("Al-Faatihah", "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ")
    assert not retasy_quranic("Al-Faatihah", "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيْمِ")
    assert not retasy_quranic("Adhan", "اللَّهُ أَكْبَرُ")
    assert retasy_quranic("Dua from the Quran", "رَبَّنَا آتِنَا")


def test_retasy_labels():
    assert [retasy_label(x) for x in ("correct", "in_correct", "in_complete", "not_match_aya", "multiple_aya", "not_related_quran", None, "")] == [
        "correct", "incorrect_pronunciation", "incomplete_verse", "wrong_verse", "multiple_verses", "unrelated_or_empty", "unlabeled", "unlabeled"]
    assert retasy_label("weird") == "other:weird"


def test_retasy_annotators():
    meta = json.dumps({"label_1": "in_correct", "annotator1_id": "1", "annotator1_MCC": "0.87", "label_2": "correct", "annotator2_id": "10", "annotator2_MCC": "0.52"})
    assert retasy_annotators(meta) == [{"label": "incorrect_pronunciation", "annotator": "1", "mcc": "0.87"}, {"label": "correct", "annotator": "10", "mcc": "0.52"}]
    assert retasy_annotators("{}") == retasy_annotators(None) == retasy_annotators("not json") == []


def test_phone_edits():
    assert phone_edits(list("abcd"), list("abcd")) == {"sub": 0, "del": 0, "ins": 0}
    assert phone_edits(list("abcd"), list("axcd")) == {"sub": 1, "del": 0, "ins": 0}
    assert phone_edits(list("abcd"), list("acd")) == {"sub": 0, "del": 1, "ins": 0}
    assert phone_edits(list("abcd"), list("abxcd")) == {"sub": 0, "del": 0, "ins": 1}
