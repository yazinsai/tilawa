import pytest

from shared.leak_guard import (
    HeldoutClip,
    LeakError,
    assert_no_leaks,
    build_twin_index,
    match_heldout_reciter,
    scan_cuts,
)


def _cut(cid, speaker, surah=1, ayah=1, dur=5.0, ayah_end=None, **custom):
    return {
        "id": cid,
        "duration": dur,
        "supervisions": [
            {
                "speaker": speaker,
                "custom": {"surah": surah, "ayah": ayah, "ayah_end": ayah_end or ayah, **custom},
            }
        ],
        "recording": {"sources": [{"source": f"/vol/audio/x/{cid}.flac"}]},
    }


@pytest.mark.parametrize(
    "name,rid",
    [
        ("Sahl_Yassin_128kbps", "sahl_yassin"),
        ("Akram_AlAlaqimy_128kbps", "akram_alalaqimy"),
        ("Muhsin_Al_Qasim_192kbps", "muhsin_al_qasim"),
        ("Mohsen Al-Qasim", "muhsin_al_qasim"),
        ("محسن القاسم", "muhsin_al_qasim"),
        ("أكرم العلاقمي", "akram_alalaqimy"),
        ("abdullah_al_nufais_qul", "alnufais"),
        ("Alafasy_128kbps", None),
        ("Husary_128kbps", None),
        ("Yasser_Ad-Dussary_128kbps", None),
        ("Sahl", None),
    ],
)
def test_match_heldout_reciter(name, rid):
    assert match_heldout_reciter(name) == rid


def test_scan_flags_reciter_in_speaker_and_custom():
    cuts = [
        _cut("a", "Alafasy_128kbps"),
        _cut("b", "Sahl_Yassin_128kbps"),
        _cut("c", "some mushaf", slug="muhsin-al-qasim-hafs"),
    ]
    rep = scan_cuts("everyayah", cuts)
    assert rep.n_cuts == 3
    assert rep.flagged_ids == {"b", "c"}
    assert rep.reciter_hits == {"sahl_yassin": 1, "muhsin_al_qasim": 1}


def test_twins_match_single_ayah_within_tolerance_only():
    twins = build_twin_index([HeldoutClip("tlog_holdout", "t1", 38, 73, 4.20)])
    cuts = [
        _cut("near", "tlog", 38, 73, 4.23),
        _cut("far", "tlog", 38, 73, 4.40),
        _cut("other_ayah", "tlog", 38, 74, 4.20),
        _cut("span", "tlog", 38, 73, 4.20, ayah_end=74),
    ]
    rep = scan_cuts("tlog", cuts, twins=twins)
    assert rep.flagged_ids == {"near"}
    assert rep.twin_hits == {"tlog_holdout": 1}


def test_check_manifests_routes_twins_by_source_prefix(tmp_path):
    import gzip
    import json

    from shared.leak_guard import check_manifests

    def write(src, cuts):
        with gzip.open(tmp_path / f"{src}_cuts_fbank.jsonl.gz", "wt") as f:
            for c in cuts:
                f.write(json.dumps(c) + "\n")

    write("everyayah_rx", [_cut("e1", "Alafasy_128kbps", 2, 5, 7.0)])
    write("tlog_clean_v3", [_cut("t1", "tlog", 2, 5, 7.0)])
    write("qua_rx", [_cut("q1", "x", 2, 5, 7.0)])
    clips = [HeldoutClip("tlog_holdout", "h", 2, 5, 7.0)]
    reps = {r.source: r for r in check_manifests(["everyayah_rx", "tlog_clean_v3", "qua_rx"], tmp_path, clips)}
    assert reps["everyayah_rx"].n_flagged == 0
    assert reps["tlog_clean_v3"].flagged_ids == {"t1"}
    assert reps["qua_rx"].n_flagged == 0


def test_assert_no_leaks():
    clean = scan_cuts("everyayah", [_cut("a", "Alafasy_128kbps")])
    assert_no_leaks([clean])
    dirty = scan_cuts("everyayah", [_cut("b", "Akram_AlAlaqimy_128kbps")])
    with pytest.raises(LeakError, match="akram_alalaqimy"):
        assert_no_leaks([clean, dirty])
    twin = scan_cuts(
        "tlog", [_cut("t", "tlog", 1, 2, 3.0)], twins=build_twin_index([HeldoutClip("tlog_holdout", "h", 1, 2, 3.0)])
    )
    with pytest.raises(LeakError):
        assert_no_leaks([twin])
    assert_no_leaks([twin], allow_twins=True)


def test_perturbed_copies_follow_their_base_cut():
    from shared.leak_guard import base_cut_id

    assert base_cut_id("tlog_00000064_100_10_sp0.9") == "tlog_00000064_100_10"
    assert base_cut_id("tlog_00000064_100_10") == "tlog_00000064_100_10"
    twins = build_twin_index([HeldoutClip("tlog_holdout", "h", 1, 2, 3.0)])
    cuts = [_cut("t", "tlog", 1, 2, 3.0), _cut("t_sp0.9", "tlog", 1, 2, 3.0 / 0.9), _cut("t_sp1.1", "tlog", 1, 2, 3.0 / 1.1)]
    rep = scan_cuts("tlog", cuts, twins=twins)
    assert rep.flagged_ids == {"t"}
    assert all(rep.is_flagged(c["id"]) for c in cuts)


def test_waqf2_rewrites_only_ayah_final_four_beat_madd():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("p0", Path(__file__).parent.parent / "scripts" / "phase0_modal.py")
    p0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0)
    assert p0.waqf2("ررَحِۦۦۦۦم") == "ررَحِۦۦم"
    assert p0.waqf2("ثَمُۥۥۥۥدڇ") == "ثَمُۥۥدڇ"
    assert p0.waqf2("ۦۦۦۦنَاا") == "ۦۦۦۦنَاا"
    assert p0.waqf2("نَسۡتَعِۦۦۦۦ") == "نَسۡتَعِۦۦ"
    assert p0.waqf2("يُوعَدُۥۥۥۥ") == "يُوعَدُۥۥ"
    assert p0.waqf2("مَآاااااا") == "مَآاااااا"
    assert p0.waqf2("مَااا") == "مَااا"
