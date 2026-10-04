"""Deterministic helpers for Zipformer CTC data staging (no Modal, no lhotse)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MOD_PATH = ROOT / "scripts" / "prepare_zipformer_data_modal.py"
_SPEC = importlib.util.spec_from_file_location("prepare_zipformer_data_modal", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

TOKENS_TXT = ROOT / "experiments" / "zipformer-ctc" / "tokens.txt"


def test_parse_qul_and_tlog_filenames():
    assert prep.parse_surah_ayah_filename("1_1.wav") == (1, 1)
    assert prep.parse_surah_ayah_filename("002_255.wav") == (2, 255)
    assert prep.parse_surah_ayah_filename("18_10_abc123.wav") == (18, 10)
    assert prep.parse_surah_ayah_filename("audio/tlog_holdout/3_4_xyz.flac") == (3, 4)
    assert prep.parse_surah_ayah_filename("hf://tarteel-ai/tlog/1_7_99.wav?download=1") == (1, 7)
    assert prep.parse_surah_ayah_filename("not-a-verse.wav") is None
    assert prep.parse_surah_ayah_filename("") is None


def test_duration_policy():
    assert prep.duration_decision(0.5) == "skip_short"
    assert prep.duration_decision(1.0) == "keep"
    assert prep.duration_decision(20.0) == "keep"
    assert prep.duration_decision(20.01) == "long"
    assert prep.duration_decision(60.0) == "long"
    assert prep.duration_decision(60.01) == "skip_long"


def test_retasy_keep_only_correct():
    assert prep.retasy_keep("correct") is True
    assert prep.retasy_keep("in_correct") is False
    assert prep.retasy_keep("not_related_quran") is False
    assert prep.retasy_keep("not_match_aya") is False
    assert prep.retasy_keep(None) is False
    assert prep.retasy_keep("multiple_aya") is False
    assert prep.BAD_RETASY_LABELS == {"in_correct", "not_related_quran", "not_match_aya"}


def test_tarteel_and_hafs_filters():
    assert prep.is_tarteel_dupe(slug="khalifa_al_tunaiji_tarteel") is True
    assert prep.is_tarteel_dupe(reciter="Tarteel EveryAyah") is True
    assert prep.is_tarteel_dupe(slug="maher_al_muaiqly_qdc") is False
    assert prep.is_hafs_riwayah("Hafs") is True
    assert prep.is_hafs_riwayah("hafs") is True
    assert prep.is_hafs_riwayah("Warsh") is False
    assert prep.is_hafs_riwayah(None) is True


def test_basmala_candidate_ayah1():
    assert prep.basmala_candidate("everyayah", 2, 1) is True
    assert prep.basmala_candidate("tlog", 1, 1) is False
    assert prep.basmala_candidate("tlog", 9, 1) is False
    assert prep.basmala_candidate("everyayah", 2, 2) is False
    assert prep.basmala_candidate("qua", 2, 1, {"recording_context": "Studio"}) is True


def test_license_npl_not_permissive():
    npl = "Quran-Lab No-Profit License\nThe Work is not for sale.\n"
    assert prep.license_is_permissive(npl) is False
    assert prep.license_is_permissive("MIT License\nPermission is hereby granted") is True
    assert prep.license_is_permissive("Creative Commons Attribution 4.0 International") is True


def test_categorize_matches_v3():
    assert prep.categorize_word_count(1) == "short"
    assert prep.categorize_word_count(5) == "short"
    assert prep.categorize_word_count(6) == "medium"
    assert prep.categorize_word_count(15) == "medium"
    assert prep.categorize_word_count(16) == "long"


def test_qlab_reciter_and_flat_name():
    assert prep.qlab_reciter("qul_alnufais") == "alnufais"
    assert prep.qlab_reciter("everyayah_heldout") == "everyayah_heldout"
    assert prep.qlab_reciter("tlog_holdout") == "tlog"
    assert prep.qlab_flat_filename("qul_alnufais", "1_1.wav") == "qul_alnufais__1_1.wav"


def test_parse_sources_csv():
    assert prep.parse_sources("everyayah,retasy") == ["everyayah", "retasy"]
    assert prep.parse_sources("qurantts") == ["qurantts"]
    assert "qurantts" not in prep.DEFAULT_SOURCES
    assert prep.parse_sources("") == list(prep.DEFAULT_SOURCES)
    assert prep.DEFAULT_SOURCES == ("everyayah", "qua", "iqra", "retasy", "tlog")
    with pytest.raises(ValueError):
        prep.parse_sources("nope")


def test_patch_hf_list_feature_registers_list_type():
    pytest.importorskip("datasets")
    prep.patch_hf_list_feature()
    from datasets.features.features import _FEATURE_TYPES

    assert "List" in _FEATURE_TYPES
    prep.patch_hf_list_feature()
    assert "List" in _FEATURE_TYPES


def test_load_tokens_txt_roundtrip():
    tokens = prep.load_token_inventory(TOKENS_TXT)
    assert len(tokens) == 251
    assert tokens[-1] == "<blank>"
    from shared.phoneme_labels import PhonemeTokenizer

    tok = PhonemeTokenizer(tokens)
    ids = tok.encode(tokens[0])
    assert ids == [0]


def test_pick_surah_ayah_column_aliases():
    assert prep.pick_surah_ayah({"sura": 2, "ayah": 255}) == (2, 255)
    assert prep.pick_surah_ayah({"chapter": 1, "verse": 7}) == (1, 7)
    assert prep.pick_surah_ayah({"surah": 114, "ayah": 6}) == (114, 6)
    assert prep.pick_surah_ayah({"text": "nope"}) is None


def test_everyayah_splits_never_test():
    assert prep.EVERYAYAH_SPLITS == ("train", "validation")
    assert "test" not in prep.EVERYAYAH_SPLITS


def test_qlab_exclusion_set_from_fake_manifest():
    samples = [
        {"source": "tlog_holdout", "file": "tlog_holdout__1_2_abc.wav"},
        {"source": "tlog_holdout", "id": "tlog_holdout__3_4_xyz"},
        {"source": "tlog_holdout", "file": "18_10_deadbeef.wav"},
        {"source": "everyayah_heldout", "file": "everyayah_heldout__test-00009-of-00013_332.wav"},
        {"source": "qul_alnufais", "file": "qul_alnufais__2_255.wav"},
    ]
    excl = prep.qlab_exclusions_from_samples(samples)
    assert excl["tlog_ids"] == {"1_2_abc", "3_4_xyz", "18_10_deadbeef"}
    assert prep.tlog_holdout_key("audio/11_90_2170378964.wav") == "11_90_2170378964"
    assert prep.tlog_holdout_key("tlog_holdout__11_90_2170378964.wav") == "11_90_2170378964"
    assert prep.is_nufais_holdout(slug="yasser_al_nufais_qul") is True
    assert prep.is_nufais_holdout(name_en="Mishary Alnufais") is True
    assert prep.is_nufais_holdout(slug="maher_al_muaiqly_qdc") is False
    assert prep.is_heldout_reciter(qari="Sahl_Yassin_128kbps") is True
    assert prep.is_heldout_reciter(qari="Akram_AlAlaqimy_128kbps", reciter=None) is True
    assert prep.is_heldout_reciter(name_en="Muhsin Al-Qasim", slug="x") is True
    assert prep.is_heldout_reciter(slug="yasser_al_nufais_qul") is True
    assert prep.is_heldout_reciter(qari="Alafasy_128kbps") is False


def test_qlab_exclusion_load_tmp_manifest(tmp_path):
    payload = {
        "samples": [
            {"source": "tlog_holdout", "file": "tlog_holdout__5_6_id1.wav"},
            {"source": "qul_alnufais", "file": "qul_alnufais__1_1.wav"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    excl = prep.load_qlab_exclusions(path)
    assert excl["tlog_ids"] == {"5_6_id1"}


def test_existing_flac_skip_path(tmp_path):
    import numpy as np
    import soundfile as sf

    audio_root = tmp_path / "audio"
    clip_id = prep.make_clip_id("iqra", 0, 1, 1)
    path = prep.flac_clip_path("iqra", clip_id, audio_root=audio_root)
    path.parent.mkdir(parents=True)
    wav = np.zeros(16000, dtype=np.float32)
    wav[100:200] = 0.05
    sf.write(str(path), wav, 16000, format="FLAC")
    dur = prep.existing_flac_duration(path)
    assert dur is not None
    assert abs(dur - 1.0) < 0.05

    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")
    assert prep.existing_flac_duration(empty) is None
    assert prep.existing_flac_duration(tmp_path / "missing.flac") is None

    stats = prep._empty_stats("iqra")
    decoded, reused_dur = prep._audio_for_clip(
        source="iqra",
        idx=0,
        surah=1,
        ayah=1,
        audio_obj={"array": np.ones(8000, dtype=np.float32), "sampling_rate": 16000},
        force=False,
        stats=stats,
        audio_root=audio_root,
    )
    assert decoded is None
    assert abs(reused_dur - 1.0) < 0.05
    assert stats.get("skipped", {}).get("audio_error") is None


def test_iqra_keep_flags_threshold():
    flags = prep.iqra_row_keep_flags(
        {
            "sentence_match": 0.4,
            "tashkeel_match": 0.7,
            "sentence_search": 0.96,
            "tashkeel_search": 0.5,
            "sentence_hamza": 0.2,
            "tashkeel_hamza": 0.94,
        }
    )
    assert flags["baseline"] is False
    assert flags["search"] is True
    assert flags["hamza"] is False
    assert flags["baseline_score"] == 0.7
    assert flags["search_score"] == 0.96


def test_skip_hf_stream_uses_skip_when_present():
    class _DS:
        def skip(self, n):
            self.n = n
            return self

    ds = _DS()
    out = prep.skip_hf_stream(ds, 12, "tlog")
    assert out is ds
    assert ds.n == 12
    assert prep.skip_hf_stream(ds, 0) is ds
    assert prep.skip_hf_stream([1, 2, 3], 5) == [1, 2, 3]


def test_restore_partial_returns_prefix_cuts_and_hours(tmp_path):
    source = "tlog"
    cuts = [
        {"id": "tlog_00000000_1_1", "duration": 2.0},
        {"id": "tlog_00000001_1_2", "duration": 3.5},
        {"id": "tlog_00000002_1_3", "duration": 1.5},
    ]
    partial = prep.partial_cuts_path(source, tmp_path)
    for d in cuts:
        prep.append_cut_dict(partial, d)
    hours = (2.0 + 3.5 + 1.5) / 3600.0
    prep.save_progress(
        source,
        {"rows_seen": 10, "hours_kept": hours, "clips_kept": 3, "split_rows": {}},
        tmp_path,
    )
    state = prep.restore_partial_state(source, tmp_path, force=False)
    assert [d["id"] for d in state["cut_dicts"]] == [c["id"] for c in cuts]
    assert state["hours_kept"] == hours
    assert state["clips_kept"] == 3
    assert state["rows_seen"] == 10

    wiped = prep.restore_partial_state(source, tmp_path, force=True)
    assert wiped["cut_dicts"] == []
    assert wiped["hours_kept"] == 0.0
    assert wiped["clips_kept"] == 0
    assert not partial.is_file()
    assert not prep.progress_path(source, tmp_path).is_file()


def test_force_wipe_also_removes_fbank_gzip(tmp_path):
    source = "everyayah"
    fbank = tmp_path / f"{source}_cuts_fbank.jsonl.gz"
    fbank.write_bytes(b"smoke-leftover")
    (tmp_path / f"{source}_cuts.jsonl.gz").write_bytes(b"cuts")
    prep.restore_partial_state(source, tmp_path, force=True)
    assert not fbank.is_file()
    assert not (tmp_path / f"{source}_cuts.jsonl.gz").is_file()


def test_force_wipe_also_removes_sharded_fbank(tmp_path):
    vol = tmp_path
    man = vol / "manifests"
    man.mkdir()
    source = "qua"
    shard0 = man / f"{source}_cuts_fbank.shard-0.jsonl.gz"
    shard3 = man / f"{source}_cuts_fbank.shard-3.jsonl.gz"
    shard0.write_bytes(b"s0")
    shard3.write_bytes(b"s3")
    (man / f"{source}_cuts_fbank.jsonl.gz").write_bytes(b"merged")
    sharded = vol / "fbank_sharded" / source / "shard-0"
    sharded.mkdir(parents=True)
    (sharded / "feats").write_bytes(b"feat")
    other = vol / "fbank_sharded" / "tlog" / "shard-0"
    other.mkdir(parents=True)
    (other / "keep").write_bytes(b"x")
    prep.restore_partial_state(source, man, force=True)
    assert not shard0.is_file()
    assert not shard3.is_file()
    assert not (man / f"{source}_cuts_fbank.jsonl.gz").is_file()
    assert not (vol / "fbank_sharded" / source).exists()
    assert (other / "keep").is_file()


def test_crash_resume_finalize_prefix_plus_suffix_no_duplicates(tmp_path):
    source = "retasy"
    prefix = [{"id": f"retasy_{i:08d}_1_1", "duration": 1.0} for i in range(12)]
    partial = prep.partial_cuts_path(source, tmp_path)
    for d in prefix:
        prep.append_cut_dict(partial, d)
    prep.save_progress(
        source,
        {"rows_seen": 100, "hours_kept": 12 / 3600.0, "clips_kept": 12, "split_rows": {}},
        tmp_path,
    )

    state = prep.restore_partial_state(source, tmp_path)
    assert len(state["cut_dicts"]) == 12
    assert state["clips_kept"] == 12
    assert state["hours_kept"] == 12 / 3600.0

    suffix = [{"id": f"retasy_{i:08d}_1_1", "duration": 1.0} for i in range(12, 30)]
    for d in suffix:
        prep.append_cut_dict(partial, d)
        state["cut_dicts"].append(d)
        state["clips_kept"] += 1
        state["hours_kept"] += 1.0 / 3600.0
    prep.append_cut_dict(partial, prefix[0])
    state["cut_dicts"].append(prefix[0])

    from_file = prep.load_cut_dicts(partial)
    final = prep.finalize_cut_dicts(from_file)
    ids = [d["id"] for d in final]
    assert ids == [f"retasy_{i:08d}_1_1" for i in range(30)]
    assert len(ids) == len(set(ids)) == 30
    assert abs(state["hours_kept"] - 30 / 3600.0) < 1e-12

    prep.remove_partial_cuts(source, tmp_path)
    assert not partial.is_file()


def test_idx_increments_on_skipped_rows():
    """QUA used to skip `idx += 1` on no_surah_ayah; clip ids drifted vs a clean run."""
    rows = [
        {"kind": "no_surah_ayah"},
        {"kind": "no_surah_ayah"},
        {"kind": "keep"},
        {"kind": "dur_none"},
        {"kind": "keep"},
    ]
    state = prep.empty_progress()
    kept_idx = []
    for row in rows:
        idx = prep.consume_row(state)
        if row["kind"] in ("no_surah_ayah", "dur_none"):
            continue
        kept_idx.append(idx)
    assert state["rows_seen"] == 5
    assert kept_idx == [2, 4]

    split_state = prep.empty_progress()
    a = prep.consume_row(split_state, split="slug_a")
    b = prep.consume_row(split_state, split="slug_a")
    c = prep.consume_row(split_state, split="slug_b")
    assert (a, b, c) == (0, 1, 0)
    assert split_state["rows_seen"] == 3
    assert split_state["split_rows"] == {"slug_a": 2, "slug_b": 1}


def test_torn_last_line_restore_repairs_partial(tmp_path):
    source = "iqra"
    good = [
        {"id": "iqra_00000000_1_1", "duration": 1.0},
        {"id": "iqra_00000001_1_2", "duration": 2.0},
        {"id": "iqra_00000002_1_3", "duration": 3.0},
    ]
    partial = prep.partial_cuts_path(source, tmp_path)
    for d in good:
        prep.append_cut_dict(partial, d)
    torn_start = partial.stat().st_size
    with partial.open("ab") as f:
        f.write(b'{"id": "iqra_00000003_1_4", "duration":')
    state = prep.restore_partial_state(source, tmp_path)
    assert [d["id"] for d in state["cut_dicts"]] == [c["id"] for c in good]
    assert len(state["cut_dicts"]) == 3
    repaired = partial.read_bytes()
    assert repaired.endswith(b"\n")
    assert b"iqra_00000003" not in repaired
    assert len(repaired) == torn_start
    again = prep.load_cut_dicts(partial)
    assert [d["id"] for d in again] == [c["id"] for c in good]


def test_torn_utf8_tail_restore_repairs_partial(tmp_path):
    source = "qua"
    good = [
        {"id": "qua_00000000_1_1", "duration": 1.0, "text": "سُ"},
        {"id": "qua_00000001_1_2", "duration": 2.0},
    ]
    partial = prep.partial_cuts_path(source, tmp_path)
    for d in good:
        prep.append_cut_dict(partial, d)
    torn_start = partial.stat().st_size
    torn = b'{"id": "qua_00000002_1_3", "text": "' + "سُ".encode()[:-1]
    with pytest.raises(UnicodeDecodeError):
        torn.decode("utf-8")
    with partial.open("ab") as f:
        f.write(torn)
    state = prep.restore_partial_state(source, tmp_path)
    assert [d["id"] for d in state["cut_dicts"]] == [c["id"] for c in good]
    assert len(state["cut_dicts"]) == 2
    repaired = partial.read_bytes()
    assert repaired.endswith(b"\n")
    assert len(repaired) == torn_start
    again = prep.load_cut_dicts(partial)
    assert [d["id"] for d in again] == [c["id"] for c in good]


def test_corrupt_progress_restore_hours_from_cuts(tmp_path):
    source = "tlog"
    cuts = [
        {"id": "tlog_00000000_1_1", "duration": 2.0},
        {"id": "tlog_00000001_1_2", "duration": 4.0},
    ]
    partial = prep.partial_cuts_path(source, tmp_path)
    for d in cuts:
        prep.append_cut_dict(partial, d)
    prog = prep.progress_path(source, tmp_path)
    prog.write_text("{not-json", encoding="utf-8")
    state = prep.restore_partial_state(source, tmp_path)
    assert len(state["cut_dicts"]) == 2
    assert state["clips_kept"] == 2
    assert state["rows_seen"] == 2
    assert abs(state["hours_kept"] - 6.0 / 3600.0) < 1e-12

    prep.save_progress(
        source,
        {"rows_seen": 9, "hours_kept": 99.0, "clips_kept": 2, "split_rows": {}},
        tmp_path,
    )
    assert not Path(str(prog) + ".tmp").exists()
    loaded = prep.load_progress(source, tmp_path)
    assert loaded is not None
    assert loaded["rows_seen"] == 9
    restored = prep.restore_partial_state(source, tmp_path)
    assert abs(restored["hours_kept"] - 6.0 / 3600.0) < 1e-12
    assert restored["rows_seen"] == 9


def test_fbank_shard_partition_exact_and_ordered():
    items = list(range(10))
    shards = [prep.shard_items(items, i, 3) for i in range(3)]
    assert shards[0] == [0, 3, 6, 9]
    assert shards[1] == [1, 4, 7]
    assert shards[2] == [2, 5, 8]
    flat = [x for s in shards for x in s]
    assert sorted(flat) == items
    assert len(flat) == len(set(flat)) == len(items)

    ids = [f"c{i}" for i in range(12)]
    parts = [prep.shard_items(ids, i, 5) for i in range(5)]
    assert parts[0] == ["c0", "c5", "c10"]
    assert parts[4] == ["c4", "c9"]
    union = [x for p in parts for x in p]
    assert sorted(union) == sorted(ids)
    assert len(union) == len(set(union)) == 12

    assert prep.shard_items(ids, 0, 1) == ids
    assert prep.shard_items([], 0, 4) == []
    for n, k in ((0, 1), (1, 1), (1, 7), (7, 3), (12, 4), (12, 5), (719925, 12), (384612, 8)):
        xs = list(range(n))
        parts = [prep.shard_items(xs, i, k) for i in range(k)]
        flat = prep.merge_shard_items(parts)
        assert sorted(flat) == xs
        assert len(flat) == n
        seen: set[int] = set()
        for p in parts:
            assert not seen.intersection(p)
            seen.update(p)
        assert seen == set(xs)
        for i, p in enumerate(parts):
            assert p == xs[i::k]

    with pytest.raises(ValueError):
        prep.shard_items([1], 0, 0)
    with pytest.raises(ValueError):
        prep.shard_items([1], 3, 3)


def test_fbank_merge_shards_preserves_order():
    shards = [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert prep.merge_shard_items(shards) == [0, 3, 6, 9, 1, 4, 7, 2, 5, 8]
    assert prep.merge_shard_items([["a"], ["b", "c"], []]) == ["a", "b", "c"]
    assert prep.merge_shard_items([]) == []

    items = [f"cut-{i}" for i in range(20)]
    parts = [prep.shard_items(items, i, 4) for i in range(4)]
    merged = prep.merge_shard_items(parts)
    assert merged == parts[0] + parts[1] + parts[2] + parts[3]
    assert merged != items
    assert sorted(merged) == sorted(items)
    assert prep.expected_fbank_cut_count(239975, False) == 719925
    assert prep.expected_fbank_cut_count(128204, True) == 128204
    assert prep.fbank_shard_storage_path("qua", 3) == Path("/vol/fbank_sharded/qua/shard-3")
    assert prep.fbank_shard_manifest_path("qua", 3) == Path(
        "/vol/manifests/qua_cuts_fbank.shard-3.jsonl.gz"
    )
