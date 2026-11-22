# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_review.py"""

import io
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_review_core import (  # noqa: E402
    byte_range, empty_reviews, fixture, pcm16_wav, safe_rel, tally, upsert, validate_queue, wav_bytes, wav_info, wav_slice,
)


def ramp_wav(n=16000, rate=16000, extra=b""):
    """PCM16 mono whose sample i is i % 30000, optionally with an extra chunk before data."""
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    data = struct.pack(f"<{n}h", *(i % 30000 for i in range(n)))
    w = wav_bytes(fmt, data)
    return w[:36] + extra + w[36:] if extra else w


def samples(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getframerate(), list(struct.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes())))


def test_slice_reads_exact_window_and_is_a_valid_wav():
    rate, s = samples(wav_slice(io.BytesIO(ramp_wav()), 0.25, 0.5))
    assert rate == 16000 and s == list(range(4000, 8000))


def test_slice_skips_unknown_chunks_and_clamps_to_the_file():
    src = io.BytesIO(ramp_wav(extra=b"LIST" + struct.pack("<I", 3) + b"abc\0"))
    assert samples(wav_slice(src, 0.9, 5.0))[1] == list(range(14400, 16000))
    assert samples(wav_slice(src, 2.0, 3.0))[1] == []
    assert samples(wav_slice(src, -1, 0.001))[1] == list(range(16))


def test_slice_whole_file_fixes_streamed_size():
    w = bytearray(ramp_wav(100))
    w[40:44] = struct.pack("<I", 0xFFFFFFFF)
    assert samples(wav_slice(io.BytesIO(bytes(w))))[1] == list(range(100))


def test_slice_keeps_float_and_stereo_frames_whole():
    fmt = struct.pack("<HHIIHH", 3, 2, 8000, 8000 * 8, 8, 32)
    data = struct.pack("<4000f", *range(4000))
    out = wav_slice(io.BytesIO(wav_bytes(fmt, data)), 0.1, 0.2)
    f, off, size = wav_info(io.BytesIO(out))
    assert f == fmt and size == 800 * 8 and struct.unpack_from("<2f", out, off) == (1600.0, 1601.0)


def test_rejects_non_wav():
    with pytest.raises(ValueError):
        wav_slice(io.BytesIO(b"ID3" + b"\0" * 40), 0, 1)


@pytest.mark.parametrize("h,want", [
    (None, None), ("", None), ("bytes=0-", (0, 99)), ("bytes=0-1", (0, 1)), ("bytes=10-500", (10, 99)),
    ("bytes=-10", (90, 99)), ("bytes=-500", (0, 99)), ("items=0-1", None), ("bytes=0-1,5-6", None),
])
def test_byte_range(h, want):
    assert byte_range(h, 100) == want


@pytest.mark.parametrize("h", ["bytes=100-", "bytes=5-2", "bytes=-0"])
def test_byte_range_unsatisfiable(h):
    with pytest.raises(ValueError):
        byte_range(h, 100)


@pytest.mark.parametrize("p", ["", "/lab/data/x.wav", "../x.wav", "a/../../x", "a//b", "./a"])
def test_safe_rel_rejects_escapes(p):
    with pytest.raises(ValueError):
        safe_rel(p)


def test_upsert_replaces_and_keeps_history_verbatim():
    doc = empty_reviews("v1", "me")
    upsert(doc, "a", "ok", "", "t1", "v1/review/q.json")
    upsert(doc, "a", "ok", "", "t2")
    assert doc["items"]["a"]["history"] == [] and doc["items"]["a"]["at"] == "t2"
    note = "  splice clicks at 3.1 s — الرحيم \n"
    r = upsert(doc, "a", "bad_splice", note, "t3")
    assert r["note"] == note and r["history"] == [{"verdict": "ok", "note": "", "at": "t2"}]
    upsert(doc, "a", "ok", "fine on relisten", "t4")
    assert [h["verdict"] for h in doc["items"]["a"]["history"]] == ["ok", "bad_splice"]
    assert json.loads(json.dumps(doc))["items"]["a"]["history"][1]["note"] == note


@pytest.mark.parametrize("v,note", [("great", ""), ("ok", 5), ("ok", "x" * 20_001)])
def test_upsert_rejects_bad_input(v, note):
    with pytest.raises(ValueError):
        upsert(empty_reviews("v1", "me"), "a", v, note, "t")


def test_tally_against_the_bar():
    q = {"items": [{"id": str(i)} for i in range(40)]}
    rv = {str(i): {"verdict": "ok"} for i in range(39)}
    assert tally(q, rv) == {"reviewed": 39, "total": 40, "bad": 0, "maxBad": 1, "state": "open"}
    rv["0"] = {"verdict": "bad_audio"}
    rv["39"] = {"verdict": "checker"}
    assert tally(q, rv)["state"] == "pass" and tally(q, rv)["bad"] == 1
    rv["1"] = {"verdict": "bad_label"}
    assert tally(q, rv)["state"] == "fail"
    assert tally(q, {"x": {"verdict": "bad_label"}})["reviewed"] == 0


def test_fixture_queue_is_valid_and_its_clips_slice():
    files = fixture(rate=4000)
    q = files["fixture-review/review/demo.json"]
    assert validate_queue(q) == []
    for it in q["items"]:
        for c in (it["audio"], it.get("before"), it.get("after")):
            if c:
                rate, s = samples(wav_slice(io.BytesIO(files[c["path"]]), c["startS"], c["endS"]))
                assert len(s) == round((c["endS"] - c["startS"]) * rate)


def test_validate_queue_reports_problems():
    ok = fixture(rate=2000)["fixture-review/review/demo.json"]
    bad = json.loads(json.dumps(ok))
    bad["testSet"] = "../x"
    bad["items"][1]["id"] = bad["items"][0]["id"]
    bad["items"][2]["audio"]["endS"] = 0
    bad["items"][3]["label"]["type"] = "typo"
    bad["items"][4]["before"] = {"path": "/abs.wav", "startS": 0, "endS": 1}
    p = validate_queue(bad)
    assert len(p) == 5 and any("duplicate" in x for x in p) and any("typo" not in x and "label.type" in x for x in p)
    assert validate_queue({"version": 1, "testSet": "v1", "items": []}) == ["items must be a non-empty list"]


def test_pcm16_wav_roundtrip():
    rate, s = samples(pcm16_wav([0, 0.5, -0.5, 2], 8000))
    assert rate == 8000 and s == [0, 16384, -16384, 32767]
