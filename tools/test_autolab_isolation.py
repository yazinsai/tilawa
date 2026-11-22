# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_isolation.py"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_isolation as I  # noqa: E402

PRACTICE, EXAM = ["lab-v1.4-p01", "lab-v1.5-p10"], ["lab-v1.4-e01", "lab-v1.4-e1"]


def session(id, split):
    return {"id": id, "split": split, "label": f"sessions/{id}.json", "session": {"id": id, "segments": []},
            "audio": {v: f"sessions/{id}.{v}.wav" for v in ("phone", "studio")}, "evidence": f"sessions/{id}.{{listener}}.phone.evidence.json"}


def manifest():
    return {"version": 1, "testSet": "lab-v1.5", "variants": ["phone", "studio"], "evidencePattern": "sessions/{id}.{listener}.{variant}.evidence.json",
            "sessions": [session(i, "practice") for i in PRACTICE] + [session(i, "exam") for i in EXAM]}


@pytest.fixture
def data(tmp_path):
    """A /lab/data with one test set: every session's files for two listeners, flags, checks and a decode record."""
    src = tmp_path / "data"
    s = src / "lab-v1.5" / "sessions"
    s.mkdir(parents=True)
    (src / "lab-v1.5/manifest-v1.json").write_text(json.dumps(manifest()))
    for id in PRACTICE + EXAM:
        for name in (f"{id}.json", f"{id}.phone.wav", f"{id}.studio.wav", f"{id}.checks.json",
                     *(f"{id}.{li}.{v}.evidence.json" for li in ("spliced-clean-int8", "teacher", "other-int8") for v in ("phone", "studio"))):
            (s / name).write_text(json.dumps({"id": id}))
    everyone = PRACTICE + EXAM
    (src / "lab-v1.5/flags-v1.json").write_text(json.dumps({"base": "lab-v1.5", "excluded": [{"session": i, "startS": 1} for i in everyone]}))
    (src / "lab-v1.5/checks-v1.json").write_text(json.dumps({"summary": {"phone": {"sessions": 4}}, "failed": [{"id": f"{i}-x1", "session": i} for i in everyone],
                                                           "sessions": [{"id": i, "variants": {}} for i in everyone], "thresholds": {"llrMin": 3}}))
    (src / "lab-v1.5/decode-v1.json").write_text(json.dumps({"listeners": {"spliced-clean-int8": {"files": {f"{i}.{v}": {"sha256": "x"} for i in everyone for v in ("phone", "studio")},
                                                                                                "totals": {"files": 8}}}}))
    (src / "lab-v1.5/triage-v1.json").write_text(json.dumps({"sessions": everyone}))
    (src / "models/spliced-clean-int8").mkdir(parents=True)
    (src / "models/spliced-clean-int8/model.onnx").write_bytes(b"onnx")
    (src / "models/spliced-clean-int8/manifest.json").write_text("{}")
    return src


SPEC = {"manifest": "lab-v1.5/manifest-v1.json", "practice": PRACTICE, "listeners": ["spliced-clean-int8", "teacher"], "models": ["models/spliced-clean-int8"]}


def test_the_practice_copy_holds_no_exam_session(data, tmp_path):
    stage = tmp_path / "stage"
    out = I.copy_practice(data, stage, SPEC)
    names = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    assert out["sessions"] == 2 and out["missing"] == [] and out["files"] == len(names)
    assert not any(e in n for n in names for e in EXAM)
    assert "lab-v1.5/sessions/lab-v1.4-p01.teacher.studio.evidence.json" in names and "models/spliced-clean-int8/model.onnx" in names
    assert not any("other-int8" in n for n in names) and "lab-v1.5/triage-v1.json" not in names
    text = "".join(p.read_text(errors="ignore") for p in stage.rglob("*.json"))
    assert I.exam_pattern(EXAM).search(text) is None
    m = json.loads((stage / SPEC["manifest"]).read_text())
    assert [s["id"] for s in m["sessions"]] == PRACTICE
    checks = json.loads((stage / "lab-v1.5/checks-v1.json").read_text())
    assert "summary" not in checks and checks["thresholds"] == {"llrMin": 3} and len(checks["failed"]) == 2
    assert sorted(json.loads((stage / "lab-v1.5/decode-v1.json").read_text())["listeners"]["spliced-clean-int8"]["files"]) == \
        sorted(f"{i}.{v}" for i in PRACTICE for v in ("phone", "studio"))


def test_the_copy_refuses_when_an_exam_id_survives_the_filters(data, tmp_path):
    (data / "lab-v1.5/flags-v2.json").write_text(json.dumps({"note": "see lab-v1.4-e01 at 3 s"}))
    with pytest.raises(ValueError, match="exam session data"):
        I.copy_practice(data, tmp_path / "stage", SPEC)


def test_the_ledgers_practice_must_be_practice_in_the_manifest(data, tmp_path):
    with pytest.raises(ValueError, match="not practice"):
        I.copy_practice(data, tmp_path / "stage", {**SPEC, "practice": PRACTICE + ["lab-v1.4-e01"]})
    with pytest.raises(ValueError, match="missing"):
        I.copy_practice(data, tmp_path / "stage", {**SPEC, "practice": ["lab-v9-p01"]})


def test_exam_ids_match_whole_ids_only():
    rx = I.exam_pattern(["lab-v1.4-e1"])
    assert rx.search("x lab-v1.4-e1.phone") and rx.search("lab-v1.4-e1-x3")
    assert not rx.search("lab-v1.4-e10") and not rx.search("lab-v1.4-p1")
    assert not I.exam_pattern([]).search("anything")


CFG = {"exec": {}}


def test_requests_are_checked_and_filled_in():
    q = I.request({"cmd": "ls", "shards": 3}, "exploit", CFG)
    assert q == {"cmd": "ls", "gpu": None, "shards": 3, "timeout": 30, "image": "node", "usdPerHour": 0.0}
    assert I.request({"cmd": "train", "gpu": "l4", "timeout": 120}, "bet", CFG) | {} == \
        {"cmd": "train", "gpu": "L4", "shards": 1, "timeout": 120, "image": "train", "usdPerHour": 0.80}
    for req, tier, why in [({"cmd": " "}, "exploit", "no command"), ({"cmd": "x", "gpu": "L4"}, "explore", "bet cards only"),
                           ({"cmd": "x", "gpu": "TPU"}, "bet", "unknown GPU"), ({"cmd": "x", "shards": 21}, "exploit", "shards"),
                           ({"cmd": "x", "gpu": "L4", "shards": 5}, "bet", "shards"), ({"cmd": "x", "timeout": 61}, "exploit", "timeout"),
                           ({"cmd": "x", "image": "gpu"}, "exploit", "image")]:
        with pytest.raises(ValueError, match=why):
            I.request(req, tier, CFG)
    assert I.request({"cmd": "x", "timeout": 600}, "bet", CFG)["timeout"] == 600


def test_gpu_cost_and_reserve():
    q = I.request({"cmd": "x", "gpu": "H100", "shards": 2, "timeout": 60}, "bet", {"exec": {"gpuUsdPerHour": {"H100": 4.0}}})
    assert I.reserve(q) == 8.0 and I.cost(q, 900) == 2.0
    assert I.reserve(I.request({"cmd": "x"}, "bet", CFG)) == 0


def test_model_paths_stay_in_the_experiments_scratch_area():
    assert I.model_path({"path": "exports/v1/"}, "listener-x") == "exports/v1"
    assert I.model_path({"path": "/scratch/listener-x/exports/v1"}, "listener-x") == "exports/v1"
    for bad in ("", ".", "../other/x", "/scratch/other/x", "/lab/data/models/x", "a/../../b"):
        with pytest.raises(ValueError):
            I.model_path({"path": bad}, "listener-x")


def test_help_train_copies_train_speakers_only(tmp_path):
    src, stage = tmp_path / "help", tmp_path / "stage"
    (src / "wav").mkdir(parents=True)
    rows = [{"id": "a", "speaker": "s1", "split": "train", "wav": "wav/a.wav"},
            {"id": "b", "speaker": "s2", "split": "eval", "wav": "wav/b.wav"}]
    for r in rows:
        (src / r["wav"]).write_bytes(b"RIFF")
    (src / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = I.copy_help_train(src, stage)
    assert out["clips"] == 1 and [f.name for f in (stage / I.HELP_DIR / "wav").iterdir()] == ["a.wav"]
    assert [json.loads(x)["id"] for x in (stage / I.HELP_DIR / "labels.jsonl").read_text().splitlines()] == ["a"]
    (src / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [*rows, {"id": "c", "speaker": "s2", "split": "train", "wav": "wav/b.wav"}]))
    with pytest.raises(ValueError, match="both splits"):
        I.copy_help_train(src, tmp_path / "stage2")
