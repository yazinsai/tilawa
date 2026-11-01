# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_ab.py"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab as A  # noqa: E402
import autolab_ab as AB  # noqa: E402
import autolab_exec as X  # noqa: E402
import autolab_state as S  # noqa: E402
from autolab_exec import FakeExecutor  # noqa: E402
from test_autolab import T0, World, card, only, states, tick, w as w  # noqa: E402, F401 - the fixture

ARMS = [{"name": "grok", "runtime": "cursor-local", "model": "grok-4.7", "fallbackModel": "grok-4.7-b"},
        {"name": "astra", "runtime": "codex-azure", "azure": {"endpoint": "https://res.openai.azure.com", "deployment": "astra-dep"}}]


def exp(id, component="checker", state="implementing", arm=None, **kw):
    return {"id": id, "state": state, "card": {"component": component}, "executor": {"arm": arm, "assignedArm": arm} if arm else None, **kw}


# --- assignment ---

def test_assignment_is_stable_per_id_and_balanced_per_component():
    arms, wts = ARMS, {"grok": 1, "astra": 1}
    st = {"experiments": {}}
    assert len({AB.assign("checker-a", "checker", arms, wts, st)["name"] for _ in range(20)}) == 1
    st = {"experiments": {"g1": exp("g1", arm="grok"), "t1": exp("t1", "tracker", arm="astra"), "old": exp("old", state="discarded", arm="astra")}}
    assert all(AB.assign(f"checker-{i}", "checker", arms, wts, st)["name"] == "astra" for i in range(20))
    assert all(AB.assign(f"tracker-{i}", "tracker", arms, wts, st)["name"] == "grok" for i in range(20))
    picks = Counter(AB.assign(f"listener-{i}", "listener", arms, wts, st)["name"] for i in range(400))
    assert 150 < picks["grok"] < 250
    assert all(AB.assign(f"listener-{i}", "listener", arms, {"grok": 3, "astra": 1}, st)["name"] == "grok" for i in range(20))
    st["experiments"] |= {f"l{i}": exp(f"l{i}", "listener", arm="grok") for i in range(2)}
    picks = Counter(AB.assign(f"listener-{i}", "listener", arms, {"grok": 3, "astra": 1}, st)["name"] for i in range(400))
    assert 250 < picks["grok"] < 350  # 2 grok running: (2+1)/3 vs (0+1)/1 is a tie, drawn 3:1
    assert all(AB.assign(f"listener-{i}", "listener", arms, {"grok": 1, "astra": 0}, st)["name"] == "grok" for i in range(50))


# --- stats ---

def gate(verdict, lost=0):
    return {"verdict": verdict, "checks": [{"name": "collateral", "lost": lost}]}


def scored(change, lost=0, verdict="fail", fold=False):
    src = {"compare": {"overall": {"change": change}}}
    return {"gate": gate(verdict, lost), **({"fold": {**src, "gate": gate(verdict, lost)}} if fold else {"branchRun": src})}


def test_stats_per_arm():
    E = lambda id, arm, state, results=None, attempts=None, **ex: {  # noqa: E731
        "id": id, "state": state, "card": {"component": "checker"}, "results": results or {}, "attempts": attempts or {"executor": 1},
        "executor": {"arm": ex.pop("now", arm), "assignedArm": arm, "assignedAt": ex.pop("at", T0), **ex}}
    st = {"experiments": {e["id"]: e for e in [
        E("g1", "grok", "folded", scored(-5, verdict="pass", fold=True), hours=2.0),
        E("g2", "grok", "discarded", scored(-2, lost=0), hours=1.0),
        E("g3", "grok", "discarded", scored(-1, lost=2), {"executor": 2, "bounces": 1}, hours=3.0),
        E("g4", "grok", "implementing", hours=0.0),
        E("a1", "astra", "discarded", {"builderOutcome": {"status": "killed"}}, hours=0.5, at="2026-11-01T09:00:00+00:00"),
        E("a2", "astra", "abandoned", now="grok", fallbacks=[{"from": "astra", "to": "grok"}], hours=0.25),
        {"id": "old", "state": "folded", "card": {}, "results": {}, "attempts": {}, "executor": {"arm": "grok", "armInferred": True}},
        {"id": "pre", "state": "folded", "card": {}, "results": {}, "attempts": {}, "executor": {"model": "grok-4.7"}},
    ]}}
    ab = AB.stats(st, ["grok", "astra", "spare"])
    g, a, s = ab["arms"]
    assert g["arm"] == "grok" and (g["started"], g["running"], g["folds"], g["bounced"], g["killedByBuilder"]) == (4, 1, 1, 1, 0)
    assert (g["gated"], g["gatePassed"], g["gatePassRate"]) == (3, 1, 0.333) and g["nearMisses"] == 1
    assert g["scored"] == 3 and g["medianChange"] == -2 and g["meanBuilderHours"] == 1.5
    assert (a["started"], a["killedByBuilder"], a["abandoned"], a["fallbacks"], a["gated"], a["medianChange"]) == (2, 1, 1, 1, 0, None)
    assert a["meanBuilderHours"] == 0.38 and a["gatePassRate"] is None
    assert s["started"] == 0 and s["meanBuilderHours"] is None and ab["since"] == "2026-11-01T09:00:00+00:00"
    assert any("Gate passed" in line for line in AB.table(ab)) and "1/3 (33%)" in AB.table(ab)[1]


# --- the supervisor ---

@pytest.fixture
def arms(w, monkeypatch):
    """Two arms with a fake executor per runtime, made through make_executor like the real ones."""
    w.executor = None
    w.cfg = {**w.cfg, "executor": {**w.cfg["executor"], "runtimes": ARMS, "split": {"grok": 1, "astra": 1}}}
    fakes = {"cursor-local": FakeExecutor(runtime="cursor-local"), "codex-azure": FakeExecutor(runtime="codex-azure")}
    made = []
    monkeypatch.setattr(A, "make_executor", lambda cfg, arm=None, check=True: made.append((arm["name"], check)) or fakes[arm["runtime"]])
    monkeypatch.setenv("CURSOR_API_KEY", "c")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setattr(X, "unready", lambda cfg, arm, environ=None: None if arm["runtime"] != "codex-azure" or (arm.get("azure") or {}).get("endpoint") else "no Azure endpoint")
    return fakes, made


def test_each_builder_gets_an_arm_and_its_handle_goes_back_to_its_runtime(w, arms):
    fakes, made = arms
    st = tick(w)  # plans six cards
    st = tick(w)  # judged, then started
    impl = [e for e in st["experiments"].values() if e["executor"]]
    assert impl and all(e["executor"]["arm"] in ("grok", "astra") and e["executor"]["assignedAt"] == T0 for e in impl)
    for e in impl:
        assert e["executor"]["handle"]["runtime"] == {"grok": "cursor-local", "astra": "codex-azure"}[e["executor"]["arm"]]
        assert e["executor"]["model"] == {"grok": "grok-4.7", "astra": "astra-dep"}[e["executor"]["arm"]]
    assert {a for a, _ in made} == {e["executor"]["arm"] for e in impl}
    assert Counter(e["executor"]["arm"] for e in impl)["grok"] in range(1, len(impl))
    assert {a["arm"] for a in st["ab"]["arms"]} == {"grok", "astra"} and sum(a["started"] for a in st["ab"]["arms"]) == len(impl)
    tick(w)
    assert sum(fakes["cursor-local"].seen.values()) and sum(fakes["codex-azure"].seen.values())


def test_an_incomplete_arm_is_skipped_with_one_notification(w, arms):
    w.cfg["executor"]["runtimes"] = [ARMS[0], {**ARMS[1], "azure": {"endpoint": "", "deployment": ""}}]
    for _ in range(3):
        st = tick(w)
    assert all(e["executor"]["arm"] == "grok" for e in st["experiments"].values() if e["executor"])
    assert [t for k, t in w.notes if k == "arm"] == ["Autolab: astra builders are skipped"]
    assert "endpoint" in st["ab"]["skipped"]["astra"]


def failing_astra(arms):
    fakes, _ = arms
    fakes["codex-azure"].script = lambda c, wt: (_ for _ in ()).throw(RuntimeError("startup: 401 Unauthorized"))


def one_on(w, arm):
    """Keeps one implementing experiment, on `arm`."""
    st = tick(w)
    st = tick(w)
    id = next(e["id"] for e in st["experiments"].values() if e["executor"] and e["executor"]["arm"] == arm)
    w.saved = only(st, id)
    return id


def test_retries_keep_the_arm_and_two_failures_fall_back_once(w, arms):
    failing_astra(arms)
    id = one_on(w, "astra")
    e = tick(w)["experiments"][id]
    assert e["state"] == "implementing" and e["executor"]["arm"] == "astra" and e["executor"]["armStarts"] == {"astra": 2}
    st = tick(w)
    e = st["experiments"][id]
    assert e["executor"]["arm"] == "grok" and e["executor"]["assignedArm"] == "astra" and e["executor"]["fallbacks"][0]["from"] == "astra"
    assert "failed twice" in e["executor"]["fallbacks"][0]["why"] and e["executor"]["model"] == "grok-4.7"
    assert st["ab"]["arms"][1]["fallbacks"] == 1 and st["ab"]["arms"][1]["arm"] == "astra"


def test_an_arm_whose_builders_keep_failing_to_start_rests(w, arms):
    failing_astra(arms)
    st = w.load()
    st["armHealth"] = {"astra": {"startupFails": 2, "lastAt": T0}}
    w.saved = st
    one_on(w, "astra")
    st = tick(w)
    assert st["armHealth"]["astra"]["startupFails"] >= 3
    st = tick(w)
    assert "failed to start" in st["ab"]["skipped"]["astra"]
    later = "2026-11-01T11:30:00+00:00"
    st = A.Tick(w, w.load(), later)
    assert st.arm_down("astra") is None


def test_a_handle_from_before_arms_keeps_working(w, arms):
    fakes, made = arms
    st = tick(w)
    id = next(iter(st["experiments"]))
    w.saved = only(st, id)
    st = w.load()
    e = st["experiments"][id]
    e.update(state="implementing", since=T0, worktree=str(w.tmp / id), branch=f"exp/{id}",
             executor={"model": "grok-4.7", "handle": {"runtime": "cursor-local", "id": f"fake-{id}"}})
    (w.tmp / id).mkdir(exist_ok=True)
    w.saved = st
    st = tick(w)
    assert ("grok", False) in made and fakes["cursor-local"].seen[f"fake-{id}"] == 1
    assert states(st)[id] != "implementing" and st["experiments"][id]["executor"].get("arm") is None


def test_one_executor_world_still_runs_every_arm(w):
    w.cfg = {**w.cfg, "executor": {**w.cfg["executor"], "runtimes": ARMS}}
    st = tick(w)
    st = tick(w)
    assert all(e["executor"]["handle"]["runtime"] == "fake" for e in st["experiments"].values() if e["executor"])


def test_old_and_new_config_both_load(w):
    assert X.arms({"executor": A.DEFAULT_CONFIG["executor"]})[1]["runtime"] == "codex-azure"
    assert card()["id"] and S.empty()["experiments"] == {}
