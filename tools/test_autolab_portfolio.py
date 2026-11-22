# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_autolab_portfolio.py (portfolio and state)"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_portfolio as P  # noqa: E402
import autolab_state as S  # noqa: E402

TODAY = {"listener": 47, "checker": 37, "tracker": 5}
T0 = "2026-11-01T00:00:00+00:00"


def at(hours):
    return f"2026-11-{1 + int(hours // 24):02d}T{int(hours % 24):02d}:00:00+00:00"


def card(id, component="checker", tier="exploit", **kw):
    return {"id": id, "component": component, "tier": tier, "hypothesis": "h", **kw}


def test_allocation_with_todays_attribution():
    a = P.allocate(TODAY, 6)
    assert Counter(c for c, _ in a) == {"listener": 3, "checker": 3}
    assert Counter(t for _, t in a) == {"exploit": 3, "explore": 2, "bet": 1}
    assert ("listener", "bet") in a


def test_no_slot_for_a_component_without_wrong_calls_and_tiers_hold_with_few_slots():
    a = P.allocate({"listener": 0, "checker": 100, "tracker": 0}, 3)
    assert Counter(c for c, _ in a) == {"checker": 3}
    assert Counter(t for _, t in a) == {"exploit": 1, "explore": 1, "bet": 1}
    assert P.allocate(TODAY, 0) == []


def test_fold_rate_moves_slots_off_a_component_that_never_folds():
    a = P.allocate({"listener": 38, "checker": 24, "tracker": 4}, 6, record={"listener": (1, 45), "checker": (3, 47), "tracker": (1, 22)})
    assert Counter(c for c, _ in a) == {"listener": 2, "checker": 3, "tracker": 1}
    assert Counter(t for _, t in a)["bet"] == 1


def test_frozen_component_trades_exploit_for_explore_not_bets():
    a = P.allocate(TODAY, 6, frozen={"listener", "checker"})
    assert not {("listener", "exploit"), ("checker", "exploit")} & set(a)
    assert Counter(t for _, t in a)["bet"] == 1


def test_fold_record_counts_decided_experiments():
    st = S.empty()
    for i, state in enumerate(("folded", "discarded", "discarded", "implementing")):
        e, _ = S.add(st, card(f"x{i}", "tracker"), T0)
        e["state"] = state
    assert P.fold_record(st)["tracker"] == (1, 3)


def test_free_slots_never_exceed_room_left_by_active_experiments():
    st = S.empty()
    for i in range(5):
        S.add(st, card(f"b{i}", "listener", "bet"), T0)
    assert len(P.free_slots(st, P.allocate(TODAY, 6))) == 1


def test_free_slots_subtract_active_experiments():
    st = S.empty()
    S.add(st, card("a", "listener", "bet"), T0)
    S.add(st, card("b", "checker", "exploit"), T0)
    e, _ = S.add(st, card("c", "checker", "explore"), T0)
    e["state"] = "discarded"
    free = P.free_slots(st, P.allocate(TODAY, 6))
    assert ("listener", "bet") not in free
    assert len(free) == 4


def decide(st, id, state, component="checker", when=T0):
    e, _ = S.add(st, card(id, component), when)
    e["state"] = state
    P.record_outcome(st, e, when)


def test_plateau_after_five_non_folds_and_thaw_on_fold():
    st = S.empty()
    for i in range(4):
        decide(st, f"d{i}", "discarded")
    assert P.update_plateau(st, at(1), T0) == []
    decide(st, "d4", "superseded")
    assert P.update_plateau(st, at(1), T0) == ["checker"]
    assert P.update_plateau(st, at(2), T0) == []
    decide(st, "f", "folded", when=at(3))
    assert "checker" not in st["counters"]["frozen"]
    assert st["counters"]["sinceFold"]["checker"] == 0


def test_a_day_without_a_fold_freezes_everything_until_any_fold():
    st = S.empty()
    assert P.update_plateau(st, at(23), T0) == []
    assert sorted(P.update_plateau(st, at(25), T0)) == ["checker", "listener", "tracker"]
    assert P.stuck_for_user(st, at(48)) == []
    assert sorted(P.stuck_for_user(st, at(50))) == ["checker", "listener", "tracker"]
    decide(st, "f", "folded", component="tracker", when=at(51))
    assert st["counters"]["frozen"] == {}


def test_reset_plateau():
    st = S.empty()
    for i in range(5):
        decide(st, f"d{i}", "discarded")
    P.update_plateau(st, at(1), T0)
    P.reset_plateau(st)
    assert st["counters"]["frozen"] == {} and st["counters"]["sinceFold"] == {}


def test_tuning_cap():
    st = S.empty()
    a, _ = S.add(st, card("a"), T0)
    S.add(st, card("b"), T0)
    assert not P.tuning_blocked(st, "b")
    a["results"]["tuning"] = True
    assert P.tuning_blocked(st, "b") and not P.tuning_blocked(st, "a")
    a["state"] = "discarded"
    assert not P.tuning_blocked(st, "b")


def test_incident_clusters_and_coverage():
    report = {"cases": [
        {"split": "practice", "variant": "phone", "kind": "mistake", "outcome": "miss", "component": "listener", "errorClass": "wrong_word", "key": "k1"},
        {"split": "practice", "variant": "phone", "kind": "mistake", "outcome": "hit", "component": None, "errorClass": "wrong_word", "key": "k2"},
        {"split": "practice", "variant": "phone", "kind": "false_warning", "component": "checker", "errorClass": "added_word", "key": "k3"},
        {"split": "practice", "variant": "studio", "kind": "mistake", "outcome": "miss", "component": "listener", "errorClass": "wrong_word", "key": "k4"},
        {"split": "exam", "variant": "phone", "kind": "mistake", "outcome": "miss", "component": "listener", "errorClass": "wrong_word", "key": "k5"},
    ]}
    cl = P.incidents(report)
    assert cl == {"listener/wrong_word/missed": ["k1"], "checker/added_word/false-warning": ["k3"]}
    st = S.empty()
    P.update_coverage(st, cl)
    S.add(st, card("x"), T0)
    P.mark_tried(st, "listener/wrong_word/missed", "x")
    st["experiments"]["x"]["state"] = "discarded"
    assert P.failed_tries(st, "listener/wrong_word/missed") == 1
    P.update_coverage(st, {"checker/added_word/false-warning": ["k3"]})
    assert st["coverage"]["listener/wrong_word/missed"]["state"] == "fixed"


def test_state_transitions_and_timeouts():
    st = S.empty()
    S.add(st, card("a", tier="explore"), T0)
    with pytest.raises(ValueError):
        S.move(st, "a", "folded", at(0))
    ev = S.move(st, "a", "accepted", at(0))
    assert ev["from"] == "proposed" and ev["to"] == "accepted"
    S.move(st, "a", "implementing", at(1))
    assert not S.timed_out(st["experiments"]["a"], at(12))
    assert S.timed_out(st["experiments"]["a"], at(14))
    S.move(st, "a", "needs-human", at(14), "stuck")
    assert st["needsHuman"][0]["what"] == "stuck"
    S.move(st, "a", "implementing", at(15))
    assert st["needsHuman"] == []
    with pytest.raises(ValueError):
        S.add(st, card("a"), T0)


def test_fold_lock_and_spend_caps():
    st = S.empty()
    S.add(st, card("a"), T0)
    S.add(st, card("b"), T0)
    st["experiments"]["a"]["state"] = "folding"
    assert S.take_lock(st, "a") and not S.take_lock(st, "b")
    st["experiments"]["a"]["state"] = "folded"
    assert S.take_lock(st, "b")
    S.charge(st, T0, runs=3, hours=1.5)
    assert S.over_cap(st, {"agentRuns": 3}, T0).startswith("daily cap on agent runs")
    assert S.over_cap(st, {"agentRuns": 3}, at(25)) is None
