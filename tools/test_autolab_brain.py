# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_autolab_brain.py"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_brain as B  # noqa: E402
import autolab_state as S  # noqa: E402

CFG = {"models": {r: f"model-{r}" for r in ("planner", "judge", "diffReviewer", "reviewer", "strategist")}}
CLUSTERS = {"listener/wrong_word/missed": ["s1/phone/mistake:wrong_word/2:8:1/73.2", "s2/phone/mistake:wrong_word/26:1:1/9.0"],
            "checker/added_word/missed": ["s3/phone/mistake:added_word/7:1:2/4.0"]}
LAB = {"experiments": {"checker-joint-additions": {"card": {"component": "checker", "hypothesis": "joint fit", "expected": {"component": -6, "overall": -8}},
                                                   "state": "folded", "decision": {"note": "keeps its gain"}}}}


def ctx(st=None):
    st = st or S.empty("lab-v1.5")
    return B.pack(st, CLUSTERS, [LAB], "## earlier\nnothing yet")


def card(id="listener-onset-boost", component="listener", tier="explore", **kw):
    return {"id": id, "component": component, "tier": tier, "hypothesis": "Weak onsets vanish; boost them. Expect -3.", "eli5": "Quiet first sounds get lost, so the app thinks a word is missing.",
            "mechanism": "Short consonants peak for one frame and lose to blank.", "cluster": "listener/wrong_word/missed",
            "targets": ["s1/phone/mistake:wrong_word/2:8:1/73.2"], "expected": {"component": -3, "overall": -3},
            "killCriterion": "no recovered onset in the targets", **kw}


def answer(obj):
    return f"Here you go.\n```json\n{json.dumps(obj)}\n```"


class Asker:
    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []

    def __call__(self, cfg, role, prompt):
        self.prompts.append((role, prompt))
        return self.texts.pop(0)


def test_pack_carries_clusters_history_and_no_exam():
    st = S.empty("lab-v1.5")
    st["main"] = {"practice": {"wrongCalls": 89}, "attribution": {"listener": 47}, "exam": {"wrongCalls": 7}}
    p = B.pack(st, CLUSTERS, [LAB], "j")
    assert p["clusters"][0]["key"] == "listener/wrong_word/missed"
    assert p["history"][0]["id"] == "checker-joint-additions"
    assert "exam" not in json.dumps(p)


def test_plan_returns_one_checked_card_per_slot():
    ask = Asker(answer({"cards": [card()]}))
    cards = B.plan(CFG, ctx(), [("listener", "explore")], ask)
    assert cards[0]["id"] == "listener-onset-boost"
    assert ask.prompts[0][0] == "planner" and "listener/wrong_word/missed" in ask.prompts[0][1]


def test_plan_retries_once_then_gives_up():
    wrong_slot = answer({"cards": [card(tier="exploit")]})
    ask = Asker(wrong_slot, answer({"cards": [card()]}))
    assert B.plan(CFG, ctx(), [("listener", "explore")], ask)[0]["tier"] == "explore"
    assert "must take its slot" in ask.prompts[1][1]
    with pytest.raises(B.BrainError):
        B.plan(CFG, ctx(), [("listener", "explore")], Asker("not json", wrong_slot))


@pytest.mark.parametrize("bad, msg", [
    ({"id": "checker-joint-additions"}, "card id"),
    ({"targets": ["made-up"]}, "targets"),
    ({"cluster": "nope"}, "cluster"),
    ({"hypothesis": "two\nlines"}, "one line"),
    ({"tier": "bet"}, "slot"),
    ({"eli5": ""}, "eli5"),
    ({"eli5": "x" * 201}, "eli5"),
])
def test_card_checks(bad, msg):
    with pytest.raises(ValueError, match=msg):
        B.check_card(card(**bad), ctx(), B.taken_ids(ctx()), ("listener", "explore"))


def test_bets_need_a_budget_and_twice_failed_clusters_need_a_reason():
    with pytest.raises(ValueError, match="gpuUsd"):
        B.check_card(card(tier="bet"), ctx(), set())
    assert B.check_card(card(tier="bet", gpuUsd=40), ctx(), set())
    st = S.empty("lab-v1.5")
    for i in range(2):
        e, _ = S.add(st, card(id=f"listener-try-{i}"), "2026-11-01T00:00:00+00:00")
        e["state"] = "discarded"
        st["coverage"].setdefault("listener/wrong_word/missed", {"state": "tried", "experiments": [], "incidents": 2})["experiments"].append(e["id"])
    with pytest.raises(ValueError, match="reason"):
        B.check_card(card(), ctx(st), set())
    assert B.check_card(card(reason="attacks the encoder, not the decoder"), ctx(st), set())


def test_the_judge_never_sees_the_card_itself():
    st = S.empty("lab-v1.5")
    S.add(st, card(), "2026-11-01T00:00:00+00:00")
    ask = Asker(answer({"accept": True, "restates": [], "why": "new"}))
    B.judge(CFG, card(), ctx(st), ask)
    assert '"listener-onset-boost"' not in ask.prompts[0][1].split("History of tried experiments")[1]


def test_judge_and_diff_review_shapes():
    j = B.judge(CFG, card(), ctx(), Asker(answer({"accept": False, "restates": ["listener-blank-margin"], "why": "same rescue"})))
    assert j == {"accept": False, "restates": ["listener-blank-margin"], "why": "same rescue"}
    ask = Asker(answer({"ok": True, "why": "ordinary change"}))
    r = B.review_diff(CFG, card(), "x" * (B.MAX_DIFF + 10), ask)
    assert r["ok"] and "10 more characters" in ask.prompts[0][1]


def test_a_silent_diff_reviewer_falls_back_to_the_reviewer_model():
    roles = []

    def ask(cfg, role, prompt):
        roles.append(role)
        if role == "diffReviewer":
            raise B.Unavailable("diffReviewer: no answer within 960 s")
        return answer({"ok": True, "why": "fine"})
    assert B.review_diff(CFG, card(), "diff", ask)["ok"] and roles == ["diffReviewer", "reviewer"]
    with pytest.raises(B.Unavailable):
        B.review_diff({"models": dict(CFG["models"], reviewer="model-diffReviewer")}, card(), "diff", ask)


def test_review_and_strategy():
    exp = {"card": card(), "tier": "explore", "results": {"executor": {"summary": "done"}}}
    r = B.review(CFG, exp, {"state": "discarded"}, ctx(), Asker(answer({"journal": "It failed.", "followUp": None, "deadEnd": "no onset"})))
    assert r["deadEnd"] == "no onset"
    s = B.strategy(CFG, ctx(), ["listener"], [("listener", "bet")], Asker(answer({"memo": "Stuck.", "cards": [card(tier="bet", gpuUsd=50)]})))
    assert s["cards"][0]["tier"] == "bet"


def test_the_judge_is_told_which_tried_experiments_never_ran():
    st = S.empty("lab-v1.5")
    for id, note in (("listener-blocked", "the diff changes no engine code (only write-ups)"), ("listener-twin", "restates listener-blocked: same"),
                     ("listener-killed", "killed by its builder: 0 of 5 targets")):
        e, _ = S.add(st, card(id), "2026-11-01T00:00:00+00:00")
        e["state"], e["note"] = "discarded", note
    h = {x["id"]: x["measured"] for x in B.history([LAB], st)}
    assert h == {"checker-joint-additions": True, "listener-blocked": False, "listener-twin": False, "listener-killed": True}


def test_of_two_twin_cards_waiting_for_the_judge_only_the_younger_sees_the_older():
    st = S.empty("lab-v1.5")
    S.add(st, card("listener-twin-a"), "2026-11-01T00:00:00+00:00")
    S.add(st, card("listener-twin-b"), "2026-11-01T00:00:00+00:00")
    seen = {}
    for id in ("listener-twin-a", "listener-twin-b"):
        ask = Asker(answer({"accept": True, "restates": [], "why": "new"}))
        B.judge(CFG, card(id), ctx(st), ask)
        seen[id] = ask.prompts[0][1].split("History of tried experiments")[1]
    assert '"listener-twin-b"' not in seen["listener-twin-a"] and '"listener-twin-a"' in seen["listener-twin-b"]
