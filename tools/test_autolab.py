# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_autolab.py"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab as A  # noqa: E402
import autolab_state as S  # noqa: E402
import lab_ledger as L  # noqa: E402
from autolab_exec import FakeExecutor  # noqa: E402

T0 = "2026-11-01T10:00:00+00:00"
CLUSTER = "checker/added_word/missed"
INCIDENT = "s1/phone/mistake:added_word/7:1:2/4.0"
PRACTICE = {"wrongCalls": 89, "missed": 64, "falseWarnings": 25, "listener": 47, "tracker": 5, "checker": 37}
CODE = "diff --git a/packages/core/src/checker.ts b/packages/core/src/checker.ts\n@@ -1 +1,2 @@\n-a\n+b\n+c\n"


def report(run):
    return {"run": run, "cases": [{"split": "practice", "variant": "phone", "kind": "mistake", "outcome": "miss",
                                   "component": "checker", "errorClass": "added_word", "key": INCIDENT}]}


def card(id="checker-edge-fit", tier="exploit", component="checker", **kw):
    return {"id": id, "component": component, "tier": tier, **({"gpuUsd": 20} if tier == "bet" else {}), "hypothesis": "Fit added words at edges. Expect -3.",
            "eli5": "An extra word gets hidden inside its neighbour; look at both neighbours.", "mechanism": "Added speech is absorbed by the neighbour.", "cluster": CLUSTER, "targets": [INCIDENT],
            "expected": {"component": -3, "overall": -3}, "killCriterion": "no target recovered", **kw}


def answer(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


def slots(prompt):
    return json.loads(prompt.split("Free slots, in order: ", 1)[1].split("\n\nAnswer", 1)[0])


class World(A.World):
    """Every side effect faked; jobs finish on the poll after `job_polls` polls with results from `job_results`."""

    def __init__(self, tmp: Path, **kw):
        cfg = {**A.DEFAULT_CONFIG, "slots": 6, "planEveryMinutes": 0}
        super().__init__(cfg, executor=kw.pop("executor", FakeExecutor()), ask=self.answer)
        self.tmp, self.saved, self.jobs, self.job_polls, self.lab_calls, self.notes = tmp, None, {}, 1, [], []
        self.job_results, self.answers, self.diff_text, self.hash, self.rebase_ok, self.journal_text = {}, {}, CODE, "h0", True, ""
        self.sender = lambda cfg, kind, title, body: self.notes.append((kind, title))

    def answer(self, cfg, role, prompt):
        a = self.answers.get(role)
        return a(prompt) if callable(a) else a

    def load(self):
        return json.loads(json.dumps(self.saved)) if self.saved else None

    def save(self, st, events):
        self.saved = json.loads(json.dumps(st))

    def events(self):
        return []

    def lab(self, *args):
        self.lab_calls.append(args)
        if args[0] == "new":
            wt = self.tmp / args[1]
            wt.mkdir(exist_ok=True)
            c = {"id": args[1], "component": args[3], "hypothesis": args[5], "expected": {"component": args[7], "overall": args[9]}}
            (wt / L.CARD).write_text(json.dumps(c))
            return {"id": args[1], "branch": f"exp/{args[1]}", "worktree": str(wt)}
        return {}

    def job(self, name, argv, env=None):
        self.jobs[name] = {"argv": argv, "polls": 0}
        return {"name": name, "pid": 0, "out": "", "err": "", "started": T0}

    def poll(self, job):
        j = self.jobs[job["name"]]
        j["polls"] += 1
        if j["polls"] < self.job_polls:
            return "running", None
        kind = job["name"].split("-")[0]
        res = self.job_results.get(kind, {})
        return ("failed", res) if "error" in res or res.get("exitCode") else ("done", res)

    def kill(self, job):
        pass

    def git(self, *args, cwd=None, check=True):
        return "abc1234" if args[:1] == ("rev-parse",) else ""

    def diff(self, e):
        return self.diff_text

    def commits(self, e):
        return 1

    def exclude(self, pattern):
        pass

    def rebase_shadow(self):
        return self.rebase_ok

    def report(self, run):
        return report(run)

    def data_hash(self, ts):
        return self.hash

    def ledgers(self):
        return []

    def modal_spend(self, day):
        return self.modal_usd

    modal_usd = 0.0
    queued: list = []

    def inbox(self):
        out, self.queued = [(f"/inbox/{i}", c) for i, c in enumerate(self.queued)], []
        return out

    def inbox_done(self, path):
        pass

    def journal(self):
        return self.journal_text

    def journal_add(self, title, body, at):
        self.journal_text += f"\n## {title}\n{body}\n"

    def transcript(self, id, text):
        pass


@pytest.fixture
def w(tmp_path, monkeypatch):
    world = World(tmp_path)
    st = S.empty("lab-v1.5")
    st.update(started=T0, day=T0[:10], dataHash="h0", main={"sha": "m1", "run": "main-run", "practice": PRACTICE,
                                                            "attribution": {c: PRACTICE[c] for c in L.COMPONENTS}})
    world.saved = st
    world.answers = {"planner": lambda p: answer({"cards": [card(f"checker-idea-{i}", component=s["component"], tier=s["tier"])
                                                            for i, s in enumerate(slots(p))]}),
                     "judge": answer({"accept": True, "restates": [], "why": "new"}),
                     "diffReviewer": answer({"ok": True, "why": "fine"}),
                     "reviewer": answer({"journal": "Learned something.", "followUp": None, "deadEnd": None}),
                     "strategist": answer({"memo": "Stuck.", "cards": []})}
    world.job_results = {"tests": {"exitCode": 0}, "score": {"run": "exp-run", "base": "main-run", "sha": "e1", "compare": {}, "verdict": "better", "practice": PRACTICE},
                         "fold": {"state": "folded", "run": "fold-run", "base": "main-run", "sha": "f1", "why": "keeps its gain", "head": "f1",
                                  "compare": {"overall": {"change": -4}}, "practice": {**PRACTICE, "wrongCalls": 85, "checker": 33}}}
    monkeypatch.setattr(A.G, "gate", lambda *a, **k: {"verdict": "pass", "tuning": False, "checks": []})
    monkeypatch.setattr(A.S, "now", lambda: T0)
    return world


def tick(w, at=T0):
    st = w.load()
    A.Tick(w, st, at).run()
    w.save(st, [])
    return st


def states(st):
    return {e["id"]: e["state"] for e in st["experiments"].values()}


def only(st, keep):
    """Keeps the one experiment named and stops planning, so a test follows a single card."""
    st["experiments"] = {k: v for k, v in st["experiments"].items() if k == keep}
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    return st


def started(w, component="checker", tier="exploit"):
    """Plans, keeps the first card of that component and tier, and ticks it through the judge to its executor."""
    st = tick(w)
    id = next(e["id"] for e in st["experiments"].values() if e["card"]["component"] == component and e["tier"] == tier)
    w.saved = only(st, id)
    st = tick(w)
    assert states(st)[id] == "implementing", states(st)
    return id


def test_a_card_walks_from_plan_to_fold(w):
    assert len(tick(w)["experiments"]) == 6
    w.saved = S.empty("lab-v1.5") | {k: v for k, v in w.load().items() if k not in ("experiments", "lastPlanAt")}
    id = started(w)
    seen = []
    for _ in range(8):
        st = tick(w)
        seen.append(states(st)[id])
    order = [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s]
    assert order[-1] == "folded"
    assert [s for s in order if s in ("validating", "scoring", "folding")] == [s for s in ("validating", "scoring", "folding") if s in order]
    assert "folding" in order
    assert st["main"]["practice"]["wrongCalls"] == 85 and st["main"]["sha"] == "f1"
    assert st["counters"]["sinceFold"]["checker"] == 0
    assert f"{id} (checker, exploit): folded" in w.journal_text
    assert any(c[0] == "new" for c in w.lab_calls)
    assert any(kind == "digest" and "kept" in title for kind, title in w.notes)
    assert st["foldLock"] is None


def test_judge_rejection_discards(w):
    w.answers["judge"] = answer({"accept": False, "restates": ["checker-joint-additions"], "why": "same joint fit"})
    st = tick(w)
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    st = tick(w)
    assert set(states(st).values()) == {"discarded"}
    assert all("restates checker-joint-additions" in e["note"] for e in st["experiments"].values())


def advance(w, n, component="checker", tier="exploit"):
    """Starts one card, ticks n more times; returns (state, id)."""
    id = started(w, component, tier)
    st = w.load()
    for _ in range(n):
        st = tick(w)
    return st, id


def test_listener_cards_are_decoded_then_scored_with_their_own_evidence(w):
    st, id = advance(w, 3, component="listener")
    decode = next(j for n, j in w.jobs.items() if n.startswith("decode-"))
    assert "modal/lab_decode.py" in " ".join(decode["argv"]) and "--listener" in decode["argv"]
    card = json.loads((w.tmp / id / L.CARD).read_text())
    assert card["listener"].startswith(f"exp-{id}")
    assert states(st)[id] in ("scoring", "folding", "folded")


def test_gate_failure_on_the_branch_discards(w, monkeypatch):
    monkeypatch.setattr(A.G, "gate", lambda *a, **k: {"verdict": "fail", "tuning": False, "checks": [
        {"name": "spread", "ok": False, "blocking": True, "why": "without reader r1: +0"}]})
    st, id = advance(w, 4)
    e = st["experiments"][id]
    assert e["state"] == "discarded" and "without reader r1" in e["note"]
    assert ("discard", id, "--force", "--note", e["note"]) in w.lab_calls


def test_gate_failure_after_rebase_discards(w):
    w.job_results["fold"] = {"state": "gate-failed", "gate": {"checks": [{"name": "collateral", "ok": False, "blocking": True, "why": "2 lost"}]},
                             "run": "r", "base": "b", "sha": "s", "compare": {}}
    st, id = advance(w, 6)
    assert states(st)[id] == "discarded"
    assert "gate after rebase: 2 lost" in st["experiments"][id]["note"]


def test_hygiene_failure_bounces_once_then_discards(w):
    w.diff_text = "diff --git a/tools/lab.py b/tools/lab.py\n@@ -1 +1 @@\n-a\n+b\n"
    st, id = advance(w, 1)
    e = st["experiments"][id]
    assert e["state"] == "implementing" and e["attempts"]["bounces"] == 1 and "protected" in e["note"]
    st = tick(w)
    assert states(st)[id] == "discarded"


def test_failed_tests_bounce_with_the_log(w):
    w.job_results["tests"] = {"error": "vitest: 2 failed"}
    st, id = advance(w, 2)
    e = st["experiments"][id]
    assert e["state"] == "implementing" and "vitest: 2 failed" in e["note"]


def test_restart_mid_job_never_starts_it_twice(w):
    w.job_polls = 3
    st, id = advance(w, 2)
    assert states(st)[id] == "validating"
    before = sorted(n for n in w.jobs if n.startswith("tests-"))
    w2 = World(w.tmp)
    w2.__dict__.update({k: v for k, v in w.__dict__.items() if k != "executor"})
    tick(w2)
    assert sorted(n for n in w2.jobs if n.startswith("tests-")) == before


def test_executor_timeout_abandons(w):
    w.executor = FakeExecutor(polls=10**6)
    id = started(w)
    assert states(tick(w, "2026-11-01T13:00:00+00:00"))[id] == "implementing"
    assert states(tick(w, "2026-11-01T15:00:00+00:00"))[id] == "abandoned"


def test_executor_failure_retries_with_the_fallback_model_once(w):
    def boom(card, wt):
        raise RuntimeError("agent crashed")
    w.executor = FakeExecutor(script=boom)
    st, id = advance(w, 1)
    e = st["experiments"][id]
    assert e["state"] == "implementing" and e["executor"]["model"] == A.DEFAULT_CONFIG["executor"]["fallbackModel"]
    st = tick(w)
    assert states(st)[id] == "abandoned"


def test_changed_test_data_pauses_everything(w):
    w.hash = "h1"
    st, _ = advance(w, 2)
    assert st["paused"] and "test data" in st["paused"]
    assert any(kind == "paused" for kind, _ in w.notes)
    before = states(st)
    assert states(tick(w)) == before


def test_rebase_conflict_pauses_on_a_new_day(w):
    w.rebase_ok = False
    st = tick(w, "2026-11-02T00:05:00+00:00")
    assert "conflicts" in st["paused"]


def test_new_day_rebases_and_rescores_the_shadow_line(w):
    w.job_results["baseline"] = {"run": "main-run-2", "sha": "m2", "practice": {**PRACTICE, "wrongCalls": 88}}
    st = tick(w, "2026-11-02T00:05:00+00:00")
    assert st["baselineJob"] and st["day"] == "2026-11-02"
    st = tick(w, "2026-11-02T00:07:00+00:00")
    assert st["main"]["sha"] == "m2" and st["main"]["practice"]["wrongCalls"] == 88


def test_modal_spend_comes_from_billing_and_caps_new_work(w):
    w.modal_usd = 3000.5
    st = tick(w)
    assert st["spend"]["modalUsd"] == 3000.5 and st["experiments"] == {}


def test_spend_cap_stops_new_work(w):
    st = w.load()
    st["spend"] = {"day": T0[:10], "agentRuns": 999, "agentHours": 0, "modalUsd": 0}
    w.saved = st
    st = tick(w)
    assert st["experiments"] == {}


def test_tuning_cap_lets_one_tuning_diff_through(w):
    w.diff_text = "diff --git a/packages/core/src/checker.ts b/packages/core/src/checker.ts\n@@ -1 +1 @@\n-const K = 1;\n+const K = 2;\n"
    st = tick(w)
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    for _ in range(3):
        st = tick(w)
    capped = [e for e in st["experiments"].values() if "tuning" in e["note"]]
    assert len(capped) == len(st["experiments"]) - 1
    assert all(e["state"] == "discarded" for e in capped)


def test_practice_score_requests_are_served_and_capped(w):
    w.executor = FakeExecutor(polls=10**6)
    w.job_results["practice"] = w.job_results["score"]
    id = started(w)
    d = w.tmp / id / ".autolab"
    d.mkdir()
    for n in range(1, 5):
        (d / f"practice-request-{n}.json").write_text("{}")
    tick(w)
    tick(w)
    assert json.loads((d / "practice-result-1.json").read_text())["verdict"] == "better"
    assert "error" in json.loads((d / "practice-result-4.json").read_text())
    assert w.load()["experiments"][id]["attempts"]["practiceScores"] == 3


def test_an_expensive_bet_waits_for_approval(w):
    w.answers["planner"] = lambda p: answer({"cards": [card(f"x-idea-{i}", component=s["component"], tier=s["tier"],
                                                            **({"gpuUsd": 150} if s["tier"] == "bet" else {})) for i, s in enumerate(slots(p))]})
    tick(w)
    st = tick(w)
    bet = next(e for e in st["experiments"].values() if e["tier"] == "bet")
    assert bet["state"] == "needs-human" and "approve" in bet["note"] and bet["worktree"] is None
    bet["results"]["approved"] = T0
    S.move(st, bet["id"], "accepted", T0)
    w.saved = st
    assert tick(w)["experiments"][bet["id"]]["state"] == "implementing"


def waiting_bet(w):
    w.answers["planner"] = lambda p: answer({"cards": [card(f"x-idea-{i}", component=s["component"], tier=s["tier"],
                                                            **({"gpuUsd": 150} if s["tier"] == "bet" else {})) for i, s in enumerate(slots(p))]})
    tick(w)
    st = tick(w)
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    return next(e["id"] for e in st["experiments"].values() if e["tier"] == "bet")


def test_inbox_approve_starts_the_bet(w):
    id = waiting_bet(w)
    w.queued = [{"op": "approve", "id": id}]
    st = tick(w)
    assert st["experiments"][id]["state"] == "implementing" and st["needsHuman"] == []


def test_inbox_retry_with_a_note_goes_back_to_the_builder(w):
    id = waiting_bet(w)
    w.queued = [{"op": "approve", "id": id}]
    tick(w)
    st = w.load()
    S.move(st, id, "needs-human", T0, "scoring failed: boom")
    w.saved = st
    w.queued = [{"op": "retry", "id": id, "note": "use the smaller recipe"}]
    st = tick(w)
    e = st["experiments"][id]
    assert e["state"] in ("implementing", "validating", "scoring", "folding") and "use the smaller recipe" in " ".join(h["note"] for h in e["history"])


def test_inbox_drop_and_rejections(w):
    id = waiting_bet(w)
    w.queued = [{"op": "drop", "id": id, "why": "too expensive"}, {"op": "approve", "id": "nope"}]
    st = tick(w)
    assert st["experiments"][id]["state"] == "discarded" and "too expensive" in st["experiments"][id]["note"]


def test_inbox_pause_and_resume(w):
    w.queued = [{"op": "pause", "why": "holiday"}]
    assert tick(w)["paused"] == "holiday"
    w.queued = [{"op": "pause", "why": ""}]
    assert tick(w)["paused"] is None


def test_a_model_outage_waits_without_pausing_or_moving_anything(w):
    def down(prompt):
        raise RuntimeError("claude-opus-5-5 did not start: [unavailable] Server at capacity")
    w.answers["planner"] = down
    st = tick(w)
    assert not st.get("paused") and st["experiments"] == {} and "capacity" in st["brainDown"]["error"]
    st = tick(w, "2026-11-01T11:30:00+00:00")
    assert any("aren't answering" in title for _, title in w.notes)
    w.answers["planner"] = lambda p: answer({"cards": [card(f"c-{i}", component=s["component"], tier=s["tier"]) for i, s in enumerate(slots(p))]})
    st = tick(w, "2026-11-01T11:32:00+00:00")
    assert "brainDown" not in st and len(st["experiments"]) == 6


def test_an_outage_mid_judging_leaves_the_card_proposed(w):
    tick(w)
    def down(prompt):
        raise RuntimeError("timeout")
    w.answers["judge"] = down
    st = w.load()
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    st = tick(w)
    assert set(states(st).values()) == {"proposed"}


def test_a_tick_that_raises_pauses_and_pages(w, monkeypatch):
    monkeypatch.setattr(A.Tick, "run", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    st = A.tick_once(w)
    assert "tick failed: RuntimeError: boom" in st["paused"]
