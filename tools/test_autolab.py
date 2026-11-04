# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_autolab.py"""

import itertools
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
IDS = itertools.count()
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
        self.background = False

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

    sessions = {"practice": ["lab-v1.5-p01", "lab-v1.5-p02"], "exam": ["lab-v1.5-e01"]}

    def test_set(self, ts):
        return {"id": ts, "base": "lab-v1.5", "manifest": "lab-v1.5/manifest-v1.json", "sessions": self.sessions}

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
    world.answers = {"planner": lambda p: answer({"cards": [card(f"checker-idea-{next(IDS)}", component=s["component"], tier=s["tier"])
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
    cmd = " ".join(decode["argv"])
    assert "modal/lab_decode.py" in cmd and "--listener exp-" in cmd and 'exitCode' in cmd
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


def test_main_moving_during_a_fold_folds_again_instead_of_paging(w):
    w.job_results["fold"] = {"error": "main moved while scoring; fold it again"}
    st, id = advance(w, 6)
    e = st["experiments"][id]
    assert e["state"] == "folding" and e["attempts"]["mainMoved"] >= 1 and not st["needsHuman"]


def test_hygiene_failure_bounces_once_then_discards(w):
    w.diff_text = "diff --git a/tools/lab.py b/tools/lab.py\n@@ -1 +1 @@\n-a\n+b\n"
    st, id = advance(w, 1)
    e = st["experiments"][id]
    assert e["state"] == "implementing" and e["attempts"]["bounces"] == 1 and "protected" in e["note"]
    st = tick(w)
    assert states(st)[id] == "discarded"


def test_a_diff_without_engine_code_goes_back_to_the_builder(w):
    w.diff_text = ("diff --git a/.scratch/lab/experiments/x.md b/.scratch/lab/experiments/x.md\n@@ -0,0 +1 @@\n+notes\n"
                   "diff --git a/packages/core/test/x.test.ts b/packages/core/test/x.test.ts\n@@ -0,0 +1 @@\n+test\n")
    st, id = advance(w, 1)
    assert st["experiments"][id]["state"] == "implementing" and "no engine code" in st["experiments"][id]["note"]


def test_a_builder_that_killed_its_experiment_is_recorded_not_scored(w):
    id = started(w)
    d = w.tmp / id / ".autolab"
    d.mkdir(exist_ok=True)
    (d / "outcome.json").write_text(json.dumps({"status": "killed", "reason": "kill criterion hit: -2 only", "measured": {"overall": -2}}))
    w.diff_text = "diff --git a/.scratch/lab/experiments/x.md b/.scratch/lab/experiments/x.md\n@@ -0,0 +1 @@\n+notes\n"
    st = tick(w)
    e = st["experiments"][id]
    assert e["state"] == "discarded" and "killed by its builder: kill criterion hit" in e["note"] and not any(n.startswith("score-") for n in w.jobs)


def test_a_builder_that_stops_with_wins_and_no_score_is_sent_back_to_score(w):
    id = started(w)
    d = w.tmp / id / ".autolab"
    d.mkdir(exist_ok=True)
    (d / "outcome.json").write_text(json.dumps({"status": "killed", "reason": "a clean seam also spells a word", "measured": {"targets won": 10}}))
    w.diff_text = "diff --git a/.scratch/lab/experiments/x.md b/.scratch/lab/experiments/x.md\n@@ -0,0 +1 @@\n+notes\n"
    st = tick(w)
    e = st["experiments"][id]
    assert e["state"] == "implementing" and e["attempts"]["bounces"] == 1 and "10 targets won and no practice score" in e["note"]
    assert not (d / "outcome.json").exists()


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
    assert e["state"] == "implementing" and e["executor"]["model"] == "gpt-6.1-sol"
    st = tick(w)
    assert states(st)[id] == "abandoned"


def test_changed_test_data_pauses_everything(w):
    w.hash = "h1"
    st, _ = advance(w, 2)
    assert st["paused"] and "test data" in st["paused"]
    assert any(kind == "paused" for kind, _ in w.notes)
    before = states(st)
    assert states(tick(w)) == before


def test_resuming_after_a_data_pause_takes_a_fresh_hash(w):
    w.hash = "h1"
    st, _ = advance(w, 2)
    assert "test data" in st["paused"]
    w.queued = [{"op": "pause", "why": ""}]
    st = tick(w)
    assert st["paused"] is None and st["dataHash"] == "h1"


def test_own_decode_output_is_not_test_data():
    assert A.OWN_DATA.search("data/lab-v1.5/sessions/lab-v1.4-p04.exp-listener-x-7085852.phone.evidence.json")
    assert A.OWN_DATA.search("data/lab-v1.5/decode-v1.json")
    assert A.OWN_DATA.search("data/lab-v1.5/review/dashboard.json")
    assert not A.OWN_DATA.search("data/lab-v1.5/sessions/lab-v1.4-p04.spliced-clean-int8.phone.evidence.json")
    assert not A.OWN_DATA.search("data/lab-v1.5/manifest-v1.json")


def test_a_new_day_leaves_an_unmoved_main_line_alone(w):
    st = w.load()
    st["main"]["head"] = "abc1234"
    w.saved = st
    st = tick(w, "2026-11-02T00:05:00+00:00")
    assert not st.get("baselineJob") and st["day"] == "2026-11-02"


def test_exam_verdicts_keep_no_exam_numbers():
    def line(p, x):
        return {"sha": f"s{p}", "practice": {"wrongCalls": p}, "exam": {"wrongCalls": x}}
    assert A.exam_verdict({"mainLine": [line(89, 150)]}, 5, T0)["verdict"] == "pass"
    v = A.exam_verdict({"mainLine": [line(89, 150), {"sha": "x", "practice": {"wrongCalls": 85}}, line(80, 154)]}, 5, T0)
    assert v["verdict"] == "worse" and v["practiceChange"] == -9 and "154" not in v["why"] and "150" not in v["why"]
    assert A.exam_verdict({"mainLine": [line(89, 150), line(80, 150)]}, 5, T0)["verdict"] == "no-transfer"
    assert A.exam_verdict({"mainLine": [line(89, 150), line(80, 141)]}, 5, T0)["verdict"] == "pass"


def test_the_exam_tripwire_runs_after_enough_folds_and_pages_when_the_exam_does_not_follow(w, monkeypatch):
    st = w.load()
    st["foldsSinceExam"], st["main"]["head"], st["lastPlanAt"] = 5, "abc1234", "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.job_results["exam"] = {"run": "main-run-3", "sha": "m3", "practice": {**PRACTICE, "wrongCalls": 80}}
    st = tick(w)
    assert st["examJob"] and st["foldsSinceExam"] == 0
    monkeypatch.setattr(A, "vol_get", lambda path: {"mainLine": [{"sha": "a", "practice": {"wrongCalls": 89}, "exam": {"wrongCalls": 150}},
                                                                   {"sha": "b", "practice": {"wrongCalls": 80}, "exam": {"wrongCalls": 151}}]})
    st = tick(w)
    assert not st["examJob"] and st["examTripwire"]["verdict"] == "no-transfer" and st["main"]["sha"] == "m3"
    assert any(k == "needs-human" and "exam" in t for k, t in w.notes)


def test_a_failed_exam_puts_its_folds_back_so_the_next_tick_runs_it_again(w):
    st = w.load()
    st["foldsSinceExam"], st["main"]["head"], st["lastPlanAt"] = 5, "abc1234", "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.job_results["exam"] = {"error": "runner killed"}
    st = tick(w)
    assert st["examJob"] and st["foldsSinceExam"] == 0
    st = tick(w)
    assert any(k == "needs-human" and "exam" in t for k, t in w.notes) and st["foldsSinceExam"] == 5
    st = tick(w)
    assert st["examJob"] and st["examJob"]["folds"] == 5


def test_a_revert_waits_for_main_to_be_still_then_reverts_and_rescores(w, monkeypatch):
    st = w.load()
    e, _ = S.add(st, card("checker-kept"), T0)
    for to in ("accepted", "implementing", "validating", "scoring", "folding", "folded"):
        S.move(st, e["id"], to, T0)
    st["main"]["head"], st["lastPlanAt"] = "abc1234", "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.reverted = []
    w.revert_fold = lambda id: w.reverted.append(id) or "def5678"
    monkeypatch.setattr(A, "vol_get", lambda path: {"mainLine": [{"listener": "spliced-clean-int8"}, {"experiment": "checker-kept", "listener": "exp-x"}]})
    w.queued = [{"op": "revert", "id": e["id"], "why": "sounds wrong"}]
    st = tick(w)
    assert w.reverted == [e["id"]] and st["experiments"][e["id"]]["state"] == "reverted"
    assert ("mark-reverted", e["id"], "--note", "reverted by you: sounds wrong") in w.lab_calls
    assert "--listener" in w.jobs[st["baselineJob"]["name"]]["argv"] and "spliced-clean-int8" in w.jobs[st["baselineJob"]["name"]]["argv"]


def test_revert_fold_takes_the_folds_commits_out_of_a_checked_out_main(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / "repo"
    g = lambda *a, cwd=repo: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    repo.mkdir()
    g("init", "-q", "-b", "main")
    g("config", "user.email", "t@t"), g("config", "user.name", "t")
    (repo / "a.txt").write_text("base\n")
    g("add", "-A"), g("commit", "-qm", "base")
    (repo / "card.json").write_text("{}\n")
    g("add", "-A"), g("commit", "-qm", "Lab experiment checker-x (checker): idea")
    (repo / "a.txt").write_text("changed\n")
    g("commit", "-qam", "the change")
    g("tag", "autolab/fold/checker-x")
    (repo / "b.txt").write_text("later\n")
    g("add", "-A"), g("commit", "-qm", "later work")
    monkeypatch.setattr(A, "REPO", repo)
    new = A.World(A.DEFAULT_CONFIG, executor=object()).revert_fold("checker-x")
    assert g("rev-parse", "main") == new and (repo / "a.txt").read_text() == "base\n"
    assert not (repo / "card.json").exists() and (repo / "b.txt").exists()


def test_new_day_rescores_a_main_line_that_moved(w):
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
    w.answers["planner"] = lambda p: answer({"cards": [card(f"x-idea-{next(IDS)}", component=s["component"], tier=s["tier"],
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
    w.answers["planner"] = lambda p: answer({"cards": [card(f"x-idea-{next(IDS)}", component=s["component"], tier=s["tier"],
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
    assert st["experiments"][id]["state"] == "implementing" and id not in [h["id"] for h in st["needsHuman"]]


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


def test_inbox_retry_of_an_abandoned_build_starts_a_fresh_builder_in_its_worktree(w, tmp_path):
    st, id = advance(w, 1)
    e = st["experiments"][id]
    e["worktree"], e["executor"]["armFailures"], e["executor"]["fallbacks"] = str(tmp_path), 1, [{"from": "a", "to": "b"}]
    S.move(st, id, "abandoned", T0, "the executor failed twice: cancelled:")
    w.saved = st
    w.queued = [{"op": "retry", "id": id, "note": ""}]
    st = tick(w)
    e = st["experiments"][id]
    seen = [h["state"] for h in e["history"]]
    assert seen[seen.index("abandoned") + 1] == "accepted" and e["state"] in ("implementing", "validating", "scoring", "folding")
    assert e["worktree"] == str(tmp_path)


def test_inbox_retry_of_a_judge_timeout_asks_the_judge_again(w):
    st = w.load()
    e, _ = S.add(st, card("checker-unjudged"), T0)
    S.move(st, e["id"], "needs-human", T0, "the novelty judge didn't answer in time")
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.queued = [{"op": "retry", "id": e["id"], "note": ""}]
    st = tick(w)
    seen = [h["state"] for h in st["experiments"][e["id"]]["history"]]
    assert seen[seen.index("needs-human") + 1] == "proposed" and "judge" in st["experiments"][e["id"]]["results"]


def test_inbox_reopen_sends_an_abandoned_build_back_to_validation(w, tmp_path):
    st = w.load()
    e, _ = S.add(st, card("checker-slow-review"), T0)
    for to in ("accepted", "implementing", "validating", "abandoned"):
        S.move(st, e["id"], to, T0)
    e["worktree"] = str(tmp_path)
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.queued = [{"op": "reopen", "id": e["id"]}]
    st = tick(w)
    assert "validating" in [h["state"] for h in st["experiments"][e["id"]]["history"]][-3:]


def test_inbox_rebuild_makes_a_new_accepted_card_from_a_builder_kill(w):
    st = w.load()
    e, _ = S.add(st, card("checker-surplus-word"), T0)
    for to in ("accepted", "implementing", "validating", "discarded"):
        S.move(st, e["id"], to, T0)
    e["branch"] = "exp/checker-surplus-word"
    e["results"]["builderOutcome"] = {"status": "killed", "reason": "one clean speller", "measured": {"targets won": 10}}
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    w.queued = [{"op": "rebuild", "id": e["id"], "note": ""}]
    st = tick(w)
    n = st["experiments"]["checker-surplus-word-r"]
    assert "accepted" in [h["state"] for h in n["history"]] and "collateral" in n["card"]["reason"]
    assert n["card"]["killCriterion"].startswith("the built change wins fewer than 3")


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
    w.answers["planner"] = lambda p: answer({"cards": [card(f"c-{next(IDS)}", component=s["component"], tier=s["tier"]) for s in slots(p)]})
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


def test_a_builder_that_cannot_start_for_an_outage_retries_later_without_a_strike(w):
    class Down(FakeExecutor):
        def collect(self, handle):
            return {"transcript": "", "summary": "startup: [unavailable] Server at capacity", "hours": 0.0}
    w.executor = Down(script=lambda c, wt: (_ for _ in ()).throw(RuntimeError("x")))
    st, id = advance(w, 1)
    e = st["experiments"][id]
    assert e["state"] == "accepted" and e["attempts"]["executor"] == 0 and e["results"]["retryAfter"] > T0
    assert tick(w, "2026-11-01T10:05:00+00:00")["experiments"][id]["state"] == "accepted"
    w.executor = FakeExecutor()
    assert tick(w, "2026-11-01T10:11:00+00:00")["experiments"][id]["state"] in ("implementing", "validating", "scoring", "folding")


def test_failed_tests_send_the_log_tail_back(w):
    w.job_results["tests"] = {"exitCode": 1, "log": "packages/core test: 2 failed in checker.test.ts"}
    st, id = advance(w, 2)
    assert "2 failed in checker.test.ts" in st["experiments"][id]["note"]


def test_outage_detection():
    assert A.outage("startup: [unavailable] Server at capacity")
    assert A.outage("runner-died: ") and A.outage("cancelled: ")
    assert not A.outage("run-error: tests kept failing")


def test_background_model_calls_never_hold_up_a_tick(w):
    import threading
    gate = threading.Event()
    plan = w.answers["planner"]
    w.answers["planner"] = lambda p: (gate.wait(5), plan(p))[1]
    w.background = True
    st = tick(w)
    assert st["experiments"] == {} and st["heartbeat"] == T0
    st = tick(w, "2026-11-01T10:02:00+00:00")
    assert st["experiments"] == {}
    gate.set()
    for k in [k for k in w.futures if k[0] == "plan"]:
        w.futures[k].result(timeout=5)
    st = tick(w, "2026-11-01T10:04:00+00:00")
    assert len(st["experiments"]) == 6


def test_reviews_wait_in_the_state_until_answered(w, monkeypatch):
    monkeypatch.setattr(A.G, "gate", lambda *a, **k: {"verdict": "fail", "tuning": False, "checks": [
        {"name": "gain", "ok": False, "blocking": True, "why": "overall +0"}]})
    down = {"on": True}
    review = w.answers["reviewer"]
    w.answers["reviewer"] = lambda p: (_ for _ in ()).throw(RuntimeError("capacity")) if down["on"] else review
    st, id = advance(w, 4)
    assert states(st)[id] == "discarded" and id in st["pendingReviews"]
    down["on"] = False
    st = tick(w)
    assert st["pendingReviews"] == {} and f"{id} (checker, exploit): discarded" in w.journal_text


def test_one_experiments_error_waits_for_you_without_pausing_the_rest(w, monkeypatch):
    id = started(w)
    monkeypatch.setattr(A.Tick, "implementing", lambda self, e: (_ for _ in ()).throw(RuntimeError("disk full")))
    st = tick(w)
    assert st["experiments"][id]["state"] == "needs-human" and "disk full" in st["experiments"][id]["note"]
    assert not st.get("paused")


def test_lab_new_that_already_happened_is_adopted(w, monkeypatch):
    real = w.lab
    def lab(*args):
        if args[0] == "new":
            (A.REPO.parent / f"tilawa-lab-exp-{args[1]}").mkdir(exist_ok=True)
            raise A.Stop(f"experiment {args[1]} already exists")
        return real(*args)
    w.lab = lab
    st = tick(w)
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = only(st, next(iter(st["experiments"])))
    id = next(iter(w.saved["experiments"]))
    try:
        st = tick(w)
        assert st["experiments"][id]["worktree"].endswith(f"tilawa-lab-exp-{id}") and st["experiments"][id]["state"] == "implementing"
    finally:
        (A.REPO.parent / f"tilawa-lab-exp-{id}").rmdir()


def test_a_tick_that_raises_pauses_and_pages(w, monkeypatch):
    monkeypatch.setattr(A.Tick, "run", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    st = A.tick_once(w)
    assert "tick failed: RuntimeError: boom" in st["paused"]


def test_a_tick_that_loses_the_network_is_skipped_without_pausing(w, monkeypatch):
    monkeypatch.setattr(A.Tick, "run", lambda self: (_ for _ in ()).throw(A.Stop("modal volume get /runs/x/score.json failed: Could not connect to the Modal server.")))
    st = A.tick_once(w)
    assert not st.get("paused")


# builders' Modal work through the supervisor (decision 14)

def builder(w, tier="exploit", component=None, polls=10**6):
    """One card of `tier` implementing, with its executor still running and the practice-only data synced."""
    w.executor = FakeExecutor(polls=polls)
    st = tick(w)
    id = next(e["id"] for e in st["experiments"].values() if e["tier"] == tier and component in (None, e["card"]["component"]))
    st = only(st, id)
    st["execData"] = {"testSet": st["testSet"]}
    w.saved = st
    assert states(tick(w))[id] == "implementing"
    return id


def ask(w, id, n, **req):
    d = w.tmp / id / ".autolab"
    d.mkdir(exist_ok=True)
    (d / f"modal-request-{n}.json").write_text(json.dumps({"cmd": "ls /lab/data", **req}))
    return d / f"modal-result-{n}.json"


def run_of(w, name):
    argv = w.jobs[name]["argv"]
    assert argv[1].endswith("modal/autolab_exec.py") and argv[2] == "run"
    return json.loads(argv[3])


RESULT = {"code": 0, "shards": [{"shard": 0, "code": 0, "seconds": 3.0, "tail": "200"}], "seconds": 1800.0, "out": "/scratch/x/runs/1", "local": "l"}


def test_modal_run_requests_run_in_a_sandbox_and_come_back(w):
    w.job_results["modal"] = {"result": RESULT, "exitCode": 0}
    id = builder(w)
    result = ask(w, id, 1, shards=2)
    tick(w)
    q = run_of(w, f"modal-{id}-1")
    assert q["exp"] == id and q["n"] == "1" and q["worktree"] == str(w.tmp / id) and q["shards"] == 2 and q["gpu"] is None and q["image"] == "node"
    assert not result.exists()
    tick(w)
    out = json.loads(result.read_text())
    assert out["code"] == 0 and out["usd"] == 0 and out["shards"][0]["tail"] == "200"
    e = w.load()["experiments"][id]
    assert e["results"]["modalJobs"] == {} and e["results"]["modalRuns"][0]["code"] == 0
    tick(w)
    assert len([n for n in w.jobs if n.startswith("modal-")]) == 1


def test_gpus_are_for_bets_only(w):
    id = builder(w, "exploit")
    result = ask(w, id, 1, gpu="L4")
    tick(w)
    assert "bet cards only" in json.loads(result.read_text())["error"]
    assert not any(n.startswith("modal-") for n in w.jobs)


def test_a_bets_gpu_runs_fit_its_budget(w):
    w.job_results["modal"] = {"result": {**RESULT, "gpu": "L4"}, "exitCode": 0}
    id = builder(w, "bet")
    big, small = ask(w, id, 1, gpu="H100", shards=4, timeout=600), ask(w, id, 2, gpu="L4", timeout=60)
    tick(w)
    assert "can cost up to $158.00" in json.loads(big.read_text())["error"]
    assert run_of(w, f"modal-{id}-2")["gpu"] == "L4"
    tick(w)
    out = json.loads(small.read_text())
    assert out["usd"] == 0.4 and out["gpuUsdLeft"] == 19.6 and out["gpuUsdBudget"] == 20
    assert w.load()["experiments"][id]["results"]["gpuUsd"] == 0.4


def test_concurrent_requests_are_capped_per_experiment(w):
    w.job_results["modal"] = {"result": RESULT, "exitCode": 0}
    id = builder(w)
    results = [ask(w, id, n) for n in (1, 2, 3)]
    w.job_polls = 10**6
    tick(w)
    assert sorted(n for n in w.jobs if n.startswith("modal-")) == [f"modal-{id}-1", f"modal-{id}-2"]
    w.job_polls = 0
    tick(w)
    assert results[0].exists() and results[1].exists() and f"modal-{id}-3" in w.jobs and not results[2].exists()


def test_requests_wait_for_the_practice_only_data(w):
    w.job_results["exec"] = {"testSet": "lab-v1.5", "sessions": 2, "files": 9, "missing": []}
    w.executor = FakeExecutor(polls=10**6)
    id = started(w)
    st = w.load()
    assert st["execData"]["testSet"] == "lab-v1.5"
    st["testSet"] = "lab-v1.5.v2"
    w.saved = st
    ask(w, id, 1)
    st = tick(w)
    sync = next(n for n in w.jobs if n.startswith("exec-sync-"))
    spec = json.loads(w.jobs[sync]["argv"][3])
    assert spec["practice"] == W_PRACTICE and "exam" not in json.dumps(spec) and spec["models"] == ["models/spliced-clean-int8"]
    assert not any(n.startswith("modal-") for n in w.jobs)
    st = tick(w)
    assert st["execData"]["testSet"] == "lab-v1.5.v2" and any(n.startswith("modal-") for n in w.jobs)


W_PRACTICE = ["lab-v1.5-p01", "lab-v1.5-p02"]


def test_a_listener_bets_own_model_is_copied_and_decoded_on_practice_only(w):
    id = builder(w, "bet", component="listener", polls=1)
    (w.tmp / id / ".autolab").mkdir(exist_ok=True)
    (w.tmp / id / ".autolab/model.json").write_text(json.dumps({"path": f"/scratch/{id}/exports/v2"}))
    for _ in range(4):
        st = tick(w)
    cmd = " ".join(next(j for n, j in w.jobs.items() if n.startswith("decode-"))["argv"])
    listener = st["experiments"][id]["results"]["listener"]
    assert f"autolab_exec.py copy-model {id} exports/v2 models/{listener} &&" in cmd
    assert f"--model models/{listener}" in cmd and "--force" in cmd
    assert f"--only {','.join(W_PRACTICE)}" in cmd and "lab-v1.5-e01" not in cmd


def test_a_listener_card_without_a_model_decodes_with_the_main_model(w):
    id = builder(w, component="listener", polls=1)
    for _ in range(4):
        tick(w)
    cmd = " ".join(next(j for n, j in w.jobs.items() if n.startswith("decode-"))["argv"])
    assert "copy-model" not in cmd and "--model models/spliced-clean-int8" in cmd and "--force" not in cmd


def test_a_decode_never_runs_without_practice_ids_or_with_exam_ones(w):
    id = builder(w, component="listener", polls=1)
    w.sessions = {"practice": ["lab-v1.5-p01", "lab-v1.5-e01"], "exam": ["lab-v1.5-e01"]}
    for _ in range(4):
        st = tick(w)
    assert not any(n.startswith("decode-") for n in w.jobs)
    assert st["experiments"][id]["state"] == "needs-human" and "also exam" in st["experiments"][id]["note"]


def test_validation_tests_run_in_the_sandbox_not_on_tilawa_lab(w):
    id = builder(w, polls=1)
    tick(w)
    q = run_of(w, next(n for n in w.jobs if n.startswith("tests-")))
    assert q["exp"] == id and q["shards"] == 6 and q["download"] is False and "vitest" in q["cmd"]


def test_the_brief_sends_modal_work_through_the_supervisor():
    ctx = {"testSet": "lab-v1.5", "history": []}
    e = {"card": card(component="listener"), "tier": "exploit", "branch": "exp/x", "results": {}}
    b = A.brief(e, ctx)
    assert "modal-run" in b and "no Modal credentials" in b and "practice-score" in b and ".autolab/model.json" in b
    assert "lab.py run . --shards" not in b and "--gpu" not in b and str(A.REPO / "tools/autolab.py") in b
    bet = A.brief({**e, "card": card(component="listener", tier="bet"), "tier": "bet"}, ctx)
    assert "--gpu L4" in bet and "$20" in bet and "--image train" in bet
    assert ".autolab/model.json" not in A.brief({**e, "card": card()}, ctx)


def test_a_rate_limit_mid_run_is_an_outage_not_a_failed_builder():
    assert A.outage("run-error: rate limit exceeded: Your requests to gpt-6-astra have exceeded the token rate limit")
    assert A.outage("startup: [unavailable] Server at capacity")
    assert not A.outage("run-error: Agent turn stopped after repeated resumes")
    assert not A.outage("run-error: tests timed out")


def test_an_arm_at_its_concurrency_cap_is_full(w):
    st = w.load()
    for i in range(2):
        e, _ = S.add(st, card(f"checker-busy-{i}"), T0)
        e["state"], e["executor"] = "implementing", {"arm": "astra"}
    t = A.Tick(w, st, T0)
    assert t.arm_full({"name": "astra", "maxConcurrent": 2}) and not t.arm_full({"name": "astra", "maxConcurrent": 3})
    assert not t.arm_full({"name": "astra"}) and not t.arm_full({"name": "grok", "maxConcurrent": 1})


def near_miss(st, id, change, won, component="tracker"):
    e, _ = S.add(st, card(id, component=component), T0)
    e["state"], e["branch"] = "discarded", f"exp/{id}"
    e["results"] = {"branchRun": {"compare": {"overall": {"change": change}, component: {"change": change}}},
                    "gate": {"checks": [{"name": "gain", "ok": False, "blocking": True, "why": "small"},
                                        {"name": "collateral", "ok": True, "blocking": True, "why": "", "won": len(won), "lost": 0, "wonKeys": won}]}}
    return e


def test_two_near_misses_with_different_wins_are_stacked_and_skip_the_builder(w, monkeypatch):
    monkeypatch.setattr(A.P, "incidents", lambda report: {"x": ["i1", "i2", "i3", "i4", "i9"]})
    st = w.load()
    near_miss(st, "tracker-a", -2, ["i1", "i2"])
    near_miss(st, "tracker-b", -2, ["i3", "i4"])
    near_miss(st, "tracker-c", -1, ["i1"])
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    st = tick(w)
    stacks = [e for e in st["experiments"].values() if e["card"].get("stack")]
    assert len(stacks) == 1 and stacks[0]["card"]["stack"] == ["tracker-a", "tracker-b"]
    assert stacks[0]["card"]["expected"]["overall"] == -4 and stacks[0]["state"] == "accepted"
    st = tick(w)
    stacks = [e for e in st["experiments"].values() if e["card"].get("stack")]
    assert len(stacks) == 1 and stacks[0]["state"] in ("validating", "scoring") and not (stacks[0].get("executor") or {}).get("handle")


def test_a_pair_whose_stack_died_on_a_merge_conflict_is_stacked_again(w, monkeypatch):
    monkeypatch.setattr(A.P, "incidents", lambda report: {"x": ["i1", "i2", "i3", "i4"]})
    st = w.load()
    near_miss(st, "tracker-a", -2, ["i1", "i2"])
    near_miss(st, "tracker-b", -1, ["i3"])
    old, _ = S.add(st, card("tracker-stack-old", component="tracker", stack=["tracker-a", "tracker-b"]), T0)
    old["state"], old["note"] = "discarded", "stack: merging tracker-b conflicts in packages/core/src/checker.ts"
    st["stacked"], st["lastPlanAt"] = [["tracker-a", "tracker-b"]], "2999-01-01T00:00:00+00:00"
    w.saved = st
    st = tick(w)
    live = [e for e in st["experiments"].values() if e["card"].get("stack") and e["state"] != "discarded"]
    assert len(live) == 1 and live[0]["card"]["stack"] == ["tracker-a", "tracker-b"]
    S.move(st, live[0]["id"], "discarded", T0, "gate on the branch: overall -1")
    w.saved = st
    st = tick(w)
    assert not [e for e in st["experiments"].values() if e["card"].get("stack") and e["state"] != "discarded"]


def test_a_stack_that_conflicts_goes_to_a_builder_with_the_merge_to_resolve(tmp_path):
    import subprocess
    repo = tmp_path / "r"
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    repo.mkdir()
    g("init", "-q", "-b", "main"), g("config", "user.email", "t@t"), g("config", "user.name", "t")
    (repo / "checker.ts").write_text("x\n")
    g("add", "-A"), g("commit", "-qm", "base")
    for part in ("a", "b"):
        g("checkout", "-q", "-b", f"exp/{part}", "main")
        (repo / "checker.ts").write_text(f"{part}\n")
        g("add", "-A"), g("commit", "-qm", f"Lab experiment {part} (tracker): x")
    g("checkout", "-q", "-b", "exp/s", "main")
    st = S.empty("lab-v1.5")
    for part in ("a", "b"):
        st["experiments"][part] = {"branch": f"exp/{part}"}
    e, _ = S.add(st, card("tracker-s", component="tracker", stack=["a", "b"]), T0)
    S.move(st, e["id"], "accepted", T0)
    e["worktree"] = str(repo)
    t = A.Tick(A.World(A.DEFAULT_CONFIG, executor=object()), st, T0)
    t.build_stack(e)
    assert e["state"] == "accepted" and e["results"]["stackConflict"] == ["checker.ts"]
    assert "git merge exp/b" in e["results"]["feedback"] and "checker.ts" in e["results"]["feedback"]
    assert (repo / "checker.ts").read_text() == "a\n" and not g("status", "--porcelain")


def test_near_misses_that_win_the_same_incidents_or_sum_to_noise_are_not_stacked(w, monkeypatch):
    monkeypatch.setattr(A.P, "incidents", lambda report: {"x": ["i1", "i2", "i3", "i4", "i9"]})
    st = w.load()
    near_miss(st, "tracker-a", -2, ["i1", "i2"])
    near_miss(st, "tracker-b", -2, ["i1", "i3"])
    near_miss(st, "tracker-c", -1, ["i9"])
    st["lastPlanAt"] = "2999-01-01T00:00:00+00:00"
    w.saved = st
    st = tick(w)
    assert st["stacked"] == [["tracker-a", "tracker-c"]]


def test_a_stack_merges_its_parts_and_keeps_its_own_card(tmp_path):
    import subprocess
    repo = tmp_path / "r"
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    repo.mkdir()
    g("init", "-q", "-b", "main"), g("config", "user.email", "t@t"), g("config", "user.name", "t")
    (repo / "base.txt").write_text("x\n")
    g("add", "-A"), g("commit", "-qm", "base")
    card_path = repo / L.CARD
    card_path.parent.mkdir(parents=True, exist_ok=True)
    for part in ("a", "b"):
        g("checkout", "-q", "-b", f"exp/{part}", "main")
        card_path.write_text(json.dumps({"id": part}))
        (repo / f"{part}.txt").write_text(part)
        g("add", "-A"), g("commit", "-qm", f"Lab experiment {part} (tracker): x")
    g("checkout", "-q", "-b", "exp/s", "main")
    card_path.write_text(json.dumps({"id": "s"}))
    g("add", "-A"), g("commit", "-qm", "Lab experiment s (tracker): stack")
    st = S.empty("lab-v1.5")
    for part in ("a", "b"):
        st["experiments"][part] = {"branch": f"exp/{part}"}
    e, _ = S.add(st, card("tracker-s", component="tracker", stack=["a", "b"]), T0)
    S.move(st, e["id"], "accepted", T0)
    e["worktree"] = str(repo)
    t = A.Tick(A.World(A.DEFAULT_CONFIG, executor=object()), st, T0)
    t.build_stack(e)
    assert e["state"] == "validating" and (repo / "a.txt").exists() and (repo / "b.txt").exists()
    assert json.loads(card_path.read_text()) == {"id": "s"} and not g("status", "--porcelain")


def test_a_near_miss_whose_wins_a_later_fold_took_is_not_stacked(w, monkeypatch):
    monkeypatch.setattr(A.P, "incidents", lambda report: {"x": ["i3", "i4"]})
    st = w.load()
    near_miss(st, "tracker-a", -2, ["i1", "i2"])
    near_miss(st, "tracker-b", -2, ["i3", "i4"])
    assert [x["id"] for x in A.Tick(w, st, T0).near_misses()] == ["tracker-b"]
