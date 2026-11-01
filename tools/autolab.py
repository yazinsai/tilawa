"""Autolab supervisor (.scratch/autolab/spec.md; ticket 04): the reconciling loop that runs the lab unattended.

    uv run --python cpython-3.12-macos-aarch64-none --with modal==1.3.4 --with cursor-sdk==1.2.0 tools/autolab.py <command>

    init                     autolab branch and worktree, state, config, practice baseline of the shadow line
    tick                     one pass (cron)
    loop [--every 120]       tick forever (phase A)
    pause [why] | resume
    status [--json]
    practice-score [worktree]   an executor's practice score through the supervisor (at most 3 per card, decision 20)
    watchdog                 notify when the heartbeat is stale (run from cron or launchd, apart from the loop)
    match <id> agree|disagree   your verdict on a shadow decision (ticket 11)
    waiting                  what needs you, with the command for each
    approve <id>             let a bet over betApprovalUsd start
    retry <id> [--note ...]  re-run checks, tests and scoring; with a note, back to the builder with it
    drop <id> [why]          end an experiment
    eli5                     plain one-liners for experiments whose cards predate the eli5 field
Your commands (pause, resume, approve, retry, drop, match) go to /lab/autolab/inbox/ and the loop applies them at the
start of its next tick, so a tick in flight never overwrites them.

Every lab operation goes through tools/lab.py on the shadow line: `--main autolab --ledger /autolab/ledger.json --json`.
Long ones (tests, decode, scoring, fold) run as detached jobs under .scratch/autolab/run/jobs/, so a restarted
supervisor finds them still running or finished. State lives on the tilawa-lab volume (/lab/autolab/state.json,
events.jsonl, journal.md); the dashboard reads it from there. A tick never decides a fold itself: `lab fold --gate`
does, with tools/lab_gate.py.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_brain as B  # noqa: E402
import autolab_notify as N  # noqa: E402
import autolab_portfolio as P  # noqa: E402
import autolab_state as S  # noqa: E402
import lab_gate as G  # noqa: E402
import lab_ledger as L  # noqa: E402
from autolab_exec import make as make_executor  # noqa: E402
from lab import REPO, Stop, vol_get, vol_ls, vol_put  # noqa: E402

LAB = REPO / "tools/lab.py"
CONFIG = REPO / ".scratch/autolab/config.json"
RUN = REPO / ".scratch/autolab/run"
BRANCH, WORKTREE = "autolab", REPO.parent / "tilawa-autolab"
VOL_STATE, VOL_EVENTS, VOL_LEDGER, VOL_JOURNAL = "/autolab/state.json", "/autolab/events.jsonl", "/autolab/ledger.json", "/autolab/journal.md"
VOL_INBOX, VOL_ELI5 = "/autolab/inbox", "/autolab/eli5.json"
OWN_DATA = re.compile(r"(\.exp-[^/]+\.(phone|studio)\.evidence\.json|/decode-v1\.json)$")
PRACTICE_SCORES = 3
GROK = {"id": "grok-4.7", "params": [{"id": "context", "value": "500k"}, {"id": "reasoning_effort", "value": "high"}, {"id": "fast", "value": "false"}]}
DEFAULT_CONFIG = {
    "slots": 6, "planEveryMinutes": 30, "tickSeconds": 120,
    "caps": {"agentRuns": 60, "agentHours": 40, "modalUsd": 150},
    "models": {**dict.fromkeys(("planner", "reviewer", "strategist"), {"id": "claude-opus-5-5", "params": [
                   {"id": "context", "value": "1m"}, {"id": "effort", "value": "medium"}, {"id": "fast", "value": "false"}]}),
               **dict.fromkeys(("judge", "diffReviewer"), GROK)},
    "executor": {"runtime": "cursor-local", "model": GROK, "fallbackModel": GROK,
                 "env": {"MODAL_ENVIRONMENT": "autolab-exec"}},
    "betApprovalUsd": 25,
    "listenerModel": "models/spliced-clean-int8",
    "tests": "bash tools/autolab_tests.sh",
    "testShards": 6,
    "notify": {"topic": "", "dashboard": ""},
}


def config() -> dict:
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    return {**DEFAULT_CONFIG, **cfg}


# Side effects


class World:
    """Everything a tick touches outside the state document: lab.py, jobs, git, the volume, models, the executor."""

    def __init__(self, cfg: dict, executor=None, ask=None, sender=None):
        self.cfg, self.ask, self.sender = cfg, ask, sender
        self.executor = executor or make_executor(cfg)
        self._reports, self._procs = {}, {}

    # state on the volume
    def load(self) -> dict | None:
        return vol_get(VOL_STATE)

    def save(self, st: dict, events: list[dict]) -> None:
        if events:
            RUN.mkdir(parents=True, exist_ok=True)
            with open(RUN / "events.jsonl", "a") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
            subprocess.run(["modal", "volume", "put", "--force", "tilawa-lab", str(RUN / "events.jsonl"), VOL_EVENTS], capture_output=True)
        vol_put(VOL_STATE, st)

    def events(self) -> list[dict]:
        f = RUN / "events.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []

    # lab.py
    def lab_argv(self, *args) -> list[str]:
        return lab_argv(*args)

    def lab(self, *args) -> dict:
        p = subprocess.run(lab_argv(*args), cwd=REPO, capture_output=True, text=True)
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": p.stderr[-2000:]}
        if p.returncode or "error" in out:
            raise Stop(out.get("error") or p.stderr[-2000:])
        return out

    def job(self, name: str, argv: list[str], env: dict | None = None) -> dict:
        jobs = RUN / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        out, err = jobs / f"{name}.out", jobs / f"{name}.err"
        with open(out, "w") as o, open(err, "w") as e:
            p = subprocess.Popen(argv, cwd=REPO, stdout=o, stderr=e, start_new_session=True, env={**os.environ, **(env or {})})
        self._procs[p.pid] = p
        return {"name": name, "pid": p.pid, "out": str(out), "err": str(err), "started": S.now()}

    def poll(self, job: dict) -> tuple[str, dict | None]:
        """('running', None) | ('done', result) | ('failed', {"error"})."""
        p = self._procs.get(job["pid"])
        if p is not None:
            if p.poll() is None:
                return "running", None
        elif alive(job["pid"]):
            return "running", None
        text = Path(job["out"]).read_text().strip() if Path(job["out"]).exists() else ""
        try:
            res = json.loads(text.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            tail = Path(job["err"]).read_text()[-2000:] if Path(job["err"]).exists() else ""
            return "failed", {"error": tail or "no output"}
        if res.get("exitCode"):
            res["log"] = text[-3000:]
        return ("failed", res) if "error" in res or res.get("exitCode") else ("done", res)

    def kill(self, job: dict | None) -> None:
        if job and alive(job["pid"]):
            try:
                os.killpg(job["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass

    # git
    def git(self, *args, cwd=None, check=True) -> str:
        p = subprocess.run(["git", *map(str, args)], cwd=cwd or REPO, capture_output=True, text=True)
        if check and p.returncode:
            raise Stop(f"git {' '.join(map(str, args))}: {(p.stderr or p.stdout).strip()}")
        return p.stdout.strip()

    def diff(self, e: dict) -> str:
        """The experiment's own change: from its card commit's parent to the worktree's HEAD."""
        wt = e["worktree"]
        start = self.git("log", "--format=%H", "--fixed-strings", f"--grep=Lab experiment {e['id']} (", "HEAD", cwd=wt).split()[-1]
        return self.git("diff", "-U0", f"{start}^", "HEAD", cwd=wt)

    def exclude(self, pattern: str) -> None:
        """Keeps supervisor files out of every worktree's commits (the shared .git/info/exclude)."""
        f = Path(self.git("rev-parse", "--git-common-dir")).resolve() / "info" / "exclude"
        if not f.is_absolute():
            f = REPO / f
        f.parent.mkdir(parents=True, exist_ok=True)
        if pattern not in (f.read_text().splitlines() if f.exists() else []):
            with open(f, "a") as out:
                out.write(f"\n{pattern}\n")

    def rebase_shadow(self) -> bool:
        """Rebases the autolab branch onto main in its worktree; False (and aborted) on a conflict."""
        if subprocess.run(["git", "rebase", "-q", "main"], cwd=WORKTREE, capture_output=True).returncode:
            subprocess.run(["git", "rebase", "--abort"], cwd=WORKTREE, capture_output=True)
            return False
        return True

    def commits(self, e: dict) -> int:
        start = self.git("log", "--format=%H", "--fixed-strings", f"--grep=Lab experiment {e['id']} (", "HEAD", cwd=e["worktree"]).split()[-1]
        return int(self.git("rev-list", "--count", f"{start}..HEAD", cwd=e["worktree"]))

    # data
    def report(self, run: str) -> dict:
        if run not in self._reports:
            r = vol_get(f"/runs/{run}/score.json")
            if not r:
                raise Stop(f"no score.json for run {run}")
            self._reports = {run: r}  # one at a time; reports are megabytes
        return self._reports[run]

    def data_hash(self, test_set: str) -> str:
        """Paths, sizes and times of every file under the test set's data directory, except what autolab writes there
        itself: an experiment's own listener evidence (`*.exp-*.evidence.json`) and the decode record."""
        import modal  # noqa: PLC0415 - only the supervisor's environment has it
        base = test_set.split(".v")[0]
        entries = modal.Volume.from_name("tilawa-lab").listdir(f"/data/{base}", recursive=True)
        h = hashlib.sha256()
        for e in sorted(entries, key=lambda e: e.path):
            if e.type.name != "FILE" or OWN_DATA.search(e.path):
                continue
            h.update(f"{e.path}\t{e.size}\t{e.mtime}\n".encode())
        return h.hexdigest()

    def inbox(self) -> list[tuple[str, dict]]:
        return [(p, c) for p in sorted(vol_ls(VOL_INBOX)) if (c := vol_get(p))]

    def inbox_done(self, path: str) -> None:
        subprocess.run(["modal", "volume", "rm", "tilawa-lab", path], capture_output=True)

    def modal_spend(self, day: str) -> float:
        """The whole workspace's Modal cost so far on a UTC day (your own runs count too)."""
        import datetime as dt  # noqa: PLC0415
        import modal.billing  # noqa: PLC0415 - only the supervisor's environment has it
        start = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
        return float(sum(r["cost"] for r in modal.billing.workspace_billing_report(start=start, end=start + dt.timedelta(days=1), resolution="h")))

    def ledgers(self) -> list[dict]:
        return [x for x in (vol_get("/ledger.json"), vol_get(VOL_LEDGER)) if x]

    def journal(self) -> str:
        f = RUN / "journal.md"
        return f.read_text() if f.exists() else ""

    def journal_add(self, title: str, body: str, at: str) -> None:
        RUN.mkdir(parents=True, exist_ok=True)
        with open(RUN / "journal.md", "a") as f:
            f.write(f"\n## {title}\n\n_{at}_\n\n{body.strip()}\n")
        subprocess.run(["modal", "volume", "put", "--force", "tilawa-lab", str(RUN / "journal.md"), VOL_JOURNAL], capture_output=True)

    def transcript(self, id: str, text: str) -> None:
        d = RUN / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{id}.md").write_text(text)

    def notify(self, st, kind, key, title, body, at) -> None:
        try:
            N.notify(st, self.cfg, kind, key, title, body, sender=self.sender, at=at)
        except OSError as err:
            print(f"notify failed: {err}", file=sys.stderr)


def lab_argv(*args) -> list[str]:
    return [sys.executable, str(LAB), "--main", BRANCH, "--ledger", VOL_LEDGER, "--json", *map(str, args)]


def alive(pid: int) -> bool:
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# The tick


class Tick:
    def __init__(self, w: World, st: dict, at: str):
        self.w, self.st, self.at, self.events = w, st, at, []
        self._ctx, self.brain_down = None, False

    def move(self, e, to, note=""):
        self.events.append(S.move(self.st, e["id"], to, self.at, note))
        if to in S.TERMINAL:
            P.record_outcome(self.st, e, self.at)

    def ctx(self) -> dict:
        if self._ctx is None:
            main = self.st.get("main") or {}
            clusters = P.incidents(self.w.report(main["run"])) if main.get("run") else {}
            self._ctx = B.pack(self.st, clusters, self.w.ledgers(), self.w.journal())
        return self._ctx

    def brain(self, e, fn, *args):
        """An orchestrator call. A model outage leaves everything where it is for a later tick (and pages after an hour);
        a malformed answer, after its retry, sends the experiment to needs-human."""
        st = self.st
        if self.brain_down:
            return None
        try:
            out = fn(self.w.cfg, *args, ask=self.w.ask)
            st.pop("brainDown", None)
            return out
        except B.Unavailable as err:
            self.brain_down = True
            down = st.setdefault("brainDown", {"since": self.at})
            down["error"] = str(err)[:300]
            if S.minutes(down["since"], self.at) > 60:
                self.w.notify(st, "needs-human", f"brain-down:{down['since']}", "Autolab: the models aren't answering",
                              f"No answer since {down['since']}: {down['error']}", self.at)
            return None
        except B.BrainError as err:
            if e:
                self.move(e, "needs-human", f"orchestrator: {err}")
            else:
                self.w.notify(self.st, "needs-human", f"brain:{self.at[:13]}", "Autolab: an orchestrator call failed", str(err), self.at)
            return None

    def run(self) -> list[dict]:
        st, at = self.st, self.at
        st["heartbeat"], st["ticks"] = at, st.get("ticks", 0) + 1
        st["slots"], st["caps"], st["tickSeconds"] = self.w.cfg["slots"], self.w.cfg["caps"], self.w.cfg["tickSeconds"]
        S.roll_day(st, at)
        self.apply_inbox()
        if not st.get("modalCheckedAt") or S.minutes(st["modalCheckedAt"], at) >= 10:
            try:
                st["spend"]["modalUsd"] = round(self.w.modal_spend(at[:10]), 2)
                st["modalCheckedAt"] = at
            except Exception as err:  # noqa: BLE001 - billing being down mustn't stop the lab
                print(f"modal billing: {err}", file=sys.stderr)
        if st.get("paused"):
            return self.events
        self.daily()
        for e in S.in_state(st, "implementing"):
            self.implementing(e)
        for e in S.in_state(st, "validating"):
            self.validating(e)
        for e in S.in_state(st, "scoring"):
            self.scoring(e)
        for e in S.in_state(st, "folding"):
            self.folding(e)
        for e in S.in_state(st, "proposed"):
            self.proposed(e)
        for e in S.in_state(st, "accepted"):
            self.accepted(e)
        self.plan()
        return self.events

    # your commands
    def apply_inbox(self):
        for path, c in self.w.inbox():
            try:
                self.apply(c)
            except (KeyError, ValueError) as err:
                self.events.append(S.event(c.get("id"), None, "rejected", self.at, f"your {c['op']}: {err}"))
            self.w.inbox_done(path)

    def apply(self, c: dict):
        st, op = self.st, c["op"]
        if op == "pause":
            if not c["why"] and "test data" in (st.get("paused") or ""):
                st["dataHash"] = self.w.data_hash(st["testSet"])
            st["paused"] = c["why"] or None
            self.events.append(S.event(None, None, "paused" if c["why"] else "resumed", self.at, c["why"] or "by you"))
            return
        e = st["experiments"][c["id"]]
        if op == "match":
            st["matches"].append({"id": e["id"], "gate": e["state"], "user": c["verdict"], "at": self.at})
            self.events.append(S.event(e["id"], e["state"], e["state"], self.at, f"you {c['verdict']}"))
        elif op == "approve":
            if e["state"] != "needs-human":
                raise ValueError(f"{e['id']} is {e['state']}, not waiting on you")
            e["results"]["approved"] = self.at
            self.move(e, "accepted", "approved by you")
        elif op == "retry":
            if e["state"] != "needs-human":
                raise ValueError(f"{e['id']} is {e['state']}, not waiting on you")
            r = e["results"]
            for k in ("testsJob", "decodeJob", "scoreJob", "foldJob"):
                self.w.kill(r.pop(k, None))
            r.pop("testsPassed", None)
            if c.get("note") or not e["worktree"]:
                r["feedback"] = c.get("note") or ""
                self.move(e, "accepted", "retry by you" + (f": {c['note'][:120]}" if c.get("note") else ""))
            else:
                self.move(e, "validating", "retry by you: checks, tests and scoring again")
        elif op == "drop":
            if e["state"] in S.TERMINAL:
                raise ValueError(f"{e['id']} already ended ({e['state']})")
            if e["state"] == "implementing":
                self.w.executor.cancel(e["executor"]["handle"])
            for k in ("testsJob", "decodeJob", "scoreJob", "foldJob"):
                self.w.kill(e["results"].pop(k, None))
            if self.st["foldLock"] == e["id"]:
                self.st["foldLock"] = None
            if e["worktree"]:
                self.discard(e, f"dropped by you: {c['why']}")
            else:
                self.move(e, "discarded", f"dropped by you: {c['why']}")
        else:
            raise ValueError(f"unknown command {op}")

    # daily: digest, rebase the shadow line onto main, re-score it
    def daily(self):
        st, at, w = self.st, self.at, self.w
        day = at[:10]
        job = st.get("baselineJob")
        if job:
            status, res = w.poll(job)
            if status != "running":
                st["baselineJob"] = None
                if status == "done":
                    self.set_main(res["run"], res["sha"], res["practice"])
                else:
                    self.pause(f"the daily re-score of the shadow line failed: {res['error'][:300]}")
        if st.get("day") == day:
            return
        if st.get("day"):
            title, body = N.digest(st, w.events(), st["day"])
            w.notify(st, "digest", st["day"], title, body, at)
        st["day"] = day
        if any(e["state"] == "folding" for e in st["experiments"].values()):
            return
        if not w.rebase_shadow():
            self.pause("rebasing the autolab branch onto main conflicts; resolve it in " + str(WORKTREE))
            return
        st["baselineJob"] = w.job(f"baseline-{day}", w.lab_argv("baseline", "--practice-only", "--note", f"daily rebase {day}"))

    def set_main(self, run: str, sha: str, practice: dict):
        report = self.w.report(run)
        teacher = ((report.get("splits") or {}).get("practice") or {}).get("teacher") or {}
        self.st["main"] = {"sha": sha, "run": run, "practice": practice,
                           "attribution": {c: practice[c] for c in L.COMPONENTS}, "teacherWrongCalls": teacher.get("wrongCalls")}
        P.update_coverage(self.st, P.incidents(report))
        self._ctx = None

    def pause(self, why: str):
        self.st["paused"] = why
        self.events.append(S.event(None, None, "paused", self.at, why))
        self.w.notify(self.st, "paused", why[:60], "Autolab paused", why, self.at)

    # proposed: the novelty judge
    def proposed(self, e):
        if S.timed_out(e, self.at):
            return self.move(e, "needs-human", "the novelty judge didn't answer in time")
        v = self.brain(e, B.judge, e["card"], self.ctx())
        if v is None:
            return
        e["results"]["judge"] = v
        self.move(e, "accepted" if v["accept"] else "discarded", v["why"] if v["accept"] else "restates " + ", ".join(v["restates"]) + ": " + v["why"])

    # accepted: worktree and executor
    def accepted(self, e):
        st, w = self.st, self.w
        if S.over_cap(st, w.cfg["caps"], self.at):
            return self.cap()
        if sum(1 for x in st["experiments"].values() if x["state"] == "implementing") >= w.cfg["slots"]:
            return
        c = e["card"]
        if e["results"].get("retryAfter", "") > self.at:
            return
        if e["tier"] == "bet" and c.get("gpuUsd", 0) > w.cfg.get("betApprovalUsd", 25) and not e["results"].get("approved"):
            return self.move(e, "needs-human", f"a bet asking for ${c['gpuUsd']} of GPU needs your approval: tools/autolab.py approve {e['id']}")
        if not e["worktree"]:
            out = w.lab("new", e["id"], "--component", c["component"], "--hypothesis", c["hypothesis"],
                        "--expect-component", c["expected"]["component"], "--expect-overall", c["expected"]["overall"], "--author", "autolab")
            e["branch"], e["worktree"] = out["branch"], out["worktree"]
            w.exclude(".autolab/")
        self.start_executor(e, e["results"].pop("feedback", ""))

    def start_executor(self, e, feedback: str = ""):
        w, cfg = self.w, self.w.cfg
        model = cfg["executor"]["model"] if e["attempts"]["executor"] == 0 else cfg["executor"]["fallbackModel"]
        e["executor"] = {"model": model, "handle": w.executor.start({**e["card"], "executorModel": model}, e["worktree"], brief(e, self.ctx(), feedback))}
        e["attempts"]["executor"] += 1
        S.charge(self.st, self.at, runs=1)
        self.move(e, "implementing", f"{model}" + (f" (feedback: {feedback[:120]})" if feedback else ""))

    def cap(self):
        why = S.over_cap(self.st, self.w.cfg["caps"], self.at)
        self.w.notify(self.st, "cap", self.st["spend"]["day"], "Autolab hit a daily cap", why, self.at)

    # implementing: serve practice-score requests, poll the executor
    def serve_practice(self, e):
        """An executor asks for a practice score by writing .autolab/practice-request-<n>.json in its worktree (it runs
        in an environment that can't reach tilawa-lab); the supervisor scores and writes practice-result-<n>.json."""
        d = Path(e["worktree"], ".autolab")
        jobs = e["results"].setdefault("practiceJobs", {})
        for req in sorted(d.glob("practice-request-*.json")) if d.exists() else []:
            n = req.stem.rsplit("-", 1)[1]
            result = d / f"practice-result-{n}.json"
            if result.exists():
                continue
            if n not in jobs:
                if e["attempts"]["practiceScores"] >= PRACTICE_SCORES:
                    result.write_text(json.dumps({"error": f"all {PRACTICE_SCORES} practice scores are used"}))
                    continue
                e["attempts"]["practiceScores"] += 1
                jobs[n] = self.w.job(f"practice-{e['id']}-{n}", lab_argv("run", e["worktree"]))
                continue
            status, res = self.w.poll(jobs[n])
            if status != "running":
                out = {k: res.get(k) for k in ("compare", "verdict", "practice", "sha")} if status == "done" else {"error": tail(res)}
                result.write_text(json.dumps(out, ensure_ascii=False))

    def implementing(self, e):
        w, h = self.w, e["executor"]["handle"]
        self.serve_practice(e)
        status = w.executor.poll(h)
        if status == "running":
            if S.timed_out(e, self.at):
                w.executor.cancel(h)
                self.finish_executor(e)
                self.move(e, "abandoned", f"the executor ran past {S.timeout(e)} minutes")
            return
        res = self.finish_executor(e)
        if status == "finished":
            return self.move(e, "validating", res["summary"][:200])
        if outage(res["summary"]):
            e["attempts"]["executor"] -= 1
            e["results"]["retryAfter"] = shift(self.at, 10)
            e["state"] = "accepted"
            self.events.append(S.event(e["id"], "implementing", "accepted", self.at, f"builder couldn't start (outage); retrying after {e['results']['retryAfter']}"))
            return
        if e["attempts"]["executor"] < 2:
            self.events.append(S.event(e["id"], "implementing", "implementing", self.at, "executor failed; retrying with the fallback model"))
            e["state"] = "accepted"
            return self.start_executor(e, f"A previous attempt failed: {res['summary'][:500]}")
        self.move(e, "abandoned", f"the executor failed twice: {res['summary'][:200]}")

    def finish_executor(self, e) -> dict:
        res = self.w.executor.collect(e["executor"]["handle"])
        self.w.transcript(e["id"], res["transcript"])
        e["results"]["executor"] = {"summary": res["summary"], "hours": res["hours"], "model": e["executor"]["model"]}
        S.charge(self.st, self.at, hours=res["hours"])
        return res

    def bounce(self, e, why: str):
        """Back to the executor once with the reason; a second time ends the experiment."""
        if e["attempts"].get("bounces", 0) >= 1 or e["attempts"]["executor"] >= 3:
            return self.discard(e, why)
        e["attempts"]["bounces"] = e["attempts"].get("bounces", 0) + 1
        e["state"] = "accepted"
        self.events.append(S.event(e["id"], "validating", "accepted", self.at, f"sent back: {why[:200]}"))
        self.start_executor(e, why)

    def discard(self, e, why: str):
        try:
            self.w.lab("discard", e["id"], "--force", "--note", why[:500])
        except Stop as err:
            print(f"lab discard {e['id']}: {err}", file=sys.stderr)
        self.move(e, "discarded", why)
        self.after(e, {"state": "discarded", "why": why, "gate": e["results"].get("gate")})

    # validating: cheap checks first, then tests on Modal, then a listener decode
    def validating(self, e):
        w, st, r = self.w, self.st, e["results"]
        if S.timed_out(e, self.at):
            w.kill(r.get("testsJob"))
            w.kill(r.get("decodeJob"))
            return self.move(e, "abandoned", "validation ran too long")
        if r.get("testsJob"):
            status, res = w.poll(r["testsJob"])
            if status == "running":
                return
            r.pop("testsJob")
            if status == "failed":
                return self.bounce(e, "build or tests fail on Modal: " + tail(res))
            r["testsPassed"] = True
        if r.get("decodeJob"):
            status, res = w.poll(r["decodeJob"])
            if status == "running":
                return
            r.pop("decodeJob")
            if status == "failed":
                return self.move(e, "needs-human", "the listener decode failed: " + tail(res))
            return self.score_with_listener(e, r["listener"])
        if r.get("testsPassed"):
            return self.after_tests(e)
        wt = e["worktree"]
        if w.git("status", "--porcelain", "--untracked-files=all", cwd=wt):
            w.git("add", "-A", cwd=wt)
            w.git("commit", "-q", "-m", f"autolab: {e['id']} executor's uncommitted work", cwd=wt)
        if w.commits(e) == 0:
            return self.discard(e, "the executor committed no change")
        card = json.loads(Path(wt, L.CARD).read_text())
        c = e["card"]
        if (card["id"], card["component"], card["hypothesis"], card["expected"]) != (c["id"], c["component"], c["hypothesis"], c["expected"]):
            return self.discard(e, "the card was edited after it was accepted")
        diff = w.diff(e)
        h = G.hygiene_check(diff)
        if not h["ok"]:
            return self.bounce(e, "hygiene: " + h["why"])
        r["tuning"] = G.is_tuning(diff)
        if r["tuning"] and P.tuning_blocked(st, e["id"]):
            return self.discard(e, "another tuning experiment is already open (at most one)")
        v = self.brain(e, B.review_diff, c, diff)
        if v is None:
            return
        r["diffReview"] = v
        if not v["ok"]:
            return self.bounce(e, "diff review: " + v["why"] + " " + "; ".join(v["concerns"][:5]))
        if st.get("dataHash") and w.data_hash(st["testSet"]) != st["dataHash"]:
            return self.pause("the test data on the volume changed since autolab started; check /lab/data before resuming")
        r["testsJob"] = w.job(f"tests-{e['id']}-{e['attempts']['executor']}", w.lab_argv("run", wt, "--shards", w.cfg.get("testShards", 6), "--name", f"autolab-tests-{e['id']}", "--cmd", test_command(w.cfg)))

    def after_tests(self, e):
        r, w, c = e["results"], self.w, e["card"]
        r.pop("testsPassed", None)
        if c["component"] != "listener" or c.get("noDecode"):
            return self.start_scoring(e)
        sha = w.git("rev-parse", "--short", "HEAD", cwd=e["worktree"])
        r["listener"] = f"exp-{e['id']}-{sha}"[:60]
        practice = (vol_get(VOL_LEDGER) or {}).get("testSets", {}).get(self.st["testSet"], {}).get("sessions", {}).get("practice", [])
        decode = ["modal", "run", str(REPO / "modal/lab_decode.py"), "--listener", r["listener"], "--model", w.cfg["listenerModel"],
                  "--test-set", self.st["testSet"].split(".v")[0], "--variants", "phone,studio", "--only", ",".join(practice)]
        r["decodeJob"] = w.job(f"decode-{e['id']}-{sha}", json_exit(decode), env={"LAB_SRC": e["worktree"]})

    def score_with_listener(self, e, listener: str):
        wt = e["worktree"]
        f = Path(wt, L.CARD)
        card = json.loads(f.read_text())
        card["listener"] = listener
        f.write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
        self.w.git("commit", "-q", "-am", f"autolab: score {e['id']} with listener evidence {listener}", cwd=wt)
        self.start_scoring(e)

    def start_scoring(self, e):
        e["results"]["scoreJob"] = self.w.job(f"score-{e['id']}-{self.w.git('rev-parse', '--short', 'HEAD', cwd=e['worktree'])}",
                                              self.w.lab_argv("run", e["worktree"]))
        self.move(e, "scoring")

    # scoring: the branch run, then the gate as a prefilter
    def scoring(self, e):
        w, r = self.w, e["results"]
        if S.timed_out(e, self.at):
            w.kill(r.get("scoreJob"))
            return self.move(e, "abandoned", "scoring ran too long")
        status, res = w.poll(r["scoreJob"])
        if status == "running":
            return
        r.pop("scoreJob")
        if status == "failed":
            return self.move(e, "needs-human", "scoring failed: " + tail(res))
        r["branchRun"] = {k: res[k] for k in ("run", "base", "sha", "compare", "verdict", "practice")}
        base = w.report(res["base"])
        g = G.gate(base, w.report(res["run"]), e["card"], w.diff(e))
        r["gate"] = g
        if g["verdict"] != "pass":
            return self.discard(e, "gate on the branch: " + "; ".join(c["why"] for c in g["checks"] if c["blocking"] and not c["ok"]))
        self.move(e, "folding", "gate passed on the branch; queued to fold")

    # folding: one at a time, rebase + re-score + gate + merge in lab.py
    def folding(self, e):
        w, st, r = self.w, self.st, e["results"]
        job = r.get("foldJob")
        if not job:
            if not S.take_lock(st, e["id"]):
                return
            r["foldJob"] = w.job(f"fold-{e['id']}-{int(time.time())}", w.lab_argv("fold", e["id"], "--practice-only", "--gate"))
            return
        if S.timed_out(e, self.at):
            w.kill(job)
            st["foldLock"] = None
            return self.move(e, "abandoned", "folding ran too long")
        status, res = w.poll(job)
        if status == "running":
            return
        r.pop("foldJob")
        st["foldLock"] = None
        if status == "failed":
            err = res.get("error", "")
            if "conflicts" in err and e["attempts"].get("conflicts", 0) == 0:
                e["attempts"]["conflicts"] = 1
                e["state"] = "accepted"
                return self.start_executor(e, "Rebasing onto the autolab branch conflicts. Rebase your branch onto `autolab`, resolve the conflicts, keep the change's intent, and commit.\n" + err[-1500:])
            return self.move(e, "needs-human", "fold failed: " + err[:300])
        r["fold"] = {k: res.get(k) for k in ("run", "base", "sha", "compare", "why", "gate", "head", "practice")}
        state = {"folded": "folded", "superseded": "superseded", "gate-failed": "discarded"}.get(res["state"])
        if not state:
            return self.move(e, "needs-human", f"fold ended in state {res['state']}")
        if state == "discarded" and Path(e["worktree"]).exists():
            w.git("worktree", "remove", "--force", e["worktree"], check=False)
        note = res.get("why", "") if state != "discarded" else "gate after rebase: " + "; ".join(
            c["why"] for c in (res.get("gate") or {}).get("checks", []) if c["blocking"] and not c["ok"])
        self.move(e, state, note)
        if state == "folded":
            self.events[-1].update(sha=res["sha"], wrongCalls=res["practice"]["wrongCalls"])
            self.set_main(res["run"], res["sha"], res["practice"])
            w.notify(st, "digest", f"fold:{e['id']}", f"Autolab kept {e['id']}",
                     f"{e['card']['hypothesis']}\nPractice change: overall {res['compare']['overall']['change']:+d}.", self.at)
        self.after(e, {"state": state, "why": note, "gate": res.get("gate"), "compare": res.get("compare")})

    # after an experiment ends: journal entry and maybe a follow-up
    def after(self, e, outcome: dict):
        for cl in [e["card"].get("cluster")] if e["card"].get("cluster") else []:
            P.mark_tried(self.st, cl, e["id"])
        v = self.brain(None, B.review, e, outcome, self.ctx())
        if v is None:
            return
        self.w.journal_add(f"{e['id']} ({e['card']['component']}, {e['tier']}): {outcome['state']}", v["journal"]
                           + (f"\n\nDead end: {v['deadEnd']}" if v.get("deadEnd") else ""), self.at)
        if v.get("followUp"):
            self.add(v["followUp"])

    def add(self, card: dict):
        if card["id"] in self.st["experiments"]:
            return
        e, ev = S.add(self.st, card, self.at)
        main = self.st.get("main") or {}
        e["base"] = {"sha": main.get("sha"), "wrongCalls": (main.get("practice") or {}).get("wrongCalls")}
        self.events.append(ev)
        self._ctx = None

    # planning: fill free slots; strategy on a plateau
    def plan(self):
        st, w, at = self.st, self.w, self.at
        if not st.get("main") or S.over_cap(st, w.cfg["caps"], at):
            return
        for c in P.stuck_for_user(st, at):
            st["counters"]["frozen"][c]["asked"] = at
            w.notify(st, "needs-human", f"stuck:{c}:{at[:10]}", f"Autolab is stuck on the {c}",
                     f"No fold in a day after a strategy change. Read the latest memo in the journal and tell autolab where to look.", at)
        alloc = P.allocate(st["main"]["attribution"], w.cfg["slots"], set(st["counters"]["frozen"]))
        free = P.free_slots(st, alloc)
        newly = P.update_plateau(st, at, st.get("started") or at)
        if newly and free:
            v = self.brain(None, B.strategy, self.ctx(), newly, free)
            if v:
                w.journal_add(f"Strategy: stuck on {', '.join(newly)}", v["memo"], at)
                for c in v["cards"]:
                    self.add(c)
                free = P.free_slots(st, P.allocate(st["main"]["attribution"], w.cfg["slots"], set(st["counters"]["frozen"])))
        last = st.get("lastPlanAt")
        if not free or (last and S.minutes(last, at) < w.cfg["planEveryMinutes"]):
            return
        st["lastPlanAt"] = at
        cards = self.brain(None, B.plan, self.ctx(), free)
        for c in cards or []:
            self.add(c)


def json_exit(argv: list[str]) -> list[str]:
    """A command that prints no JSON, wrapped so its job ends in a JSON line saying how it exited (what poll reads)."""
    import shlex  # noqa: PLC0415
    return ["bash", "-c", f'{shlex.join(argv)}; code=$?; echo "{{\\"exitCode\\": $code}}"']


def test_command(cfg: dict) -> str:
    """The validation command; a `bash tools/<script>` is sent as the main repo's script text, so older worktrees work
    and an experiment's own copy of the script never decides its tests."""
    cmd = cfg["tests"]
    if cmd.startswith("bash tools/") and (REPO / cmd[5:]).exists():
        return (REPO / cmd[5:]).read_text()
    return cmd


def outage(summary: str) -> bool:
    """A builder that never got going for reasons outside the experiment (the model service at capacity, network)."""
    s = (summary or "").lower()
    return s.startswith(("startup", "sdk-error")) and any(k in s for k in ("unavailable", "capacity", "timeout", "timed out", "network", "connect"))


def shift(at: str, minutes: int) -> str:
    from datetime import datetime, timedelta  # noqa: PLC0415
    return (datetime.fromisoformat(at) + timedelta(minutes=minutes)).isoformat()


def tail(res: dict | None) -> str:
    res = res or {}
    return (res.get("log") or res.get("error") or json.dumps(res))[-2500:]


def brief(e: dict, ctx: dict, feedback: str = "") -> str:
    """What the executor is told: the card, where to look, the rules, and how to finish."""
    c = e["card"]
    targets = "\n".join(f"- `{t}`" for t in c.get("targets", []))
    tried = [h for h in ctx["history"] if h.get("cluster") == c.get("cluster") and h["id"] != c["id"]]
    return f"""You are implementing one experiment in Tilawa's lab, in this git worktree (branch {e.get('branch')}).
Read AGENTS.md first. The experiment's card is lab-experiment.json; do not edit it.

{B.ABOUT}

## The card

- id: {c['id']} ({c['component']}, {e['tier']})
- hypothesis: {c['hypothesis']}
- mechanism: {c.get('mechanism', '')}
- expected change on practice: component {c['expected']['component']:+d}, overall {c['expected']['overall']:+d}
- kill criterion: {c.get('killCriterion', '')}

## Target incidents (cluster {c.get('cluster')})

{targets}

Inspect a case with the trace tool (LAB_DATA must point at a copy of the practice data):
`pnpm --filter @tilawa/core exec tsx src/bench/lab-trace-cli.ts --set {ctx['testSet'].split('.v')[0]} --session <session> --at <seconds>`
Run code on Modal with `python3 tools/lab.py run . --shards N --cmd '<command>'`.

## Tried before on this cluster

{json.dumps(tried, ensure_ascii=False, indent=1) if tried else 'nothing yet'}

## Rules

- Change engine code only. Never touch the scorer or lab tooling (packages/core/src/bench/lab*.ts, tools/lab*.py,
  tools/autolab*.py, modal/lab_*.py, docs/benchmarks/lab-*), test data, manifests or evidence.
- Never special-case sessions, readers, verses or anything else specific to the test set. A reviewer and a code check
  reject that.
- Add or adjust unit tests for the behaviour you change; keep existing tests passing
  (`pnpm --filter @tilawa/core test`, `pnpm --filter @tilawa/sdk test`).
- You may score the practice set at most {PRACTICE_SCORES} times in total: `python3 tools/autolab.py practice-score .`
  (about 11 minutes each). Don't tune constants against it.
- Write `.scratch/lab/experiments/{c['id']}.md` like the other write-ups there: the prediction, the mechanism, what you
  validated and its limits.
- Commit everything on this branch with clear messages. Finish with a short summary of what you changed and why.
{f'''
## Feedback from the previous attempt

{feedback}
''' if feedback else ''}"""


# Commands


def cmd_init(a):
    cfg, w = config(), None
    if not CONFIG.exists():
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=1) + "\n")
        print(f"wrote {CONFIG}; set notify.topic and the executor's env there")
    git = lambda *x, **k: subprocess.run(["git", *x], cwd=REPO, capture_output=True, text=True, **k)
    if git("rev-parse", "--verify", "--quiet", BRANCH).returncode:
        git("branch", BRANCH, "main", check=True)
    if not WORKTREE.exists():
        git("worktree", "add", str(WORKTREE), BRANCH, check=True)
    w = World(cfg, executor=object())
    lab_led = vol_get("/ledger.json")
    if not vol_get(VOL_LEDGER):
        ts = L.test_set(lab_led)
        a2 = ["baseline", "--practice-only", "--manifest", ts["manifest"], "--note", "autolab start"] + (["--exclude", ts["exclude"]] if ts.get("exclude") else [])
        print("scoring the shadow line's practice baseline (about 11 minutes)...", flush=True)
        res = w.lab(*a2)
    else:
        m = L.main_score(vol_get(VOL_LEDGER))
        res = {"run": m["run"], "sha": m["sha"], "practice": L.brief(m["practice"])}
    st = w.load() or S.empty(lab_led["testSet"])
    st.update(started=st.get("started") or S.now(), tickSeconds=cfg["tickSeconds"], caps=cfg["caps"], day=S.now()[:10])
    t = Tick(w, st, S.now())
    t.set_main(res["run"], res["sha"], res["practice"])
    st["dataHash"] = w.data_hash(st["testSet"])
    w.save(st, [S.event(None, None, "started", S.now(), f"shadow line {res['sha']}")])
    print(f"autolab ready: shadow line {res['sha']}, practice {res['practice']['wrongCalls']} wrong calls; run `tools/autolab.py loop`")


def tick_once(w: World) -> dict:
    st = w.load()
    if not st:
        raise Stop("no autolab state: run `tools/autolab.py init` first")
    t = Tick(w, st, S.now())
    try:
        events = t.run()
    except Exception as err:  # noqa: BLE001 - one bad tick must not stop the loop; it pauses and pages instead
        t.pause(f"tick failed: {type(err).__name__}: {err}")
        events = t.events
    w.save(st, events)
    return st


def cmd_tick(a):
    st = tick_once(World(config()))
    print(f"tick {st['ticks']}: " + ", ".join(f"{k} {v}" for k, v in sorted(count_states(st).items())))


def count_states(st):
    out = {}
    for e in st["experiments"].values():
        out[e["state"]] = out.get(e["state"], 0) + 1
    return out


TICK_LIMIT_S = 25 * 60


def cmd_loop(a):
    """Ticks forever. A tick stuck past TICK_LIMIT_S exits the process (run it under a restarting shell loop): state is
    on the volume and jobs are detached, so a fresh process picks up where this one hung."""
    import threading  # noqa: PLC0415
    w, started = World(config()), [time.time()]

    def watchdog():
        while True:
            time.sleep(60)
            if time.time() - started[0] > TICK_LIMIT_S:
                print(f"{S.now()} tick stuck for over {TICK_LIMIT_S // 60} min; exiting so the loop restarts", flush=True)
                os.killpg(os.getpgid(0), signal.SIGTERM) if os.getpgid(0) == os.getpid() else None
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    while True:
        t0 = started[0] = time.time()
        w.cfg = config()
        try:
            st = tick_once(w)
            print(f"{S.now()} tick {st['ticks']}: {count_states(st)}" + (f" PAUSED: {st['paused']}" if st.get("paused") else ""), flush=True)
        except Stop as err:
            print(f"{S.now()} {err}", flush=True)
        time.sleep(max(5, w.cfg["tickSeconds"] - (time.time() - t0)))


def send(op: str, **args) -> None:
    """Your commands go to an inbox the loop applies at the start of its next tick, so a tick in flight can't overwrite them."""
    vol_put(f"{VOL_INBOX}/{S.now().replace(':', '')}-{op}-{os.getpid()}.json", {"op": op, "at": S.now(), **args})
    print(f"{op} queued; the loop applies it within one tick (about {config()['tickSeconds'] // 60} min)")


def cmd_pause(a, why=None):
    send("pause", why=why if why is not None else (" ".join(a.why) or "paused by you"))


def cmd_waiting(a):
    st = World(config(), executor=object()).load()
    if not st["needsHuman"] and not st.get("paused"):
        print("nothing is waiting on you")
    if st.get("paused"):
        print(f"autolab is paused: {st['paused']}\n  -> fix it, then: tools/autolab.py resume")
    for h in st["needsHuman"]:
        e = st["experiments"].get(h["id"], {})
        c = e.get("card", {})
        print(f"\n{h['id']} ({c.get('component')}, {e.get('tier')}), waiting since {h['at']}\n  {c.get('hypothesis', '')}\n  why: {h['what']}")
        if "approval" in h["what"]:
            print(f"  -> tools/autolab.py approve {h['id']}      or   tools/autolab.py drop {h['id']} 'too expensive'")
        else:
            print(f"  -> tools/autolab.py retry {h['id']}        (re-run its checks, tests and scoring)")
            print(f"     tools/autolab.py retry {h['id']} --note 'what to change'   (back to the builder with your note)")
            print(f"     tools/autolab.py drop {h['id']} 'why'")


def cmd_status(a):
    st = World(config(), executor=object()).load()
    if a.json:
        print(json.dumps(st, ensure_ascii=False))
        return
    m = (st.get("main") or {}).get("practice") or {}
    print(f"heartbeat {st['heartbeat']} (tick {st['ticks']}){'; PAUSED: ' + st['paused'] if st.get('paused') else ''}")
    print(f"shadow line {(st.get('main') or {}).get('sha')}: practice {m.get('wrongCalls')} wrong calls")
    for e in sorted(st["experiments"].values(), key=lambda e: e["since"], reverse=True)[:30]:
        print(f"  {e['state']:12} {e['id']:40} {e['card']['component']:8} {e['tier']:8} {e['note'][:80]}")


def cmd_practice_score(a):
    """Run by an executor: asks the supervisor for a practice score of this worktree and waits for it (spec decision 20)."""
    d = Path(a.worktree).resolve() / ".autolab"
    d.mkdir(exist_ok=True)
    n = len(list(d.glob("practice-request-*.json"))) + 1
    if n > PRACTICE_SCORES:
        sys.exit(f"autolab: all {PRACTICE_SCORES} practice scores are used")
    (d / f"practice-request-{n}.json").write_text(json.dumps({"at": S.now()}))
    result = d / f"practice-result-{n}.json"
    print(f"asked the supervisor for practice score {n} of {PRACTICE_SCORES}; this takes about 15 minutes", flush=True)
    for _ in range(60 * 60 // 15):
        if result.exists():
            out = json.loads(result.read_text())
            if "error" in out:
                sys.exit(f"autolab: {out['error']}")
            print("\n".join(L.format_compare(out["compare"])))
            print(f"verdict: {out['verdict']}; {PRACTICE_SCORES - n} practice score(s) left")
            return
        time.sleep(15)
    sys.exit("autolab: no practice result within an hour; the supervisor may be paused")


def cmd_watchdog(a):
    cfg = config()
    w = World(cfg, executor=object())
    st = w.load()
    if not st or not st.get("heartbeat"):
        return
    age = S.minutes(st["heartbeat"], S.now())
    if age * 60 > 3 * st.get("tickSeconds", 120):
        w.notify(st, "stale", st["heartbeat"], "Autolab has stopped", f"No tick for {age:.0f} minutes (last {st['heartbeat']}).", S.now())
        vol_put(VOL_STATE, st)


def known(id: str) -> None:
    st = World(config(), executor=object()).load()
    if id not in st["experiments"]:
        sys.exit(f"autolab: no experiment {id}")


def cmd_eli5(a):
    """Plain one-liners for every experiment without a card eli5 (the hand-run lab's and early autolab cards), kept in
    /lab/autolab/eli5.json for the dashboard."""
    w = World(config(), executor=object())
    have = vol_get(VOL_ELI5) or {}
    items = {}
    for led in w.ledgers():
        for id, e in led.get("experiments", {}).items():
            items[id] = e["card"]["hypothesis"]
    for id, e in (w.load() or {"experiments": {}})["experiments"].items():
        if not e["card"].get("eli5"):
            items[id] = e["card"]["hypothesis"]
    todo = {k: v for k, v in items.items() if k not in have}
    for i in range(0, len(todo), 12):
        batch = dict(list(todo.items())[i:i + 12])
        have.update(B.eli5(w.cfg, batch))
        print(f"{len(have)} explained", flush=True)
    vol_put(VOL_ELI5, have)
    for k in sorted(todo):
        print(f"{k}: {have[k]}")


def cmd_approve(a):
    known(a.id)
    send("approve", id=a.id)


def cmd_retry(a):
    known(a.id)
    send("retry", id=a.id, note=a.note)


def cmd_drop(a):
    known(a.id)
    send("drop", id=a.id, why=" ".join(a.why) or "dropped by you")


def cmd_match(a):
    known(a.id)
    send("match", id=a.id, verdict=a.verdict)


def main():
    p = argparse.ArgumentParser(description="Autolab supervisor")
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("init")
    sub.add_parser("tick")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=int, default=config()["tickSeconds"])
    pp = sub.add_parser("pause")
    pp.add_argument("why", nargs="*")
    sub.add_parser("resume")
    sp = sub.add_parser("status")
    sp.add_argument("--json", action="store_true")
    ps = sub.add_parser("practice-score")
    ps.add_argument("worktree", nargs="?", default=".")
    sub.add_parser("watchdog")
    sub.add_parser("eli5")
    ap = sub.add_parser("approve")
    ap.add_argument("id")
    sub.add_parser("waiting")
    rp = sub.add_parser("retry")
    rp.add_argument("id")
    rp.add_argument("--note", default="")
    dp = sub.add_parser("drop")
    dp.add_argument("id")
    dp.add_argument("why", nargs="*")
    mp = sub.add_parser("match")
    mp.add_argument("id")
    mp.add_argument("verdict", choices=("agree", "disagree"))
    a = p.parse_args()
    try:
        {"init": cmd_init, "tick": cmd_tick, "loop": cmd_loop, "pause": cmd_pause, "resume": lambda a: cmd_pause(a, why=""),
         "status": cmd_status, "practice-score": cmd_practice_score, "watchdog": cmd_watchdog, "match": cmd_match,
         "approve": cmd_approve, "waiting": cmd_waiting, "retry": cmd_retry, "drop": cmd_drop, "eli5": cmd_eli5}[a.op](a)
    except Stop as err:
        sys.exit(f"autolab: {err}")


if __name__ == "__main__":
    main()
