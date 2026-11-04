"""Autolab supervisor (.scratch/autolab/spec.md; ticket 04): the reconciling loop that runs the lab unattended.

    uv run --python cpython-3.12-macos-aarch64-none --with modal==1.3.4 tools/autolab.py <command>

    init                     state, config, practice baseline of the main line
    tick                     one pass (cron)
    loop [--every 120]       tick forever (phase A)
    pause [why] | resume
    status [--json]
    ab [--json]              the builder A/B: per arm, what its builders did and what came of it (tools/autolab_ab.py)
    practice-score [worktree]   an executor's practice score through the supervisor (at most 3 per card, decision 20)
    modal-run [--shards N] [--gpu L4] [--image node|train] [--timeout MIN] [--no-wait] -- '<command>'
                             an executor's command on Modal through the supervisor (decision 14; modal/autolab_exec.py)
    modal-result <n>         wait for an executor's modal-run started with --no-wait
    sync-exec-data           refresh tilawa-lab-exec, the builders' practice-only data (the loop also does it per test set)
    watchdog                 notify when the heartbeat is stale (run from cron or launchd, apart from the loop)
    match <id> agree|disagree   your verdict on a gate decision (ticket 11)
    waiting                  what needs you, with the command for each
    approve <id>             let a bet of betApprovalUsd or more start
    retry <id> [--note ...]  re-run checks, tests and scoring; with a note, back to the builder with it; an abandoned
                             experiment gets a fresh builder in its worktree
    drop <id> [why]          end an experiment
    revert <id> [why]        take a folded experiment back out of main (git revert of its commits, then a re-score)
    reopen <id>              send an abandoned experiment with its worktree back to validation
    eli5                     plain one-liners for experiments whose cards predate the eli5 field
Your commands (pause, resume, approve, retry, drop, match) go to /lab/autolab/inbox/ and the loop applies them at the
start of its next tick, so a tick in flight never overwrites them.

Every lab operation goes through tools/lab.py on the main line: `--main main --ledger /ledger.json --json`. A fold lands
on `main` with the tag autolab/fold/<id>; `revert <id>` takes it back out. Every `examEvery` folds the main line is
scored on the exam too, as a tripwire that only says whether the exam followed practice (never fed to the models).
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
import autolab_ab as AB  # noqa: E402
import autolab_brain as B  # noqa: E402
import autolab_exec as X  # noqa: E402
import autolab_isolation as I  # noqa: E402
import autolab_notify as N  # noqa: E402
import autolab_portfolio as P  # noqa: E402
import autolab_state as S  # noqa: E402
import lab_gate as G  # noqa: E402
import lab_ledger as L  # noqa: E402
from autolab_exec import make as make_executor  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from lab import REPO, Stop, vol_get, vol_ls, vol_put  # noqa: E402

LAB = REPO / "tools/lab.py"
EXEC = REPO / "modal/autolab_exec.py"
CONFIG = REPO / ".scratch/autolab/config.json"
RUN = REPO / ".scratch/autolab/run"
BRANCH, WORKTREE = "main", REPO.parent / "tilawa-autolab"
VOL_STATE, VOL_EVENTS, VOL_LEDGER, VOL_JOURNAL = "/autolab/state.json", "/autolab/events.jsonl", "/ledger.json", "/autolab/journal.md"
SHADOW_LEDGER = "/autolab/ledger.json"
FOLD_TAG = "autolab/fold/"
VOL_INBOX, VOL_ELI5 = "/autolab/inbox", "/autolab/eli5.json"
NOT_ENGINE = re.compile(r"(^\.scratch/|^docs/|(^|/)lab-experiment\.json$|\.md$|(^|/)test/|\.test\.ts$|^tools/test_)")
OWN_DATA = re.compile(r"(\.exp-[^/]+\.(phone|studio)\.evidence\.json|/decode-v1\.json|/review/dashboard\.json)$")  # review queue: scoring reads it only via lab apply-flags
PRACTICE_SCORES = 3
OFFLINE = re.compile(r"could not connect|cannot connect|timed out after|connection (reset|refused|error)|name resolution", re.I)
ARM_WAIT_MINUTES = 20  # a card whose arm is full and whose builder never started moves to an arm with room
UNSCORED_KILL_WINS = 3  # a builder that stops with this many targets won and no practice score is sent back to score
ARM_DOWN_AFTER, ARM_DOWN_MINUTES = 3, 60  # an arm whose last builders all failed to start rests, then gets one more try
SOL = {"id": "gpt-6.1-sol", "params": [{"id": "effort", "value": "high"}]}
DEFAULT_CONFIG = {
    "slots": 6, "planEveryMinutes": 30, "tickSeconds": 120,
    "caps": {"agentRuns": 60, "agentHours": 40, "modalUsd": 150},
    "models": {**dict.fromkeys(("planner", "reviewer", "strategist"), {"id": "claude-opus-5-5", "params": [
                   {"id": "context", "value": "1m"}, {"id": "effort", "value": "medium"}, {"id": "fast", "value": "false"}]}),
               **dict.fromkeys(("judge", "diffReviewer"), SOL)},
    "executor": {"env": {},
                 "runtimes": [{"name": "sol", "runtime": "codex-azure", "azure": {"endpoint": "", "deployment": "gpt-6.1-sol", "apiVersion": ""}, "effort": "high"}],
                 "split": {"sol": 1}},
    "exec": dict(I.LIMITS),
    "betApprovalUsd": 100,
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
        self.executor = executor  # one executor for every arm (tests, dry runs); None: one per arm, made on first use
        self._executors = {}
        self._reports, self._procs = {}, {}
        self.background, self.futures = True, {}
        self.pool = ThreadPoolExecutor(8)

    # builders
    def arm_executor(self, arm: dict):
        """The executor that starts `arm`'s builders; made again when the arm's config changes."""
        if self.executor is not None:
            return self.executor
        key = json.dumps(X.arm_config(self.cfg, arm)["executor"], sort_keys=True, default=str)
        if key not in self._executors:
            self._executors[key] = make_executor(self.cfg, arm)
        return self._executors[key]

    def builder(self, handle: dict):
        """The executor that owns a stored handle, by its runtime."""
        if self.executor is not None:
            return self.executor
        rt, arms = handle.get("runtime"), X.arms(self.cfg)
        arm = next((a for a in arms if a["runtime"] == rt and a["name"] == handle.get("arm")), None) \
            or next((a for a in arms if a["runtime"] == rt), {"name": rt, "runtime": rt})
        key = ("handles", rt, arm["name"])
        if key not in self._executors:
            self._executors[key] = make_executor(self.cfg, arm, check=False)
        return self._executors[key]

    # state on the volume
    def load(self) -> dict | None:
        return vol_get(VOL_STATE)

    def save(self, st: dict, events: list[dict]) -> None:
        if events:
            RUN.mkdir(parents=True, exist_ok=True)
            with open(RUN / "events.jsonl", "a") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
            subprocess.run(["modal", "volume", "put", "--force", "tilawa-lab", str(RUN / "events.jsonl"), VOL_EVENTS], capture_output=True)
        for wait in (2, 5, 10, 20, 40, 0):
            try:
                return vol_put(VOL_STATE, st)
            except Stop:
                if not wait:
                    raise
                time.sleep(wait)  # e.g. "too many layers in volume": Modal compacts, then accepts writes again

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

    def exec_argv(self, *args) -> list[str]:
        return [sys.executable, str(EXEC), *map(str, args)]

    def test_set(self, ts: str) -> dict:
        """The autolab ledger's record of a test set version: manifest, exclude, sessions by split."""
        t = ((vol_get(VOL_LEDGER) or {}).get("testSets") or {}).get(ts)
        if not t:
            raise Stop(f"test set {ts} is not in {VOL_LEDGER}")
        return t

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

    def revert_fold(self, id: str) -> str:
        """Reverts a fold's commits (its card commit through its tag) on main, in a scratch checkout, then moves main
        (fast-forward in the worktree that has it checked out); returns the new tip. Stop on a conflict."""
        tip = self.git("rev-parse", FOLD_TAG + id)
        start = self.git("log", "--format=%H", "--fixed-strings", f"--grep=Lab experiment {id} (", tip).split()
        if not start:
            raise Stop(f"no card commit for {id} under {FOLD_TAG + id}")
        tmp = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip()) / "main"
        self.git("worktree", "add", "--detach", "-q", tmp, BRANCH)
        try:
            p = subprocess.run(["git", "revert", "--no-edit", f"{start[-1]}^..{tip}"], cwd=tmp, capture_output=True, text=True)
            if p.returncode:
                subprocess.run(["git", "revert", "--abort"], cwd=tmp, capture_output=True)
                raise Stop(f"reverting {id} conflicts with later work on main:\n{(p.stderr or p.stdout)[-1200:]}")
            new = self.git("rev-parse", "HEAD", cwd=tmp)
        finally:
            self.git("worktree", "remove", "--force", tmp, check=False)
        checked = [blk.splitlines()[0][9:] for blk in self.git("worktree", "list", "--porcelain").split("\n\n")
                   if f"branch refs/heads/{BRANCH}" in blk.splitlines()]
        if checked:
            self.git("merge", "-q", "--ff-only", new, cwd=checked[0])
        else:
            self.git("branch", "-f", BRANCH, new)
        return new

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
        return [x for x in (vol_get(VOL_LEDGER),) if x]

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


def practice_ids(t: dict) -> list[str]:
    """The practice sessions of a ledger test set; never empty and never an exam session (an empty --only decodes all)."""
    practice, exam = t["sessions"].get("practice") or [], set(t["sessions"].get("exam") or [])
    if not practice or exam & set(practice):
        raise Stop(f"test set {t.get('id') or t.get('manifest')}: no practice sessions, or some are also exam")
    return practice


def exec_spec(t: dict, cfg: dict, listener: str = L.DEFAULT_LISTENER) -> dict:
    """What modal/autolab_exec.py's sync copies into tilawa-lab-exec: the main line's current listener evidence among them."""
    return {"base": t["base"], "manifest": t["manifest"], "practice": practice_ids(t),
            "listeners": sorted({L.DEFAULT_LISTENER, listener, "teacher"}), "models": sorted({cfg["listenerModel"], *cfg.get("execModels", [])})}


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
        self._ctx, self.brain_down, self._arms, self.skipped = None, False, None, {}

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

    def brain(self, e, fn, *args, key: tuple):
        """An orchestrator call, run in the background in the real loop: the first tick starts it and returns PENDING,
        a later tick picks up its answer, so a tick never waits minutes on a model. A model outage leaves everything
        where it is for a later tick (and pages after an hour); a malformed answer, after its retry, sends the experiment
        to needs-human. Returns the answer, PENDING, or None (outage or malformed)."""
        st, w = self.st, self.w
        if self.brain_down:
            return None
        if w.background:
            f = w.futures.get(key)
            if f is None:
                w.futures[key] = w.pool.submit(fn, w.cfg, *args, ask=w.ask)
                return PENDING
            if not f.done():
                return PENDING
            del w.futures[key]
            call = f.result
        else:
            call = lambda: fn(w.cfg, *args, ask=w.ask)  # noqa: E731
        try:
            out = call()
            st.pop("brainDown", None)
            return out
        except B.Unavailable as err:
            print(f"{self.at} brain unavailable ({fn.__name__}): {str(err)[:300]}", file=sys.stderr, flush=True)
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
        self.ab()
        if st.get("paused"):
            return self.events
        self.daily()
        self.reverts()
        self.exam_check()
        self.exec_data()
        for state, step in (("implementing", self.implementing), ("validating", self.validating), ("scoring", self.scoring),
                            ("folding", self.folding), ("proposed", self.proposed), ("accepted", self.accepted)):
            for e in S.in_state(st, state):
                self.guard(e, step)
        self.reviews()
        self.stack()
        self.plan()
        self.ab()
        return self.events

    def ab(self):
        self.arms()
        self.st["ab"] = {**AB.stats(self.st, [a["name"] for a in X.arms(self.w.cfg)]), "skipped": self.skipped,
                         "split": X.weights(self.w.cfg)}

    # builder arms (the A/B)
    def arms(self) -> list[dict]:
        """The arms that can start a builder now. One that can't (its config incomplete, its key missing, or its last
        builders all failed to start) is skipped, with one notification per reason."""
        if self._arms is None:
            ready = []
            for a in X.arms(self.w.cfg):
                why = X.unready(self.w.cfg, a) or self.arm_down(a["name"])
                if not why:
                    ready.append(a)
                    continue
                self.skipped[a["name"]] = why
                self.w.notify(self.st, "arm", f"{a['name']}:{why[:100]}", f"Autolab: {a['name']} builders are skipped",
                              f"{why}. New experiments go to the other builders until this is fixed.", self.at)
            self._arms = ready
        return self._arms

    def arm_down(self, name: str) -> str | None:
        h = (self.st.get("armHealth") or {}).get(name) or {}
        if h.get("startupFails", 0) >= ARM_DOWN_AFTER and S.minutes(h["lastAt"], self.at) < ARM_DOWN_MINUTES:
            return f"its last {ARM_DOWN_AFTER} builders failed to start ({h.get('why', '')[:120]}); one more try after {ARM_DOWN_MINUTES} min"
        return None

    def arm_health(self, name: str, res: dict | None):
        """Counts an arm's builders that failed to start in a row (res None: one got going)."""
        if not name:
            return
        h = self.st.setdefault("armHealth", {}).setdefault(name, {"startupFails": 0})
        if res is None:
            h["startupFails"] = 0
        elif res["summary"].startswith("startup"):
            h.update(startupFails=h["startupFails"] + 1, lastAt=self.at, why=res["summary"][:300])

    def arm_of(self, e) -> dict:
        """The arm that builds `e`: chosen when its builder first starts and kept for retries and bounces; another arm
        only when its own can't run now (recorded as a fallback)."""
        cfg, every, ready = self.w.cfg, X.arms(self.w.cfg), self.arms()
        ex = e["executor"] = e.get("executor") or {}
        if not ex.get("arm"):
            if ex.get("handle"):  # its builder started before arms existed
                rt = ex["handle"].get("runtime")
                a = next((a for a in every if a["runtime"] == rt), every[0])
                ex.update(arm=a["name"], assignedArm=a["name"], armInferred=True)
            else:
                a = AB.assign(e["id"], e["card"]["component"], ready or every[:1], X.weights(cfg), self.st)
                ex.update(arm=a["name"], assignedArm=a["name"], assignedAt=self.at)
        arm = next((a for a in every if a["name"] == ex["arm"]), None)
        if arm is None:
            rt = (ex.get("handle") or {}).get("runtime")
            to = next((a for a in ready if a["runtime"] == rt), None) or (ready or every)[0]
            self.fall_back(e, to["name"], f"arm {ex['arm']} is no longer in executor.runtimes")
            return to
        if ready and arm not in ready:
            to = AB.assign(e["id"], e["card"]["component"], ready, X.weights(cfg), self.st)
            self.fall_back(e, to["name"], f"{arm['name']} can't run now: {X.unready(cfg, arm) or self.arm_down(arm['name'])}")
            return to
        return arm

    def fall_back(self, e, to: str, why: str):
        ex = e["executor"]
        ex.setdefault("fallbacks", []).append({"from": ex["arm"], "to": to, "why": why[:300], "at": self.at})
        self.events.append(S.event(e["id"], e["state"], e["state"], self.at, f"builder {ex['arm']} -> {to}: {why[:200]}"))
        ex["arm"], ex["armFailures"] = to, 0

    def guard(self, e, step):
        """One experiment's step; an error in it sends that experiment to needs-human instead of pausing everything."""
        try:
            step(e)
        except Exception as err:  # noqa: BLE001
            if e["state"] in S.TERMINAL or e["state"] == "needs-human":
                raise
            if self.st["foldLock"] == e["id"]:
                self.st["foldLock"] = None
            self.events.append(S.move(self.st, e["id"], "needs-human", self.at, f"{e['state']} failed: {type(err).__name__}: {str(err)[:300]}"))

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
        elif op == "retry" and e["state"] == "abandoned":
            if not e["worktree"] or not Path(e["worktree"]).exists():
                raise ValueError(f"{e['id']} is abandoned without its worktree; nothing to build in")
            e["executor"].update(armFailures=0, fallbacks=[])
            e["results"]["feedback"] = c.get("note") or "A previous attempt was cut off from outside the experiment; carry on from the worktree."
            self.move(e, "accepted", "retry by you: a fresh builder in the same worktree")
        elif op == "retry":
            if e["state"] != "needs-human":
                raise ValueError(f"{e['id']} is {e['state']}, not waiting on you")
            r = e["results"]
            for k in ("testsJob", "decodeJob", "scoreJob", "foldJob"):
                self.w.kill(r.pop(k, None))
            r.pop("testsPassed", None)
            if "judge" not in r and not e["worktree"]:
                self.move(e, "proposed", "retry by you: the novelty judge again")
            elif c.get("note") or not e["worktree"]:
                r["feedback"] = c.get("note") or ""
                self.move(e, "accepted", "retry by you" + (f": {c['note'][:120]}" if c.get("note") else ""))
            else:
                self.move(e, "validating", "retry by you: checks, tests and scoring again")
        elif op == "reopen":
            if e["state"] != "abandoned" or not e["worktree"] or not Path(e["worktree"]).exists():
                raise ValueError(f"{e['id']} is {e['state']}; only an abandoned experiment with its worktree can be reopened")
            for k in ("testsJob", "decodeJob", "scoreJob", "foldJob", "testsPassed"):
                e["results"].pop(k, None)
            self.move(e, "validating", "reopened by you: checks, tests and scoring again")
        elif op == "rebuild":
            if e["state"] != "discarded" or "builderOutcome" not in e["results"]:
                raise ValueError(f"{e['id']} is {e['state']}; only an experiment its builder killed can be rebuilt")
            new = f"{e['id'][:37]}-r"
            if new in st["experiments"]:
                raise ValueError(f"{new} already exists")
            won = (e["results"]["builderOutcome"].get("measured") or {}).get("targets won")
            self.add({**e["card"], "id": new, "killCriterion": "the built change wins fewer than 3 targets, or its practice score fails the gate",
                      "reason": f"rebuild of {e['id']}: its builder stopped on a collateral counterexample with {won} targets won, before any score"})
            n = st["experiments"][new]
            n["results"]["feedback"] = (f"This rebuilds {e['id']}, which stopped before scoring: {e['results']['builderOutcome'].get('reason', '')} "
                                        f"Its write-up is on branch {e['branch']} (`git show {e['branch']}:.scratch/lab/experiments/{e['id']}.md`). "
                                        "Build the change, run practice-score once, and let the gate weigh the counterexample. " + (c.get("note") or ""))
            self.move(n, "accepted", f"rebuilt by you from {e['id']}")
        elif op == "revert":
            if e["state"] != "folded":
                raise ValueError(f"{e['id']} is {e['state']}, not folded")
            if self.w.git("rev-parse", "--verify", "--quiet", FOLD_TAG + e["id"], check=False) == "":
                raise ValueError(f"{e['id']} has no {FOLD_TAG}{e['id']} tag")
            if any(x["id"] == e["id"] for x in st.setdefault("pendingReverts", [])):
                raise ValueError(f"{e['id']} is already queued to revert")
            st["pendingReverts"].append({"id": e["id"], "why": c.get("why", ""), "at": self.at})
            self.events.append(S.event(e["id"], "folded", "folded", self.at, "revert queued by you"))
        elif op == "drop":
            if e["state"] in S.TERMINAL:
                raise ValueError(f"{e['id']} already ended ({e['state']})")
            if e["state"] == "implementing":
                self.w.builder(e["executor"]["handle"]).cancel(e["executor"]["handle"])
            self.stop_modal(e)
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

    # daily: digest, and a re-score when main moved without autolab (your own commits)
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
                    self.pause(f"the re-score of the main line failed: {res['error'][:300]}")
        if st.get("day") == day:
            return
        if st.get("day"):
            title, body = N.digest(st, w.events(), st["day"])
            w.notify(st, "digest", st["day"], title, body, at)
        st["day"] = day
        if any(e["state"] == "folding" for e in st["experiments"].values()) or st.get("examJob"):
            return
        main = st.get("main") or {}
        if not w.git("rev-parse", BRANCH).startswith((main.get("head") or main.get("sha") or "-")[:7]):
            st["baselineJob"] = w.job(f"baseline-{day}", w.lab_argv("baseline", "--practice-only", "--note", f"daily re-score {day}"))

    def tag_fold(self, e, head: str | None):
        if head:
            self.w.git("tag", "-f", FOLD_TAG + e["id"], head, check=False)
        st = self.st
        st["foldsSinceExam"] = st.get("foldsSinceExam", 0) + 1

    def main_busy(self) -> bool:
        """Something is moving or measuring main: a fold, the exam tripwire, a re-score."""
        st = self.st
        holder = st["experiments"].get(st.get("foldLock") or "")
        return bool((holder and holder["state"] == "folding") or st.get("examJob") or st.get("baselineJob"))

    # reverts: queued by you, applied when nothing else is moving main
    def reverts(self):
        st, w = self.st, self.w
        queue = st.setdefault("pendingReverts", [])
        if not queue or self.main_busy():
            return
        c = queue.pop(0)
        e = st["experiments"][c["id"]]
        try:
            tip = w.revert_fold(e["id"])
        except Stop as err:
            return w.notify(st, "needs-human", f"revert:{e['id']}", f"Autolab couldn't revert {e['id']}", str(err)[:1500], self.at)
        led = vol_get(VOL_LEDGER) or {}
        line = led.get("mainLine", [])
        i = next((k for k, m in enumerate(line) if m.get("experiment") == e["id"]), None)
        listener = (line[i - 1] if i else {}).get("listener") if i is not None else None
        w.lab("mark-reverted", e["id"], "--note", f"reverted by you: {c.get('why') or ''}".strip())
        self.move(e, "reverted", f"reverted by you{': ' + c['why'] if c.get('why') else ''} (main is now {tip[:7]})")
        st["baselineJob"] = w.job(f"baseline-revert-{e['id']}", w.lab_argv("baseline", "--practice-only", "--note", f"after reverting {e['id']}",
                                                                          *(["--listener", listener] if listener else [])))

    # exam tripwire: every examEvery folds, main on the exam too; only whether the exam followed practice is kept
    def exam_check(self):
        st, w = self.st, self.w
        job = st.get("examJob")
        if job:
            status, res = w.poll(job)
            if status == "running":
                return
            st["examJob"] = None
            if status != "done":
                st["foldsSinceExam"] = st.get("foldsSinceExam", 0) + job["folds"]
                return w.notify(st, "needs-human", f"exam:{self.at[:13]}", "Autolab: the exam tripwire run failed", tail(res), self.at)
            self.set_main(res["run"], res["sha"], res["practice"], w.git("rev-parse", "--short", BRANCH))
            st["examTripwire"] = trip = exam_verdict(vol_get(VOL_LEDGER) or {}, job["folds"], self.at)
            self.events.append(S.event(None, None, "exam-tripwire", self.at, trip["why"]))
            if trip["verdict"] != "pass":
                w.notify(st, "needs-human", f"exam:{self.at[:13]}", "Autolab: the exam didn't follow practice",
                         trip["why"] + "\nCheck the recent folds; `tools/autolab.py revert <id>` takes one back out.", self.at)
            return
        every = w.cfg.get("examEvery", 5)
        if not every or st.get("foldsSinceExam", 0) < every or self.main_busy():
            return
        listener, folds = (st.get("main") or {}).get("listener") or L.DEFAULT_LISTENER, st["foldsSinceExam"]
        st["foldsSinceExam"] = 0
        st["examJob"] = {**w.job(f"exam-{self.at[:16].replace(':', '')}", exam_argv(w, st, listener)), "folds": folds}

    def set_main(self, run: str, sha: str, practice: dict, head: str | None = None):
        report = self.w.report(run)
        teacher = ((report.get("splits") or {}).get("practice") or {}).get("teacher") or {}
        self.st["main"] = {"sha": sha, "head": head or sha, "run": run, "practice": practice, "listener": report.get("listener") or L.DEFAULT_LISTENER,
                           "attribution": {c: practice[c] for c in L.COMPONENTS}, "teacherWrongCalls": teacher.get("wrongCalls")}
        P.update_coverage(self.st, P.incidents(report))
        self._ctx = None

    def exec_data(self):
        """Keeps tilawa-lab-exec, the builders' practice-only data, on the current test set version; modal-run requests
        wait until it is."""
        st, w = self.st, self.w
        job = st.get("execSyncJob")
        if job:
            status, res = w.poll(job)
            if status == "running":
                return
            st["execSyncJob"] = None
            if status == "done":
                st["execData"] = {"testSet": job["testSet"], "listener": job.get("listener"), "at": self.at, **{k: res.get(k) for k in ("sessions", "files", "missing")}}
                self.events.append(S.event(None, None, "exec-data", self.at, f"practice-only data for builders: {res.get('sessions')} sessions, {res.get('files')} files"))
            else:
                st["execSyncFailed"] = self.at
                w.notify(st, "needs-human", f"exec-sync:{self.at[:13]}", "Autolab: the builders' practice data didn't sync", tail(res), self.at)
            return
        listener = (st.get("main") or {}).get("listener") or L.DEFAULT_LISTENER
        have = st.get("execData") or {}
        if (have.get("testSet") == st["testSet"] and have.get("listener", L.DEFAULT_LISTENER) == listener) or \
                (st.get("execSyncFailed") and S.minutes(st["execSyncFailed"], self.at) < 30):
            return
        spec = exec_spec(w.test_set(st["testSet"]), w.cfg, listener)
        st["execSyncJob"] = {**w.job(f"exec-sync-{int(time.time())}", w.exec_argv("sync", json.dumps(spec))), "testSet": st["testSet"], "listener": listener}

    def pause(self, why: str):
        self.st["paused"] = why
        self.events.append(S.event(None, None, "paused", self.at, why))
        self.w.notify(self.st, "paused", why[:60], "Autolab paused", why, self.at)

    # proposed: the novelty judge
    def proposed(self, e):
        if S.timed_out(e, self.at):
            return self.move(e, "needs-human", "the novelty judge didn't answer in time")
        v = self.brain(e, B.judge, e["card"], self.ctx(), key=("judge", e["id"]))
        if v is None or v is PENDING:
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
        if e["tier"] == "bet" and c.get("gpuUsd", 0) >= w.cfg.get("betApprovalUsd", 100) and not e["results"].get("approved"):
            return self.move(e, "needs-human", f"a bet asking for ${c['gpuUsd']} of GPU needs your approval: tools/autolab.py approve {e['id']}")
        if not e["worktree"]:
            try:
                out = w.lab("new", e["id"], "--component", c["component"], "--hypothesis", c["hypothesis"],
                            "--expect-component", c["expected"]["component"], "--expect-overall", c["expected"]["overall"], "--author", "autolab")
            except Stop as err:
                wt = REPO.parent / f"tilawa-lab-exp-{e['id']}"
                if "already exists" not in str(err) or not wt.exists():
                    raise
                out = {"branch": f"exp/{e['id']}", "worktree": str(wt)}  # an earlier tick made it, then its state save failed
            e["branch"], e["worktree"] = out["branch"], out["worktree"]
            w.exclude(".autolab/")
        if c.get("stack") and not e["results"].get("stackConflict"):
            return self.build_stack(e)
        arm = self.arm_of(e)
        if self.arm_full(arm):
            others = [a for a in self.arms() if a["name"] != arm["name"] and not self.arm_full(a) and X.weights(self.w.cfg).get(a["name"], 1) > 0]
            waited = S.minutes(e["since"], self.at)
            if others and not (e.get("executor") or {}).get("handle") and waited >= ARM_WAIT_MINUTES:
                self.fall_back(e, others[0]["name"], f"{arm['name']} was at its {arm.get('maxConcurrent')} builders for {waited:.0f} min")
            else:
                return
        self.start_executor(e, e["results"].pop("feedback", ""))

    # stacks: two near misses that win different incidents, merged and scored as one experiment (no builder)
    def near_misses(self) -> list[dict]:
        """Discarded experiments that measured -1 or -2 on their branch, lost no incident, and failed only on size (the
        gain bar, or the reader spread that a gain this small can't clear); their branches still exist."""
        st, out = self.st, []
        broken = {k for ks in P.incidents(self.w.report(st["main"]["run"])).values() for k in ks} if (st.get("main") or {}).get("run") else set()
        used = {p for x in st["experiments"].values() if x["card"].get("stack") and x["state"] not in ("discarded", "abandoned", "superseded")
                for p in x["card"]["stack"]}
        for e in st["experiments"].values():
            r = e.get("results") or {}
            change = (((r.get("branchRun") or {}).get("compare") or {}).get("overall") or {}).get("change")
            checks = {c["name"]: c for c in (r.get("gate") or {}).get("checks", [])}
            col = checks.get("collateral") or {}
            failed = {n for n, c in checks.items() if c["blocking"] and not c["ok"]}
            if (e["state"] != "discarded" or change not in (-1, -2) or col.get("lost") or not col.get("won") or not failed <= {"gain", "spread"}
                    or e["id"] in used or r.get("model") or not e.get("branch")
                    or not self.w.git("rev-parse", "--verify", "--quiet", e["branch"], check=False)):
                continue
            if "wonKeys" not in col:
                try:
                    col["wonKeys"] = G.wins(self.w.report(r["branchRun"]["base"]), self.w.report(r["branchRun"]["run"]))
                except (KeyError, ValueError, TypeError, Stop):
                    continue
            still = set(col["wonKeys"]) & broken  # wins a later fold already took are not this branch's any more
            if not still:
                continue
            out.append({"id": e["id"], "change": -len(still) if len(still) < -change else change, "won": still, "component": e["card"]["component"],
                        "compare": r["branchRun"]["compare"]})
        return out

    def stack(self):
        """One new stack per tick, when a slot is free: the pair of near misses with disjoint wins and the largest summed
        change that hasn't been stacked before."""
        st, w = self.st, self.w
        if not st.get("main") or S.over_cap(st, w.cfg["caps"], self.at) or len(S.active(st)) >= w.cfg["slots"]:
            return
        stacks = [x for x in st["experiments"].values() if x["card"].get("stack")]
        unmerged = lambda x: x["state"] == "discarded" and x.get("note", "").startswith("stack: merging ")  # noqa: E731
        retry = ({tuple(sorted(x["card"]["stack"])) for x in stacks if unmerged(x)}
                 - {tuple(sorted(x["card"]["stack"])) for x in stacks if not unmerged(x)})
        tried = {tuple(p) for p in st.setdefault("stacked", [])} - retry
        nm = self.near_misses()
        pairs = sorted(((a["change"] + b["change"], a, b) for i, a in enumerate(nm) for b in nm[i + 1:]
                        if not a["won"] & b["won"] and tuple(sorted((a["id"], b["id"]))) not in tried), key=lambda x: (x[0], x[1]["id"]))
        if not pairs or pairs[0][0] > -3:
            return
        total, a, b = pairs[0]
        comp = "listener" if "listener" in (a["component"], b["component"]) else (a if a["change"] <= b["change"] else b)["component"]
        ca, cb = (st["experiments"][x["id"]]["card"] for x in (a, b))
        key = tuple(sorted((a["id"], b["id"])))
        cid = f"{comp}-stack-{hashlib.sha1('+'.join(key).encode()).hexdigest()[:6]}"
        while cid in st["experiments"]:
            cid += "-m"
        card = {"id": cid, "component": comp, "tier": "exploit",
                "hypothesis": f"Stack {a['id']} ({a['change']:+d}) and {b['id']} ({b['change']:+d}): each won different incidents and lost none, "
                              f"so together they should clear the fold bar; predict overall {total:+d}.",
                "eli5": "Two small fixes that each helped a little, tried together.",
                "mechanism": f"{ca['hypothesis'][:400]} | {cb['hypothesis'][:400]}", "cluster": ca.get("cluster") or cb.get("cluster"),
                "targets": sorted(a["won"] | b["won"]),
                "expected": {"component": sum(x["compare"].get(comp, {}).get("change", 0) for x in (a, b)), "overall": total},
                "killCriterion": "the stack measures above -3 overall or loses an incident", "stack": list(key)}
        st["stacked"].append(list(key))
        self.add(card)
        e = st["experiments"][card["id"]]
        self.move(e, "accepted", f"stack of near misses {key[0]} and {key[1]}; no judge or builder: their branches are merged")

    def build_stack(self, e):
        w, wt, parts = self.w, e["worktree"], e["card"]["stack"]
        for i, part in enumerate(parts):
            branch = self.st["experiments"][part]["branch"]
            p = subprocess.run(["git", "merge", "--no-edit", "-q", branch], cwd=wt, capture_output=True, text=True)
            if p.returncode:
                bad = [f for f in w.git("diff", "--name-only", "--diff-filter=U", cwd=wt, check=False).splitlines() if f != L.CARD]
                if bad:
                    subprocess.run(["git", "merge", "--abort"], cwd=wt, capture_output=True)
                    left = [self.st["experiments"][x]["branch"] for x in parts[i:]]
                    e["results"]["stackConflict"] = bad
                    e["results"]["feedback"] = (
                        f"This stack has no new idea: it is {' and '.join(parts)} applied together. Run "
                        + "; ".join(f"`git merge {b}`" for b in left)
                        + f". Git can't merge {', '.join(bad[:5])} on its own: resolve it so both changes apply exactly as on their branches, "
                        f"keep this card's own {L.CARD}, change nothing else, then run the tests and practice-score once.")
                    self.events.append(S.event(e["id"], e["state"], e["state"], self.at, f"stack: merging {part} conflicts in {', '.join(bad[:5])}; a builder resolves it"))
                    return
                w.git("checkout", "--ours", "--", L.CARD, cwd=wt)
                w.git("add", L.CARD, cwd=wt)
                w.git("commit", "-q", "--no-edit", cwd=wt)
            elif w.git("diff", "--name-only", "HEAD~1", "HEAD", "--", L.CARD, cwd=wt, check=False):
                w.git("checkout", "HEAD~1", "--", L.CARD, cwd=wt)
                w.git("commit", "-q", "-m", f"autolab: keep the {e['id']} card", cwd=wt)
        e["results"]["stackedAt"] = self.at
        self.move(e, "validating", f"merged {' and '.join(e['card']['stack'])}")

    def arm_full(self, arm: dict) -> bool:
        """An arm at its `maxConcurrent` builders (a model quota, e.g. Azure tokens per minute): the card waits for it
        rather than going to another arm, so the A/B stays as assigned."""
        cap = arm.get("maxConcurrent")
        busy = sum(1 for x in self.st["experiments"].values() if x["state"] == "implementing" and (x.get("executor") or {}).get("arm") == arm["name"])
        return bool(cap) and busy >= cap

    def start_executor(self, e, feedback: str = ""):
        w = self.w
        arm = self.arm_of(e)
        ex = e["executor"]
        starts = ex.get("armStarts") or ({ex["arm"]: e["attempts"]["executor"]} if e["attempts"]["executor"] else {})
        first, retry = X.arm_models(X.arm_config(w.cfg, arm)["executor"])
        model = first if starts.get(arm["name"], 0) == 0 else retry
        handle = w.arm_executor(arm).start({**e["card"], "executorModel": model}, e["worktree"], brief(e, self.ctx(), feedback))
        ex.update(runtime=arm["runtime"], model=model, handle=handle, armStarts={**starts, arm["name"]: starts.get(arm["name"], 0) + 1})
        e["attempts"]["executor"] += 1
        S.charge(self.st, self.at, runs=1)
        self.move(e, "implementing", f"{arm['name']}: {model['id'] if isinstance(model, dict) else model}" + (f" (feedback: {feedback[:120]})" if feedback else ""))

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

    def serve_modal(self, e):
        """An executor has no Modal access: `tools/autolab.py modal-run` writes .autolab/modal-request-<n>.json, the
        supervisor runs it in a sandbox (modal/autolab_exec.py) and writes modal-result-<n>.json. GPUs only for bets,
        within the card's gpuUsd (a run reserves its whole timeout up front); at most exec.concurrent at a time."""
        d = Path(e["worktree"], ".autolab")
        r, cfg = e["results"], self.w.cfg
        jobs, runs = r.setdefault("modalJobs", {}), r.setdefault("modalRuns", [])
        for n, j in list(jobs.items()):
            status, res = self.w.poll(j["job"])
            if status == "running":
                continue
            out = (res or {}).get("result") or {"error": tail(res)}
            usd = I.cost(j["q"], out.get("seconds") or S.minutes(j["job"]["started"], self.at) * 60)
            r["gpuUsd"] = round(r.get("gpuUsd", 0) + usd, 4)
            runs.append({"n": n, "cmd": j["q"]["cmd"][:200], "gpu": j["q"]["gpu"], "code": out.get("code"), "usd": usd, "at": self.at})
            del jobs[n]
            write_json(d / f"modal-result-{n}.json", {**out, "usd": usd, **budget(e)})
        if (self.st.get("execData") or {}).get("testSet") != self.st["testSet"]:
            return
        lim = I.limits(cfg)
        for req in sorted(d.glob("modal-request-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])) if d.exists() else []:
            n = req.stem.rsplit("-", 1)[1]
            if n in jobs or (d / f"modal-result-{n}.json").exists():
                continue
            if len(jobs) >= lim["concurrent"]:
                return
            try:
                q = I.request(json.loads(req.read_text()), e["tier"], cfg)
                if q["gpu"] and I.reserve(q) > budget(e)["gpuUsdLeft"]:
                    raise ValueError(f"this run can cost up to ${I.reserve(q):.2f} ({q['shards']} x {q['gpu']} for {q['timeout']} min); "
                                     f"${budget(e)['gpuUsdLeft']:.2f} of the card's ${e['card'].get('gpuUsd', 0)} GPU budget is left")
            except (ValueError, TypeError, json.JSONDecodeError) as err:
                write_json(d / f"modal-result-{n}.json", {"error": str(err), **budget(e)})
                continue
            run = {**q, "exp": e["id"], "n": n, "worktree": e["worktree"]}
            jobs[n] = {"q": q, "job": self.w.job(f"modal-{e['id']}-{n}", self.w.exec_argv("run", json.dumps(run)))}

    def stop_modal(self, e):
        """Ends an experiment's runs still going (their job terminates its sandboxes), charging the time they ran."""
        r = e["results"]
        for n, j in (r.pop("modalJobs", None) or {}).items():
            self.w.kill(j["job"])
            r["gpuUsd"] = round(r.get("gpuUsd", 0) + I.cost(j["q"], S.minutes(j["job"]["started"], self.at) * 60), 4)

    def implementing(self, e):
        w, h = self.w, e["executor"]["handle"]
        self.serve_practice(e)
        self.serve_modal(e)
        status = w.builder(h).poll(h)
        if status == "running":
            if S.timed_out(e, self.at):
                w.builder(h).cancel(h)
                self.finish_executor(e)
                self.stop_modal(e)
                self.move(e, "abandoned", f"the executor ran past {S.timeout(e)} minutes")
            return
        self.stop_modal(e)
        res = self.finish_executor(e)
        ex = e["executor"]
        if status == "finished":
            self.arm_health(ex.get("arm"), None)
            return self.move(e, "validating", res["summary"][:200])
        if outage(res["summary"]):
            e["attempts"]["executor"] -= 1
            e["results"]["retryAfter"] = shift(self.at, 10)
            e["state"] = "accepted"
            self.events.append(S.event(e["id"], "implementing", "accepted", self.at, f"builder couldn't start (outage); retrying after {e['results']['retryAfter']}"))
            return
        self.arm_health(ex.get("arm"), res)
        ex["armFailures"] = ex.get("armFailures", 0) + 1
        if ex["armFailures"] < 2:
            self.events.append(S.event(e["id"], "implementing", "implementing", self.at, "executor failed; retrying with the fallback model"))
            e["state"] = "accepted"
            return self.start_executor(e, f"A previous attempt failed: {res['summary'][:500]}")
        others = [a for a in self.arms() if a["name"] != ex.get("arm") and X.weights(self.w.cfg).get(a["name"], 1) > 0]
        if others and not ex.get("fallbacks"):
            to = AB.assign(e["id"], e["card"]["component"], others, X.weights(self.w.cfg), self.st)
            self.fall_back(e, to["name"], f"its builder failed twice: {res['summary'][:200]}")
            e["state"] = "accepted"
            return self.start_executor(e, f"A previous attempt failed: {res['summary'][:500]}")
        self.move(e, "abandoned", f"the executor failed twice: {res['summary'][:200]}")

    def finish_executor(self, e) -> dict:
        ex = e["executor"]
        res = self.w.builder(ex["handle"]).collect(ex["handle"])
        self.w.transcript(e["id"], res["transcript"])
        e["results"]["executor"] = {"summary": res["summary"], "hours": res["hours"], "model": ex["model"], "arm": ex.get("arm")}
        ex["hours"] = round(ex.get("hours", 0) + res["hours"], 4)
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
                return self.bounce(e, "build or tests fail on Modal: " + tests_tail(res))
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
        outcome = read_json(Path(wt, ".autolab", "outcome.json"))
        if outcome.get("status") == "killed":
            r["builderOutcome"] = outcome
            m = outcome.get("measured") if isinstance(outcome.get("measured"), dict) else {}
            won = m.get("targets won")
            if isinstance(won, int) and won >= UNSCORED_KILL_WINS and m.get("overall") is None and not e["attempts"]["practiceScores"]:
                Path(wt, ".autolab", "outcome.json").unlink()
                return self.bounce(e, f"you stopped with {won} targets won and no practice score: {str(outcome.get('reason', ''))[:200]} "
                                      "Restore the change (or build it if you stopped at step 0), run practice-score once, and stop "
                                      "only if that score loses more than the gate allows.")
            return self.discard(e, f"killed by its builder: {str(outcome.get('reason', ''))[:300]}"
                                   + (f" (measured: {json.dumps(outcome['measured'])[:200]})" if outcome.get("measured") else ""))
        diff = w.diff(e)
        if not any(not NOT_ENGINE.search(f) for f in G.diff_files(diff)):
            return self.bounce(e, "the diff changes no engine code (only write-ups, tests or docs); implement the change, or explain in the write-up why it can't be done")
        h = G.hygiene_check(diff)
        if not h["ok"]:
            return self.bounce(e, "hygiene: " + h["why"])
        r["tuning"] = G.is_tuning(diff)
        if r["tuning"] and P.tuning_blocked(st, e["id"]):
            return self.discard(e, "another tuning experiment is already open (at most one)")
        v = self.brain(e, B.review_diff, c, diff, key=("diff", e["id"], e["attempts"]["executor"]))
        if v is None or v is PENDING:
            return
        r["diffReview"] = v
        if not v["ok"]:
            return self.bounce(e, "diff review: " + v["why"] + " " + "; ".join(v["concerns"][:5]))
        if st.get("dataHash") and w.data_hash(st["testSet"]) != st["dataHash"]:
            return self.pause("the test data on the volume changed since autolab started; check /lab/data before resuming")
        n = e["attempts"]["executor"]
        run = {"exp": e["id"], "n": f"tests-{n}", "worktree": wt, "cmd": test_command(w.cfg), "shards": w.cfg.get("testShards", 6),
               "gpu": None, "image": "node", "timeout": 30, "download": False}
        r["testsJob"] = w.job(f"tests-{e['id']}-{n}", w.exec_argv("run", json.dumps(run)))

    def after_tests(self, e):
        r, w, c = e["results"], self.w, e["card"]
        r.pop("testsPassed", None)
        if c["component"] != "listener" or c.get("noDecode"):
            return self.start_scoring(e)
        sha = w.git("rev-parse", "--short", "HEAD", cwd=e["worktree"])
        r["listener"] = f"exp-{e['id']}-{sha}"[:60]
        practice = practice_ids(w.test_set(self.st["testSet"]))
        model, copy = w.cfg["listenerModel"], []
        f = Path(e["worktree"], ".autolab", "model.json")
        if f.exists():
            try:
                path = I.model_path(json.loads(f.read_text()), e["id"])
            except (ValueError, json.JSONDecodeError) as err:
                return self.bounce(e, str(err))
            model = f"models/{r['listener']}"
            r["model"] = {"scratch": f"/scratch/{e['id']}/{path}", "path": model}
            copy = [w.exec_argv("copy-model", e["id"], path, model)]
        self.st.setdefault("listenerModels", {})[r["listener"]] = model
        decode = ["modal", "run", str(REPO / "modal/lab_decode.py"), "--listener", r["listener"], "--model", model,
                  "--test-set", self.st["testSet"].split(".v")[0], "--variants", "phone,studio", "--only", ",".join(practice)] + (["--force"] if copy else [])
        r["decodeJob"] = w.job(f"decode-{e['id']}-{sha}", json_exit(*copy, decode), env={"LAB_SRC": e["worktree"]})

    def score_with_listener(self, e, listener: str):
        wt = e["worktree"]
        f = Path(wt, L.CARD)
        card = json.loads(f.read_text())
        card["listener"] = listener
        f.write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
        if self.w.git("status", "--porcelain", "--", L.CARD, cwd=wt):
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
            if "moved while scoring" in err and e["attempts"].get("mainMoved", 0) < 3:
                e["attempts"]["mainMoved"] = e["attempts"].get("mainMoved", 0) + 1
                self.events.append(S.event(e["id"], "folding", "folding", self.at, "main moved while the fold was scoring; folding again"))
                return
            if "conflicts" in err and e["attempts"].get("conflicts", 0) == 0:
                e["attempts"]["conflicts"] = 1
                e["state"] = "accepted"
                return self.start_executor(e, "Rebasing onto main conflicts. Rebase your branch onto `main`, resolve the conflicts, keep the change's intent, and commit.\n" + err[-1500:])
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
            self.set_main(res["run"], res["sha"], res["practice"], res.get("head"))
            self.tag_fold(e, res.get("head"))
            w.notify(st, "digest", f"fold:{e['id']}", f"Autolab kept {e['id']}",
                     f"{e['card']['hypothesis']}\nPractice change: overall {res['compare']['overall']['change']:+d}.", self.at)
        self.after(e, {"state": state, "why": note, "gate": res.get("gate"), "compare": res.get("compare")})

    # after an experiment ends: journal entry and maybe a follow-up (reviewed on this or a later tick)
    def after(self, e, outcome: dict):
        for cl in [e["card"].get("cluster")] if e["card"].get("cluster") else []:
            P.mark_tried(self.st, cl, e["id"])
        self.st.setdefault("pendingReviews", {})[e["id"]] = outcome
        self.reviews()

    def reviews(self):
        pending = self.st.setdefault("pendingReviews", {})
        for id, outcome in list(pending.items()):
            e = self.st["experiments"].get(id)
            if not e:
                pending.pop(id)
                continue
            v = self.brain(None, B.review, e, outcome, self.ctx(), key=("review", id))
            if v is PENDING or (v is None and self.brain_down):
                continue
            pending.pop(id)
            if v is None:
                continue
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
        free_now = lambda: P.free_slots(st, P.allocate(st["main"]["attribution"], w.cfg["slots"], set(st["counters"]["frozen"]), P.fold_record(st)))  # noqa: E731
        free = free_now()
        newly = P.update_plateau(st, at, st.get("started") or at)
        if newly:
            st["pendingStrategy"] = sorted(set(st.get("pendingStrategy") or []) | set(newly))
        if st.get("pendingStrategy") and free:
            stuck = st["pendingStrategy"]
            v = self.brain(None, B.strategy, self.ctx(), stuck, free, key=("strategy",))
            if v is PENDING or (v is None and self.brain_down):
                return
            st["pendingStrategy"] = []
            if v:
                w.journal_add(f"Strategy: stuck on {', '.join(stuck)}", v["memo"], at)
                for c in v["cards"][: len(free_now())]:
                    self.add(c)
                free = free_now()
        last, todo = st.get("lastPlanAt"), st.get("planSlots") or []
        if not todo:
            if not free or (last and S.minutes(last, at) < w.cfg["planEveryMinutes"]):
                return
            todo = [[c, t, i] for i, (c, t) in enumerate(free)]
            st["lastPlanAt"], st["planSlots"] = at, todo
        # one call per slot, in parallel: one card takes Opus about a minute, five in one answer never finished
        for slot in list(todo):
            cards = self.brain(None, B.plan, self.ctx(), [tuple(slot[:2])], key=("plan", *slot))
            if cards is PENDING or (cards is None and self.brain_down):
                continue
            todo.remove(slot)
            if cards and free_now():
                c, base, n = cards[0], cards[0]["id"], 2
                while c["id"] in st["experiments"]:
                    c["id"], n = f"{base}-v{n}", n + 1
                self.add(c)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def json_exit(*argvs: list[str]) -> list[str]:
    """Commands that print no JSON, run in turn until one fails, wrapped so the job ends in a JSON line saying how it
    exited (what poll reads)."""
    import shlex  # noqa: PLC0415
    return ["bash", "-c", f'{" && ".join(map(shlex.join, argvs))}; code=$?; echo "{{\\"exitCode\\": $code}}"']


def exam_argv(w, st: dict, listener: str) -> list[str]:
    """The tripwire job: in a clean checkout of main, decode the exam with main's listener when it is an experiment's
    own (the shipped one's exam evidence exists), then a full baseline. Only the baseline's JSON reaches stdout."""
    import shlex  # noqa: PLC0415
    t = w.test_set(st["testSet"])
    lines = ["set -eo pipefail", 'wt="$(mktemp -d)/main"', f'git -C {shlex.quote(str(REPO))} worktree add --detach -q "$wt" {BRANCH}',
             f'trap \'git -C {shlex.quote(str(REPO))} worktree remove --force "$wt"\' EXIT']
    if listener.startswith("exp-"):
        model = (st.get("listenerModels") or {}).get(listener) or w.cfg["listenerModel"]
        decode = ["modal", "run", str(REPO / "modal/lab_decode.py"), "--listener", listener, "--model", model,
                  "--test-set", st["testSet"].split(".v")[0], "--variants", "phone,studio", "--only", ",".join(t["sessions"]["exam"])]
        lines.append(f'LAB_SRC="$wt" {shlex.join(decode)} >&2')
    lines.append(shlex.join(w.lab_argv("baseline", "--note", "autolab exam tripwire")))
    return ["bash", "-c", "\n".join(lines)]


def exam_verdict(led: dict, folds: int, at: str) -> dict:
    """Whether the exam followed practice since the previous exam reading of the main line. Keeps the verdict and the
    practice change, never the exam numbers (nothing downstream should steer by them)."""
    read = [m for m in led.get("mainLine", []) if m.get("exam") and m.get("practice")]
    out = {"at": at, "folds": folds}
    if len(read) < 2:
        return {**out, "verdict": "pass", "why": "first exam reading of the main line; nothing to compare yet"}
    ref, new = read[-2], read[-1]
    p = new["practice"]["wrongCalls"] - ref["practice"]["wrongCalls"]
    x = new["exam"]["wrongCalls"] - ref["exam"]["wrongCalls"]
    out.update(practiceChange=p, since=ref.get("head") or ref["sha"], head=new.get("head") or new["sha"])
    if x > G.EXAM_SLACK:
        return {**out, "verdict": "worse", "why": f"the exam got worse while practice changed {p:+d} over {folds} fold(s)"}
    if p <= -6 and x >= 0:
        return {**out, "verdict": "no-transfer", "why": f"practice improved {p:+d} over {folds} fold(s) but the exam did not improve"}
    return {**out, "verdict": "pass", "why": f"the exam followed practice ({p:+d}) over {folds} fold(s)"}


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
    if s.startswith(("runner-died", "cancelled")):  # the host killed the builder's process; the supervisor's own cancels never get here
        return True
    if any(k in s for k in ("too many requests", "rate limit", "429")):  # a quota, at start or mid-run: wait, same arm
        return True
    return s.startswith(("startup", "sdk-error")) and any(k in s for k in ("unavailable", "capacity", "timeout", "timed out", "network", "connect"))


def shift(at: str, minutes: int) -> str:
    from datetime import datetime, timedelta  # noqa: PLC0415
    return (datetime.fromisoformat(at) + timedelta(minutes=minutes)).isoformat()


def budget(e: dict) -> dict:
    """A card's GPU dollars: spent, and left after what running requests hold."""
    r = e["results"]
    held = sum(I.reserve(j["q"]) for j in (r.get("modalJobs") or {}).values())
    total = e["card"].get("gpuUsd", 0) if e["tier"] == "bet" else 0
    return {"gpuUsdBudget": total, "gpuUsdSpent": round(r.get("gpuUsd", 0), 2), "gpuUsdLeft": round(max(0, total - r.get("gpuUsd", 0) - held), 2)}


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    os.replace(tmp, path)


def tail(res: dict | None) -> str:
    res = res or {}
    return (res.get("log") or res.get("error") or json.dumps(res))[-2500:]


def tests_tail(res: dict | None) -> str:
    """A sharded test run's failing shards, not the log's end (the last shard's, often one that passed or whose only
    failure was excused): this text goes back to the builder."""
    bad = [s for s in (((res or {}).get("result") or {}).get("shards") or []) if s.get("code")]
    return "\n".join(f"shard {s.get('shard')}: {(s.get('tail') or '')[-1500:]}" for s in bad)[-2500:] if bad else tail(res)


def brief(e: dict, ctx: dict, feedback: str = "") -> str:
    """What the executor is told: the card, where to look, the rules, and how to finish."""
    c = e["card"]
    targets = "\n".join(f"- `{t}`" for t in c.get("targets", []))
    tried = [h for h in ctx["history"] if h.get("cluster") == c.get("cluster") and h["id"] != c["id"]]
    base, lim, b, al = ctx["testSet"].split(".v")[0], I.limits(config()), budget(e), f"python3 {REPO / 'tools/autolab.py'}"
    gpu = f"""
GPUs (this is a bet): add `--gpu L4|A10|L40S|A100|H100` (and `--image train`: torch, k2 and icefall, with the training
data read-only under /student: s1/, aug/, eval/; /student/runs is in your scratch area; earlier training runs, each with
its ckpt.pt and log.jsonl, are read-only under /student/prior-runs: copy one into /student/runs to resume or fine-tune
it). Earlier exported models are read-only under /lab/data/models. The card's GPU budget is
${b['gpuUsdBudget']} (${b['gpuUsdLeft']} left); a run reserves shards x the GPU's hourly rate x its timeout before it
starts, and one that doesn't fit is refused. There is no Modal client inside a run: call training code directly, e.g.
`python -c 'import types, student_train as T; T.vol = types.SimpleNamespace(commit=lambda: None); T.train.local({{...}})'`.
""" if e["tier"] == "bet" else ""
    model = f"""
To be scored with a model of your own, export it like /lab/data/models/spliced-clean-int8 (model.onnx + manifest.json
with its sha256) to a directory under /scratch/{c['id']}/, and write `.autolab/model.json` as
`{{"path": "<that directory, relative to /scratch/{c['id']}>"}}`. The supervisor copies it and decodes the practice set
with it itself; without model.json it decodes with the main listener model.
""" if c["component"] == "listener" else ""
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

## Running code on Modal

You have no Modal credentials: `modal ...` and `tools/lab.py run` fail here. The supervisor runs things for you:

    {al} modal-run [--shards N] [--timeout MIN] -- '<command>'

runs the command under bash in /app, over this worktree's tracked and untracked files, and waits for it (add
`--no-wait`, then `{al} modal-result <n>`, for long runs). Each shard gets LAB_SHARD and LAB_SHARDS,
LAB_DATA=/lab/data (read-only: the practice sessions of {base} with their labels, audio and evidence, and
models/spliced-clean-int8; the exam is not there; and help-train/: real learners' phone and laptop recordings of
single ayahs with scripted mistakes, labels.jsonl in core's event vocabulary, repeats in non_errors; for training and
analysis only, its held-out speakers are the release benchmark's and are not there), and LAB_OUT, which comes back to `.autolab/modal-out-<n>/<shard>/`
with the command's log.txt. /scratch/{c['id']} keeps files across your runs. At most {lim['concurrent']} runs at a time,
{lim['maxShards']} shards and {lim['maxMinutes']} minutes each{' (a bet: up to ' + str(lim['betMaxMinutes'] // 60) + ' h)' if e['tier'] == 'bet' else ''}.
This worktree has no node_modules; run the tests there too: `{al} modal-run --shards 6 -- 'bash tools/autolab_tests.sh'`.
Inspect a case with the trace tool:
`{al} modal-run -- 'pnpm --filter @tilawa/core exec tsx src/bench/lab-trace-cli.ts --set {base} --session <session> --at <seconds>'`
{gpu}{model}
## Tried before on this cluster

{json.dumps(tried, ensure_ascii=False, indent=1) if tried else 'nothing yet'}

## Rules

- Change engine code only. Never touch the scorer or lab tooling (packages/core/src/bench/lab*.ts, tools/lab*.py,
  tools/autolab*.py, modal/lab_*.py, docs/benchmarks/lab-*), test data, manifests or evidence.
- Never special-case sessions, readers, verses or anything else specific to the test set. A reviewer and a code check
  reject that.
- Add or adjust unit tests for the behaviour you change; keep existing tests passing
  (`pnpm --filter @tilawa/core test`, `pnpm --filter @tilawa/sdk test`).
- You may score the practice set at most {PRACTICE_SCORES} times in total: `{al} practice-score .`
  (about 11 minutes each). Don't tune constants against it. Your own numbers from modal-run are notes; the supervisor
  scores the branch itself.
- Don't look for another way to Modal, the exam or files outside this worktree; writes outside it are blocked.
- Write `.scratch/lab/experiments/{c['id']}.md` like the other write-ups there: the prediction, the mechanism, what you
  validated and its limits.
- Commit everything on this branch with clear messages. Finish with a short summary of what you changed and why.

## If it doesn't work

Kill only on the premise: the targets don't have what the mechanism relies on, or the built change wins fewer than 3
of them. Losses elsewhere (a clean seam, a pause, a planted mistake caught before) are not a reason to stop, even if
the card's kill criterion names them: the fold gate trades wins against losses (losses up to 30% of wins, studio up to
2 worse; only a new false warning on a fully clean session is a hard stop). If the change wins 3 or more targets, run
`{al} practice-score .` before deciding; narrow the change only when that score shows the gate failing.

If your kill criterion fires that way, or you find the mechanism is wrong, stop: that is a useful result, not a failure.
Revert your engine changes, explain what you found in the write-up, commit, and write `.autolab/outcome.json`:
`{{"status": "killed", "reason": "<one sentence>", "measured": {{"overall": <practice change, if you scored>, "targets won": <n>}}}}`.
Otherwise leave the change in place; the supervisor scores it and the fold gate decides.
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
    w = World(cfg, executor=object())
    lab_led = vol_get(VOL_LEDGER)
    if not L.main_score(lab_led):
        ts = L.test_set(lab_led)
        a2 = ["baseline", "--practice-only", "--manifest", ts["manifest"], "--note", "autolab start"] + (["--exclude", ts["exclude"]] if ts.get("exclude") else [])
        print("scoring the main line's practice baseline (about 11 minutes)...", flush=True)
        res = w.lab(*a2)
    else:
        m = L.main_score(vol_get(VOL_LEDGER))
        res = {"run": m["run"], "sha": m["sha"], "practice": L.brief(m["practice"])}
    st = w.load() or S.empty(lab_led["testSet"])
    st.update(started=st.get("started") or S.now(), tickSeconds=cfg["tickSeconds"], caps=cfg["caps"], day=S.now()[:10])
    t = Tick(w, st, S.now())
    t.set_main(res["run"], res["sha"], res["practice"])
    st["dataHash"] = w.data_hash(st["testSet"])
    out = sync_exec_data(cfg, st["testSet"])
    st["execData"] = {"testSet": st["testSet"], "at": S.now(), **{k: out.get(k) for k in ("sessions", "files", "missing")}}
    w.save(st, [S.event(None, None, "started", S.now(), f"main line {res['sha']}")])
    print(f"autolab ready: main line {res['sha']}, practice {res['practice']['wrongCalls']} wrong calls; run `tools/autolab.py loop`")


def tick_once(w: World) -> dict:
    st = w.load()
    if not st:
        raise Stop("no autolab state: run `tools/autolab.py init` first")
    t = Tick(w, st, S.now())
    try:
        events = t.run()
    except Exception as err:  # noqa: BLE001 - one bad tick must not stop the loop; it pauses and pages instead
        if OFFLINE.search(str(err)):  # the network or Modal is down: this tick is lost, the next one tries again
            print(f"{S.now()} tick skipped, offline: {str(err)[:200]}", flush=True)
            events = t.events
        else:
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
PENDING = object()  # an orchestrator call still running in the background


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
    trip = st.get("examTripwire") or {}
    print(f"main line {(st.get('main') or {}).get('head') or (st.get('main') or {}).get('sha')}: practice {m.get('wrongCalls')} wrong calls; "
          f"{st.get('foldsSinceExam', 0)} fold(s) since the exam tripwire" + (f" (last: {trip['verdict']}: {trip['why']})" if trip else ""))
    for e in sorted(st["experiments"].values(), key=lambda e: e["since"], reverse=True)[:30]:
        print(f"  {e['state']:12} {e['id']:40} {e['card']['component']:8} {e['tier']:8} {e['note'][:80]}")


def cmd_ab(a):
    """The builder A/B so far, computed from the state on the volume."""
    cfg = config()
    st = World(cfg, executor=object()).load() or S.empty()
    ab = AB.stats(st, [x["name"] for x in X.arms(cfg)])
    if a.json:
        print(json.dumps({**ab, "skipped": (st.get("ab") or {}).get("skipped", {})}, ensure_ascii=False))
        return
    print("\n".join(AB.table(ab)))
    for name, why in ((st.get("ab") or {}).get("skipped") or {}).items():
        print(f"skipped at the last tick: {name}: {why}")


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


def worktree_root() -> Path:
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())


def cmd_modal_run(a):
    """Run by an executor: asks the supervisor to run a command on Modal for this worktree, then waits for it."""
    cmd = " ".join(a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd)
    if not cmd.strip():
        sys.exit("autolab: modal-run -- '<command>'")
    d = worktree_root() / ".autolab"
    d.mkdir(exist_ok=True)
    n = 1
    while True:
        try:
            fd = os.open(d / f"modal-request-{n}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            n += 1
    with os.fdopen(fd, "w") as f:
        json.dump({"cmd": cmd, "shards": a.shards, "gpu": a.gpu, "image": a.image, "timeout": a.timeout, "at": S.now()}, f)
    print(f"modal run {n} asked of the supervisor ({a.shards} shard(s){', ' + a.gpu if a.gpu else ''}, up to {a.timeout} min)", flush=True)
    if a.no_wait:
        print(f"wait for it with: python3 {Path(__file__).resolve()} modal-result {n}")
        return
    wait_modal(d, n, a.timeout)


def cmd_modal_result(a):
    wait_modal(worktree_root() / ".autolab", a.n, a.timeout)


def wait_modal(d: Path, n: int, minutes: int):
    result, deadline = d / f"modal-result-{n}.json", time.time() + (minutes + 60) * 60
    while time.time() < deadline:
        if result.exists():
            out = json.loads(result.read_text())
            if "error" in out:
                sys.exit(f"autolab: modal run {n}: {out['error']}")
            for s in out["shards"]:
                print(f"--- shard {s['shard']}: exit {s['code']} after {s['seconds']} s\n{s['tail'].rstrip()}")
            print(f"outputs (LAB_OUT per shard): {out.get('local')}; on Modal: {out.get('out')}")
            if out.get("gpu"):
                print(f"GPU ${out['usd']:.2f}; ${out['gpuUsdLeft']:.2f} of ${out['gpuUsdBudget']} left")
            sys.exit(out["code"])
        time.sleep(10)
    sys.exit(f"autolab: no result for modal run {n} in {minutes + 60} minutes; the supervisor may be paused")


def sync_exec_data(cfg: dict, ts: str) -> dict:
    w = World(cfg, executor=object())
    p = subprocess.run(w.exec_argv("sync", json.dumps(exec_spec(w.test_set(ts), cfg))), cwd=REPO, capture_output=True, text=True)
    out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": p.stderr[-2000:]}
    if "error" in out:
        raise Stop(f"syncing the builders' practice data: {out['error']}")
    return out


def cmd_sync_exec_data(a):
    """The practice-only copy of the current test set in tilawa-lab-exec. The loop also does this whenever the test set
    version changes."""
    st = World(config(), executor=object()).load()
    out = sync_exec_data(config(), st["testSet"])
    print(f"tilawa-lab-exec: {out['sessions']} practice sessions of {out['testSet']}, {out['files']} files, {out['bytes'] / 1e9:.2f} GB"
          + (f"; not on tilawa-lab: {len(out['missing'])} ({', '.join(out['missing'][:3])}...)" if out["missing"] else ""))


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


def cmd_reopen(a):
    known(a.id)
    send("reopen", id=a.id)


def cmd_rebuild(a):
    known(a.id)
    send("rebuild", id=a.id, note=a.note)


def cmd_revert(a):
    known(a.id)
    send("revert", id=a.id, why=" ".join(a.why))


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
    abp = sub.add_parser("ab")
    abp.add_argument("--json", action="store_true")
    ps = sub.add_parser("practice-score")
    ps.add_argument("worktree", nargs="?", default=".")
    mr = sub.add_parser("modal-run")
    mr.add_argument("--shards", type=int, default=1)
    mr.add_argument("--gpu", default=None)
    mr.add_argument("--image", choices=I.IMAGES, default=None)
    mr.add_argument("--timeout", type=int, default=30, help="minutes")
    mr.add_argument("--no-wait", action="store_true")
    mr.add_argument("cmd", nargs=argparse.REMAINDER)
    mq = sub.add_parser("modal-result")
    mq.add_argument("n", type=int)
    mq.add_argument("--timeout", type=int, default=24 * 60, help="minutes")
    sub.add_parser("sync-exec-data")
    sub.add_parser("watchdog")
    sub.add_parser("eli5")
    ap = sub.add_parser("approve")
    ap.add_argument("id")
    sub.add_parser("waiting")
    rp = sub.add_parser("retry")
    rp.add_argument("id")
    rp.add_argument("--note", default="")
    op = sub.add_parser("reopen")
    op.add_argument("id")
    rbp = sub.add_parser("rebuild")
    rbp.add_argument("id")
    rbp.add_argument("--note", default="")
    rvp = sub.add_parser("revert")
    rvp.add_argument("id")
    rvp.add_argument("why", nargs="*")
    dp = sub.add_parser("drop")
    dp.add_argument("id")
    dp.add_argument("why", nargs="*")
    mp = sub.add_parser("match")
    mp.add_argument("id")
    mp.add_argument("verdict", choices=("agree", "disagree"))
    a = p.parse_args()
    try:
        {"init": cmd_init, "tick": cmd_tick, "loop": cmd_loop, "pause": cmd_pause, "resume": lambda a: cmd_pause(a, why=""),
         "status": cmd_status, "ab": cmd_ab, "practice-score": cmd_practice_score, "watchdog": cmd_watchdog, "match": cmd_match,
         "approve": cmd_approve, "waiting": cmd_waiting, "retry": cmd_retry, "reopen": cmd_reopen, "rebuild": cmd_rebuild, "revert": cmd_revert, "drop": cmd_drop, "eli5": cmd_eli5,
         "modal-run": cmd_modal_run, "modal-result": cmd_modal_result, "sync-exec-data": cmd_sync_exec_data}[a.op](a)
    except Stop as err:
        sys.exit(f"autolab: {err}")


if __name__ == "__main__":
    main()
