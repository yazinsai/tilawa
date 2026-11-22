"""Lab CLI (.scratch/lab/spec.md, tickets 09 and 10). Git runs here; scoring runs on Modal; the ledger lives on the
tilawa-lab volume at /lab/ledger.json (logic in tools/lab_ledger.py).

    python3 tools/lab.py baseline                      # score the main line (practice + exam); first run sets the test set
    python3 tools/lab.py baseline --manifest lab-v1.5/manifest-v1.json --exclude lab-v1.5/flags-v1.json   # a new test set, flags carried
    python3 tools/lab.py new <id> --component checker --hypothesis '...' --expect-component -5 --expect-overall -3
    python3 tools/lab.py run [worktree]                # score an experiment on the practice set, verdict vs the main line
    python3 tools/lab.py status [--all]
    python3 tools/lab.py fold <id> [--yes]             # rebase, re-score practice + exam, merge on confirmation
    python3 tools/lab.py fold <id> --practice-only    # re-score practice only; leave exam unmeasured
    python3 tools/lab.py discard <id>
    python3 tools/lab.py apply-flags [--dry-run]       # bad-data verdicts -> next test set version, re-score everything open
    python3 tools/lab.py run [worktree] --shards 4 --cmd '<command>'   # any command over shards (ticket 09)

Global options: --main <branch> (default main, the main line), --ledger <volume path> (default /ledger.json),
--json (one JSON object on stdout, prose on stderr, never prompts; for tools/autolab.py). `fold --no-merge` rebases,
re-scores and records, then stops before merging; its exam result is only "pass" or "blocked" (tools/lab_gate.py).
`fold --gate` merges only when the gate passes on the rebased run, and discards otherwise.
`baseline --practice-only` scores practice alone, which is what the autolab ledger does (no exam per fold).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_gate as G  # noqa: E402
import lab_ledger as L  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "modal" / "lab_run.py"
VOLUME = "tilawa-lab"
SUMMARY = "docs/benchmarks/lab-ledger.json"
MAX_SHARDS = 200  # one shard per session (lab-v1.5 has 200), so a run takes as long as its longest session


class Stop(Exception):
    pass


VOLUME_TIMEOUT_S = 180  # a volume call that hangs (the network or Modal down) fails instead of holding a tick


def sh(*cmd, cwd=None, check=True, env=None, timeout=None) -> str:
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Stop(f"{' '.join(map(str, cmd[:4]))} timed out after {timeout} s (could not connect?)") from None
    if check and p.returncode:
        raise Stop(f"{' '.join(map(str, cmd))} failed:\n{(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


git = lambda *a, cwd=REPO, check=True: sh("git", *a, cwd=cwd, check=check)


# Volume


def vol_get(path: str, tries: int = 2):
    """A read is safe to repeat, and a hung `modal volume get` usually answers on a fresh call, so a timeout is tried
    again before it fails (it once threw away a finished 100-shard score)."""
    for k in range(tries):
        try:
            p = subprocess.run(["modal", "volume", "get", VOLUME, path, "-"], capture_output=True, timeout=VOLUME_TIMEOUT_S)
            break
        except subprocess.TimeoutExpired:
            if k == tries - 1:
                raise Stop(f"modal volume get {path} timed out after {tries} tries of {VOLUME_TIMEOUT_S} s (could not connect?)") from None
    if p.returncode:
        if b"No such file" in p.stderr or b"No such file" in p.stdout:
            return None
        raise Stop(f"modal volume get {path} failed: {p.stderr.decode().strip()}")
    return json.JSONDecoder().raw_decode(p.stdout.decode().lstrip())[0]  # the CLI appends a "Finished downloading" line


def vol_put(path: str, obj) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    try:
        for wait in (2, 5, 10, 20, 40, 0):
            try:
                sh("modal", "volume", "put", "--force", VOLUME, f.name, path, timeout=VOLUME_TIMEOUT_S)
                break
            except Stop as err:
                if not wait or "too many layers" not in str(err):
                    raise
                time.sleep(wait)  # Modal compacts the volume's layers, then accepts writes again
    finally:
        os.unlink(f.name)


def vol_ls(path: str) -> list[str]:
    try:
        p = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, path], capture_output=True, text=True, timeout=VOLUME_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Stop(f"modal volume ls {path} timed out after {VOLUME_TIMEOUT_S} s (could not connect?)") from None
    return [] if p.returncode else ["/" + x["Filename"] for x in json.loads(p.stdout) if x["Type"] == "file"]


def load(a) -> dict:
    led = vol_get(a.ledger) or L.empty(a.main)
    led["main"] = a.main
    return led


def update(a, fn):
    """Re-reads the ledger, applies `fn`, writes it back: long runs don't overwrite each other's records."""
    led = load(a)
    out = fn(led)
    vol_put(a.ledger, led)
    return out


# Scoring on Modal


APP, FANOUT = "tilawa-lab-run", "fanout"  # deployed with `modal deploy modal/lab_run.py`
RUNS = Path(tempfile.gettempdir()) / "tilawa-lab-runs"  # run files outlive the caller, so a rerun resumes its call
POLL = 60


def runner():
    """modal/lab_run.py as a module, for its `source` and `run_id`; importing it builds nothing."""
    spec = importlib.util.spec_from_file_location("lab_run", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_key(src: bytes, *parts) -> str:
    """Hash of the tarball's file names and contents plus `parts`: the same for a fresh checkout of the same commit."""
    h = hashlib.sha256(json.dumps(parts).encode())
    with tarfile.open(fileobj=io.BytesIO(src)) as t:
        for m in sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: m.name):
            h.update(f"{m.name}\0{m.size}\0".encode())
            h.update(t.extractfile(m).read())
    return h.hexdigest()


def read_run(f: Path) -> dict:
    """A run file: JSON {run, callId, key}; older ones hold just the run id."""
    if not f.exists():
        return {}
    text = f.read_text().strip()
    try:
        return json.loads(text)
    except ValueError:
        return {"run": text}


def live_call(modal, call_id: str):
    """The call if it is running or finished (Modal keeps results 7 days), else None."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.get(timeout=0)
    except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
        return None
    except Exception:
        pass
    return call


def wait_call(modal, call, say) -> dict:
    """Polls `call` until it returns fanout's summary; a call that raised comes back as {ok: False, error}."""
    t0 = time.time()
    while True:
        try:
            return call.get(timeout=POLL)
        except modal.exception.OutputExpiredError:
            return {"ok": False, "error": f"call {call.object_id} expired"}
        except modal.exception.FunctionTimeoutError as e:
            return {"ok": False, "error": f"call {call.object_id} timed out: {e}"}
        except (TimeoutError, modal.exception.TimeoutError):  # modal 1.3 raises the builtin one for a running call
            say(f"waiting on call {call.object_id}: {(time.time() - t0) / 60:.0f} min")
        except Exception as e:
            return {"ok": False, "error": f"call {call.object_id} failed: {type(e).__name__}: {e}"}


BORROW = """import contextlib, json, sys
sys.path.insert(0, sys.argv[1])
import lab
wt, name, cmd, merge, n, log, what = json.loads(sys.argv[2])
try:
    with contextlib.redirect_stdout(sys.stderr):
        out = {"run": lab.spawned(lab.Path(wt), name, cmd, merge, n, log and lab.Path(log), what)}
except lab.Stop as e:
    out = {"error": str(e)}
print(json.dumps(out))
"""


def borrowed(*args) -> str:
    """`spawned` in a uv Python that has the modal package; the CLI's own interpreter usually lacks it (the `modal` CLI has its own)."""
    p = subprocess.run(["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "modal>=1.3", "python", "-P", "-c", BORROW,
                        str(Path(__file__).resolve().parent), json.dumps([str(a) if isinstance(a, Path) else a for a in args])],
                       stdout=subprocess.PIPE, text=True)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise Stop(f"scoring through uv failed (exit {p.returncode}): {p.stdout.strip()[-2000:]}")
    if "error" in out:
        raise Stop(out["error"])
    return out["run"]


def spawned(wt: Path, name: str, cmd: str, merge: str, n: int, log: Path | None, what: str) -> str:
    """Runs `cmd` over `n` shards of the worktree through lab_run's deployed `fanout`, polling the call; returns the run id.
    Rerunning with the same source and commands resumes that call, even after the first caller died."""
    try:
        import modal
        modal.Function
    except (ImportError, AttributeError):  # AttributeError: the repo's modal/ directory shadowing the package
        return borrowed(wt, name, cmd, merge, n, log, what)

    R = runner()
    src, sha = R.source(wt)
    key = source_key(src, cmd, merge, n)
    RUNS.mkdir(parents=True, exist_ok=True)
    run_file = RUNS / f"{name}-{key[:16]}.run"
    log = log or run_file.with_suffix(".log")
    print(f"scoring {name} ({what}) on Modal: {n} shards; log {log}", flush=True)
    with open(log, "a") as f:
        say = lambda s: print(s, file=f, flush=True)
        prev = read_run(run_file)
        call = prev.get("key") == key and prev.get("callId") and live_call(modal, prev["callId"])
        if call:
            run = prev["run"]
            say(f"run {run}: resuming call {call.object_id}")
        else:
            run = R.run_id(name, sha)
            try:
                call = modal.Function.from_name(APP, FANOUT).spawn(src, cmd, n, run, merge)
            except modal.exception.NotFoundError as e:
                raise Stop(f"{e}; deploy the runner: modal deploy {RUNNER.relative_to(REPO)}")
            run_file.write_text(json.dumps({"run": run, "callId": call.object_id, "key": key}) + "\n")
            say(f"run {run}: {n} shard(s), source {len(src) / 1e6:.1f} MB, call {call.object_id}")
        out = wait_call(modal, call, say)
        for s in out.get("shards", []) + [m for m in [out.get("merge")] if m]:
            say(f"{'merge' if s is out.get('merge') else 'shard ' + str(s['shard'])}: exit {s['code']} in {s['seconds']} s")
            if s["code"]:
                say(s["tail"])
        if out.get("error"):
            say(out["error"])
    if not out["ok"]:
        run_file.unlink(missing_ok=True)
        raise Stop(f"scoring {wt.name} failed; last lines of {log}:\n" + "\n".join(log.read_text().splitlines()[-25:]))
    return run


def score(a, wt: Path, *, name: str, listener: str, ts: dict, splits: tuple, log: Path | None = None) -> dict:
    """Scores a worktree's source over the test set version's sessions of `splits`, one shard per session; returns score.json."""
    n = min(MAX_SHARDS, sum(len(ts["sessions"][s]) for s in splits))
    cmd = f"pnpm --filter @tilawa/core lab --manifest /lab/data/{ts['manifest']} --listener {listener}"
    if splits == ("practice",):
        cmd += " --split practice"
    if ts["exclude"]:
        cmd += f" --exclude /lab/data/{ts['exclude']}"
    merge = "pnpm --filter @tilawa/core lab --merge /lab/runs/$LAB_RUN"
    run = spawned(wt, name, cmd, merge, n, log, f"{' + '.join(splits)}, {ts['id']}")
    report = vol_get(f"/runs/{run}/score.json")
    if not report:
        raise Stop(f"no merged score.json for run {run}")
    if report["testSet"] != ts["id"]:
        raise Stop(f"scored test set {report['testSet']}, expected {ts['id']}")
    if report["sessions"]["excluded"]:
        print(f"  {len(report['sessions']['excluded'])} session(s) not scored: " + "; ".join(f"{x['id']}: {x['reason']}" for x in report["sessions"]["excluded"][:5]))
    return report


@contextlib.contextmanager
def main_checkout(a):
    """A clean detached checkout of the main line's tip (never the live `main` worktree, which may hold uncommitted work)."""
    sha = git("rev-parse", a.main)
    path = Path(tempfile.mkdtemp(prefix="lab-main-")) / "src"
    git("worktree", "add", "--detach", path, sha)
    try:
        yield path, sha
    finally:
        git("worktree", "remove", "--force", path, check=False)


def score_main(a, ts: dict, log: Path | None = None, *, splits=("practice", "exam"), listener: str | None = None) -> dict:
    listener = listener or L.main_listener(load(a))
    with main_checkout(a) as (path, _):
        return L.from_report(score(a, path, name="main", listener=listener, ts=ts, splits=splits, log=log), splits)


def main_head(a) -> str:
    return git("rev-parse", "--short", a.main)


def stale(entry: dict, head: str) -> bool:
    return not head.startswith((entry.get("head") or entry["sha"]).removesuffix("-dirty"))


def card_of(wt: Path) -> dict:
    f = wt / L.CARD
    if not f.exists():
        raise Stop(f"no {L.CARD} in {wt}: create experiments with `lab new`")
    return L.validate_card(json.loads(f.read_text()))


def ask(a, prompt: str) -> str:
    if getattr(a, "json", False):
        raise Stop(f"needs a decision ({prompt.strip()}); pass --yes or --no-merge")
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def counts(s: dict) -> str:
    return f"{s['wrongCalls']} wrong calls ({s['missed']} missed + {s['falseWarnings']} false warnings)"


# Commands


def cmd_baseline(a):
    led, new_ts = load(a), None
    if led["testSet"] is None or L.test_set(led)["manifest"] != a.manifest:
        m = vol_get(f"/data/{a.manifest}")
        if not m:
            raise Stop(f"no manifest at /lab/data/{a.manifest}")
        sessions = {s: [e["id"] for e in m["sessions"] if e["split"] == s] for s in ("practice", "exam")}
        new_ts = dict(base=m["testSet"], version=1, manifest=a.manifest, sessions=sessions)
        if a.exclude:
            ex = vol_get(f"/data/{a.exclude}")
            if not ex or ex.get("base") != m["testSet"]:
                raise Stop(f"/lab/data/{a.exclude} is not an exclusion list for {m['testSet']}")
            new_ts.update(exclude=a.exclude, flags=len(ex["excluded"]))
        L.add_test_set(led, **new_ts)
    ts = L.test_set(led)
    practice_only = getattr(a, "practice_only", False)
    res = score_main(a, ts, splits=("practice",) if practice_only else ("practice", "exam"), listener=getattr(a, "listener", None))

    def save(led):
        if new_ts and (led["testSet"] is None or L.test_set(led)["manifest"] != new_ts["manifest"]):
            L.add_test_set(led, **new_ts)
        return L.record_main(led, {**res, "head": res["sha"], "note": a.note or "baseline"})

    update(a, save)
    print(f"main line {res['sha']} on {ts['id']}:")
    print(f"  practice: {counts(res['practice'])} over {res['practice']['sessions']} sessions")
    if not practice_only:
        print(f"  exam:     {counts(res['exam'])} over {res['exam']['sessions']} sessions")
    return {"sha": res["sha"], "run": res["run"], "testSet": ts["id"], "practice": L.brief(res["practice"])}


def drop_leftover_branch(branch: str, base: str, cwd=REPO) -> bool:
    """A branch with no commits of its own, checked out nowhere, is what an interrupted `new` leaves (the branch made,
    the card never committed); it's deleted so `new` can start over from `base`. A branch with work, or in use, stays."""
    if not sh("git", "branch", "--list", branch, cwd=cwd) or sh("git", "rev-list", f"{base}..{branch}", cwd=cwd):
        return False
    sh("git", "worktree", "prune", cwd=cwd)  # a worktree whose directory is gone no longer holds the branch
    if f"branch refs/heads/{branch}\n" in sh("git", "worktree", "list", "--porcelain", cwd=cwd) + "\n":
        return False
    sh("git", "branch", "-D", branch, cwd=cwd)
    return True


def cmd_new(a):
    led = load(a)
    card = L.new_card(a.id, a.component, a.hypothesis, a.expect_component, a.expect_overall, test_set=led["testSet"] or "lab-v1.4",
                      author=a.author or git("config", "user.name", check=False) or os.environ.get("USER", "unknown"),
                      listener=a.listener or L.main_listener(led))
    if a.id in led["experiments"]:
        raise Stop(f"experiment {a.id} already exists")
    wt, branch = REPO.parent / f"tilawa-lab-exp-{a.id}", f"exp/{a.id}"
    drop_leftover_branch(branch, a.main)
    git("worktree", "add", "-b", branch, wt, a.main)
    (wt / L.CARD).write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
    git("add", L.CARD, cwd=wt)
    git("commit", "-q", "-m", f"Lab experiment {a.id} ({a.component}): {a.hypothesis}", cwd=wt)
    update(a, lambda led: L.add_experiment(led, card, branch=branch, worktree=str(wt)))
    print(f"experiment {a.id}: branch {branch} at {wt}")
    print(f"change the {a.component} there, then: python3 tools/lab.py run {wt}")
    return {"id": a.id, "branch": branch, "worktree": str(wt)}


def run_experiment(a, wt: Path, card: dict, ts: dict, log: Path | None = None) -> dict:
    listener = L.scoring_listener(load(a), card["listener"])
    return L.from_report(score(a, wt, name=f"exp-{card['id']}", listener=listener, ts=ts, splits=("practice",), log=log))


def print_verdict(card: dict, res: dict, base: dict):
    cmp = L.compare(res["practice"], base["practice"], card)
    p = res["practice"]
    print(f"\n{card['id']} ({card['component']}): practice set of {res['testSet']}, {p['sessions']} sessions, {p['minutes']} min; {res['sha']} vs main line {base['sha']}")
    print("\n".join(L.format_compare(cmp, card)))
    print(L.format_details(res["practice"], base["practice"]))
    if p["sessions"] != base["practice"]["sessions"]:
        print(f"warning: {p['sessions']} sessions scored here, {base['practice']['sessions']} on the main line, so these counts don't compare (missing evidence?)")
    print(f"verdict: {L.headline(cmp, card)} (a change of {L.NOISE} or fewer is noise)")
    return cmp


def cmd_run(a):
    wt = Path(a.worktree).resolve()
    if a.cmd:
        return {"exitCode": subprocess.run(["modal", "run", str(RUNNER), "--shards", str(a.shards or 1), "--name", a.name, "--cmd", a.cmd],
                                           cwd=REPO, env={**os.environ, "LAB_SRC": str(wt)}).returncode}
    card = card_of(wt)
    led = load(a)
    ts, base = L.test_set(led), L.main_score(led)
    if not base:
        raise Stop(f"the main line has no score on {ts['id']}: run `lab baseline` first")
    if stale(base, main_head(a)):
        print(f"note: the main line moved since its last score ({base['sha']} -> {main_head(a)}); `lab baseline` refreshes it")
    res = run_experiment(a, wt, card, ts)

    def save(led):
        if card["id"] not in led["experiments"]:
            L.add_experiment(led, card, branch=git("branch", "--show-current", cwd=wt), worktree=str(wt))
        L.record_result(led, card["id"], res, vs=base)

    update(a, save)
    cmp = print_verdict(card, res, base)
    print(f"decide: python3 tools/lab.py fold {card['id']}   or   python3 tools/lab.py discard {card['id']}")
    return {"id": card["id"], "sha": res["sha"], "run": res["run"], "base": base["run"], "testSet": ts["id"],
            "practice": L.brief(res["practice"]), "compare": cmp, "verdict": L.headline(cmp, card)}


def cmd_status(a):
    led = load(a)
    rows = L.status_rows(led, all=a.all)
    print("\n".join(L.format_status(led, rows)))
    return {"testSet": led["testSet"], "rows": rows}


def remove_worktree(wt: str, force=False):
    if Path(wt).exists():
        git("worktree", "remove", *(["--force"] if force else []), wt)


def cmd_discard(a):
    led = load(a)
    e = L.experiment(led, a.id)
    if e["state"] == "folded":
        raise Stop(f"{a.id} is already folded into the main line")
    if Path(e["worktree"]).exists() and git("status", "--porcelain", cwd=e["worktree"]) and not a.force:
        raise Stop(f"{e['worktree']} has uncommitted changes; commit them or pass --force")
    update(a, lambda led: L.decide(led, a.id, "discarded", note=a.note or "", testSet=led["testSet"]))
    remove_worktree(e["worktree"], force=a.force)
    print(f"discarded {a.id}: card and results kept in the ledger, worktree removed, branch {e['branch']} kept")
    return {"id": a.id, "state": "discarded", "branch": e["branch"]}


def main_worktree(a) -> Path | None:
    """The worktree that has the main line checked out, if any."""
    cur = None
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            cur = Path(line[9:])
        elif line == f"branch refs/heads/{a.main}":
            return cur
    return None


def fold_summary(led: dict, wt: Path, practice_only: bool) -> dict:
    if not practice_only:
        return L.summary(led)

    # Do not inspect historical exam scores during a restricted fold. Generate the new rows from a practice-only
    # projection, then retain already-committed history rows as opaque records rather than rewriting their scores.
    def practice(value):
        if isinstance(value, dict):
            return {k: practice(v) for k, v in value.items() if k != "exam"}
        if isinstance(value, list):
            return [practice(v) for v in value]
        return value

    out = L.summary(practice(led))
    path = wt / SUMMARY
    prior = json.loads(path.read_text()) if path.exists() else {"rows": []}
    key = lambda row: tuple(row.get(k) for k in ("kind", "at", "sha", "id", "testSet", "state"))
    historic = {key(row): row for row in prior["rows"]}
    out["rows"] = [historic.get(key(row), row) for row in out["rows"]]
    return out


def cmd_fold(a):
    practice_only = getattr(a, "practice_only", False)
    splits = ("practice",) if practice_only else ("practice", "exam")
    led = load(a)
    e = L.experiment(led, a.id)
    card, ts, base = e["card"], L.test_set(led), L.main_score(led)
    if e["state"] != "open":
        raise Stop(f"{a.id} is {e['state']}")
    b = L.latest(e, ts["id"])
    if not b or not base:
        raise Stop(f"{a.id} has no practice result on {ts['id']} yet: python3 tools/lab.py run {e['worktree']}")
    b_base = L.base_for(led, b) or base
    card = {**card, "listener": L.scoring_listener(led, b.get("listener") or card["listener"])}  # its own decode, or the current main line's
    bcmp = L.compare(b["practice"], b_base["practice"], card)
    realuser = getattr(a, "realuser", False)
    b_ru = L.realuser_change(b["practice"], b_base["practice"])
    if realuser:
        state, why = L.realuser_fold_check(card, bcmp, bcmp, b_ru, b_ru)
        if state != "ok":
            raise Stop(f"{a.id}: {why}: nothing to fold")
    elif not L.gains(bcmp, card):
        raise Stop(f"{a.id} has no practice gain on its branch ({L.headline(bcmp, card)}): nothing to fold; discard it")
    wt = Path(e["worktree"])
    if git("status", "--porcelain", "--untracked-files=no", cwd=wt):
        raise Stop(f"{wt} has uncommitted changes; commit them first")
    p = subprocess.run(["git", "rebase", "-q", a.main], cwd=wt, capture_output=True, text=True)
    if p.returncode:
        git("rebase", "--abort", cwd=wt, check=False)
        raise Stop(f"rebasing {e['branch']} onto {a.main} conflicts; resolve it in {wt} and fold again:\n{p.stdout}{p.stderr}")
    head = main_head(a)
    refresh = stale(base, head) or (not practice_only and not base.get("exam"))
    print(f"rebased {e['branch']} onto {a.main} ({head})" + ("; refreshing the main-line measurements too" if refresh else ""))
    logs = Path(tempfile.mkdtemp(prefix="lab-fold-"))
    with ThreadPoolExecutor(2) as pool:
        exp_f = pool.submit(lambda: L.from_report(score(a, wt, name=f"exp-{a.id}-rebased", listener=card["listener"], ts=ts,
                                                         splits=splits, log=logs / "exp.log"), splits))
        main_f = pool.submit(score_main, a, ts, logs / "main.log", splits=splits) if refresh else None
        r = exp_f.result()
        if main_f:
            base = {**main_f.result(), "note": f"re-scored for fold of {a.id}", "at": L.now()}
            base["head"] = base["sha"]
    rcmp = L.compare(r["practice"], base["practice"], card)
    if realuser:
        state, why = L.realuser_fold_check(card, bcmp, rcmp, b_ru, L.realuser_change(r["practice"], base["practice"]))
    else:
        state, why = L.fold_check(card, bcmp, rcmp)
    if practice_only:
        r.update(measurementScope=["practice"], examStatus="unmeasured_for_this_head")
    else:
        exam = L.compare(r["exam"], base["exam"], card)
    out = {"id": a.id, "run": r["run"], "base": base["run"], "branchRun": b["run"], "sha": r["sha"], "testSet": ts["id"],
           "compare": rcmp, "why": why, "practice": L.brief(r["practice"]), "examTripwire": None if practice_only else G.exam_tripwire(base["exam"], r["exam"], card["component"])}

    print(f"\n{a.id} ({card['component']}): {card['hypothesis']}")
    print(f"{'practice':10} {'expected':>9} {'on branch':>10} {'after rebase':>13}")
    for w in ("overall", card["component"]):
        print(f"{w:10} {card['expected']['component' if w != 'overall' else 'overall']:>+9d} {bcmp[w]['change']:>+10d} {rcmp[w]['change']:>+13d}")
    if practice_only:
        print("exam not measured (--practice-only)")
    else:
        print(f"exam (numbers only): main line {counts(base['exam'])}; after fold {counts(r['exam'])}; "
              f"change {exam['overall']['change']:+d} overall, {exam[card['component']]['change']:+d} {card['component']}")

    def record(led, *decision):
        if refresh:
            L.record_main(led, base, at=base["at"])
        L.record_result(led, a.id, r, kind="rebased", vs=base)
        if decision:
            L.decide(led, a.id, decision[0], note=decision[1], testSet=ts["id"])

    if state == "superseded":
        update(a, lambda led: record(led, "superseded", why))
        print(f"superseded: {why}. Not merged.")
        return {**out, "state": "superseded"}
    print(f"after rebase it {why}.")
    if getattr(a, "gate", False):
        verdict = G.gate(vol_get(f"/runs/{base['run']}/score.json"), vol_get(f"/runs/{r['run']}/score.json"), card,
                         git("diff", "-U0", f"{a.main}...HEAD", cwd=wt))
        out["gate"] = verdict
        print("\n".join(G.format_gate(verdict)))
        if verdict["verdict"] != "pass":
            update(a, lambda led: record(led, "discarded", "autolab gate: " + "; ".join(c["why"] for c in verdict["checks"] if c["blocking"] and not c["ok"])))
            return {**out, "state": "gate-failed"}
    if not a.yes and not getattr(a, "gate", False) and (getattr(a, "no_merge", False) or ask(a, f"fold {a.id} into {a.main}? [y/N] ") not in ("y", "yes")):
        update(a, record)
        print("not folded; still waiting for your decision")
        return {**out, "state": "waiting", "worktree": str(wt)}

    def folded(led):
        record(led, "folded", why)
        note = card["hypothesis"] + ("; practice-only fold, exam not measured" if practice_only else "")
        L.record_main(led, {**r, "experiment": a.id, "note": note})

    doc = load(a)
    folded(doc)
    (wt / SUMMARY).write_text(json.dumps(fold_summary(doc, wt, practice_only), indent=1, ensure_ascii=False) + "\n")
    git("add", SUMMARY, cwd=wt)
    git("rm", "-q", "--ignore-unmatch", L.CARD, cwd=wt)  # cards stay on experiment branches; the ledger keeps a copy
    git("commit", "-q", "-m", f"Lab ledger: fold {a.id} (practice {rcmp['overall']['change']:+d} overall, {rcmp[card['component']]['change']:+d} {card['component']})", cwd=wt)
    tip = git("rev-parse", "HEAD", cwd=wt)
    if subprocess.run(["git", "merge-base", "--is-ancestor", a.main, tip], cwd=REPO).returncode:
        git("reset", "-q", "--hard", "HEAD~1", cwd=wt)
        raise Stop(f"{a.main} moved while scoring; fold {a.id} again")
    mwt = main_worktree(a)
    if mwt:
        git("merge", "-q", "--ff-only", tip, cwd=mwt)
    else:
        git("branch", "-f", a.main, tip)

    def save(led):
        folded(led)
        led["mainLine"][-1]["head"] = tip[:7]

    update(a, save)
    remove_worktree(str(wt))
    exam_text = "exam not measured" if practice_only else f"exam {counts(r['exam'])}"
    print(f"folded {a.id}: {a.main} is now {tip[:7]}; main line practice {counts(r['practice'])}, {exam_text}")
    return {**out, "state": "folded", "head": tip[:7]}


def cmd_mark_reverted(a):
    def save(led):
        e = L.experiment(led, a.id)
        if e["state"] != "folded":
            raise Stop(f"{a.id} is {e['state']}, not folded")
        L.decide(led, a.id, "reverted", note=a.note)
    update(a, save)
    return {"id": a.id, "state": "reverted"}


def cmd_apply_flags(a):
    led = load(a)
    ts = L.test_set(led)
    reviews = [x for x in (vol_get(f) for f in vol_ls(f"/reviews/{ts['base']}") if f.endswith(".json")) if x]
    queues = {q: vol_get(f"/data/{q}") for q in {r.get("queue") for rv in reviews for r in rv.get("items", {}).values()} if q}
    applied = (vol_get(f"/data/{ts['exclude']}") or {"excluded": []})["excluded"] if ts["exclude"] else []
    new, warn = L.pending_flags(reviews, {k: v for k, v in queues.items() if v}, set(ts["sessions"]["practice"] + ts["sessions"]["exam"]), applied)
    for w in warn:
        print(f"skipped {w}")
    if not new:
        print(f"no pending flags on {ts['id']} ({len(applied)} already applied)")
        return {"testSet": ts["id"], "pending": []}
    print(f"{len(new)} pending flag(s) on {ts['id']}:")
    for f in new:
        print(f"  {f['item']}: {f['verdict']} by {f['reviewer']} ({f['session']} {f['startS']}-{f['endS']} s){' - ' + f['note'] if f['note'] else ''}")
    if a.dry_run:
        return {"testSet": ts["id"], "pending": new}
    version = ts["version"] + 1
    doc = L.flags_doc(ts["base"], version, applied + new)
    path = f"{ts['base']}/flags-v{version}.json"
    vol_put(f"/data/{path}", doc)
    new_id = update(a, lambda led: L.add_test_set(led, base=ts["base"], version=version, manifest=ts["manifest"], sessions=ts["sessions"],
                                                   exclude=path, flags=len(doc["excluded"])))
    led = load(a)
    nts = led["testSets"][new_id]
    print(f"test set {new_id}: {len(doc['excluded'])} excluded windows (/lab/data/{path}); re-scoring the main line and every open experiment")
    open_exps = {id: e for id, e in led["experiments"].items() if e["state"] == "open"}
    for id, e in list(open_exps.items()):
        if not Path(e["worktree"]).exists():
            print(f"  skipped {id}: worktree {e['worktree']} is gone")
            open_exps.pop(id)
    logs = Path(tempfile.mkdtemp(prefix="lab-flags-"))
    with ThreadPoolExecutor(1 + len(open_exps)) as pool:
        main_f = pool.submit(score_main, a, nts, logs / "main.log")
        exp_f = {id: pool.submit(run_experiment, a, Path(e["worktree"]), e["card"], nts, logs / f"{id}.log") for id, e in open_exps.items()}
        base = main_f.result()
        results, failed = {}, {}
        for id, f in exp_f.items():
            try:
                results[id] = f.result()
            except Stop as err:
                failed[id] = str(err)

    def save(led):
        e = L.record_main(led, {**base, "head": base["sha"], "note": f"re-scored on {new_id}"})
        for id, r in results.items():
            L.record_result(led, id, r, vs=e)

    update(a, save)
    print(f"main line on {new_id}: practice {counts(base['practice'])}; exam {counts(base['exam'])}")
    for id, r in results.items():
        cmp = L.compare(r["practice"], base["practice"], open_exps[id]["card"])
        print(f"  {id}: overall {cmp['overall']['change']:+d}, {open_exps[id]['card']['component']} {cmp[open_exps[id]['card']['component']]['change']:+d} -> {L.headline(cmp, open_exps[id]['card'])}")
    for id, err in failed.items():
        print(f"  {id}: scoring failed: {err.splitlines()[0]}")
    return {"testSet": new_id, "run": base["run"], "practice": L.brief(base["practice"]),
            "experiments": {id: r["run"] for id, r in results.items()}, "failed": failed}


def parser():
    p = argparse.ArgumentParser(description="Lab experiments: cards, practice runs, fold-in, flags.")
    p.add_argument("--main", default=os.environ.get("LAB_MAIN", "main"), help="main line branch")
    p.add_argument("--ledger", default=os.environ.get("LAB_LEDGER", "/ledger.json"), help="ledger path on the tilawa-lab volume")
    p.add_argument("--json", action="store_true", help="one JSON object on stdout, prose on stderr; never prompts")
    sub = p.add_subparsers(dest="op", required=True)
    b = sub.add_parser("baseline", help="score the main line on practice and exam")
    b.add_argument("--manifest", default="lab-v1.5/manifest-v1.json", help="under /lab/data; registers its test set when the ledger has none or another one")
    b.add_argument("--exclude", default="", help="under /lab/data: flags carried into a new test set (e.g. lab-v1.5/flags-v1.json, from lab-v1.4's)")
    b.add_argument("--note", default="")
    b.add_argument("--practice-only", action="store_true", help="score practice only (the autolab ledger never scores the exam)")
    b.add_argument("--listener", default=None, help="listener evidence to score with (default: the main line's); e.g. the one before a reverted listener fold")
    n = sub.add_parser("new", help="create an experiment worktree and card")
    n.add_argument("id")
    n.add_argument("--component", required=True, choices=L.COMPONENTS)
    n.add_argument("--hypothesis", required=True)
    n.add_argument("--expect-component", type=int, required=True, help="expected change in the component's wrong calls (negative = fewer)")
    n.add_argument("--expect-overall", type=int, required=True, help="expected change in overall wrong calls")
    n.add_argument("--listener", default=None, help="decoded listener evidence to score with (default: the main line's)")
    n.add_argument("--author", default="")
    r = sub.add_parser("run", help="score an experiment on the practice set (or --cmd: any command over shards)")
    r.add_argument("worktree", nargs="?", default=".")
    r.add_argument("--cmd", help="run this command instead of the scorer (ticket 09)")
    r.add_argument("--shards", type=int)
    r.add_argument("--name", default="adhoc")
    s = sub.add_parser("status", help="open experiments, waiting for your decision first")
    s.add_argument("--all", action="store_true", help="include folded and discarded")
    f = sub.add_parser("fold", help="rebase onto the main line, re-score, merge on confirmation")
    f.add_argument("id")
    f.add_argument("--yes", action="store_true", help="skip the confirmation (only when the user approved this fold)")
    f.add_argument("--practice-only", action="store_true", help="score only practice and leave exam unmeasured for the new main revision")
    f.add_argument("--no-merge", action="store_true", help="rebase, re-score and record, then stop before merging (the autolab gate decides)")
    f.add_argument("--realuser", action="store_true",
                   help="the gain is fewer false warnings on real users' clean recitation, with practice not worse (practice-neutral fixes)")
    f.add_argument("--gate", action="store_true", help="merge only if tools/lab_gate.py passes on the rebased run; discard otherwise (autolab)")
    d = sub.add_parser("discard", help="mark discarded, remove the worktree, keep the branch")
    d.add_argument("id")
    d.add_argument("--note", default="")
    d.add_argument("--force", action="store_true", help="remove the worktree even with uncommitted changes")
    rv = sub.add_parser("mark-reverted", help="record that a folded experiment's commits were reverted on the main line")
    rv.add_argument("id")
    rv.add_argument("--note", default="")
    af = sub.add_parser("apply-flags", help="bad-data verdicts -> next test set version; re-score the main line and open experiments")
    af.add_argument("--dry-run", action="store_true")
    return p


def main():
    a = parser().parse_args()
    fn = {"baseline": cmd_baseline, "new": cmd_new, "run": cmd_run, "status": cmd_status, "fold": cmd_fold,
          "discard": cmd_discard, "mark-reverted": cmd_mark_reverted, "apply-flags": cmd_apply_flags}[a.op]
    try:
        if a.json:
            with contextlib.redirect_stdout(sys.stderr):
                out = fn(a) or {}
            print(json.dumps(out, ensure_ascii=False))
        else:
            out = fn(a) or {}
    except (Stop, LookupError, ValueError) as err:
        if a.json:
            print(json.dumps({"error": str(err)}, ensure_ascii=False))
            sys.exit(1)
        sys.exit(f"lab: {err}")
    if out.get("exitCode"):
        sys.exit(out["exitCode"])


if __name__ == "__main__":
    main()
