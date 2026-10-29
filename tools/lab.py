"""Lab CLI (.scratch/lab/spec.md, tickets 09 and 10). Git runs here; scoring runs on Modal; the ledger lives on the
tilawa-lab volume at /lab/ledger.json (logic in tools/lab_ledger.py).

    python3 tools/lab.py baseline                      # score the main line (practice + exam); first run sets the test set
    python3 tools/lab.py new <id> --component checker --hypothesis '...' --expect-component -5 --expect-overall -3
    python3 tools/lab.py run [worktree]                # score an experiment on the practice set, verdict vs the main line
    python3 tools/lab.py status [--all]
    python3 tools/lab.py fold <id> [--yes]             # rebase, re-score practice + exam, merge on confirmation
    python3 tools/lab.py discard <id>
    python3 tools/lab.py apply-flags [--dry-run]       # bad-data verdicts -> next test set version, re-score everything open
    python3 tools/lab.py run [worktree] --shards 4 --cmd '<command>'   # any command over shards (ticket 09)

Global options: --main <branch> (default lab, the main line), --ledger <volume path> (default /ledger.json).
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_ledger as L  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "modal" / "lab_run.py"
VOLUME = "tilawa-lab"
SUMMARY = "docs/benchmarks/lab-ledger.json"
MAX_SHARDS = 60


class Stop(Exception):
    pass


def sh(*cmd, cwd=None, check=True, env=None) -> str:
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env)
    if check and p.returncode:
        raise Stop(f"{' '.join(map(str, cmd))} failed:\n{(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


git = lambda *a, cwd=REPO, check=True: sh("git", *a, cwd=cwd, check=check)


# Volume


def vol_get(path: str):
    p = subprocess.run(["modal", "volume", "get", VOLUME, path, "-"], capture_output=True)
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
        sh("modal", "volume", "put", "--force", VOLUME, f.name, path)
    finally:
        os.unlink(f.name)


def vol_ls(path: str) -> list[str]:
    p = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, path], capture_output=True, text=True)
    return [] if p.returncode else ["/" + x["Filename"] for x in json.loads(p.stdout) if x["Type"] == "file"]


def load(a) -> dict:
    return vol_get(a.ledger) or L.empty(a.main)


def update(a, fn):
    """Re-reads the ledger, applies `fn`, writes it back: long runs don't overwrite each other's records."""
    led = load(a)
    out = fn(led)
    vol_put(a.ledger, led)
    return out


# Scoring on Modal


def score(a, wt: Path, *, name: str, listener: str, ts: dict, splits: tuple, log: Path | None = None) -> dict:
    """Scores a worktree's source over the test set version's sessions of `splits`, one shard per session; returns score.json."""
    n = min(MAX_SHARDS, sum(len(ts["sessions"][s]) for s in splits))
    cmd = f"pnpm --filter @tilawa/core lab --manifest /lab/data/{ts['manifest']} --listener {listener}"
    if splits == ("practice",):
        cmd += " --split practice"
    if ts["exclude"]:
        cmd += f" --exclude /lab/data/{ts['exclude']}"
    merge = "pnpm --filter @tilawa/core lab --merge /lab/runs/$LAB_RUN"
    log = log or Path(tempfile.mkdtemp(prefix="lab-")) / f"{name}.log"
    run_file = log.with_suffix(".run")
    print(f"scoring {name} ({' + '.join(splits)}, {ts['id']}) on Modal: {n} shards; log {log}", flush=True)
    with open(log, "w") as f:
        p = subprocess.run(["modal", "run", str(RUNNER), "--shards", str(n), "--name", name, "--cmd", cmd, "--merge", merge, "--run-file", str(run_file)],
                           cwd=REPO, env={**os.environ, "LAB_SRC": str(wt)}, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode or not run_file.exists():
        raise Stop(f"scoring {wt.name} failed; last lines of {log}:\n" + "\n".join(log.read_text().splitlines()[-25:]))
    report = vol_get(f"/runs/{run_file.read_text().strip()}/score.json")
    if not report:
        raise Stop(f"no merged score.json for run {run_file.read_text().strip()}")
    if report["testSet"] != ts["id"]:
        raise Stop(f"scored test set {report['testSet']}, expected {ts['id']}")
    if report["sessions"]["excluded"]:
        print(f"  {len(report['sessions']['excluded'])} session(s) not scored: " + "; ".join(f"{x['id']}: {x['reason']}" for x in report["sessions"]["excluded"][:5]))
    return report


@contextlib.contextmanager
def main_checkout(a):
    """A clean detached checkout of the main line's tip (never the live `lab` worktree, which may hold uncommitted work)."""
    sha = git("rev-parse", a.main)
    path = Path(tempfile.mkdtemp(prefix="lab-main-")) / "src"
    git("worktree", "add", "--detach", path, sha)
    try:
        yield path, sha
    finally:
        git("worktree", "remove", "--force", path, check=False)


def score_main(a, ts: dict, log: Path | None = None) -> dict:
    with main_checkout(a) as (path, _):
        return L.from_report(score(a, path, name="main", listener=L.DEFAULT_LISTENER, ts=ts, splits=("practice", "exam"), log=log), ("practice", "exam"))


def main_head(a) -> str:
    return git("rev-parse", "--short", a.main)


def stale(entry: dict, head: str) -> bool:
    return not head.startswith((entry.get("head") or entry["sha"]).removesuffix("-dirty"))


def card_of(wt: Path) -> dict:
    f = wt / L.CARD
    if not f.exists():
        raise Stop(f"no {L.CARD} in {wt}: create experiments with `lab new`")
    return L.validate_card(json.loads(f.read_text()))


def ask(prompt: str) -> str:
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
        L.add_test_set(led, **new_ts)
    ts = L.test_set(led)
    res = score_main(a, ts)

    def save(led):
        if new_ts and (led["testSet"] is None or L.test_set(led)["manifest"] != new_ts["manifest"]):
            L.add_test_set(led, **new_ts)
        return L.record_main(led, {**res, "head": res["sha"], "note": a.note or "baseline"})

    update(a, save)
    print(f"main line {res['sha']} on {ts['id']}:")
    print(f"  practice: {counts(res['practice'])} over {res['practice']['sessions']} sessions")
    print(f"  exam:     {counts(res['exam'])} over {res['exam']['sessions']} sessions")


def cmd_new(a):
    led = load(a)
    card = L.new_card(a.id, a.component, a.hypothesis, a.expect_component, a.expect_overall, test_set=led["testSet"] or "lab-v1.4",
                      author=a.author or git("config", "user.name", check=False) or os.environ.get("USER", "unknown"), listener=a.listener)
    if a.id in led["experiments"]:
        raise Stop(f"experiment {a.id} already exists")
    wt, branch = REPO.parent / f"tilawa-lab-exp-{a.id}", f"exp/{a.id}"
    git("worktree", "add", "-b", branch, wt, a.main)
    (wt / L.CARD).write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
    git("add", L.CARD, cwd=wt)
    git("commit", "-q", "-m", f"Lab experiment {a.id} ({a.component}): {a.hypothesis}", cwd=wt)
    update(a, lambda led: L.add_experiment(led, card, branch=branch, worktree=str(wt)))
    print(f"experiment {a.id}: branch {branch} at {wt}")
    print(f"change the {a.component} there, then: python3 tools/lab.py run {wt}")


def run_experiment(a, wt: Path, card: dict, ts: dict, log: Path | None = None) -> dict:
    return L.from_report(score(a, wt, name=f"exp-{card['id']}", listener=card["listener"], ts=ts, splits=("practice",), log=log))


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
        sys.exit(subprocess.run(["modal", "run", str(RUNNER), "--shards", str(a.shards or 1), "--name", a.name, "--cmd", a.cmd],
                                cwd=REPO, env={**os.environ, "LAB_SRC": str(wt)}).returncode)
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
    print_verdict(card, res, base)
    print(f"decide: python3 tools/lab.py fold {card['id']}   or   python3 tools/lab.py discard {card['id']}")


def cmd_status(a):
    led = load(a)
    print("\n".join(L.format_status(led, L.status_rows(led, all=a.all))))


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


def main_worktree(a) -> Path | None:
    """The worktree that has the main line checked out, if any."""
    cur = None
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            cur = Path(line[9:])
        elif line == f"branch refs/heads/{a.main}":
            return cur
    return None


def cmd_fold(a):
    led = load(a)
    e = L.experiment(led, a.id)
    card, ts, base = e["card"], L.test_set(led), L.main_score(led)
    if e["state"] != "open":
        raise Stop(f"{a.id} is {e['state']}")
    b = L.latest(e, ts["id"])
    if not b or not base:
        raise Stop(f"{a.id} has no practice result on {ts['id']} yet: python3 tools/lab.py run {e['worktree']}")
    b_base = L.base_for(led, b) or base
    bcmp = L.compare(b["practice"], b_base["practice"], card)
    if not L.gains(bcmp, card):
        raise Stop(f"{a.id} has no practice gain on its branch ({L.headline(bcmp, card)}): nothing to fold; discard it")
    wt = Path(e["worktree"])
    if git("status", "--porcelain", "--untracked-files=no", cwd=wt):
        raise Stop(f"{wt} has uncommitted changes; commit them first")
    p = subprocess.run(["git", "rebase", "-q", a.main], cwd=wt, capture_output=True, text=True)
    if p.returncode:
        git("rebase", "--abort", cwd=wt, check=False)
        raise Stop(f"rebasing {e['branch']} onto {a.main} conflicts; resolve it in {wt} and fold again:\n{p.stdout}{p.stderr}")
    head = main_head(a)
    refresh = stale(base, head)
    print(f"rebased {e['branch']} onto {a.main} ({head})" + ("; the main line moved since its last score, re-scoring it too" if refresh else ""))
    logs = Path(tempfile.mkdtemp(prefix="lab-fold-"))
    with ThreadPoolExecutor(2) as pool:
        exp_f = pool.submit(lambda: L.from_report(score(a, wt, name=f"exp-{a.id}-rebased", listener=card["listener"], ts=ts,
                                                         splits=("practice", "exam"), log=logs / "exp.log"), ("practice", "exam")))
        main_f = pool.submit(score_main, a, ts, logs / "main.log") if refresh else None
        r = exp_f.result()
        if main_f:
            base = {**main_f.result(), "note": f"re-scored for fold of {a.id}", "at": L.now()}
            base["head"] = base["sha"]
    rcmp = L.compare(r["practice"], base["practice"], card)
    state, why = L.fold_check(card, bcmp, rcmp)
    exam = L.compare(r["exam"], base["exam"], card)

    print(f"\n{a.id} ({card['component']}): {card['hypothesis']}")
    print(f"{'practice':10} {'expected':>9} {'on branch':>10} {'after rebase':>13}")
    for w in ("overall", card["component"]):
        print(f"{w:10} {card['expected']['component' if w != 'overall' else 'overall']:>+9d} {bcmp[w]['change']:>+10d} {rcmp[w]['change']:>+13d}")
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
        return
    print(f"after rebase it {why}.")
    if not a.yes and ask(f"fold {a.id} into {a.main}? [y/N] ") not in ("y", "yes"):
        update(a, record)
        print("not folded; still waiting for your decision")
        return

    def folded(led):
        record(led, "folded", why)
        L.record_main(led, {**r, "experiment": a.id, "note": card["hypothesis"]})

    doc = load(a)
    folded(doc)
    (wt / SUMMARY).write_text(json.dumps(L.summary(doc), indent=1, ensure_ascii=False) + "\n")
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
    print(f"folded {a.id}: {a.main} is now {tip[:7]}; main line practice {counts(r['practice'])}, exam {counts(r['exam'])}")


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
        return
    print(f"{len(new)} pending flag(s) on {ts['id']}:")
    for f in new:
        print(f"  {f['item']}: {f['verdict']} by {f['reviewer']} ({f['session']} {f['startS']}-{f['endS']} s){' - ' + f['note'] if f['note'] else ''}")
    if a.dry_run:
        return
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


def parser():
    p = argparse.ArgumentParser(description="Lab experiments: cards, practice runs, fold-in, flags.")
    p.add_argument("--main", default=os.environ.get("LAB_MAIN", "lab"), help="main line branch")
    p.add_argument("--ledger", default=os.environ.get("LAB_LEDGER", "/ledger.json"), help="ledger path on the tilawa-lab volume")
    sub = p.add_subparsers(dest="op", required=True)
    b = sub.add_parser("baseline", help="score the main line on practice and exam")
    b.add_argument("--manifest", default="lab-v1.4/manifest-v1.json", help="under /lab/data; registers its test set when the ledger has none or another one")
    b.add_argument("--note", default="")
    n = sub.add_parser("new", help="create an experiment worktree and card")
    n.add_argument("id")
    n.add_argument("--component", required=True, choices=L.COMPONENTS)
    n.add_argument("--hypothesis", required=True)
    n.add_argument("--expect-component", type=int, required=True, help="expected change in the component's wrong calls (negative = fewer)")
    n.add_argument("--expect-overall", type=int, required=True, help="expected change in overall wrong calls")
    n.add_argument("--listener", default=L.DEFAULT_LISTENER, help="decoded listener evidence to score with")
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
    d = sub.add_parser("discard", help="mark discarded, remove the worktree, keep the branch")
    d.add_argument("id")
    d.add_argument("--note", default="")
    d.add_argument("--force", action="store_true", help="remove the worktree even with uncommitted changes")
    af = sub.add_parser("apply-flags", help="bad-data verdicts -> next test set version; re-score the main line and open experiments")
    af.add_argument("--dry-run", action="store_true")
    return p


def main():
    a = parser().parse_args()
    try:
        {"baseline": cmd_baseline, "new": cmd_new, "run": cmd_run, "status": cmd_status, "fold": cmd_fold,
         "discard": cmd_discard, "apply-flags": cmd_apply_flags}[a.op](a)
    except (Stop, LookupError, ValueError) as err:
        sys.exit(f"lab: {err}")


if __name__ == "__main__":
    main()
