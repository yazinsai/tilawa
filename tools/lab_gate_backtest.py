"""Backtest the autolab fold gate (tools/lab_gate.py) on every experiment in the lab ledger (.scratch/autolab ticket 02).

    python3 tools/lab_gate_backtest.py [--out .scratch/autolab/backtest.md]

For each experiment: the gate on its latest branch run against the main-line score it was measured on, and on its
rebased run where there is one, next to the decision you recorded. Score reports are cached under .scratch/lab/cache/runs/.
Only the practice split is read; the exam tripwire is not backtested.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_gate as G  # noqa: E402
import lab_ledger as L  # noqa: E402
from lab import REPO, vol_get  # noqa: E402

CACHE = REPO / ".scratch/lab/cache/runs"


def report(run: str) -> dict | None:
    f = CACHE / f"{run}.json"
    if not f.exists():
        r = vol_get(f"/runs/{run}/score.json")
        if not r:
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, ensure_ascii=False))
    return json.loads(f.read_text())


def commit(ref: str) -> str | None:
    p = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip() or None


def card_commit(id: str, tip: str) -> str | None:
    """The `lab new` commit that started the experiment, reachable from `tip`."""
    out = subprocess.run(["git", "log", "--format=%H", "--fixed-strings", f"--grep=Lab experiment {id} (", tip],
                         cwd=REPO, capture_output=True, text=True).stdout.split()
    return out[-1] if out else None


def diff(id: str, sha: str, branch: str) -> tuple[str | None, str]:
    """The experiment's own commits: from its card commit's parent to the scored commit (the branch tip for a dirty run).
    Not from the main-line score's sha, which can be older than the branch point when main moved by docs only."""
    tip = None if sha.endswith("-dirty") else commit(sha)
    note = ""
    if not tip:
        tip, note = commit(branch), "diff to branch tip (scored code not committed)"
    start = tip and card_commit(id, tip)
    if not start:
        return None, "no diff: commit, branch or card commit gone"
    return subprocess.run(["git", "diff", "-U0", f"{start}^", tip], cwd=REPO, capture_output=True, text=True, check=True).stdout, note


def run_gate(led: dict, e: dict, r: dict) -> dict:
    base_entry = L.base_for(led, r)
    if not base_entry:
        return {"error": "no main-line score it was measured against"}
    b, n = report(base_entry["run"]), report(r["run"])
    if not b or not n:
        return {"error": f"score.json gone for {base_entry['run'] if not b else r['run']}"}
    d, note = diff(e["card"]["id"], r["sha"], e["branch"])
    res = G.gate(b, n, e["card"], d)
    if d is None:
        res["checks"] = [c for c in res["checks"] if c["name"] != "hygiene"]
        res["verdict"] = G.verdict(res["checks"])
    return {**res, "note": note, "run": r["run"], "base": base_entry["run"]}


def cell(g: dict | None) -> str:
    if not g:
        return "-"
    if "error" in g:
        return f"n/a ({g['error']})"
    bad = [c["name"] for c in g["checks"] if not c["ok"] and c["blocking"]]
    return g["verdict"] + (f": {', '.join(bad)}" if bad else "") + (" (tuning)" if g["tuning"] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / ".scratch/autolab/backtest.md"))
    ap.add_argument("--ledger", default="/ledger.json")
    a = ap.parse_args()
    led = vol_get(a.ledger)
    rows, details = [], []
    for id, e in sorted(led["experiments"].items(), key=lambda kv: kv[1]["card"]["created"]):
        ts = (e.get("decision") or {}).get("testSet") or e["card"]["testSet"]
        tsets = [ts] + [t for t in led["testSets"] if t != ts]
        br = next((x for t in tsets if (x := L.latest(e, t))), None)
        rb = next((x for t in tsets if (x := L.latest(e, t, "rebased"))), None)
        print(f"{id}: gate on {br and br['run']}" + (f" and {rb['run']}" if rb else ""), file=sys.stderr, flush=True)
        g_br = run_gate(led, e, br) if br else None
        g_rb = run_gate(led, e, rb) if rb else None
        decided = e["state"] if e["state"] != "open" else "open (undecided)"
        final = g_rb or g_br
        gate_folds = bool(final and final.get("verdict") == "pass")
        agree = "-" if e["state"] == "open" or not final or "error" in final else ("yes" if gate_folds == (e["state"] == "folded") else "NO")
        c = e["card"]
        rows.append(f"| {id} | {c['component']} | {c['expected']['overall']:+d} / {c['expected']['component']:+d} | {decided} | {cell(g_br)} | {cell(g_rb)} | {agree} |")
        for label, g in (("branch", g_br), ("after rebase", g_rb)):
            if g and "error" not in g:
                details.append(f"### {id}, {label}\n\n`{g['run']}` against `{g['base']}`" + (f"; {g['note']}" if g["note"] else "") + "\n\n```text\n"
                               + "\n".join(G.format_gate(g)) + "\n```\n")
    agreed = sum(r.endswith("| yes |") for r in rows)
    disagreed = [r for r in rows if r.endswith("| NO |")]
    out = ["# Fold gate backtest", "",
           f"Generated by `tools/lab_gate_backtest.py` from the lab ledger (test set {led['testSet']}). The gate (`tools/lab_gate.py`) on each experiment's",
           "latest branch run and, where there is one, its rebased run, each against the main-line score it was measured on. Practice only; the exam",
           "tripwire is not backtested. Agreement compares the gate's final verdict (after rebase where measured) with your recorded decision.", "",
           f"Agreed on {agreed} of {agreed + len(disagreed)} decided experiments with usable reports.", "",
           "| experiment | component | expected overall / component | your decision | gate on branch | gate after rebase | agree |",
           "|---|---|---|---|---|---|---|", *rows, "", "## Disagreements", "", *(disagreed or ["None."]), "", "## Gate details", "", *details]
    Path(a.out).write_text("\n".join(out) + "\n")
    print(f"wrote {a.out}: agreed on {agreed}, disagreed on {len(disagreed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
