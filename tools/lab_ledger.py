"""Lab experiment ledger (.scratch/lab/issues/10-experiment-ledger.md): cards, scores, verdicts, states and flags.

Pure logic; `tools/lab.py` does the git and Modal work. The ledger is one JSON document (on the tilawa-lab volume at
/lab/ledger.json); every function here takes and returns plain dicts.

Counts, not rates: every number compared is a count of wrong calls (missed + false warnings) on the practice set,
and a change of NOISE or fewer is noise. A component's count is the wrong calls the attribution charges to it, so
listener + tracker + checker = overall. Details beside them: the listener's phoneme errors, the tracker's false jumps
and the checker's wrong calls with perfect inputs (oracle evidence, true position).
"""

import re
from datetime import datetime, timezone

COMPONENTS = ("listener", "tracker", "checker")
NOISE = 2
DEFAULT_LISTENER = "spliced-clean-int8"
CARD = "lab-experiment.json"
STATES = ("open", "superseded", "folded", "discarded")
BAD = ("bad_label", "bad_audio", "bad_splice")
WAITING = "waiting for your decision"
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Cards


def new_card(id: str, component: str, hypothesis: str, expect_component: int, expect_overall: int, *, test_set: str,
             author: str, listener: str = DEFAULT_LISTENER, created: str | None = None) -> dict:
    return validate_card({"id": id, "component": component, "hypothesis": hypothesis,
                          "expected": {"component": expect_component, "overall": expect_overall},
                          "listener": listener, "testSet": test_set, "created": created or now(), "author": author})


def validate_card(c: dict) -> dict:
    if not ID.match(str(c.get("id", ""))):
        raise ValueError(f"experiment id {c.get('id')!r}: lowercase letters, digits and dashes, up to 40")
    if c.get("component") not in COMPONENTS:
        raise ValueError(f"component must be one of {', '.join(COMPONENTS)}")
    h = str(c.get("hypothesis", "")).strip()
    if not h or "\n" in h:
        raise ValueError("hypothesis must be one line")
    e = c.get("expected") or {}
    if not all(isinstance(e.get(k), int) and not isinstance(e.get(k), bool) for k in ("component", "overall")):
        raise ValueError("expected.component and expected.overall must be whole counts (negative = fewer wrong calls)")
    for k in ("listener", "testSet", "created", "author"):
        if not c.get(k):
            raise ValueError(f"card lacks {k}")
    return c


# Scores


def score(totals: dict) -> dict:
    """The ledger's compact copy of one split's `Totals` from the scorer's score.json."""
    o, a, c = totals["overall"], totals["attribution"], totals["components"]
    part = lambda k: {"wrongCalls": a[k], "missed": a["missed"][k], "falseWarnings": a["falseWarnings"][k]}
    return {
        "sessions": totals["sessions"], "minutes": totals["recitedMinutes"],
        "wrongCalls": o["wrongCalls"], "missed": o["missed"], "falseWarnings": o["falseWarnings"], "mistakes": o["mistakes"], "per5Min": o["per5Min"],
        "listener": {**part("listener"), "phonemeErrors": c["listener"]["edits"], "phonemes": c["listener"]["phonemes"], "per": c["listener"]["per"]},
        "tracker": {**part("tracker"), "falseJumps": c["tracker"]["falseJumps"], "onPlaceShare": c["tracker"]["onPlaceShare"]},
        "checker": {**part("checker"), "alone": {k: c["checker"][k] for k in ("wrongCalls", "missed", "falseWarnings")}},
    }


def from_report(report: dict, splits=("practice",)) -> dict:
    """What a run records: its identity and the compact score of each split asked for (exam only when asked)."""
    if report.get("kind") != "lab-score":
        raise ValueError("not a lab score report")
    out = {"sha": report["sha"], "run": report["run"], "testSet": report["testSet"], "listener": report["listener"],
           "excluded": len(report["sessions"]["excluded"])}
    for s in splits:
        if not report["splits"][s]["sessions"]:
            raise ValueError(f"the report has no {s} sessions")
        out[s] = score(report["splits"][s])
    return out


def count(s: dict, what: str) -> int:
    return s["wrongCalls"] if what == "overall" else s[what]["wrongCalls"]


def verdict(change: int) -> str:
    return "noise" if abs(change) <= NOISE else "better" if change < 0 else "worse"


def compare(new: dict, base: dict, card: dict | None = None) -> dict:
    """Per row (overall, then each component): main line, this, change and verdict; expected where the card says."""
    rows = {}
    for what in ("overall", *COMPONENTS):
        b, n = (base, new) if what == "overall" else (base[what], new[what])
        change = n["wrongCalls"] - b["wrongCalls"]
        rows[what] = {"main": [b["wrongCalls"], b["missed"], b["falseWarnings"]], "this": [n["wrongCalls"], n["missed"], n["falseWarnings"]],
                      "change": change, "verdict": verdict(change)}
    if card:
        rows["overall"]["expected"] = card["expected"]["overall"]
        rows[card["component"]]["expected"] = card["expected"]["component"]
    return rows


def gains(cmp: dict, card: dict) -> list[str]:
    """The card's numbers (its component, overall) that got better."""
    return [w for w in ("overall", card["component"]) if cmp[w]["verdict"] == "better"]


def headline(cmp: dict, card: dict) -> str:
    """One verdict for an experiment: overall's, or its component's when overall is noise."""
    o, c = cmp["overall"]["verdict"], cmp[card["component"]]["verdict"]
    return o if o != "noise" or c == "noise" else f"noise overall, {card['component']} {c}"


# Ledger


def empty(main: str = "lab") -> dict:
    return {"version": 1, "kind": "lab-ledger", "main": main, "testSet": None, "testSets": {}, "mainLine": [], "experiments": {}}


def add_test_set(led: dict, *, base: str, version: int, manifest: str, sessions: dict, exclude: str | None = None,
                 flags: int = 0, at: str | None = None) -> str:
    """Registers a test set version and makes it current. Version 1 is the base name; later ones are `<base>.v<n>`."""
    id = base if version == 1 else f"{base}.v{version}"
    led["testSets"][id] = {"id": id, "base": base, "version": version, "manifest": manifest, "exclude": exclude, "flags": flags,
                           "sessions": {k: list(sessions[k]) for k in ("practice", "exam")}, "created": at or now(), "previous": led["testSet"]}
    led["testSet"] = id
    return id


def test_set(led: dict) -> dict:
    if not led["testSet"]:
        raise LookupError("the ledger has no test set yet: run `lab baseline` first")
    return led["testSets"][led["testSet"]]


def record_main(led: dict, entry: dict, at: str | None = None) -> dict:
    """A main-line score: {sha, testSet, run, practice, exam, note?, experiment?}."""
    e = {**entry, "at": at or now()}
    led["mainLine"].append(e)
    return e


def main_score(led: dict, ts: str | None = None) -> dict | None:
    """The latest main-line score on a test set version (the current one by default)."""
    ts = ts or led["testSet"]
    return next((e for e in reversed(led["mainLine"]) if e["testSet"] == ts), None)


def add_experiment(led: dict, card: dict, *, branch: str, worktree: str) -> dict:
    if card["id"] in led["experiments"]:
        raise ValueError(f"experiment {card['id']} already exists ({led['experiments'][card['id']]['state']})")
    e = {"card": validate_card(card), "branch": branch, "worktree": worktree, "state": "open", "results": [], "decision": None}
    led["experiments"][card["id"]] = e
    return e


def experiment(led: dict, id: str) -> dict:
    if id not in led["experiments"]:
        raise LookupError(f"no experiment {id}")
    return led["experiments"][id]


def record_result(led: dict, id: str, result: dict, kind: str = "branch", at: str | None = None) -> dict:
    """A scored run of the experiment: kind `branch` (as written) or `rebased` (at fold-in)."""
    r = {**result, "kind": kind, "at": at or now()}
    experiment(led, id)["results"].append(r)
    return r


def latest(exp: dict, ts: str, kind: str = "branch") -> dict | None:
    return next((r for r in reversed(exp["results"]) if r["testSet"] == ts and r["kind"] == kind), None)


def decide(led: dict, id: str, state: str, note: str = "", at: str | None = None, **extra) -> dict:
    if state not in STATES[1:]:
        raise ValueError(f"bad state {state}")
    e = experiment(led, id)
    e["state"], e["decision"] = state, {"state": state, "at": at or now(), "note": note, **extra}
    return e


def shown_state(exp: dict, ts: str) -> str:
    if exp["state"] != "open":
        return exp["state"]
    return WAITING if latest(exp, ts) else "open"


def status_rows(led: dict, all: bool = False) -> list[dict]:
    """Experiments for `lab status`: waiting for a decision first, then open, then superseded (then decided ones with `all`)."""
    ts, base = led["testSet"], main_score(led)
    order = {WAITING: 0, "open": 1, "superseded": 2, "folded": 3, "discarded": 4}
    rows = []
    for id, e in led["experiments"].items():
        state = shown_state(e, ts)
        if not all and state in ("folded", "discarded"):
            continue
        c = e["card"]
        r = latest(e, ts)
        cmp = compare(r["practice"], base["practice"], c) if r and base else None
        rows.append({"id": id, "component": c["component"], "state": state, "expected": c["expected"],
                     "actual": {"component": cmp[c["component"]]["change"], "overall": cmp["overall"]["change"]} if cmp else None,
                     "verdict": headline(cmp, c) if cmp else "-", "created": c["created"]})
    return sorted(rows, key=lambda r: (order[r["state"]], r["created"]))


def fold_check(card: dict, branch: dict, rebased: dict) -> tuple[str, str]:
    """('superseded' | 'ok', why), from the branch's and the rebased run's comparisons with the main line (practice)."""
    had, kept = gains(branch, card), gains(rebased, card)
    if not had:
        return "no-gain", "no practice gain on its branch"
    lost = [w for w in had if w not in kept]
    if not kept:
        return "superseded", f"the {' and '.join(had)} gain is gone after rebase ({', '.join(f'{w} {rebased[w]['change']:+d}' for w in had)})"
    return "ok", "keeps its " + " and ".join(kept) + " gain" + (f" (lost {', '.join(lost)})" if lost else "")


# Flags


def latest_verdicts(reviews: list[dict]) -> dict[str, dict]:
    """item id -> the most recent verdict record across reviewers, with `reviewer` filled in."""
    out = {}
    for rv in reviews:
        for item, rec in rv.get("items", {}).items():
            if item not in out or rec["at"] > out[item]["at"]:
                out[item] = {**rec, "reviewer": rv.get("reviewer")}
    return out


def session_of(item: dict, sessions: set[str]) -> str | None:
    """A review item's session: the id the queue item starts with, else its audio file's stem."""
    best = max((s for s in sessions if item["id"] == s or item["id"].startswith(s + "-")), key=len, default=None)
    if best:
        return best
    stem = item["audio"]["path"].rsplit("/", 1)[-1].split(".", 1)[0]
    return stem if stem in sessions else None


def pending_flags(reviews: list[dict], queues: dict[str, dict], sessions: set[str], applied: list[dict]) -> tuple[list[dict], list[str]]:
    """Bad-data verdicts not yet in the current exclusion list, as exclusion entries (the review window), plus warnings."""
    done = {e["item"] for e in applied if e.get("item")}
    out, warn = [], []
    for item_id, rec in sorted(latest_verdicts(reviews).items()):
        if rec["verdict"] not in BAD or item_id in done:
            continue
        q = queues.get(rec.get("queue", ""))
        item = next((i for i in (q or {}).get("items", []) if i["id"] == item_id), None)
        if not item:
            warn.append(f"{item_id}: queue {rec.get('queue')} not found or lacks the item")
            continue
        s = session_of(item, sessions)
        if not s:
            warn.append(f"{item_id}: no session of this test set")
            continue
        out.append({"session": s, "startS": item["audio"]["startS"], "endS": item["audio"]["endS"], "item": item_id,
                    "verdict": rec["verdict"], "note": rec.get("note", ""), "reviewer": rec["reviewer"], "at": rec["at"]})
    return out, warn


def flags_doc(base: str, version: int, excluded: list[dict], at: str | None = None) -> dict:
    """The next test set version's exclusion list, read by the scorer's `--exclude`."""
    return {"version": 1, "kind": "lab-flags", "testSet": f"{base}.v{version}", "base": base, "created": at or now(),
            "excluded": sorted(excluded, key=lambda e: (e["session"], e.get("startS") or 0, e.get("item") or ""))}


# Committed summary (docs/benchmarks/lab-ledger.json)


def brief(s: dict | None) -> dict | None:
    return None if s is None else {"wrongCalls": s["wrongCalls"], "missed": s["missed"], "falseWarnings": s["falseWarnings"],
                                   **{c: s[c]["wrongCalls"] for c in COMPONENTS}}


def summary(led: dict) -> dict:
    rows = []
    for e in led["mainLine"]:
        rows.append({"kind": "main", "at": e["at"], "sha": e["sha"], "testSet": e["testSet"], "experiment": e.get("experiment"),
                     "note": e.get("note", ""), "practice": brief(e.get("practice")), "exam": brief(e.get("exam"))})
    for id, e in led["experiments"].items():
        if not e["decision"]:
            continue
        c, d = e["card"], e["decision"]
        ts = d.get("testSet") or led["testSet"]
        b, r = latest(e, ts), latest(e, ts, "rebased")
        rows.append({"kind": "experiment", "at": d["at"], "id": id, "component": c["component"], "hypothesis": c["hypothesis"],
                     "expected": c["expected"], "state": d["state"], "testSet": ts, "note": d.get("note", ""),
                     "branch": brief(b and b.get("practice")), "rebased": {"practice": brief(r and r.get("practice")), "exam": brief(r and r.get("exam"))} if r else None})
    return {"version": 1, "kind": "lab-ledger-summary", "main": led["main"], "testSet": led["testSet"],
            "note": "Mirror of the lab ledger (tilawa-lab volume /lab/ledger.json), written on fold-in. Counts of wrong calls; a change of 2 or fewer is noise.",
            "rows": sorted(rows, key=lambda r: r["at"])}


# Plain-language output


def _wc(x: list[int]) -> str:
    return f"{x[0]} ({x[1]} missed + {x[2]} false)"


def format_compare(cmp: dict, card: dict | None = None) -> list[str]:
    lines = [f"{'wrong calls':12} {'main line':>22} {'this':>22} {'change':>7}  verdict"]
    for what in ("overall", *COMPONENTS):
        r = cmp[what]
        mark = " <" if card and what == card["component"] else ""
        exp = f"  (expected {r['expected']:+d})" if "expected" in r else ""
        lines.append(f"{('lost by ' if what != 'overall' else '') + what + mark:12} {_wc(r['main']):>22} {_wc(r['this']):>22} {r['change']:>+7d}  {r['verdict']}{exp}")
    return lines


def format_details(new: dict, base: dict) -> str:
    arrow = lambda a, b: f"{a} -> {b}" if a != b else f"{a} (same)"
    al, bl = base["checker"]["alone"], new["checker"]["alone"]
    return (f"details: listener phoneme errors {arrow(base['listener']['phonemeErrors'], new['listener']['phonemeErrors'])}; "
            f"tracker false jumps {arrow(base['tracker']['falseJumps'], new['tracker']['falseJumps'])}; "
            f"checker with perfect inputs {arrow(_wc([al['wrongCalls'], al['missed'], al['falseWarnings']]), _wc([bl['wrongCalls'], bl['missed'], bl['falseWarnings']]))}")


def format_status(led: dict, rows: list[dict]) -> list[str]:
    ts, m = led["testSet"], main_score(led)
    head = f"test set {ts}; main line {m['sha']}: {m['practice']['wrongCalls']} wrong calls on practice ({m['practice']['missed']} missed + {m['practice']['falseWarnings']} false warnings)" if m else f"test set {ts}; main line not scored yet (lab baseline)"
    if not rows:
        return [head, "no open experiments"]
    pair = lambda d, c: f"{c} {d['component']:+d}, overall {d['overall']:+d}"
    table = [("id", "expected", "actual (practice)", "verdict", "state")] + [
        (r["id"], pair(r["expected"], r["component"]), pair(r["actual"], r["component"]) if r["actual"] else "-", r["verdict"], r["state"]) for r in rows]
    w = [max(len(row[i]) for row in table) for i in range(5)]
    return [head, ""] + ["  ".join(x.ljust(w[i]) for i, x in enumerate(row)).rstrip() for row in table]
