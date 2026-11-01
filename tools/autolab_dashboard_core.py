"""Pure logic for the Autolab tab (.scratch/autolab/spec.md decision 13; ticket 07): shapes the supervisor's state
document (its schema is in tools/autolab_state.py), the tail of its events and the autolab ledger's main line into
what the page shows.

    view(state, events, now, mainline) -> {"started", "testSet", "heartbeat", "main", "needsHuman", "running", "queue",
                                           "verdicts", "trend", "portfolio", "coverage", "spend", "builders", "activity"}

Served by modal/lab_review.py (GET /api/autolab); tested by tools/test_autolab_dashboard.py. Standard library plus
autolab_state.

Practice numbers only: fields are picked by name, none of them exam, and `scrub` drops any exam key a later input
might carry.
"""

import hmac
import re
from datetime import datetime

import autolab_state as S

COMPONENTS = ("listener", "tracker", "checker")
NAMES = {"listener": "Listener", "tracker": "Tracker", "checker": "Checker"}
TIERS = {"exploit": "Quick fix", "explore": "New idea", "bet": "Big bet"}
SHARES = {"exploit": 50, "explore": 30, "bet": 20}  # spec decision 7: per cent of the slots
RUNNING = ("implementing", "validating", "scoring", "folding")
QUEUED = ("proposed", "accepted")
STATE_TEXT = {
    "proposed": "Proposed, waiting to be checked",
    "accepted": "Accepted, waiting for a free agent",
    "implementing": "An agent is writing the change",
    "validating": "Checking it builds and plays fair",
    "scoring": "Being scored on the practice set",
    "folding": "Being folded into main",
    "folded": "Folded in",
    "discarded": "Discarded",
    "superseded": "Gain gone after catching up with the line",
    "needs-human": "Waiting for you",
    "abandoned": "Given up",
    "reverted": "Taken back out of main",
    "rejected": "Your command was turned down",
    "paused": "Autolab paused",
    "resumed": "Autolab resumed",
}
COVERAGE_TEXT = {"untouched": "Not tried yet", "tried": "Tried, not fixed yet", "fixed": "Fixed"}
CHECKS = {
    "gain": "Fewer wrong calls",
    "spread": "Not one reader's luck",
    "collateral": "Breaks nothing else",
    "hygiene": "Plays fair",
    "significance": "Clear of chance",
    "surah spread": "Not one surah's luck",
    "tests": "Builds and tests pass",
}
SPEND = (("agentRuns", "Agent runs", "runs"), ("agentHours", "Agent hours", "hours"), ("modalUsd", "Modal", "usd"))
JOURNAL = ".scratch/autolab/journal.md"
STALE_TICKS = 3
VERDICTS = 50
ACTIVITY = 30


def seconds(a: str | None, b: str) -> int | None:
    return None if not a else round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())


def fewer(n: int) -> str:
    return "no change" if n == 0 else f"{-n} fewer" if n < 0 else f"{n} more"


def wc(x) -> int | None:
    """Wrong calls from a ledger brief (an int per part) or a full ledger score (a dict per part)."""
    return x if isinstance(x, int) else x.get("wrongCalls") if isinstance(x, dict) else None


# Plain words for an experiment: what it is about, what it expects and what came of it (practice counts only).

NOISE = 2
KEPT = {"folded": "kept", "discarded": "dropped", "superseded": "dropped: the gain was gone after catching up with the line",
        "abandoned": "given up", "reverted": "taken back out of main"}
REASON_CHARS = 140


def eli5(id: str, card: dict | None, extra: dict | None = None) -> dict:
    """{"text", "plain"}: the card's own eli5, else the explained-later file's line, else the hypothesis (not plain)."""
    c = card or {}
    for t in (c.get("eli5"), (extra or {}).get(id)):
        if isinstance(t, str) and t.strip():
            return {"text": t.strip(), "plain": True}
    return {"text": (c.get("hypothesis") or "").strip(), "plain": False}


def about(n: int) -> str:
    return f"{abs(n)} {'fewer' if n < 0 else 'more'} wrong call{'' if abs(n) == 1 else 's'}"


def expect_text(expected: dict | None, component: str | None) -> str | None:
    """'Expected: about 4 fewer wrong calls overall (5 fewer charged to the checker)'."""
    e = expected or {}
    o, c = e.get("overall"), e.get("component")
    if not isinstance(o, int):
        return None
    head = "Expected: no change overall" if o == 0 else f"Expected: about {about(o)} overall"
    if isinstance(c, int) and component:
        part = "no change" if c == 0 else f"{abs(c)} {'fewer' if c < 0 else 'more'}"
        head += f" ({part} charged to the {component})"
    return head


def short(reason: str) -> str:
    r = " ".join((reason or "").split())
    return r if len(r) <= REASON_CHARS else r[:REASON_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def result_text(state: str | None, change: int | None, expected: dict | None, reason: str = "") -> str | None:
    """'Result: 10 fewer (expected 8 fewer), kept' or 'Result: no real change (1 fewer), dropped: <reason>'; None while running."""
    if state not in KEPT and state != "open":
        return None
    o = (expected or {}).get("overall")
    exp = f"expected {fewer(o)}" if isinstance(o, int) else ""
    if change is None:
        got = "not scored"
    elif abs(change) <= NOISE:
        got = "no real change" + (f" ({'; '.join(x for x in (fewer(change) if change else '', exp) if x)})" if change or exp else "")
    else:
        got = fewer(change) + (f" ({exp})" if exp else "")
    if state == "open":
        return f"So far: {got} on its branch, waiting for a decision"
    why = short(reason) if state in ("discarded", "abandoned") else ""
    return f"Result: {got}, {KEPT[state]}{f': {why[0].lower() + why[1:]}' if why else ''}"


def scrub(x):
    """Drops every exam key, at any depth. The tripwire's word (pass or blocked) is the one exam fact the page shows."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if not str(k).lower().startswith("exam") or k == "examTripwire"}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


# Heartbeat


def heartbeat(st: dict, now: str) -> dict:
    tick = st.get("tickSeconds") or 120
    age = seconds(st.get("heartbeat"), now)
    return {"at": st.get("heartbeat"), "ageSeconds": age, "tickSeconds": tick, "stale": age is None or age > STALE_TICKS * tick,
            "paused": st.get("paused"), "ticks": st.get("ticks", 0)}


# Experiments


def card_of(e: dict) -> dict:
    return e.get("card") or {}


def row(e: dict, now: str, explained: dict | None = None) -> dict:
    c = card_of(e)
    limit = S.timeout(e) if e.get("state") in S.TIMEOUTS else None
    age = seconds(e.get("since"), now)
    ex = e.get("executor") or {}
    model = ex.get("model")
    return {"id": e["id"], "arm": ex.get("arm"), "component": c.get("component"), "tier": e.get("tier") or c.get("tier"), "tierText": TIERS.get(e.get("tier"), e.get("tier")),
            "eli5": eli5(e["id"], c, explained), "expectText": expect_text(c.get("expected"), c.get("component")),
            "state": e["state"], "stateText": STATE_TEXT.get(e["state"], e["state"]), "since": e.get("since"), "ageSeconds": age,
            "limitMinutes": limit, "late": bool(limit and age is not None and age > limit * 60), "hypothesis": c.get("hypothesis", ""),
            "cluster": c.get("cluster"), "expected": c.get("expected"), "executor": (model.get("id") if isinstance(model, dict) else model) or ex.get("runtime"),
            "attempts": (e.get("attempts") or {}).get("executor", 0), "note": e.get("note", "")}


def check_text(c: dict) -> str:
    """One gate check (tools/lab_gate.py) in plain words, from its numbers; its own `why` when they're missing."""
    try:
        n = c["name"]
        if n == "gain":
            ch, comp = c["change"], next(k for k in c["change"] if k != "overall")
            return (f"{fewer(ch['overall'])} wrong calls overall (needs {c['need']} fewer or better); "
                    f"{NAMES.get(comp, comp).lower()} {fewer(ch[comp])} (may not get worse beyond noise)")
        if n in ("spread", "surah spread"):
            who = "reader" if n == "spread" else "surah"
            w = c["worst"]
            return (f"Leaving out the {who} who gained most ({w['left out']}) still leaves {fewer(w['remaining'])}" if c["ok"]
                    else f"Leaving out the {who} who gained most ({w['left out']}) leaves {fewer(w['remaining'])}: the gain rests on that {who}")
        if n == "collateral":
            parts = [f"{c['won']} cases put right, {c['lost']} newly wrong"]
            if c.get("cleanNewWarnings"):
                parts.append(f"{c['cleanNewWarnings']} new false warnings on recitations with no mistakes")
            parts.append(f"studio audio {fewer(c['studio'])}")
            if c.get("realuser") is not None:
                parts.append(f"false warnings on real users' clean recitation: {fewer(c['realuser'])}")
            return "; ".join(parts)
        if n == "hygiene":
            if c["ok"]:
                return "Leaves the lab's own files alone and doesn't write test-set details into the app"
            out = [f"Changes the lab's own files ({', '.join(c['protected'])})"] if c.get("protected") else []
            if c.get("literals"):
                out.append(f"Writes test-set details into the app ({', '.join(sorted({x['text'] for x in c['literals']}))})")
            return "; ".join(out) or c.get("why", "")
        if n == "significance":
            return f"{c['sessionsMoved']} sessions changed; a cautious estimate of the change is {fewer(c['upper'])}"
        if n == "tests":
            return "The build and its tests pass" if c["ok"] else "The build or its tests fail"
    except (KeyError, TypeError, StopIteration):
        pass
    return c.get("why", "")


def gate_view(g: dict | None) -> dict | None:
    if not isinstance(g, dict) or "checks" not in g:
        return None
    checks = [{"name": c.get("name"), "title": CHECKS.get(c.get("name"), c.get("name")), "ok": bool(c.get("ok")), "blocking": c.get("blocking", True),
               "result": "passed" if c.get("ok") else "failed" if c.get("blocking", True) else "note", "text": check_text(c), "detail": c.get("why", "")}
              for c in g["checks"]]
    return {"verdict": g.get("verdict"), "tuning": bool(g.get("tuning")), "checks": checks,
            "failed": [c["title"] for c in checks if c["result"] == "failed"]}


def practice_change(res: dict, card: dict) -> dict | None:
    """The practice change against the line: after rebase when the fold ran, else on the branch."""
    for where, src in (("after rebase", res.get("fold")), ("on its branch", res.get("branchRun"))):
        cmp = src.get("compare") if isinstance(src, dict) else None
        comp = card.get("component")
        if isinstance(cmp, dict) and "overall" in cmp:
            return {"where": where, "overall": cmp["overall"].get("change"), "component": (cmp.get(comp) or {}).get("change"),
                    "expected": card.get("expected"), "why": src.get("why")}
    return None


def why_ended(e: dict, gate: dict | None) -> str:
    return f"failed the gate: {', '.join(gate['failed']).lower()}" if gate and gate["failed"] else e.get("note", "")


def verdict_row(e: dict, now: str, explained: dict | None = None) -> dict:
    res, c = e.get("results") or {}, card_of(e)
    fold = res.get("fold") if isinstance(res.get("fold"), dict) else {}
    gate, pc = gate_view(res.get("gate") or fold.get("gate")), practice_change(res, c)
    return {**{k: v for k, v in row(e, now, explained).items() if k not in ("limitMinutes", "late")}, "at": e.get("since"),
            "gate": gate, "practice": pc, "resultText": result_text(e["state"], pc and pc["overall"], c.get("expected"), why_ended(e, gate)),
            "examTripwire": fold.get("examTripwire"), "journal": f"{JOURNAL}#{e['id']}"}


# Trend, portfolio, coverage, spend


def trend(mainline: list[dict] | None, test_set: str | None) -> dict:
    """Each part's practice wrong calls over the main line's scores on one test set (counts on two versions don't compare)."""
    rows = [e for e in mainline or [] if isinstance(e.get("practice"), dict)]
    ts = test_set if any(e.get("testSet") == test_set for e in rows) else rows[-1].get("testSet") if rows else None
    rows = sorted((e for e in rows if e.get("testSet") == ts), key=lambda e: e.get("at") or "")
    series = {}
    for k in ("overall", *COMPONENTS):
        pts = [{"at": e.get("at"), "sha": e.get("sha"), "experiment": e.get("experiment"), "n": n}
               for e in rows if (n := wc(e["practice"] if k == "overall" else e["practice"].get(k))) is not None]
        if pts:
            series[k] = {"points": pts, "first": pts[0]["n"], "last": pts[-1]["n"], "change": pts[-1]["n"] - pts[0]["n"],
                         "best": min(p["n"] for p in pts)}
    return {"testSet": ts, "series": series}


def main_view(m: dict | None) -> dict | None:
    if not isinstance(m, dict):
        return None
    p, a = m.get("practice") or {}, m.get("attribution") or {}
    parts = {k: wc(p.get(k)) if wc(p.get(k)) is not None else wc(a.get(k)) for k in COMPONENTS}
    return {"sha": m.get("sha"), "run": m.get("run"), "wrongCalls": wc(p), "missed": p.get("missed"), "falseWarnings": p.get("falseWarnings"), "parts": parts}


def portfolio(st: dict) -> dict:
    """Slots held now (autolab_state.active: not finished and not waiting on a person), by part and tier."""
    held = S.active(st)
    cnt = st.get("counters") or {}
    by = {k: {t: sum(1 for e in held if card_of(e).get("component") == k and e.get("tier") == t) for t in TIERS} for k in COMPONENTS}
    return {"held": len(held), "slots": st.get("slots") or (st.get("caps") or {}).get("slots"), "shares": SHARES, "tiers": TIERS,
            "byTier": {t: sum(1 for e in held if e.get("tier") == t) for t in TIERS},
            "components": [{"key": k, "name": NAMES[k], "tiers": by[k], "held": sum(by[k].values()),
                            "sinceFold": (cnt.get("sinceFold") or {}).get(k, 0), "frozenSince": (cnt.get("frozen") or {}).get(k)} for k in COMPONENTS],
            "lastFoldAt": cnt.get("lastFoldAt"), "foldsSinceRefresh": cnt.get("foldsSinceRefresh", 0)}


def coverage(st: dict) -> dict:
    order = {"tried": 0, "untouched": 1, "fixed": 2}
    rows = []
    for key, c in (st.get("coverage") or {}).items():
        parts = key.split("/")
        comp = parts[0] if parts[0] in COMPONENTS else None
        label = " · ".join(p.replace("_", " ") for p in (parts[1:] if comp else parts)) or key
        state = c.get("state", "untouched")
        exps = list(c.get("experiments") or [])
        rows.append({"cluster": key, "component": comp, "label": label, "state": state, "stateText": COVERAGE_TEXT.get(state, state),
                     "incidents": c.get("incidents", 0), "experiments": exps, "triedTwice": state == "tried" and len(exps) >= 2})
    rows.sort(key=lambda r: (order.get(r["state"], 3), -r["incidents"], r["cluster"]))
    return {"clusters": rows, "counts": {s: sum(r["state"] == s for r in rows) for s in COVERAGE_TEXT}}


def spend(st: dict) -> dict:
    s, caps = st.get("spend") or {}, st.get("caps") or {}
    rows = []
    for k, label, unit in SPEND:
        used, cap = s.get(k, 0) or 0, caps.get(k)
        rows.append({"key": k, "label": label, "unit": unit, "used": used, "cap": cap,
                     "share": round(min(used / cap, 1), 3) if cap else None, "over": cap is not None and used >= cap})
    return {"day": s.get("day"), "rows": rows, "atCap": [r["label"] for r in rows if r["over"]]}


def builders(st: dict) -> dict | None:
    """The builder A/B the supervisor keeps in state["ab"] (tools/autolab_ab.py), or None with fewer than two arms."""
    ab = st.get("ab") if isinstance(st.get("ab"), dict) else {}
    arms = [a for a in ab.get("arms") or [] if isinstance(a, dict)]
    skipped = ab.get("skipped") or {}
    if len(arms) < 2 and not skipped:
        return None
    return {"since": ab.get("since"), "split": ab.get("split") or {}, "skipped": [{"arm": k, "why": v} for k, v in skipped.items()],
            "arms": [{k: a.get(k) for k in ("arm", "started", "running", "killedByBuilder", "bounced", "gated", "gatePassed", "gatePassRate",
                                            "nearMisses", "folds", "abandoned", "fallbacks", "scored", "medianChange", "meanBuilderHours")} for a in arms]}


def activity(events: list[dict]) -> list[dict]:
    out = []
    for ev in reversed(events[-ACTIVITY:]):
        to = ev.get("to")
        out.append({"at": ev.get("at"), "id": ev.get("id"), "from": ev.get("from"), "to": to, "toText": STATE_TEXT.get(to, to), "note": ev.get("note", "")})
    return out


# Waiting for you: what each item needs, in plain words, and the commands it takes

APPROVAL = re.compile(r"^a bet asking for \$(\d+(?:\.\d+)?) of GPU")
STUCK = (  # the supervisor's `what` prefix -> what broke, in plain words
    ("orchestrator:", "The planning model failed while handling it"),
    ("the listener decode failed:", "Re-running the listener on the practice audio failed"),
    ("scoring failed:", "Scoring it on the practice set failed"),
    ("fold failed:", "Folding it into main failed"),
    ("fold ended in state", "Folding it didn't finish"),
    ("the novelty judge didn't answer in time", "The model that checks it's a new idea didn't answer in time"),
    ("proposed failed:", "Checking the proposal crashed"),
    ("accepted failed:", "Starting it crashed"),
    ("implementing failed:", "Watching the agent write it crashed"),
    ("validating failed:", "Checking that it builds and plays fair crashed"),
    ("folding failed:", "Folding it crashed"),
)

def why_waiting(what: str) -> dict:
    """{"kind": "approval" | "stuck", "usd", "text"}: a bet over the approval budget, or anything else that stopped."""
    what = (what or "").strip()
    if m := APPROVAL.match(what):
        usd = float(m.group(1))
        usd = int(usd) if usd.is_integer() else usd
        return {"kind": "approval", "usd": usd,
                "text": f"A big bet that wants about ${usd} of GPU time, more than autolab may spend without asking you. Approve it to start, or drop it."}
    for pre, plain in STUCK:
        if what.lower().startswith(pre):
            rest = what[len(pre):].strip(" :")
            return {"kind": "stuck", "usd": None, "text": f"{plain}{f': {short(rest)}' if rest else '.'}"}
    return {"kind": "stuck", "usd": None, "text": sentence_of(short(what)) or "It stopped and needs you to look."}


def sentence_of(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


ACTIONS = {"approval": ["approve", "drop"], "stuck": ["retry", "retryNote", "drop"]}


def waiting(state: dict | None, now: str, explained: dict | None = None, inbox: list[dict] | None = None) -> list[dict]:
    """state.needsHuman plus any experiment in needs-human it doesn't list, oldest first, each with the commands it
    takes and any command of yours still in the inbox for it."""
    if not state:
        return []
    exps = state.get("experiments") or {}
    asks = {h.get("id"): h for h in state.get("needsHuman") or [] if h.get("id")}
    for e in exps.values():
        if e.get("state") == "needs-human" and e.get("id") not in asks:
            asks[e["id"]] = {"id": e["id"], "what": e.get("note", ""), "at": e.get("since")}
    out = []
    for id, h in asks.items():
        e = exps.get(id)
        c = card_of(e) if e else {}
        w = why_waiting(h.get("what", ""))
        acts = ACTIONS[w["kind"]] if e and e.get("state") == "needs-human" else ["drop"] if e and e.get("state") not in S.TERMINAL else []
        out.append({"id": id, "what": h.get("what", ""), "at": h.get("at"), "ageSeconds": seconds(h.get("at"), now), "kind": w["kind"], "usd": w["usd"],
                    "whyText": w["text"], "actions": acts, "state": e.get("state") if e else None,
                    "component": c.get("component"), "tierText": TIERS.get(e.get("tier")) if e else None, "hypothesis": c.get("hypothesis", ""),
                    "eli5": eli5(id, c, explained), "expectText": expect_text(c.get("expected"), c.get("component")),
                    "sent": [x for x in inbox or [] if x.get("id") == id]})
    return sorted(out, key=lambda x: x["at"] or "")


# Your commands from the page: validated here, written to the supervisor's inbox by modal/lab_review.py, applied by
# tools/autolab.py at the start of its next tick (it rejects anything that went stale meanwhile, as an event).

OPS = ("approve", "retry", "drop", "pause", "resume", "match")
NOTE_CHARS = 1000
WHY_CHARS = 300
VERDICTS_MATCH = ("folded", "discarded")


class CommandError(ValueError):
    pass


def text_arg(body: dict, key: str, cap: int) -> str:
    v = body.get(key, "")
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise CommandError(f"{key} must be text")
    v = " ".join(v.split())
    if len(v) > cap:
        raise CommandError(f"{key} is {len(v)} characters; keep it under {cap}")
    return v


def command(state: dict | None, body: dict, now: str, inbox: list[dict] | None = None) -> dict:
    """The inbox document for a command from the page, or CommandError saying in plain words why not."""
    if not isinstance(body, dict):
        raise CommandError("send a JSON object")
    op = body.get("op")
    if op not in OPS:
        raise CommandError(f"op is one of {', '.join(OPS)}")
    if not state:
        raise CommandError("autolab hasn't started")
    pending = inbox or []
    if op in ("pause", "resume"):
        if op == "resume" and not state.get("paused"):
            raise CommandError("autolab isn't paused")
        why = text_arg(body, "why", WHY_CHARS) if op == "pause" else ""
        if op == "pause" and not why:
            raise CommandError("say why you're pausing")
        if any(x.get("op") == "pause" for x in pending):
            raise CommandError("a pause or resume is already on its way")
        return {"op": "pause", "at": now, "why": why}
    id = body.get("id")
    e = (state.get("experiments") or {}).get(id) if isinstance(id, str) else None
    if not e:
        raise CommandError(f"no experiment {id!r}")
    st = e.get("state")
    if any(x.get("id") == id and x.get("op") in ("approve", "retry", "drop") for x in pending) and op != "match":
        raise CommandError(f"a command for {id} is already on its way")
    if op in ("approve", "retry") and st != "needs-human":
        raise CommandError(f"{id} isn't waiting for you ({STATE_TEXT.get(st, st).lower()})")
    if op == "approve":
        return {"op": "approve", "at": now, "id": id}
    if op == "retry":
        return {"op": "retry", "at": now, "id": id, "note": text_arg(body, "note", NOTE_CHARS)}
    if op == "drop":
        if st in S.TERMINAL:
            raise CommandError(f"{id} already ended ({STATE_TEXT.get(st, st).lower()})")
        return {"op": "drop", "at": now, "id": id, "why": text_arg(body, "why", WHY_CHARS) or "dropped from the dashboard"}
    if st not in VERDICTS_MATCH:
        raise CommandError(f"{id} wasn't folded or discarded, so there's no decision to agree with")
    if body.get("verdict") not in ("agree", "disagree"):
        raise CommandError("verdict is agree or disagree")
    return {"op": "match", "at": now, "id": id, "verdict": body["verdict"]}


def inbox_name(cmd: dict, rand: str) -> str:
    """Same shape as `tools/autolab.py`'s own commands, so the supervisor applies them in time order."""
    return f"{cmd['at'].replace(':', '')}-{cmd['op']}-{rand}.json"


def pending(docs: list[dict]) -> list[dict]:
    """Inbox documents not applied yet, as the page shows them (no free text)."""
    return [{"op": d.get("op"), "id": d.get("id"), "at": d.get("at"), "resume": d.get("op") == "pause" and not d.get("why")}
            for d in docs if isinstance(d, dict) and d.get("op") in OPS]


def token_status(stored, given: str | None) -> int:
    """200 when `given` matches the access code in ui-token.json, 401 when it doesn't, 503 when there is no code."""
    tok = stored.get("token") if isinstance(stored, dict) else None
    if not isinstance(tok, str) or len(tok) < 8:
        return 503
    return 200 if isinstance(given, str) and hmac.compare_digest(given.encode(), tok.encode()) else 401


def matches(state: dict | None) -> dict:
    """Your latest agree/disagree per experiment."""
    return {m.get("id"): m.get("user") for m in (state or {}).get("matches") or [] if m.get("id") and m.get("user") in ("agree", "disagree")}


# The page


def view(state: dict | None, events: list[dict], now: str, mainline: list[dict] | None = None, explained: dict | None = None,
         inbox: list[dict] | None = None) -> dict:
    """Everything the Autolab tab shows. `state` None means the supervisor hasn't written state.json yet; `explained` is
    /lab/autolab/eli5.json (experiment id -> plain sentence) for cards written before they carried one."""
    if not state:
        return scrub({"started": False, "testSet": None, "heartbeat": {"at": None, "ageSeconds": None, "tickSeconds": None, "stale": False, "paused": None, "ticks": 0},
                      "main": None, "needsHuman": [], "pending": [], "matches": {}, "running": [], "queue": [], "verdicts": [], "trend": trend(mainline, None),
                      "portfolio": None, "coverage": {"clusters": [], "counts": {}}, "spend": None, "activity": activity(events)})
    exps = state.get("experiments") or {}
    by_since = lambda xs, rev=False: sorted(xs, key=lambda e: e.get("since") or "", reverse=rev)
    return scrub({
        "started": True, "testSet": state.get("testSet"), "heartbeat": heartbeat(state, now), "main": main_view(state.get("main")),
        "needsHuman": waiting(state, now, explained, inbox), "pending": inbox or [], "matches": matches(state),
        "running": [row(e, now, explained) for e in by_since(e for e in exps.values() if e.get("state") in RUNNING)],
        "queue": [row(e, now, explained) for e in by_since(e for e in exps.values() if e.get("state") in QUEUED)],
        "verdicts": [verdict_row(e, now, explained) for e in by_since((e for e in exps.values() if e.get("state") in S.TERMINAL), rev=True)][:VERDICTS],
        "trend": trend(mainline, state.get("testSet")), "portfolio": portfolio(state), "coverage": coverage(state), "spend": spend(state),
        "builders": builders(state), "activity": activity(events),
    })
