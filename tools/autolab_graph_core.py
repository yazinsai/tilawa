"""Pure logic for the Autolab progress page (/autolab/live): the main line and every experiment drawn as a git-style
graph, and the practice score with how close it is to the goal.

    progress(state, events, lab, autolab, now, report=None, explained=None)
        -> {"now", "testSet", "heartbeat", "views", "score", "ticker"}

`lab` is the hand-run lab's ledger (/lab/ledger.json), `autolab` the shadow line's (/lab/autolab/ledger.json), both in
tools/lab_ledger.py's shape; `report` is the autolab line's latest score.json, of which only the practice teacher count
is read (state.main.teacherWrongCalls first); `explained` is /lab/autolab/eli5.json, the plain line for cards written
before they carried one. Each of `views` ("autolab", "testSet", "all") is one graph: spine points, test-set boundaries, branches with
non-overlapping lanes, and the stretches of time worth drawing (quiet ones longer than GAP_S become breaks).

Served by modal/lab_review.py (GET /api/autolab/progress); tested by tools/test_autolab_graph.py. Practice numbers
only: every field is picked by name and the result goes through autolab_dashboard_core.scrub.
"""

from datetime import datetime, timedelta

import autolab_dashboard_core as AD

LIVE = ("proposed", "accepted", "implementing", "validating", "scoring", "folding", "needs-human")
LAB_TEXT = {"open": "Open in the hand-run lab, waiting for a decision", "folded": "Folded in", "discarded": "Discarded",
            "superseded": "Gain gone after catching up with the line"}
GAP_S = 2 * 3600
PAD_S = 10 * 60
LANE_GAP_S = 20 * 60
TICKER = 12
DAY = 86400


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def iso(d: datetime) -> str:
    return d.replace(microsecond=0).isoformat()


def component_of(id: str, card: dict) -> str | None:
    return card.get("component") or next((c for c in AD.COMPONENTS if id.startswith(c + "-")), None)


# The main line


def spine(lab: dict | None, auto: dict | None) -> list[dict]:
    """Every scored main-line entry of both ledgers, oldest first; a fold names its experiment."""
    pts = []
    for src, led in (("lab", lab), ("autolab", auto)):
        for e in (led or {}).get("mainLine") or []:
            p = e.get("practice")
            if AD.wc(p) is None or not e.get("at"):
                continue
            pts.append({"at": e["at"], "sha": e.get("sha"), "testSet": e.get("testSet"), "n": AD.wc(p), "missed": p.get("missed"),
                        "falseWarnings": p.get("falseWarnings"), "parts": {k: AD.wc(p.get(k)) for k in AD.COMPONENTS},
                        "experiment": e.get("experiment"), "kind": "fold" if e.get("experiment") else "score", "source": src, "note": e.get("note", "")})
    return sorted(pts, key=lambda p: ts(p["at"]))


def boundaries(pts: list[dict]) -> list[dict]:
    return [{"at": b["at"], "from": a["testSet"], "to": b["testSet"], "before": a["n"], "after": b["n"]}
            for a, b in zip(pts, pts[1:]) if a["testSet"] != b["testSet"]]


# Experiments


def lab_branches(lab: dict | None, now: str, explained: dict | None = None) -> list[dict]:
    out = []
    for id, e in ((lab or {}).get("experiments") or {}).items():
        c, d, state = e.get("card") or {}, e.get("decision") or {}, e.get("state")
        tset = d.get("testSet") or c.get("testSet")
        run = next((r for r in reversed(e.get("results") or []) if r.get("kind") == "branch" and r.get("testSet") == tset), None)
        out.append({"id": id, "source": "lab", "component": component_of(id, c), "tier": None, "tierText": None, "state": state,
                    "stateText": LAB_TEXT.get(state, state), "outcome": "folded" if state == "folded" else "open" if state == "open" else "ended",
                    "start": c.get("created"), "end": now if state == "open" else d.get("at"), "testSet": tset, "hypothesis": c.get("hypothesis", ""),
                    "expected": (c.get("expected") or {}).get("overall"), "result": AD.wc((run or {}).get("practice")), "change": None,
                    "reason": d.get("note", ""), "stages": [], "ageSeconds": None, "eli5": AD.eli5(id, c, explained),
                    "expectText": AD.expect_text(c.get("expected"), component_of(id, c)), "baseN": None, "foldEventN": None})
    return out


def autolab_experiments(st: dict | None, events: list[dict]) -> dict:
    """The supervisor's experiments, plus any the events still show that state.json no longer holds; `removed` ones are
    dropped (the supervisor withdrew them as never real)."""
    exps = dict((st or {}).get("experiments") or {})
    seen, gone = {}, set()
    for ev in events:
        i = ev.get("id")
        if not i or i in exps:
            continue
        if ev.get("to") == "removed":
            gone.add(i)
            continue
        e = seen.setdefault(i, {"id": i, "created": ev.get("at"), "card": {}, "history": []})
        if ev.get("from") is None and not e["card"]:
            e["card"] = {"hypothesis": ev.get("note", "")}
        e["history"].append({"state": ev.get("to"), "at": ev.get("at"), "note": ev.get("note", "")})
    for i, e in seen.items():
        if i not in gone:
            h = e["history"][-1]
            exps[i] = {**e, "state": h["state"], "since": h["at"], "note": h["note"]}
    return exps


def autolab_branches(st: dict | None, events: list[dict], now: str, explained: dict | None = None) -> list[dict]:
    folds = {ev.get("id"): ev.get("wrongCalls") for ev in events if ev.get("to") == "folded" and isinstance(ev.get("wrongCalls"), int)}
    out = []
    for id, e in autolab_experiments(st, events).items():
        c, state, hist = e.get("card") or {}, e.get("state"), e.get("history") or []
        res = e.get("results") or {}
        fold = res.get("fold") if isinstance(res.get("fold"), dict) else {}
        gate = AD.gate_view(res.get("gate") or fold.get("gate"))
        pc = AD.practice_change(res, c)
        live = state in LIVE
        reason = AD.why_ended(e, gate)
        base = e.get("base") if isinstance(e.get("base"), dict) else {}
        out.append({"id": id, "source": "autolab", "component": component_of(id, c), "tier": e.get("tier") or c.get("tier"),
                    "tierText": AD.TIERS.get(e.get("tier") or c.get("tier")), "state": state, "stateText": AD.STATE_TEXT.get(state, state),
                    "outcome": "folded" if state == "folded" else "live" if live else "ended",
                    "start": e.get("created") or (hist[0]["at"] if hist else e.get("since")), "end": now if live else e.get("since"),
                    "testSet": c.get("testSet") or (st or {}).get("testSet"), "hypothesis": c.get("hypothesis", ""),
                    "expected": (c.get("expected") or {}).get("overall"), "result": None, "change": pc["overall"] if pc else None,
                    "reason": "" if live else reason, "stages": [{"state": h.get("state"), "at": h.get("at")} for h in hist],
                    "ageSeconds": AD.seconds(e.get("since"), now) if live else None, "waiting": state == "needs-human",
                    "eli5": AD.eli5(id, c, explained), "expectText": AD.expect_text(c.get("expected"), component_of(id, c)),
                    "baseN": base.get("wrongCalls") if isinstance(base.get("wrongCalls"), int) else None, "foldEventN": folds.get(id)})
    return out


# Layout


def lanes(branches: list[dict]) -> int:
    """Gives each branch the lowest lane free at its start (a lane frees LANE_GAP_S after its branch ends); returns the count."""
    ends: list[datetime] = []
    for b in sorted(branches, key=lambda b: (ts(b["start"]), b["id"])):
        s, e = ts(b["start"]), ts(b["end"] or b["start"])
        i = next((i for i, x in enumerate(ends) if x + timedelta(seconds=LANE_GAP_S) <= s), len(ends))
        ends[i:i + 1] = [max(e, s)]
        b["lane"] = i
    return len(ends)


def windows(spans: list[tuple[str, str]], now: str | None = None) -> list[dict]:
    """Stretches of activity: spans (start, end) closer than GAP_S join, each padded by PAD_S (never past `now`); the
    space between is a break."""
    out: list[list[datetime]] = []
    for a, b in sorted((ts(a), ts(b or a)) for a, b in spans if a):
        if out and (a - out[-1][1]).total_seconds() <= GAP_S:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    pad = timedelta(seconds=PAD_S)
    merged: list[list[datetime]] = []
    for a, b in out:
        a, b = a - pad, b + pad
        if merged and a <= merged[-1][1]:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if now and merged:
        merged[-1][1] = min(merged[-1][1], max(ts(now), merged[-1][0]))
    return [{"from": iso(a), "to": iso(b), "seconds": round((b - a).total_seconds())} for a, b in merged]


def graph(pts: list[dict], branches: list[dict], now: str) -> dict:
    bs = [dict(b) for b in branches if b.get("start")]
    for b in bs:
        base = next((p for p in reversed(pts) if p["testSet"] == b["testSet"] and ts(p["at"]) <= ts(b["start"])), None)
        fold = next((p for p in pts if p["experiment"] == b["id"]), None) if b["outcome"] == "folded" else None
        b["base"] = b.pop("baseN") if b.get("baseN") is not None else base["n"] if base else None
        fn = b.pop("foldEventN", None)
        b["foldAt"], b["foldN"] = (fold["at"], fold["n"]) if fold else (b["end"], fn) if b["outcome"] == "folded" else (None, None)
        b.pop("baseN", None)
        if b["change"] is None and b["result"] is not None and b["base"] is not None:
            b["change"] = b["result"] - b["base"]
        if b["change"] is None and fn is not None and b["base"] is not None:
            b["change"] = fn - b["base"]
        b["resultText"] = AD.result_text(b["state"], b["change"], {"overall": b["expected"]}, b["reason"])
    n = lanes(bs)
    # a hand-run branch left open isn't activity: only its start counts, so the nights after it still fold away
    ws = windows([(p["at"], p["at"]) for p in pts] + [(b["start"], b["start"] if b["outcome"] == "open" else b["end"]) for b in bs] + [(now, now)], now)
    auto = next((p["at"] for p in pts if p["source"] == "autolab"), None) or min((b["start"] for b in bs if b["source"] == "autolab"), key=ts, default=None)
    return {"from": ws[0]["from"], "to": ws[-1]["to"], "windows": ws, "spine": pts, "boundaries": boundaries(pts),
            "branches": sorted(bs, key=lambda b: (b["lane"], ts(b["start"]))), "lanes": n, "autolabStart": auto,
            "counts": {k: sum(b["outcome"] == k for b in bs) for k in ("folded", "ended", "live", "open")}}


# The score


def teacher(report: dict | None, test_set: str | None) -> int | None:
    """The big reference listener's practice wrong calls from a score report on the same test set; nothing else is read."""
    if not isinstance(report, dict) or report.get("kind") != "lab-score" or report.get("testSet") not in (None, test_set):
        return None
    return AD.wc(((report.get("splits") or {}).get("practice") or {}).get("teacher"))


def score(pts: list[dict], st: dict | None, test_set: str | None, now: str, report: dict | None) -> dict:
    series = [{k: p[k] for k in ("at", "n", "missed", "falseWarnings", "parts", "experiment", "source", "sha")} for p in pts if p["testSet"] == test_set]
    m = AD.main_view((st or {}).get("main"))
    cur = ({**m, "at": now} if m and m["wrongCalls"] is not None
           else {"wrongCalls": series[-1]["n"], "missed": series[-1]["missed"], "falseWarnings": series[-1]["falseWarnings"],
                 "parts": series[-1]["parts"], "sha": series[-1]["sha"], "at": series[-1]["at"]} if series else None)
    g = (st or {}).get("goal")
    goal = g if isinstance(g, int) and not isinstance(g, bool) else AD.wc(g) if isinstance(g, dict) else None
    goal = {"n": goal if goal is not None else 0, "source": "state" if goal is not None else "zero"}
    tw = ((st or {}).get("main") or {}).get("teacherWrongCalls")
    out = {"testSet": test_set, "series": series, "current": cur, "goal": goal, "teacher": tw if isinstance(tw, int) else teacher(report, test_set),
           "start": None, "change": None, "toGo": None, "rate": None, "projection": None, "lastDay": None, "autolab": None}
    if not series or not cur:
        return out
    n, first = cur["wrongCalls"], series[0]
    out.update(start={"n": first["n"], "at": first["at"]}, change=n - first["n"], toGo=max(n - goal["n"], 0))
    days = (ts(now) - ts(first["at"])).total_seconds() / DAY
    if days >= 0.25:
        per = (first["n"] - n) / days
        out["rate"] = {"perDay": round(per, 1), "since": first["at"], "days": round(days, 2)}
        if per > 0 and out["toGo"]:
            d = out["toGo"] / per
            out["projection"] = {"days": round(d, 1), "at": iso(ts(now) + timedelta(days=d))}
    day_ago = ts(now) - timedelta(days=1)
    then = next((p for p in reversed(series) if ts(p["at"]) <= day_ago), None)
    if then:
        out["lastDay"] = n - then["n"]
    a = [p for p in series if p["source"] == "autolab"]
    if a:
        out["autolab"] = {"start": a[0]["n"], "at": a[0]["at"], "change": n - a[0]["n"], "folds": sum(p["experiment"] is not None for p in a[1:])}
    return out


# Live events


def ticker(events: list[dict], branches: list[dict]) -> list[dict]:
    comp = {b["id"]: b["component"] for b in branches}
    text = {**AD.STATE_TEXT, "removed": "Withdrawn by the supervisor"}
    return [{"key": f"{ev.get('at')}|{ev.get('id')}|{ev.get('to')}", "at": ev.get("at"), "id": ev.get("id"),
             "component": comp.get(ev.get("id")) or component_of(ev.get("id") or "", {}), "from": ev.get("from"), "to": ev.get("to"),
             "toText": text.get(ev.get("to"), ev.get("to")), "note": ev.get("note", "")}
            for ev in reversed(events[-TICKER:])]


# The page


def progress(state: dict | None, events: list[dict], lab: dict | None, auto: dict | None, now: str, report: dict | None = None,
             explained: dict | None = None) -> dict:
    pts = spine(lab, auto)
    cur = (state or {}).get("testSet") or (auto or {}).get("testSet") or (pts[-1]["testSet"] if pts else None)
    lb, ab = lab_branches(lab, now, explained), autolab_branches(state, events, now, explained)
    views = {"autolab": graph([p for p in pts if p["source"] == "autolab"], ab, now),
             "testSet": graph([p for p in pts if p["testSet"] == cur], [b for b in lb + ab if b["testSet"] == cur], now),
             "all": graph(pts, lb + ab, now)}
    return AD.scrub({"now": now, "testSet": cur, "started": bool(state), "heartbeat": AD.heartbeat(state, now) if state else None,
                     "views": views, "score": score(pts, state, cur, now, report), "ticker": ticker(events, lb + ab)})
