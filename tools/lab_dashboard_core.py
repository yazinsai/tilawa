"""Pure logic for the lab dashboard (.scratch/lab/issues/11-dashboard.md): shapes the ledger and a merged score.json
into what the page shows, counts pending flags, orders the drill-down and turns scorer cases into review items.

Served by modal/lab_review.py; tested by tools/test_lab_dashboard.py. Standard library plus lab_ledger and lab_review_core.

Numbers are counts of wrong calls (missed mistakes + false warnings). The headline is the exam set on phone audio; the
drill-down only ever shows practice cases.
"""

import re

import lab_ledger as L
import lab_review_core as R

COMPONENTS = L.COMPONENTS
DASH_QUEUE = "dashboard.json"
CHECK_QUEUE = "v1-check.json"
CHECK_QUEUES = ("triage.json", CHECK_QUEUE)  # a test set's listening check: the two-model triage queue once it exists (ticket 17)
CLI = "python3 tools/lab.py"

JOBS = {
    "listener": ("Hears the sounds", "Turns the audio into speech sounds. Its own score: how many sounds it gets wrong."),
    "tracker": ("Follows where you are", "Works out where in the Quran you are. Its own score: time on the right place when it hears perfectly."),
    "checker": ("Decides if it's a mistake", "Says whether you made a mistake. Its own score: wrong calls when it hears perfectly and knows the place."),
}
WHY = {
    "listener": "with perfect hearing and the true place the engine gets it right, but with the app's listener it doesn't",
    "tracker": "with the app's listener and the true place the engine gets it right; only the real tracker gets it wrong",
    "checker": "the engine gets it wrong even with perfect hearing and the true place",
}
CLASSES = {"wrong_word": "a wrong word", "omitted_word": "a left-out word", "added_word": "an extra word", "skipped_passage": "a skipped passage"}
STATE_TEXT = {L.WAITING: "waiting for your decision", "open": "running or not scored yet", "superseded": "gain gone after rebase",
              "folded": "folded in", "discarded": "discarded"}


def base_of(test_set: str) -> str:
    """The base name of a test set version (`lab-v1.1.v3` -> `lab-v1.1`): where its reviews and queues live."""
    return re.sub(r"\.v\d+$", "", test_set or "")


def minutes(s: float) -> str:
    return f"{round(s)} s" if s < 90 else f"{s / 60:.1f} min" if s < 600 else f"{round(s / 60)} min"


def n_of(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


def where(word: str) -> str:
    s, a, w = (word or "::").split(":")[:3]
    return f"{s}:{a}, word {w}" if w else word


# Header, cards, bar


def headline_totals(score: dict) -> tuple[str, dict]:
    """('exam', Totals), or ('all', Totals) when the run has no exam sessions (an experiment's practice-only run)."""
    sp = score["splits"]
    for name in ("exam", "all", "practice"):
        if sp.get(name, {}).get("sessions"):
            return name, sp[name]
    return "exam", sp["exam"]


def header(score: dict | None, *, led: dict | None = None, pending: int = 0, check: dict | None = None, run: str | None = None) -> dict:
    ts = (led or {}).get("testSet") or (score or {}).get("testSet")
    tsd = ((led or {}).get("testSets") or {}).get(ts) or {}
    out = {"testSet": ts, "flagsApplied": tsd.get("flags", 0), "pending": pending, "check": check, "inLedger": bool(led and led.get("testSet")),
           "applyFlags": f"{CLI} apply-flags", "baseline": f"{CLI} baseline", "score": None}
    if score:
        split, t = headline_totals(score)
        o = t["overall"]
        out["score"] = {"split": split, "audio": score.get("headline", {}).get("variant", "phone"), "per5Min": o["per5Min"],
                        "wrongCalls": o["wrongCalls"], "missed": o["missed"], "falseWarnings": o["falseWarnings"], "mistakes": o["mistakes"],
                        "caught": o["caught"], "misclassified": o.get("misclassified", 0), "minutes": round(t["recitedMinutes"]),
                        "sessions": t["sessions"], "sha": score.get("sha"), "run": run or score.get("run"), "recordedAt": score.get("recordedAt"),
                        "listener": score.get("listener")}
    return out


def cards(t: dict) -> list[dict]:
    """The three component cards from one split's Totals: wrong calls charged to it (they add up) and its own score."""
    a, c = t["attribution"], t["components"]
    lis, trk, chk = c["listener"], c["tracker"], c["checker"]
    perfect = trk.get("oracleEvidence") or trk
    own = {
        "listener": {"text": f"Mishears {round(lis['per'] * 1000)} of every 1,000 sounds",
                     "detail": f"{lis['edits']:,} of {lis['phonemes']:,} sounds wrong"},
        "tracker": {"text": f"Lost its place for {minutes(perfect['scoredS'] - perfect['onPlaceS'])} of {minutes(perfect['scoredS'])}, "
                            f"{n_of(perfect['falseJumps'], 'wrong jump')}",
                    "detail": f"With the app's listener: lost for {minutes(trk['scoredS'] - trk['onPlaceS'])}, {n_of(trk['falseJumps'], 'wrong jump')}"},
        "checker": {"text": f"On its own, {n_of(chk['wrongCalls'], 'wrong call')} when it hears perfectly and knows the place",
                    "detail": f"{chk['missed']} missed + {chk['falseWarnings']} false warnings, {chk['per5Min']:.1f} per 5 min"},
    }
    return [{"key": k, "name": k.capitalize(), "job": JOBS[k][0], "explain": JOBS[k][1], "wrongCalls": a[k],
             "missed": a["missed"][k], "falseWarnings": a["falseWarnings"][k], "own": own[k]} for k in COMPONENTS]


def bar(t: dict) -> dict:
    a = t["attribution"]
    total = t["overall"]["wrongCalls"]
    parts = [{"key": k, "n": a[k], "share": a[k] / total if total else 0} for k in COMPONENTS]
    return {"total": total, "parts": parts, "adds": sum(p["n"] for p in parts) == total}


def teacher(t: dict) -> dict | None:
    te = t.get("teacher")
    if not te:
        return None
    return {"wrongCalls": te["wrongCalls"], "missed": te["missed"], "falseWarnings": te["falseWarnings"], "per5Min": te["per5Min"],
            "mishears": round(te["per"] * 1000) if te.get("per") is not None else None}


def levers(cs: list[dict]) -> list[str]:
    """The biggest lever: the component charged with the most wrong calls, plus any within noise of it."""
    top = max((c["wrongCalls"] for c in cs), default=0)
    return [c["key"] for c in cs if top and top - c["wrongCalls"] <= L.NOISE]


def overview(score: dict) -> dict:
    split, t = headline_totals(score)
    cs = cards(t)
    return {"split": split, "cards": cs, "bar": bar(t), "teacher": teacher(t), "levers": levers(cs)}


# Experiments and history


def _counts(s: dict | None) -> dict | None:
    return None if not s else {"wrongCalls": s["wrongCalls"], "missed": s["missed"], "falseWarnings": s["falseWarnings"]}


def _change(new: dict, base: dict, card: dict) -> dict:
    cmp = L.compare(new, base, card)
    k = card["component"]
    return {"overall": cmp["overall"]["change"], "component": cmp[k]["change"], "verdict": L.headline(cmp, card),
            "overallVerdict": cmp["overall"]["verdict"], "componentVerdict": cmp[k]["verdict"], "main": _counts(base), "this": _counts(new)}


def experiment_rows(led: dict) -> list[dict]:
    """Every experiment: expected, practice result on its branch, practice and exam after rebase (fold-in), verdict,
    state and the next command. Waiting for your decision first, then open, superseded, folded, discarded."""
    ts, cur = led["testSet"], L.main_score(led)
    order = {L.WAITING: 0, "open": 1, "superseded": 2, "folded": 3, "discarded": 4}
    rows = []
    for id, e in led["experiments"].items():
        c, state = e["card"], L.shown_state(e, ts)
        on = (e.get("decision") or {}).get("testSet") or ts
        r = L.latest(e, on)
        base = L.base_for(led, r) if r else None
        rb = L.latest(e, on, "rebased")
        rbase = L.base_for(led, rb) if rb else None
        if state == L.WAITING:
            nxt = [f"{CLI} fold {id}", f"{CLI} discard {id}"]
        elif state == "open":
            nxt = [f"{CLI} run {e['worktree']}"]
        elif state == "superseded":
            nxt = [f"{CLI} discard {id}"]
        else:
            nxt = []
        rows.append({
            "id": id, "component": c["component"], "hypothesis": c["hypothesis"], "expected": c["expected"], "listener": c.get("listener"),
            "state": state, "stateText": STATE_TEXT.get(state, state), "created": c["created"], "testSet": on,
            "moved": bool(state == L.WAITING and base and cur and base is not cur),
            "practice": _change(r["practice"], base["practice"], c) if r and base else None,
            "rebased": _change(rb["practice"], rbase["practice"], c) if rb and rbase else None,
            "exam": _change(rb["exam"], rbase["exam"], c) if rb and rbase and rb.get("exam") and rbase.get("exam") else None,
            "note": (e.get("decision") or {}).get("note", ""), "next": nxt,
        })
    return sorted(rows, key=lambda r: (order.get(r["state"], 5), r["created"]))


def history(led: dict) -> list[dict]:
    """Main-line scores, newest first, each with its change in exam wrong calls from the one before (same test set only)."""
    out, prev = [], None
    for e in led["mainLine"]:
        ex = e.get("exam")
        same = prev is not None and prev["testSet"] == e["testSet"]
        change = ex["wrongCalls"] - prev["exam"]["wrongCalls"] if same and ex and prev.get("exam") else None
        kind = "fold" if e.get("experiment") else "first score" if prev is None else "new test set" if not same else "re-score"
        out.append({"at": e["at"], "sha": e.get("head") or e["sha"], "testSet": e["testSet"], "kind": kind, "experiment": e.get("experiment"),
                    "note": e.get("note", ""), "exam": _counts(ex), "practice": _counts(e.get("practice")), "change": change,
                    "verdict": L.verdict(change) if change is not None else None})
        prev = e
    return out[::-1]


def main_run(led: dict | None) -> str | None:
    """The run whose merged score.json the dashboard shows: the latest main-line score on the current test set."""
    if not led or not led.get("mainLine"):
        return None
    e = (L.main_score(led) if led.get("testSet") else None) or led["mainLine"][-1]
    return e.get("run")


# Flags and the listening check


def pending_flags(reviews: list[dict], queues: dict[str, dict], sessions: set[str], applied: list[dict]) -> tuple[list[dict], list[str]]:
    """Exactly what `lab apply-flags` would apply next (lab_ledger.pending_flags)."""
    return L.pending_flags(reviews, queues, sessions, applied)


def check_queue(base: str, exists) -> str:
    """The test set's listening-check queue path (`<base>/review/<name>`): the first of CHECK_QUEUES that `exists`."""
    return f"{base}/review/" + next((n for n in CHECK_QUEUES if exists(n)), CHECK_QUEUE)


def check_status(queue: dict | None, reviews: list[dict], path: str | None = None) -> dict | None:
    """The test set's listening check (spec decision 16): latest verdict per item across reviewers, against the bar."""
    if not queue:
        return None
    t = R.tally(queue, L.latest_verdicts(reviews))
    if t["state"] == "pass":
        text = f"listening check passed {t['reviewed']}/{t['total']}" + (f", {t['bad']} bad" if t["bad"] else "")
    elif t["state"] == "fail":
        text = f"listening check failed: {t['bad']} bad of {t['total']} (at most {t['maxBad']})"
    else:
        text = f"listening check: {t['reviewed']} of {t['total']} heard, {t['bad']} bad"
    return {**t, "text": text, **({"queue": path} if path else {})}


# Drill-down


def case_time(c: dict) -> float:
    return c["startS"] if c["kind"] == "mistake" else c["timeS"]


def failing(score: dict, component: str | None = None, variant: str = "phone") -> list[dict]:
    """Practice cases that are wrong calls (a miss or a false warning), charged to `component` when given. Never exam."""
    return [c for c in score.get("cases", []) if c.get("split") == "practice" and c.get("component")
            and (component is None or c["component"] == component) and c.get("variant") == variant]


def failing_keys(score: dict) -> set[str]:
    return {c["key"] for c in score.get("cases", []) if c.get("split") == "practice" and c.get("component")}


def fails_everywhere(main: set[str], experiments: list[set[str]]) -> set[str]:
    """Cases failing on the main line and in every experiment run (none when there are no experiment runs)."""
    if not experiments:
        return set()
    out = set(main)
    for s in experiments:
        out &= s
    return out


def _plain(reason: str) -> str:
    """A data-check reason without its raw score, in the page's words."""
    return re.sub(r"\s*\((?:local )?PER [\d.]+\)", "", reason).replace("teacher", "the big reference listener")


def suspects(c: dict, everywhere: set[str] = frozenset()) -> list[dict]:
    """Why a failing case may be bad data rather than an engine fault (spec decision 14). Only a failed data check
    counts: the perfect-hearing arm is built from the label, so failing there too says nothing against the label."""
    if not c.get("dataCheck"):
        return []
    reasons = c["dataCheck"].get("reasons") or ["no reason given"]
    return [{"key": "dataCheck", "text": "Failed an automatic data check: " + "; ".join(map(_plain, reasons)), "raw": "; ".join(reasons)}]


def insights(c: dict, everywhere: set[str] = frozenset()) -> list[dict]:
    """What a failing case says about the engine, strongest first."""
    out = []
    if c.get("allFail"):
        out.append({"key": "checkerAlways", "text": "The engine gets this wrong even with perfect hearing and the true place, so the checker's rules are the cause, not the listeners or the data"})
    if c["key"] in everywhere:
        out.append({"key": "everywhere", "text": "No experiment fixes this yet"})
    return out


def case_title(c: dict) -> str:
    cls = CLASSES.get(c.get("errorClass"), c.get("errorClass") or "a mistake")
    if c["kind"] == "mistake":
        return f"The app didn't warn about {cls}"
    return f"False warning: said {cls} on a {c['nonError']}" if c.get("nonError") else f"False warning: said {cls}, but nothing was wrong"


def order_cases(cases: list[dict], everywhere: set[str] = frozenset(), verdicts: dict | None = None) -> list[dict]:
    """The drill-down list: likely bad data (a failed data check) first, then cases the checker misses even with perfect
    hearing, then cases no experiment fixes, then the rest; by session and time within each. Each row carries its reasons and the latest verdict, if any."""
    verdicts = verdicts or {}
    rank = {"dataCheck": 0, "checkerAlways": 1, "everywhere": 2}
    rows = []
    for c in cases:
        sus, ins = suspects(c, everywhere), insights(c, everywhere)
        v = verdicts.get(c["key"])
        tags = sus + ins
        strongest = min(tags, key=lambda s: rank[s["key"]])["key"] if tags else "other"
        rows.append({"key": c["key"], "title": case_title(c), "where": where(c.get("word")), "session": c["session"], "timeS": case_time(c),
                     "variant": c["variant"], "kind": c["kind"], "suspects": sus, "insights": ins, "group": strongest,
                     "verdict": v and v["verdict"], "verdictText": v and R.VERDICTS.get(v["verdict"]),
                     "_sort": (rank.get(strongest, 9), c["session"], case_time(c))})
    rows.sort(key=lambda r: r.pop("_sort"))
    return rows


def said_text(sdoc: dict, a: float, b: float) -> str:
    return " ".join(w["text"] for w in sdoc.get("said", []) if w["endS"] > a and w["startS"] < b and w.get("text"))


def _loc(at: dict | None) -> str:
    return f"{at['surah']}:{at['ayah']}:{at['word']}" if at else ""


def event_of(c: dict, sdoc: dict | None) -> dict | None:
    """The labelled mistake behind a mistake case: same class and word, nearest in time."""
    if not sdoc or c["kind"] != "mistake":
        return None
    evs = [e for e in sdoc.get("events", []) if e.get("errorClass") == c["errorClass"] and _loc(e.get("location")) == c["word"]]
    return min(evs, key=lambda e: abs(e["startS"] - c["startS"]), default=None)


def session_doc_path(c: dict) -> str | None:
    """The session label file next to the case's audio (`<set>/sessions/<id>.json`), relative to /lab/data."""
    p = (c.get("review") or {}).get("path")
    return f"{p.rsplit('/', 1)[0]}/{c['session']}.json" if p and "/" in p else None


def case_item(c: dict, sdoc: dict | None = None) -> dict:
    """A scorer case as a review item (lab-review.md queue item; id = the case key): the case's audio window, the Arabic
    the labels say is recited there, the label, and for a planted edit the original recording before and after it."""
    rv = c["review"]
    a, b = rv["startS"], rv["endS"]
    ev = event_of(c, sdoc)
    if c["kind"] == "mistake":
        label = {"type": c["errorClass"], "expected": (ev or {}).get("expected"), "said": (ev or {}).get("observed")}
    elif c.get("nonError") in R.LABELS:
        label = {"type": c["nonError"]}
    else:
        label = {"type": "clean"}
    splice = next((x for x in (sdoc or {}).get("splices", []) if ev and x.get("id") == ev.get("splice")), None)
    meta = (sdoc or {}).get("meta") or {}
    item = {"id": c["key"], "kind": "splice" if splice else "case", "title": case_title(c),
            "audio": {"path": rv["path"], "startS": a, "endS": b}, "text": said_text(sdoc, a, b) if sdoc else "", "label": label,
            "context": " · ".join(x for x in (where(c.get("word")), f"session {c['session']} at {case_time(c):.1f} s", meta.get("reciter"),
                                              f"{c['variant']} copy") if x)}
    seam = (splice or {}).get("seams", [{}])[0].get("out") if splice else None
    if seam and seam.get("audio") is not None and seam.get("srcS") is not None:
        item["before"] = {"path": seam["audio"], "startS": round(max(0.0, seam["srcS"] - 3), 3), "endS": round(seam["srcS"], 3)}
        item["after"] = {"path": seam["audio"], "startS": round(seam["srcS"], 3), "endS": round(seam["srcS"] + 3.2, 3)}
    return item


def _arm(v) -> str:
    return {"hit": "caught it", "misclassified": "caught it (as another kind of mistake)", "miss": "missed it", True: "warned here too",
            False: "stayed quiet", None: "not run"}.get(v, str(v))


def _what(c: dict) -> str:
    cls = CLASSES.get(c.get("errorClass"), c.get("errorClass"))
    if c["kind"] == "mistake":
        return f"The labels say {cls} was planted here ({c['startS']:.1f}-{c['endS']:.1f} s). The engine did not warn."
    return (f"The engine warned \"{cls}\" at {c['timeS']:.1f} s" + (f" (sure at {c['confirmedS']:.1f} s)" if c.get("confirmedS") else "")
            + (f", on a {c['nonError']}, which is not a mistake." if c.get("nonError") else ", but the labels say nothing is wrong here."))


def engine_facts(c: dict) -> dict:
    """engine_story as fields for the case view: what happened, each arm (did it get this case right?), the tracker's
    time on place around the case (percent), and the component charged with why."""
    mistake = c["kind"] == "mistake"
    arms = []
    for who, k in (("Perfect hearing, true place", "oracle"), ("App's listener, true place", "truePosition"), ("Big reference listener", "teacher")):
        v = c.get(k)
        arms.append({"who": who, "did": _arm(v), "right": None if v is None else v in ("hit", "misclassified") if mistake else v is False})
    on = c.get("trackerOnPlace")
    return {"what": _what(c), "arms": arms, "onPlace": None if on is None else round(on * 100), "charged": c.get("component"),
            "why": WHY.get(c.get("component"), "")}


def engine_story(c: dict) -> list[str]:
    """What the engine did, in plain words, and why the wrong call is charged where it is."""
    out = [_what(c),
           f"Perfect hearing and the true place: {_arm(c.get('oracle'))}. The app's listener and the true place: {_arm(c.get('truePosition'))}. "
           f"The big reference listener: {_arm(c.get('teacher'))}."]
    if c.get("trackerOnPlace") is not None:
        out.append(f"The tracker was on the right place {round(c['trackerOnPlace'] * 100)}% of the time around here.")
    if c.get("component"):
        out.append(f"Charged to the {c['component']}: {WHY[c['component']]}.")
    return out


def upsert_queue_item(doc: dict | None, test_set: str, item: dict) -> dict:
    """The dashboard's review queue (`<set>/review/dashboard.json`): every case reviewed from the drill-down, so
    `lab apply-flags` finds each verdict's window. Replaces an item with the same id."""
    doc = doc or {"version": 1, "testSet": test_set, "items": []}
    doc["items"] = [i for i in doc["items"] if i["id"] != item["id"]] + [item]
    return doc


# Release tab


def release(md: str) -> dict | None:
    """The 'Release readiness' section of docs/benchmarks/scoreboard.md: title, summary, gate rows and notes."""
    m = re.search(r"^## (Release readiness.*?)$(.*?)(?=^## |\Z)", md, re.M | re.S)
    if not m:
        return None
    head, gates, notes, summary = None, [], [], []
    for line in m[2].strip().splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            if head is None:
                head = cells
            elif not all(set(x) <= set("-: ") for x in cells):
                gates.append(dict(zip(head, cells)))
        elif line:
            (notes if head else summary).append(line)
    return {"title": m[1].strip(), "summary": " ".join(summary), "columns": head or [], "gates": gates, "notes": notes}
