"""Autolab state (.scratch/autolab/spec.md decision 3; ticket 04). Pure logic over one JSON document, kept on the
tilawa-lab volume at /lab/autolab/state.json; every transition also yields an event line for /lab/autolab/events.jsonl.

The document:

    {"version": 1, "kind": "autolab-state", "testSet": "lab-v1.5", "paused": null | "why", "heartbeat": iso, "ticks": n,
     "tickSeconds": 120 (the loop's interval; the dashboard calls the heartbeat stale after 3 missed ticks),
     "main": {"sha", "run", "practice": brief, "attribution": {listener, tracker, checker}} | null,
     "dataHash": str | null,
     "experiments": {id: {"id", "state", "since", "created", "tier", "card", "branch", "worktree", "executor": {...} | null,
                          "attempts": {"executor": n, "practiceScores": n}, "results": {...}, "note": str,
                          "history": [{"state", "at", "note"}]}},
     "foldLock": id | null,
     "counters": {"sinceFold": {component: n}, "lastFoldAt": iso | null, "foldsSinceRefresh": n, "frozen": {component: iso}},
     "coverage": {cluster: {"state": "untouched" | "tried" | "fixed", "experiments": [id], "incidents": n}},
     "spend": {"day": "YYYY-MM-DD", "agentRuns": n, "agentHours": x, "modalUsd": x},
     "matches": [{"id", "gate", "user", "at"}],
     "needsHuman": [{"id", "what", "at"}]}

`card` is the lab card (`lab_ledger.new_card`) plus the autolab fields: tier, mechanism, targets (incident keys),
cluster, killCriterion, and for bets gpuUsd.
"""

from datetime import datetime, timezone

STATES = ("proposed", "accepted", "implementing", "validating", "scoring", "folding",
          "folded", "discarded", "superseded", "needs-human", "abandoned")
TERMINAL = ("folded", "discarded", "superseded", "abandoned")
TIERS = ("exploit", "explore", "bet")
NEXT = {
    "proposed": ("accepted", "discarded", "needs-human"),
    "accepted": ("implementing", "discarded", "needs-human"),
    "implementing": ("validating", "abandoned", "discarded", "needs-human"),
    "validating": ("scoring", "implementing", "discarded", "abandoned", "needs-human"),
    "scoring": ("folding", "discarded", "abandoned", "needs-human"),
    "folding": ("folded", "discarded", "superseded", "implementing", "abandoned", "needs-human"),
    "needs-human": ("accepted", "implementing", "validating", "scoring", "folding", "discarded", "abandoned"),
}
# Minutes in a state before the supervisor gives up on it; implementing depends on the tier.
TIMEOUTS = {"proposed": 60, "validating": 90, "scoring": 90, "folding": 150,
            "implementing": {"exploit": 240, "explore": 720, "bet": 4320}}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def minutes(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60


def empty(test_set: str | None = None) -> dict:
    return {"version": 1, "kind": "autolab-state", "testSet": test_set, "paused": None, "heartbeat": None, "ticks": 0, "tickSeconds": 120,
            "main": None, "dataHash": None, "experiments": {}, "foldLock": None,
            "counters": {"sinceFold": {}, "lastFoldAt": None, "foldsSinceRefresh": 0, "frozen": {}},
            "coverage": {}, "spend": {"day": None, "agentRuns": 0, "agentHours": 0.0, "modalUsd": 0.0},
            "matches": [], "needsHuman": []}


def event(id: str | None, frm: str | None, to: str, at: str, note: str = "", **extra) -> dict:
    return {"at": at, "id": id, "from": frm, "to": to, "note": note, **extra}


def add(st: dict, card: dict, at: str | None = None) -> tuple[dict, dict]:
    """A new experiment in `proposed`; returns (experiment, event)."""
    at = at or now()
    id = card["id"]
    if id in st["experiments"]:
        raise ValueError(f"experiment {id} already exists")
    if card.get("tier") not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    e = {"id": id, "state": "proposed", "since": at, "created": at, "tier": card["tier"], "card": card, "branch": None,
         "worktree": None, "executor": None, "attempts": {"executor": 0, "practiceScores": 0}, "results": {}, "note": "",
         "history": [{"state": "proposed", "at": at, "note": ""}]}
    st["experiments"][id] = e
    return e, event(id, None, "proposed", at, card.get("hypothesis", ""))


def move(st: dict, id: str, to: str, at: str | None = None, note: str = "") -> dict:
    """Moves an experiment to `to` if the transition is allowed; returns the event."""
    at = at or now()
    e = st["experiments"][id]
    if to not in NEXT.get(e["state"], ()):
        raise ValueError(f"{id}: {e['state']} -> {to} is not allowed")
    frm = e["state"]
    e["state"], e["since"], e["note"] = to, at, note
    e["history"].append({"state": to, "at": at, "note": note})
    if to == "needs-human":
        st["needsHuman"].append({"id": id, "what": note, "at": at})
    elif frm == "needs-human":
        st["needsHuman"] = [h for h in st["needsHuman"] if h["id"] != id]
    if to in TERMINAL and st["foldLock"] == id:
        st["foldLock"] = None
    return event(id, frm, to, at, note)


def timeout(e: dict) -> float | None:
    t = TIMEOUTS.get(e["state"])
    return t.get(e["tier"]) if isinstance(t, dict) else t


def timed_out(e: dict, at: str) -> bool:
    t = timeout(e)
    return t is not None and minutes(e["since"], at) > t


def active(st: dict) -> list[dict]:
    """Experiments holding a slot: everything not terminal and not waiting on a person."""
    return [e for e in st["experiments"].values() if e["state"] not in TERMINAL and e["state"] != "needs-human"]


def in_state(st: dict, state: str) -> list[dict]:
    return sorted((e for e in st["experiments"].values() if e["state"] == state), key=lambda e: e["since"])


# Fold queue


def take_lock(st: dict, id: str) -> bool:
    if st["foldLock"] in (None, id):
        st["foldLock"] = id
        return True
    holder = st["experiments"].get(st["foldLock"])
    if not holder or holder["state"] != "folding":
        st["foldLock"] = id
        return True
    return False


# Spend


def roll_day(st: dict, at: str) -> None:
    day = at[:10]
    if st["spend"]["day"] != day:
        st["spend"] = {"day": day, "agentRuns": 0, "agentHours": 0.0, "modalUsd": 0.0}


def charge(st: dict, at: str, runs: int = 0, hours: float = 0.0, modal: float = 0.0) -> None:
    roll_day(st, at)
    s = st["spend"]
    s["agentRuns"] += runs
    s["agentHours"] = round(s["agentHours"] + hours, 3)
    s["modalUsd"] = round(s["modalUsd"] + modal, 2)


def over_cap(st: dict, caps: dict, at: str) -> str | None:
    """The first daily cap reached, in words, or None. caps: {"agentRuns", "agentHours", "modalUsd"} (missing = no cap)."""
    roll_day(st, at)
    for k, label in (("agentRuns", "agent runs"), ("agentHours", "agent hours"), ("modalUsd", "Modal dollars")):
        if caps.get(k) is not None and st["spend"][k] >= caps[k]:
            return f"daily cap on {label} reached ({st['spend'][k]} of {caps[k]})"
    return None
