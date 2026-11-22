"""Autolab portfolio (.scratch/autolab/spec.md decisions 7, 8, 9; ticket 08). Pure logic over the state document
(tools/autolab_state.py): how many experiment slots each component and tier gets, the plateau freeze, the tuning cap,
and incident clusters with their coverage.
"""

from collections import Counter

from autolab_state import TERMINAL, active, minutes

COMPONENTS = ("listener", "tracker", "checker")
TIER_SHARE = {"exploit": 0.5, "explore": 0.3, "bet": 0.2}
PLATEAU_RUN = 5  # consecutive experiments in a component without a fold
PLATEAU_HOURS = 24  # hours without any fold
ASK_AFTER_HOURS = 24  # hours frozen with no fold before one question to the user
FAILED_TRIES = 2  # a cluster tried this often without a fold needs a written reason


def largest_remainder(weights: dict[str, float], n: int, floor: int = 0) -> dict[str, int]:
    """`n` split in proportion to `weights`, each key at least `floor` (while n allows), rounding by largest remainder."""
    keys = list(weights)
    if not sum(weights.values()):
        weights = dict.fromkeys(keys, 1)
    out = {k: floor for k in keys} if n >= floor * len(keys) else dict.fromkeys(keys, 0)
    left = n - sum(out.values())
    share = {k: left * weights[k] / sum(weights.values()) for k in keys}
    for k in keys:
        out[k] += int(share[k])
    rest = n - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(share[k] - int(share[k])), -weights[k], k))[:rest]:
        out[k] += 1
    return out


def fold_record(st: dict) -> dict[str, tuple[int, int]]:
    """component -> (folds, decided experiments)."""
    out = {c: [0, 0] for c in COMPONENTS}
    for e in st["experiments"].values():
        if e["state"] in TERMINAL and (c := e["card"]["component"]) in out:
            out[c][0] += e["state"] == "folded"
            out[c][1] += 1
    return {c: (f, n) for c, (f, n) in out.items()}


def allocate(attribution: dict[str, int], slots: int, frozen: set[str] = frozenset(),
             record: dict[str, tuple[int, int]] | None = None) -> list[tuple[str, str]]:
    """(component, tier) per slot. Components get slots in proportion to their wrong calls times their smoothed fold
    rate ((folds + 1) / (decided + 2), from `record`), with no floor; tiers split all slots 50/30/20. Bets go to the
    components with the most wrong calls first, then explore, then exploit. A frozen component's exploit slots become
    explore: a plateau needs a new mechanism, and bets fold least."""
    rate = {c: (f + 1) / (n + 2) for c, (f, n) in (record or {}).items()}
    per = largest_remainder({c: attribution.get(c, 0) * rate.get(c, 0.5) for c in COMPONENTS}, slots)
    tiers = largest_remainder(TIER_SHARE, slots)
    order = sorted(COMPONENTS, key=lambda c: (-attribution.get(c, 0), c))
    room, out = dict(per), []
    for tier in ("bet", "explore"):
        for _ in range(tiers[tier]):
            c = max(order, key=lambda c: (room[c] / max(per[c], 1), attribution.get(c, 0)))
            if room[c] == 0:
                break
            room[c] -= 1
            out.append((c, tier))
    out += [(c, "exploit") for c in order for _ in range(room[c])]
    return sorted(((c, "explore" if t == "exploit" and c in frozen else t) for c, t in out), key=lambda s: (order.index(s[0]), s[1]))


def free_slots(st: dict, alloc: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The allocated slots no active experiment holds, no more than the slots left once every active experiment
    (including ones the allocation no longer has a slot for) holds one."""
    held, now = Counter((e["card"]["component"], e["tier"]) for e in active(st)), len(active(st))
    out = []
    for s in alloc:
        if held[s]:
            held[s] -= 1
        else:
            out.append(s)
    return out[:max(0, len(alloc) - now)]


# Plateau


def record_outcome(st: dict, e: dict, at: str) -> None:
    """Counts a decided experiment toward its component's plateau run; a fold resets the run and thaws the component."""
    c, k = e["card"]["component"], st["counters"]
    if e["state"] == "folded":
        k["sinceFold"][c] = 0
        k["lastFoldAt"] = at
        k["foldsSinceRefresh"] = k.get("foldsSinceRefresh", 0) + 1
        k["frozen"].pop(c, None)
        k["frozen"] = {x: v for x, v in k["frozen"].items() if not v.get("global")}
    elif e["state"] in TERMINAL:
        k["sinceFold"][c] = k["sinceFold"].get(c, 0) + 1


def update_plateau(st: dict, at: str, started: str) -> list[str]:
    """Freezes components on a plateau; returns the newly frozen ones (each triggers a strategy call). `started` is when
    autolab began, standing in for the last fold before the first one."""
    k, new = st["counters"], []
    for c in COMPONENTS:
        if c not in k["frozen"] and k["sinceFold"].get(c, 0) >= PLATEAU_RUN:
            k["frozen"][c] = {"at": at, "why": f"{PLATEAU_RUN} experiments without a fold", "global": False}
            new.append(c)
    if minutes(k["lastFoldAt"] or started, at) > PLATEAU_HOURS * 60:
        for c in COMPONENTS:
            if c not in k["frozen"]:
                k["frozen"][c] = {"at": at, "why": f"{PLATEAU_HOURS} hours without a fold", "global": True}
                new.append(c)
    return new


def stuck_for_user(st: dict, at: str) -> list[str]:
    """Components frozen longer than ASK_AFTER_HOURS that haven't been asked about yet."""
    return [c for c, v in st["counters"]["frozen"].items() if not v.get("asked") and minutes(v["at"], at) > ASK_AFTER_HOURS * 60]


def reset_plateau(st: dict) -> None:
    """A new test set version: counters start over."""
    st["counters"].update(sinceFold={}, frozen={}, foldsSinceRefresh=0)


# Tuning cap


def tuning_blocked(st: dict, id: str) -> bool:
    """True when another open experiment is already a tuning diff (at most one at a time)."""
    return any(e["id"] != id and e["results"].get("tuning") for e in active(st))


# Incident clusters


def incidents(report: dict, variant: str = "phone") -> dict[str, list[str]]:
    """cluster -> incident keys from a main-line score report's practice cases: missed mistakes and false warnings,
    keyed by charged component, error class and kind."""
    out: dict[str, list[str]] = {}
    for c in report["cases"]:
        if c["split"] != "practice" or c["variant"] != variant:
            continue
        if c["kind"] == "mistake" and c["outcome"] == "miss":
            kind = "missed"
        elif c["kind"] == "false_warning":
            kind = "false-warning"
        else:
            continue
        out.setdefault(f"{c.get('component') or 'unattributed'}/{c['errorClass']}/{kind}", []).append(c["key"])
    return out


def update_coverage(st: dict, clusters: dict[str, list[str]]) -> None:
    """Incident counts from the latest main-line score; a cluster with none left is fixed."""
    cov = st["coverage"]
    for k, keys in clusters.items():
        cov.setdefault(k, {"state": "untouched", "experiments": [], "incidents": 0})["incidents"] = len(keys)
    for k, v in cov.items():
        if k not in clusters:
            v.update(incidents=0, state="fixed")
        elif v["state"] == "fixed":
            v["state"] = "tried" if v["experiments"] else "untouched"


def mark_tried(st: dict, cluster: str, id: str) -> None:
    v = st["coverage"].setdefault(cluster, {"state": "untouched", "experiments": [], "incidents": 0})
    if id not in v["experiments"]:
        v["experiments"].append(id)
    if v["state"] != "fixed":
        v["state"] = "tried"


def failed_tries(st: dict, cluster: str) -> int:
    """Decided experiments on a cluster that didn't fold."""
    ids = st["coverage"].get(cluster, {}).get("experiments", [])
    return sum(1 for i in ids if (e := st["experiments"].get(i)) and e["state"] in TERMINAL and e["state"] != "folded")
