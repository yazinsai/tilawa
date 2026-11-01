"""The builder A/B (Grok through Cursor against GPT-6 Astra through Codex, or any arms in executor.runtimes). Pure
functions over the state document (tools/autolab_state.py); tools/autolab.py calls them.

    assign(id, component, arms, weights, st) -> arm    which arm builds a new experiment
    stats(st, names) -> {"arms": [...], "since"}      per arm: what its builders did and what came of it

An experiment's arm is chosen once, when its builder first starts, and kept in e["executor"]["assignedArm"]; a fallback
to another arm (after two failures of its own, or when its arm can't run) moves e["executor"]["arm"] but the stats keep
counting it for the arm it was assigned to (intent to treat), with the fallback counted there too. Experiments whose
builder started before arms existed (armInferred) are left out.
"""

import hashlib
import random
import statistics

import autolab_state as S


def assign(id: str, component: str, arms: list[dict], weights: dict, st: dict) -> dict:
    """The arm furthest below its share of this component's running experiments ((running + 1) / weight, lowest wins);
    ties go to a draw by weight, seeded by the experiment id, so the same state always gives the same arm."""
    pool = [a for a in arms if weights.get(a["name"], 1) > 0] or arms
    running = {a["name"]: 0 for a in pool}
    for e in S.active(st):
        arm = (e.get("executor") or {}).get("arm")
        if e["id"] != id and arm in running and (e.get("card") or {}).get("component") == component:
            running[arm] += 1
    load = {a["name"]: (running[a["name"]] + 1) / (weights.get(a["name"], 1) or 1) for a in pool}
    least = min(load.values())
    tied = [a for a in pool if load[a["name"]] == least]
    rng = random.Random(int(hashlib.sha256(id.encode()).hexdigest()[:16], 16))
    return rng.choices(tied, weights=[weights.get(a["name"], 1) or 1 for a in tied])[0]


def measured(res: dict) -> tuple[int | None, int | None]:
    """(overall practice change, incidents lost) as the supervisor scored them: after rebase if the fold ran, else on the
    branch."""
    for src in (res.get("fold"), res.get("branchRun")):
        cmp = (src or {}).get("compare") if isinstance(src, dict) else None
        if isinstance(cmp, dict) and isinstance((cmp.get("overall") or {}).get("change"), int):
            gate = (src.get("gate") if isinstance(src.get("gate"), dict) else None) or res.get("gate") or {}
            lost = next((c.get("lost") for c in gate.get("checks") or [] if c.get("name") == "collateral"), None)
            return cmp["overall"]["change"], lost
    return None, None


def _blank() -> dict:
    return {"started": 0, "running": 0, "killedByBuilder": 0, "bounced": 0, "gated": 0, "gatePassed": 0, "nearMisses": 0,
            "folds": 0, "abandoned": 0, "fallbacks": 0, "_changes": [], "_hours": []}


def stats(st: dict, names: list[str] | None = None) -> dict:
    rows, at = {n: None for n in names or []}, []
    for e in st.get("experiments", {}).values():
        ex = e.get("executor") or {}
        arm = ex.get("assignedArm") or ex.get("arm")
        if not arm or ex.get("armInferred"):
            continue
        at += [ex["assignedAt"]] if ex.get("assignedAt") else []
        r = rows.get(arm) or _blank()
        rows[arm] = r
        res, state = e.get("results") or {}, e.get("state")
        r["started"] += 1
        r["running"] += state not in S.TERMINAL and state != "needs-human"
        r["killedByBuilder"] += (res.get("builderOutcome") or {}).get("status") == "killed"
        r["bounced"] += (e.get("attempts") or {}).get("bounces", 0) > 0
        gate = res.get("gate")
        if isinstance(gate, dict) and gate.get("verdict"):
            r["gated"] += 1
            r["gatePassed"] += gate["verdict"] == "pass"
        change, lost = measured(res)
        if change is not None:
            r["_changes"].append(change)
            r["nearMisses"] += change in (-1, -2) and lost == 0
        r["folds"] += state == "folded"
        r["abandoned"] += state == "abandoned"
        r["fallbacks"] += len(ex.get("fallbacks") or [])
        r["_hours"].append(float(ex.get("hours") or 0))
    out = []
    for name, r in rows.items():
        r = r or _blank()
        ch, hrs = r.pop("_changes"), r.pop("_hours")
        out.append({"arm": name, **r, "gatePassRate": round(r["gatePassed"] / r["gated"], 3) if r["gated"] else None,
                    "scored": len(ch), "medianChange": statistics.median(ch) if ch else None,
                    "meanBuilderHours": round(statistics.fmean(hrs), 2) if hrs else None})
    return {"arms": out, "since": min(at) if at else None}


def table(ab: dict) -> list[str]:
    """Plain-text table of stats() for the CLI."""
    cols = (("arm", "Arm"), ("started", "Started"), ("running", "Running"), ("killedByBuilder", "Killed by builder"),
            ("bounced", "Bounced"), ("gatePass", "Gate passed"), ("nearMisses", "Near misses"), ("folds", "Folds"),
            ("medianChange", "Median change"), ("meanBuilderHours", "Builder h (mean)"), ("fallbacks", "Fallbacks"))
    rows = [{**a, "gatePass": f"{a['gatePassed']}/{a['gated']}" + (f" ({a['gatePassRate']:.0%})" if a["gated"] else ""),
             "medianChange": "–" if a["medianChange"] is None else f"{a['medianChange']:+g}",
             "meanBuilderHours": "–" if a["meanBuilderHours"] is None else a["meanBuilderHours"]} for a in ab["arms"]]
    width = {k: max(len(h), *(len(str(r[k])) for r in rows)) if rows else len(h) for k, h in cols}
    return [("  ".join(h.ljust(width[k]) for k, h in cols)),
            *("  ".join(str(r[k]).ljust(width[k]) for k, _ in cols) for r in rows),
            f"since {ab['since'] or '–'}; median change is practice wrong calls overall (negative is better); a near miss "
            "measured -1 or -2 and lost nothing"]
