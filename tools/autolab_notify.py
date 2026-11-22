"""Autolab notifications (.scratch/autolab/spec.md decision 12; ticket 07). One channel, ntfy: a POST of plain text to
https://ntfy.sh/<topic> (or cfg["notify"]["server"]). Kinds: digest, needs-human, stale (the watchdog), cap, paused.

`notify(st, cfg, kind, key, title, body)` sends once per (kind, key): repeats are remembered in st["notified"], so a
stuck item doesn't page every tick. Tests pass their own `sender`.
"""

import urllib.request
from collections import Counter

PRIORITY = {"digest": "default", "needs-human": "high", "stale": "urgent", "cap": "default", "paused": "high"}


def ntfy(cfg: dict, kind: str, title: str, body: str) -> None:
    n = cfg.get("notify") or {}
    if not n.get("topic"):
        return
    req = urllib.request.Request(f"{n.get('server', 'https://ntfy.sh').rstrip('/')}/{n['topic']}", data=body.encode(),
                                 headers={"Title": title.encode("ascii", "replace").decode(), "Priority": PRIORITY.get(kind, "default"),
                                          "Tags": kind, **({"Click": n["dashboard"]} if n.get("dashboard") else {})})
    urllib.request.urlopen(req, timeout=15).read()


def notify(st: dict, cfg: dict, kind: str, key: str, title: str, body: str, sender=None, at: str = "") -> bool:
    sent = st.setdefault("notified", {})
    k = f"{kind}:{key}"
    if k in sent:
        return False
    (sender or ntfy)(cfg, kind, title, body)
    sent[k] = at
    return True


def digest(st: dict, events: list[dict], day: str) -> tuple[str, str]:
    """The daily summary from the day's events and the current state."""
    today = [e for e in events if e["at"].startswith(day)]
    ended = Counter(e["to"] for e in today if e["to"] in ("folded", "discarded", "superseded", "abandoned"))
    folded = [e["id"] for e in today if e["to"] == "folded"]
    main = (st.get("main") or {}).get("practice") or {}
    s = st["spend"]
    lines = [f"Practice wrong calls now: {main.get('wrongCalls', '?')} ({main.get('missed', '?')} missed + {main.get('falseWarnings', '?')} false warnings)."
             if main else "No main-line score yet.",
             "Ended today: " + (", ".join(f"{n} {k}" for k, n in sorted(ended.items())) or "none") + ".",
             *([f"Kept: {', '.join(folded)}."] if folded else []),
             f"Running: {sum(1 for e in st['experiments'].values() if e['state'] in ('implementing', 'validating', 'scoring', 'folding'))}.",
             f"Spend: {s['agentRuns']} agent runs, {s['agentHours']} agent hours, ${s['modalUsd']} Modal.",
             *([f"Waiting on you: {len(st['needsHuman'])} item(s)."] if st["needsHuman"] else []),
             *([f"Paused: {st['paused']}."] if st.get("paused") else [])]
    return f"Autolab {day}: {len(folded)} kept, {sum(ended.values()) - len(folded)} dropped", "\n".join(lines)
