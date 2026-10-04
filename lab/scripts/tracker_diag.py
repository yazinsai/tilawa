"""Classify why acted mistakes are missed: tracker or detector.

Reads ``tracker_correction_eval.py --diag`` rows (help-acted.jsonl) and the
acted labels, and prints aggregate counts per label kind. Private inputs; the
output is aggregates only.

Per label (word kinds), at the labelled word's forced-alignment span:
- ``caught``: an issue hits the label (acted_eval.hits).
- ``seen:<state>``: the word reached a controller verdict snapshot but no
  flag; the most "error-like" state it reached (wrong > skipped > unsure > ok).
- not seen: ``no_lock`` (the take never locked), ``other_ayah`` (every lock
  went to another ayah, usually a near-identical one), ``late_lock`` (locked
  on the ayah only after the word was said); else by tracker state over the
  span: ``searching``, ``other_surah``, ``far`` (same surah, cursor more than
  one ayah away), ``lost``, ``near`` (cursor on or next to the ayah but the
  word never got a verdict).

skip_ayah labels: whether N+1 (the skipped ayah) was emitted as a verse,
whether N+2 was emitted, and whether the ayah-gap rule raised.

    ../.venv/bin/python scripts/tracker_diag.py --rows /tmp/tc/base-dev-r --split dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
STATES = ["ok", "unsure", "wrong", "skipped", "pending"]


def _acted_eval():
    spec = importlib.util.spec_from_file_location("acted_eval", LAB / "scripts" / "acted_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def classify(label: dict, row: dict, ae) -> str:
    issues = row.get("issues") or []
    if any(ae.hits(i, label) for i in issues):
        return "caught"
    s, a = int(label["surah"]), int(label["ayah"])
    key = f"{s}:{a}:{label['word_index']}"
    seen = (row.get("seen") or {}).get(key)
    if seen:
        mask = seen[0]
        for st in ("wrong", "skipped", "unsure", "ok"):
            if mask & (1 << STATES.index(st)):
                return f"seen:{st}"
        return "seen:pending"
    span = label.get("span_s") or [0, row.get("duration_s") or 0]
    lo, hi = float(span[0]), float(span[1]) + 1.0
    events = row.get("events") or []
    locks = [(e.get("t", 0.0), e.get("surah"), e.get("ayah")) for e in events if e.get("type") == "located"]
    locks += [(e.get("t", 0.0), e["to"]["surah"], e["to"]["ayah"]) for e in events
              if e.get("type") == "relocated" and isinstance(e.get("to"), dict)]
    near = [lk for lk in locks if lk[1] == s and abs(int(lk[2] or 0) - a) <= 1]
    if not locks:
        return "no_lock"
    if not near:
        return "other_ayah"
    if min(lk[0] for lk in near) > lo:
        return "late_lock"
    diag = [d for d in row.get("diag") or [] if lo <= d[0] <= hi + 0.5] or [
        d for d in row.get("diag") or [] if d[0] >= lo][:2]
    if not diag:
        return "no_diag"
    votes = Counter()
    for t, tracking, _w, ds, da, _dw, lost, _rate in diag:
        if not tracking:
            votes["searching"] += 1
        elif ds != s:
            votes["other_surah"] += 1
        elif lost:
            votes["lost"] += 1
        elif abs(da - a) > 1:
            votes["far"] += 1
        else:
            votes["near"] += 1
    return votes.most_common(1)[0][0]


def skip_ayah_info(label: dict, row: dict, ae) -> str:
    s, a = int(label["surah"]), int(label["ayah"])
    verses = {tuple(v) for v in row.get("verses") or []}
    issues = row.get("issues") or []
    if any(ae.hits(i, label) for i in issues):
        return "caught"
    n1 = (s, a) in verses
    n2 = (s, a + 1) in verses
    n0 = (s, a - 1) in verses
    return f"prev={'y' if n0 else 'n'},skipped={'y' if n1 else 'n'},next={'y' if n2 else 'n'}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("/tmp/help_slips/acted/acted_located.jsonl"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    ae = _acted_eval()
    rows = {str(r["id"]): r for r in ae.load_jsonl(args.rows / "help-acted.jsonl") if not r.get("error")}
    out: dict[str, Counter] = defaultdict(Counter)
    for label in ae.load_jsonl(args.labels):
        if label.get("split") != args.split or not label.get("mapped") or label.get("error"):
            continue
        row = rows.get(str(label["id"]))
        if row is None:
            continue
        kind = label["label_kind"]
        out[kind][skip_ayah_info(label, row, ae) if kind == "skip_ayah" else classify(label, row, ae)] += 1
    for kind in ae.ALL_KINDS:
        c = out.get(kind, Counter())
        print(f"{kind} (n={sum(c.values())}): " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    if args.json:
        args.json.write_text(json.dumps({k: dict(v) for k, v in out.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
