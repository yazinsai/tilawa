"""Re-score `qlab_per_eval.py` per-clip output under several gold/length views.

Views (all over the *same* clip subset per slice, so models stay comparable):
- ``ordered`` / ``text``: the two golds as in qlab_per_eval.
- ``ordered_maddfree``: ordered gold with madd runs (ا / ۥ / ۦ repeated)
  collapsed to one unit in gold and hyp — PER that ignores madd length,
  which the two golds disagree on at waqf (4 vs 2 harakat) and which v3.1's
  madd fine-tune specifically moved.

Also prints the per-clip interp-vs-parents table for the interp artifact check.

  ../.venv/bin/python scripts/qlab_per_report.py --qlab-dir /tmp/qlab --per-dir /tmp/phase0/qlab_per \\
      --models v3,v31,ft-gentle,interp-gentle-a05 --interp interp-gentle-a05=v31+ft-gentle
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "scripts"))

MADD_CHARS = frozenset("اۥۦ")


def madd_free_map(tokens: list[str]) -> list[int]:
    """Token id -> canonical id where pure madd runs share one id."""
    canon: dict[str, int] = {}
    out = []
    for i, t in enumerate(tokens):
        key = t[0] if t and set(t) <= MADD_CHARS and len(set(t)) == 1 else t
        out.append(canon.setdefault(key, len(canon)))
    return out


def collapse(ids, m):
    out = []
    for i in ids:
        j = m[i]
        if out and out[-1] == j and j in _MADD_IDS:
            continue
        out.append(j)
    return out


_MADD_IDS: set[int] = set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlab-dir", required=True)
    ap.add_argument("--per-dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--interp", default="", help="name=parentA+parentB")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from Levenshtein import editops

    from qlab_per_eval import build_rows
    from shared.paths import resolve_data_file
    from shared.phoneme_labels import PhonemeTokenizer, load_tokens

    tokens = load_tokens()
    m = madd_free_map(tokens)
    for i, t in enumerate(tokens):
        if t and set(t) <= MADD_CHARS and len(set(t)) == 1:
            _MADD_IDS.add(m[i])
    ref_dir = resolve_data_file("zipformer/reference/quran_text2phoneme.json").parent
    rows = {r["id"]: r for r in build_rows(Path(args.qlab_dir), LAB / "benchmark" / "test_corpus_qlab" / "manifest.json",
                                            ref_dir, PhonemeTokenizer(tokens))}
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    per_clip = {}
    for name in models:
        per_clip[name] = {json.loads(l)["id"]: json.loads(l) for l in (Path(args.per_dir) / f"{name}.jsonl").read_text().splitlines() if l.strip()}

    def s(ids):
        return "".join(chr(0x100 + i) for i in ids)

    def counts(ref, hyp):
        c = {"replace": 0, "insert": 0, "delete": 0}
        for op, _, _ in editops(s(ref), s(hyp)):
            c[op] += 1
        return c["replace"], c["insert"], c["delete"], len(ref)

    views = {
        "ordered": lambda r, h: (r["gold_ordered"], h) if r["gold_ordered"] and r["gold_text"] else None,
        "text": lambda r, h: (r["gold_text"], h) if r["gold_ordered"] and r["gold_text"] else None,
        "ordered_maddfree": lambda r, h: (collapse(r["gold_ordered"], m), collapse(h, m)) if r["gold_ordered"] and r["gold_text"] else None,
        "ordered_all": lambda r, h: (r["gold_ordered"], h) if r["gold_ordered"] else None,
        "ordered_all_maddfree": lambda r, h: (collapse(r["gold_ordered"], m), collapse(h, m)) if r["gold_ordered"] else None,
    }
    table: dict = {}
    clip_err: dict = defaultdict(dict)
    for view, fn in views.items():
        for name in models:
            agg = defaultdict(lambda: [0, 0, 0, 0, 0])
            for cid, pc in per_clip[name].items():
                pair = fn(rows[cid], pc["hyp"])
                if pair is None:
                    continue
                sub, ins, dele, n = counts(*pair)
                clip_err[(view, name)][cid] = (sub + ins + dele, n)
                for src in (pc["source"], "ALL"):
                    a = agg[src]
                    a[0] += sub; a[1] += ins; a[2] += dele; a[3] += n; a[4] += 1
            table.setdefault(view, {})[name] = {
                src: {"clips": a[4], "per": round(100 * (a[0] + a[1] + a[2]) / max(a[3], 1), 2),
                      "S": round(100 * a[0] / max(a[3], 1), 2), "I": round(100 * a[1] / max(a[3], 1), 2),
                      "D": round(100 * a[2] / max(a[3], 1), 2)}
                for src, a in sorted(agg.items())
            }
    for view in views:
        print(f"\n== {view}")
        srcs = sorted({src for n in models for src in table[view][n]})
        print(f"{'model':22s}" + "".join(f"{src[:16]:>28s}" for src in srcs))
        for n in models:
            cells = []
            for src in srcs:
                c = table[view][n].get(src)
                cells.append(f"{c['per']:6.2f} S{c['S']:.2f}/I{c['I']:.2f}/D{c['D']:.2f} n{c['clips']}" if c else "")
            print(f"{n:22s}" + "".join(f"{x:>28s}" for x in cells))

    interp_out = {}
    if args.interp:
        name, _, parents = args.interp.partition("=")
        pa, pb = parents.split("+")
        for view in ("ordered_all", "ordered_all_maddfree", "text"):
            e = clip_err
            buckets = defaultdict(int)
            for cid, (ei, n) in e[(view, name)].items():
                if per_clip[name][cid]["source"] != "tlog_holdout" or cid not in e[(view, pa)] or cid not in e[(view, pb)]:
                    continue
                ea, eb = e[(view, pa)][cid][0], e[(view, pb)][cid][0]
                if ei < min(ea, eb):
                    buckets["better_than_both"] += 1
                elif ei > max(ea, eb):
                    buckets["worse_than_both"] += 1
                elif ei == ea == eb:
                    buckets["tie_all"] += 1
                else:
                    buckets["between"] += 1
            interp_out[view] = dict(buckets)
            print(f"\n[interp tlog_holdout {view}] {name} vs {pa},{pb}: {dict(buckets)}")
    if args.json:
        Path(args.json).write_text(json.dumps({"table": table, "interp": interp_out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
