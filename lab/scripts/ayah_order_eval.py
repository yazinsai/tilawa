"""Score the ayah-order check on top of an engine run (private data, aggregates out).

Engine rows come from ``tracker_correction_eval.py`` (with ``ZIPFORMER_TOKENS=1``);
candidates from ``ayah_order.py detect`` over the same dir. A rule keeps the
candidates that pass ``margin`` / ``fit`` / ``between``, drops any on an ayah
the engine already flagged, and adds the rest as issues. Scoring follows
``unseen_scoreboard.py`` (caught per kind, seen / unseen) plus clean false
flags, new-flag precision and latency.

    ../.venv/bin/python scripts/ayah_order_eval.py grid --run /tmp/ao/run/a0w-nopass --cands /tmp/ao/cand/a0w-nopass --split dev
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acted_eval import IN_SCOPE, hits, load_jsonl  # noqa: E402
from ayah_order import flags_of  # noqa: E402
from unseen_scoreboard import ERROR_KINDS, test_labels  # noqa: E402

SETS = ("help-acted", "help-clean", "tlog-dev", "v1")


def merged(run: Path, cands: Path | None, rule: dict | None, extra: dict[str, dict[str, list]] | None = None
           ) -> dict[str, list[dict]]:
    """Per set, rows with the check's flags (and any ``extra`` issues by id) added."""
    out = {}
    for name in SETS:
        rows = [r for r in load_jsonl(run / f"{name}.jsonl") if not r.get("error")]
        cand = {}
        if rule is not None and cands is not None:
            cand = {str(c["id"]): c["cand"] for c in load_jsonl(cands / f"{name}.jsonl")}
        res = []
        for r in rows:
            issues = list(r.get("issues") or [])
            for i in (extra or {}).get(name, {}).get(str(r["id"]), []):
                issues.append(i)
            if rule is not None:
                for f in flags_of(cand.get(str(r["id"])), rule):
                    if any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                           and int(i.get("ayah", 0)) == f["ayah"] for i in issues):
                        continue
                    issues.append(f)
            res.append({**r, "issues": issues})
        out[name] = res
    return out


# Frozen similar-verse rule (docs: similar-verse check, test row).
SV_RULE = {"families": ["lookalike", "drop"], "e": {"default": 3.5, "drop": 10.0}, "max_span": 1,
           "loc": 0.25, "ayah_d": 0.5}


def sv_extra(run: Path, cands: Path, no_passage: bool) -> dict[str, dict[str, list]]:
    """Similar-verse flags per set and id, deduplicated against engine issues as in its own eval."""
    from similar_verse_eval import merged_rows

    rule = {**SV_RULE, "all": no_passage}
    out = {}
    for name in SETS:
        rows = [r for r in load_jsonl(run / f"{name}.jsonl") if not r.get("error")]
        cc = {str(c["id"]): c["cands"] for c in load_jsonl(cands / f"{name}.jsonl")}
        rows, _ = merged_rows(rows, cc, rule)
        out[name] = {str(r["id"]): [i for i in r["issues"] if i.get("source") == "similar_verse"] for r in rows}
    return out


def evaluate(sets: dict[str, list[dict]], labels: list[dict], split: str) -> dict:
    acted = {str(r["id"]): r for r in sets["help-acted"]}
    labs = test_labels(labels, split)
    by = {p: Counter() for p in ("all_n", "all_c", "seen_n", "seen_c", "unseen_n", "unseen_c")}
    lats, new_lats = [], []
    labels_by_id: dict[str, list[dict]] = {}
    for lab in labs:
        r = acted.get(str(lab["id"]))
        if r is None:
            continue
        labels_by_id.setdefault(str(lab["id"]), []).append(lab)
        hit = [i for i in r["issues"] if i.get("kind") in ERROR_KINDS and hits(i, lab)]
        part = "unseen" if lab["unseen"] else "seen"
        for p in ("all", part):
            by[f"{p}_n"][lab["label_kind"]] += 1
            by[f"{p}_c"][lab["label_kind"]] += int(bool(hit))
        if hit and lab.get("span_s"):
            t = min(float(i["atSeconds"]) for i in hit if i.get("atSeconds") is not None) - float(lab["span_s"][0])
            lats.append(t)
            if all(i.get("source") == "ayah_order" for i in hit):
                new_lats.append(t)
    new_flags = new_tp = 0
    for cid, r in acted.items():
        if r.get("split") != split:
            continue
        for i in r["issues"]:
            if i.get("source") == "ayah_order":
                new_flags += 1
                new_tp += any(hits(i, lab) for lab in labels_by_id.get(cid, []))
    ff = {}
    for name in ("help-clean", "tlog-dev", "v1"):
        rows = [r for r in sets[name] if name != "help-clean" or r.get("split") == split]
        ff[name] = sum(1 for r in rows for i in r["issues"] if i.get("kind") in ERROR_KINDS)
        ff[name + "_new"] = sum(1 for r in rows for i in r["issues"] if i.get("source") == "ayah_order")
    return {"by": {k: dict(v) for k, v in by.items()}, "ff": ff, "new_flags": new_flags, "new_tp": new_tp,
            "latency": statistics.median(lats) if lats else None,
            "new_latency": statistics.median(new_lats) if new_lats else None}


def line(res: dict) -> str:
    b = res["by"]
    sa = f"skip_ayah {b['all_c'].get('skip_ayah', 0)}/{b['all_n'].get('skip_ayah', 0)}"
    un = f"unseen {sum(b['unseen_c'].values())}/{sum(b['unseen_n'].values())}"
    al = f"all {sum(b['all_c'].values())}/{sum(b['all_n'].values())}"
    ff = res["ff"]
    return (f"{sa} | {un} | {al} | new {res['new_tp']}/{res['new_flags']} | FF help {ff['help-clean']} "
            f"(+{ff['help-clean_new']}) tlog {ff['tlog-dev']} (+{ff['tlog-dev_new']}) v1 {ff['v1']} (+{ff['v1_new']})")


def grid() -> list[dict]:
    out = []
    for margin, rel, fit, between, min_post in itertools.product(
            (3, 5, 8, 99), (0.05, 0.075, 0.1, 0.15, 99), (0.2, 0.25, 0.3, 0.4), (0, 8), (0, 8)):
        out.append({"margin": margin, "rel": rel, "fit": fit, "between": between, "min_post": min_post})
    return out


def detect_grid() -> list[dict]:
    out = []
    for gap, jump, restart, garbage, back in itertools.product((10, 12, 14), (0.5, 1.0), (2.0, 3.0), (0.4, 0.5),
                                                               (1, 2)):
        out.append({"gap": gap, "jump": jump, "restart": restart, "garbage": garbage, "back": back})
    return out


def tune(runs: list[tuple[Path, str]], labels: list[dict], out: Path | None) -> None:
    """Dev only: detection params x flag thresholds over several runs (model x passage mode).
    Guards are help clean dev and v1; TLOG is not used for tuning."""
    from ayah_order import DEFAULTS, Corpus, detect_take

    corpus = Corpus()
    data = []
    for run, mode in runs:
        sets = {}
        for name in ("help-acted", "help-clean", "v1"):
            rows = [r for r in load_jsonl(run / f"{name}.jsonl") if not r.get("error")]
            sets[name] = [r for r in rows if name == "v1" or r.get("split") == "dev"]
        sets["tlog-dev"] = []
        data.append((run, mode, sets, evaluate(sets, labels, "dev")))
    table = []
    for dp in detect_grid():
        p = {**DEFAULTS, **dp}
        cands = [{name: {str(r["id"]): detect_take(corpus, r, mode, p) for r in rows} for name, rows in sets.items()}
                 for _run, mode, sets, _b in data]
        for rule in grid():
            tot = {"tp": 0, "flags": 0, "sa": 0, "ff": 0}
            per = []
            for (_run, _mode, sets, base), cc in zip(data, cands):
                merged_sets = {}
                for name, rows in sets.items():
                    res = []
                    for r in rows:
                        issues = list(r.get("issues") or [])
                        for f in flags_of(cc.get(name, {}).get(str(r["id"])), rule):
                            if not any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                                       and int(i.get("ayah", 0)) == f["ayah"] for i in issues):
                                issues.append(f)
                        res.append({**r, "issues": issues})
                    merged_sets[name] = res
                ev = evaluate(merged_sets, labels, "dev")
                d_sa = ev["by"]["all_c"].get("skip_ayah", 0) - base["by"]["all_c"].get("skip_ayah", 0)
                ff = ev["ff"]["help-clean_new"] + ev["ff"]["v1_new"]
                tot["tp"] += ev["new_tp"]
                tot["flags"] += ev["new_flags"]
                tot["sa"] += d_sa
                tot["ff"] += ff
                per.append((d_sa, ev["new_tp"], ev["new_flags"], ff))
            table.append({"detect": dp, "rule": rule, **tot, "per": per})
        best = max((t for t in table if t["detect"] == dp), key=lambda t: (t["ff"] == 0, t["sa"], -t["flags"]))
        print(json.dumps(dp), "best", json.dumps(best["rule"]), "sa", best["sa"], "tp/flags", best["tp"], best["flags"],
              "ff", best["ff"], best["per"], flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("grid", "score", "tune"))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--cands", type=Path)
    ap.add_argument("--runs", default="", help="tune: dir=passage_mode,... (passage_mode tracker|expected)")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--rule", help="JSON rule (score); default the frozen ayah_order.RULE")
    ap.add_argument("--out", type=Path, help="JSON aggregates (private)")
    ap.add_argument("--sv-cands", type=Path, help="similar_verse.py detect dir: add its frozen flags first")
    ap.add_argument("--sv-all", action="store_true", help="similar-verse with no passage (margin_all)")
    args = ap.parse_args(argv)
    labels = load_jsonl(args.labels)
    if args.cmd == "tune":
        runs = [(Path(d), m) for d, m in (s.split("=", 1) for s in args.runs.split(",") if s)]
        tune(runs, labels, args.out)
        return
    extra = sv_extra(args.run, args.sv_cands, args.sv_all) if args.sv_cands else None
    base = evaluate(merged(args.run, None, None), labels, args.split)
    print("base:", line(base))
    if extra is not None:
        sv = evaluate(merged(args.run, None, None, extra), labels, args.split)
        print("similar-verse:", line(sv))
        print("similar-verse unseen per kind:", ", ".join(
            f"{k} {sv['by']['unseen_c'].get(k, 0)}/{sv['by']['unseen_n'].get(k, 0)}" for k in IN_SCOPE))
        base = sv
    from ayah_order import RULE

    rules = [json.loads(args.rule) if args.rule else RULE] if args.cmd == "score" else grid()
    results = []
    for rule in rules:
        res = evaluate(merged(args.run, args.cands, rule, extra), labels, args.split)
        results.append({"rule": rule, **res})
        print(json.dumps(rule), line(res), flush=True)
        if args.cmd == "score":
            kinds = ", ".join(f"{k} {res['by']['unseen_c'].get(k, 0)}/{res['by']['unseen_n'].get(k, 0)}"
                              for k in IN_SCOPE)
            print("unseen per kind:", kinds)
            print("latency all hits", base["latency"], "->", res["latency"], "| new hits", res["new_latency"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"base": base, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
