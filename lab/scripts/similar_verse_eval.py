"""Score the similar-verse check on top of an engine run (private data, aggregates out).

Engine rows come from ``tracker_correction_eval.py`` (one dir per model and
passage mode); candidates from ``similar_verse.py detect`` over the same dir.
A rule set picks candidates by family, kind and margin, keeps the best per
ayah slot, drops any that sit within ±1 word of an engine issue, and adds the
rest as issues. Scoring is ``acted_eval.py`` on the merged rows, plus the
seen / unseen split: a test label is unseen when no acted-dev label of the
same kind names its ``surah:ayah:word``.

    ../.venv/bin/python scripts/similar_verse_eval.py grid --run /tmp/tc/a0w-pass --cands /tmp/sv/a0w-pass --split dev
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
SETS = ("help-acted", "help-clean", "tlog-dev", "v1")


def _acted_eval():
    spec = importlib.util.spec_from_file_location("acted_eval", LAB / "scripts" / "acted_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["acted_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _acted_eval()


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def pick(cands: list[dict], rule: dict) -> list[dict]:
    """Best qualifying candidate per ayah slot."""
    best: dict[int, dict] = {}
    for c in cands:
        if c["family"] not in rule["families"] or c.get("at") is None:
            continue
        key = "margin_all" if (rule.get("all") or not c.get("known")) else "margin"
        m = c[key]
        need = rule["e"].get(c["family"], rule["e"]["default"]) if isinstance(rule["e"], dict) else rule["e"]
        if m < need or c["span"] > rule.get("max_span", 99):
            continue
        if rule.get("loc") is not None and (c.get("loc") is None or c["loc"] > rule["loc"]):
            continue
        if rule.get("ayah_d") is not None and (c.get("ayah_d") is None or c["ayah_d"] > rule["ayah_d"]):
            continue
        if c["family"] == "drop" and c["span"] > rule.get("max_drop", 2):
            continue
        cur = best.get(c["slot"])
        if cur is None or m > cur["_m"]:
            best[c["slot"]] = {**c, "_m": m}
    return list(best.values())


def merged_rows(rows: list[dict], cands: dict[str, list[dict]], rule: dict) -> tuple[list[dict], int]:
    out, added = [], 0
    for r in rows:
        issues = list(r.get("issues") or [])
        for c in pick(cands.get(str(r["id"]), []), rule):
            if any(int(i.get("surah", 0)) == c["surah"] and int(i.get("ayah", 0)) == c["ayah"]
                   and abs(int(i.get("word", -9)) - c["word"]) <= 1 for i in issues):
                continue
            issues.append({"kind": c["kind"], "surah": c["surah"], "ayah": c["ayah"], "word": c["word"],
                           "atSeconds": c["at"], "source": "similar_verse", "family": c["family"]})
            added += 1
        out.append({**r, "issues": issues})
    return out, added


def write_run(run: Path, cands_dir: Path, rule: dict | None, dest: Path) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    added = {}
    for name in SETS:
        rows = [r for r in load(run / f"{name}.jsonl") if not r.get("error")]
        if rule is not None:
            cands = {str(c["id"]): c["cands"] for c in load(cands_dir / f"{name}.jsonl")}
            rows, added[name] = merged_rows(rows, cands, rule)
        with (dest / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                slim = {k: r[k] for k in ("id", "duration_s", "split", "speaker", "set", "issues", "notes") if k in r}
                fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    return added


def unseen_keys(labels: list[dict]) -> dict[str, set]:
    seen: dict[str, set] = defaultdict(set)
    for lab in labels:
        if lab.get("split") == "dev" and lab.get("mapped"):
            seen[lab["label_kind"]].add((lab["surah"], lab["ayah"], lab.get("word_index")))
    return seen


def per_label(run_dir: Path, labels: list[dict], split: str) -> dict[str, dict]:
    rows = ev.clip_rows(run_dir, "help-acted")
    out = {}
    for lab in labels:
        if lab.get("split") != split or not lab.get("mapped") or lab.get("error"):
            continue
        row = rows.get(str(lab["id"]))
        if row is None:
            continue
        st = ev.acted_clip_stats(lab, row)
        sv = [i for i in row.get("issues") or [] if i.get("source") == "similar_verse" and ev.hits(i, lab)]
        out[str(lab["id"])] = {**st, "label": lab, "sv": bool(sv)}
    return out


def score(run: Path, cands_dir: Path, rule: dict | None, labels: list[dict], split: str,
          speakers: dict[str, str], base_stats: dict | None = None, ci: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        added = write_run(run, cands_dir, rule, d)
        summ = ev.summarize(ev.collect(d, labels, split, speakers), ci=ci)
        pl = per_label(d, labels, split)
        # sv flag latencies on hits, relative to the label span start
        lats = []
        rows = ev.clip_rows(d, "help-acted")
        for cid, st in pl.items():
            lab = st["label"]
            if not st["sv"] or not lab.get("span_s"):
                continue
            ts = [float(i["atSeconds"]) for i in rows[cid]["issues"]
                  if i.get("source") == "similar_verse" and ev.hits(i, lab)]
            lats.append(min(ts) - float(lab["span_s"][0]))
        sv_flags = sv_tp = 0
        for name in ("help-acted", "help-clean"):
            for r in ev.clip_rows(d, name).values():
                if r.get("split") != split:
                    continue
                for i in r.get("issues") or []:
                    if i.get("source") != "similar_verse":
                        continue
                    sv_flags += 1
                    if name == "help-acted":
                        labs = [st["label"] for cid, st in pl.items() if cid == str(r["id"])]
                        sv_tp += any(ev.hits(i, lab) for lab in labs)
    seen = unseen_keys(labels)
    split_counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for cid, st in pl.items():
        lab = st["label"]
        kind = lab["label_kind"]
        part = "seen" if (lab["surah"], lab["ayah"], lab.get("word_index")) in seen[kind] else "unseen"
        cell = split_counts[kind][part]
        cell[0] += 1
        cell[1] += int(st["caught"])
        if base_stats is not None:
            cell[2] += int(st["caught"] and not base_stats[cid]["caught"])
    ff = {k: v["issues"] for k, v in summ["ff"].items()}
    return {"summary": summ, "per_label": pl, "split": {k: dict(v) for k, v in split_counts.items()},
            "ff": ff, "added": added, "sv_flags": sv_flags, "sv_tp": sv_tp,
            "sv_latency": statistics.median(lats) if lats else None}


def rules_grid() -> list[dict]:
    out = []
    fams = [("lookalike",), ("lookalike", "eyeskip"), ("lookalike", "eyeskip", "drop")]
    for fam, e, max_span, loc in itertools.product(fams, (2.0, 2.5, 3.0, 4.0, 5.0), (1, 99), (None, 0.35, 0.25)):
        base = {"families": list(fam), "e": {"default": e}, "max_span": max_span, "loc": loc, "ayah_d": 0.5}
        out.append(base)
        if "drop" in fam:
            for ed in (3.0, 4.0, 5.0, 6.0):
                if ed > e:
                    out.append({**base, "e": {"default": e, "drop": ed}})
    return out


def fmt_split(res: dict, kinds=("substitution", "skip_word")) -> str:
    parts = []
    for k in kinds:
        s = res["split"].get(k, {})
        u, se = s.get("unseen", [0, 0, 0]), s.get("seen", [0, 0, 0])
        parts.append(f"{k[:4]} seen {se[1]}/{se[0]} (+{se[2]}) unseen {u[1]}/{u[0]} (+{u[2]})")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("grid", "score"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--rule", help="JSON rule (score)")
    ap.add_argument("--all", action="store_true", help="no passage: margin over every pure look-alike reading")
    ap.add_argument("--out", type=Path, help="JSON summary (private)")
    args = ap.parse_args(argv)
    labels = ev.load_labels(args.labels)
    speakers = ev.load_speakers(args.manifest)
    base = score(args.run, args.cands, None, labels, args.split, speakers)
    bstats = base["per_label"]
    print(f"base: caught {base['summary']['agg']['all']['caught']}/{base['summary']['agg']['all']['labels']} "
          f"FF {base['ff']} | {fmt_split(base)}")
    if args.cmd == "score":
        rule = json.loads(args.rule)
        rule["all"] = args.all
        res = score(args.run, args.cands, rule, labels, args.split, speakers, bstats, ci=True)
        h = res["summary"]["headline"]
        print(json.dumps({"rule": rule, "p": h["p"], "r": h["r"], "f1": h["f1"], "ff": res["ff"],
                          "caught": res["summary"]["agg"]["all"]["caught"], "sv_flags": res["sv_flags"],
                          "sv_tp": res["sv_tp"], "sv_latency": res["sv_latency"], "split": res["split"],
                          "base_f1": base["summary"]["headline"]["f1"], "base_p": base["summary"]["headline"]["p"],
                          "base_r": base["summary"]["headline"]["r"],
                          "base_latency": base["summary"]["agg"]["all"]["median_latency_s"],
                          "latency": res["summary"]["agg"]["all"]["median_latency_s"],
                          "kinds": {k: res["summary"]["agg"]["by_kind"][k] for k in ("substitution", "skip_word")},
                          "base_kinds": {k: base["summary"]["agg"]["by_kind"][k] for k in ("substitution", "skip_word")}},
                         indent=1, default=str))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            slim = {k: v for k, v in res.items() if k != "per_label"}
            args.out.write_text(json.dumps({"rule": rule, "res": slim, "base": {k: v for k, v in base.items() if k != "per_label"}},
                                           indent=1, default=str), encoding="utf-8")
        return
    rows = []
    for rule in rules_grid():
        rule["all"] = args.all
        res = score(args.run, args.cands, rule, labels, args.split, speakers, bstats)
        sub = res["split"]
        extra = sum(sub.get(k, {}).get(p, [0, 0, 0])[2] for k in ("substitution", "skip_word") for p in ("seen", "unseen"))
        rows.append((rule, res["ff"], extra, res["sv_flags"], res["sv_tp"], fmt_split(res)))
        print(json.dumps({k: v for k, v in rule.items() if k != "all"}), res["ff"], f"extra {extra}",
              f"sv {res['sv_tp']}/{res['sv_flags']}", fmt_split(res), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
