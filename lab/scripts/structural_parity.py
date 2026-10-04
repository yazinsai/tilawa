"""Parity of the SDK structural rules (TS port) with the lab prototypes (aggregates only).

``cands``: field-by-field diff of candidates written by ``ayah_order.py detect`` /
``similar_verse.py detect`` against ``structural-parity.ts`` on the same rows
(look-alike and drop families; eye-skip is not ported), plus the frozen-rule flags.

``stack``: engine + similar-verse + ayah-order issues per set, from the lab candidate
files (``ayah_order_stress.stack_rows``, optional near-skip suppression) or from an SDK
run (issues with ``source``), and the confirmatory recall / control counts.

    ../.venv/bin/python scripts/structural_parity.py cands --py /tmp/sr/py --ts /tmp/sr/ts --cfg nopass
    ../.venv/bin/python scripts/structural_parity.py stack --run /tmp/sr/base/nopass --sv /tmp/sr/py/sv/nopass \\
        --ao /tmp/sr/py/ao/nopass [--near-skip] | --sdk /tmp/sr/sdk/nopass
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acted_eval import load_jsonl  # noqa: E402
from ayah_order import RULE, flags_of  # noqa: E402
from ayah_order_stress import SV_RULE  # noqa: E402
from similar_verse_eval import pick  # noqa: E402
from unseen_scoreboard import ERROR_KINDS  # noqa: E402

SETS = ("help-acted", "help-clean", "v1", "everyayah", "tlog-qc", "skip", "skip-ctl", "sister", "sister-ctl")
SV_KEYS = ("slot", "surah", "ayah", "word", "kind", "family", "span", "from", "margin", "margin_all", "known",
           "at", "loc", "ayah_d")


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= 1e-9
    return a == b


def cands(py: Path, ts: Path, cfg: str) -> dict:
    out: dict = {}
    sv_rule = {**SV_RULE, "all": cfg == "nopass"}
    for name in SETS:
        pa = {r["id"]: r["cand"] for r in load_jsonl(py / "ao" / cfg / f"{name}.jsonl")}
        ta = {r["id"]: r["cand"] for r in load_jsonl(ts / cfg / "ao" / f"{name}.jsonl")}
        ps = {r["id"]: [c for c in r["cands"] if c["family"] in ("lookalike", "drop")]
              for r in load_jsonl(py / "sv" / cfg / f"{name}.jsonl")}
        tsv = {r["id"]: r["cands"] for r in load_jsonl(ts / cfg / "sv" / f"{name}.jsonl")}
        st = Counter()
        for cid, a in pa.items():
            b = ta.get(cid)
            st["ao_takes"] += 1
            same = (a is None and b is None) or (a is not None and b is not None and _eq(a["margin"], b["margin"])
                                                  and a["segs"] == b["segs"] and len(a["jumps"]) == len(b["jumps"])
                                                  and all(all(_eq(x[k], y[k]) for k in x) for x, y in zip(a["jumps"], b["jumps"])))
            st["ao_same"] += int(same)
            fa = [(f["surah"], f["ayah"]) for f in flags_of(a, RULE)]
            fb = [(f["surah"], f["ayah"]) for f in flags_of(b, RULE)]
            st["ao_flags_py"] += len(fa)
            st["ao_flags_ts"] += len(fb)
            st["ao_flags_same"] += int(fa == fb)
        for cid, a in ps.items():
            b = tsv.get(cid, [])
            st["sv_takes"] += 1
            st["sv_cands"] += len(a)
            ok = len(a) == len(b) and all(all(_eq(x[k], y[k]) for k in SV_KEYS) for x, y in zip(a, b))
            st["sv_same"] += int(ok)
            pk = sorted((c["slot"], c["word"], c["kind"]) for c in pick(a, sv_rule))
            tk = sorted((c["slot"], c["word"], c["kind"]) for c in pick(b, sv_rule))
            st["sv_picks_py"] += len(pk)
            st["sv_picks_ts"] += len(tk)
            st["sv_picks_same"] += int(pk == tk)
            if not ok and st["sv_diff_shown"] < 2:
                st["sv_diff_shown"] += 1
                for x, y in zip(a, b):
                    bad = [k for k in SV_KEYS if not _eq(x[k], y[k])]
                    if bad:
                        print(name, "sv first diff", {k: (x[k], y[k]) for k in bad}, file=sys.stderr)
                        break
                else:
                    print(name, "sv len", len(a), len(b), file=sys.stderr)
        out[name] = dict(st)
    return out


def _lab_stack(run: Path, sv: Path, ao: Path, name: str, near_skip: bool) -> list[dict]:
    """``ayah_order_stress.stack_rows`` with the SDK's near-skip suppression as an option: similar-verse
    flags on or next to an ayah the engine or ayah-order flags as skipped are dropped."""
    from similar_verse_eval import merged_rows

    rows = [r for r in load_jsonl(run / f"{name}.jsonl") if not r.get("error")]
    ccv = {str(c["id"]): c["cands"] for c in load_jsonl(sv / f"{name}.jsonl")}
    cca = {str(c["id"]): c["cand"] for c in load_jsonl(ao / f"{name}.jsonl")}
    out = []
    for r in rows:
        eng = list(r.get("issues") or [])
        aof = [f for f in flags_of(cca.get(str(r["id"])), RULE)
               if not any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                          and int(i.get("ayah", 0)) == f["ayah"] for i in eng)]
        merged, _ = merged_rows([r], {str(r["id"]): ccv.get(str(r["id"]), [])}, {**SV_RULE, "all": True})
        issues = merged[0]["issues"]
        if near_skip:
            skips = [(int(i["surah"]), int(i["ayah"])) for i in eng if i.get("kind") == "possible_skipped_ayah"]
            skips += [(f["surah"], f["ayah"]) for f in aof]
            issues = [i for i in issues if i.get("source") != "similar_verse"
                      or not any(s == i["surah"] and abs(a - i["ayah"]) <= 1 for s, a in skips)]
        for f in aof:
            if any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                   and int(i.get("ayah", 0)) == f["ayah"] for i in issues):
                continue
            issues.append(f)
        out.append({**r, "issues": issues})
    return out


def stack(args) -> dict:
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    labels = [x for x in load_jsonl(Path(args.labels)) if x.get("split") == "dev" and x.get("mapped") and not x.get("error")]
    res: dict = {}
    rows_by: dict[str, list[dict]] = {}
    for name in SETS:
        if args.sdk:
            rows_by[name] = [r for r in load_jsonl(Path(args.sdk) / f"{name}.jsonl") if not r.get("error")]
        else:
            rows_by[name] = _lab_stack(Path(args.run), Path(args.sv), Path(args.ao), name, args.near_skip)
    for name, rows in rows_by.items():
        c = Counter()
        mins = sum(float(r.get("duration_s") or 0) for r in rows) / 60
        for r in rows:
            for i in r["issues"]:
                if i.get("kind") in ERROR_KINDS:
                    c[i.get("source") or "engine"] += 1
        res[name] = {"takes": len(rows), "min": round(mins, 1), **dict(c)}
    # confirmatory recall (confirm_synth.score definitions)
    for fam, pos in (("skip", "skip"), ("sister", "sister")):
        hit = eng = n = wrong = 0
        for r in rows_by[pos]:
            m = meta[str(r["id"])]
            if fam == "skip":
                ok = lambda i, m=m: int(i.get("surah", 0)) == m["surah"] and int(i.get("ayah", 0)) == m["ayah"]  # noqa
            else:
                ok = lambda i, m=m: (int(i.get("surah", 0)) == m["surah"] and int(i.get("ayah", 0)) == m["ayah"]  # noqa
                                     and m["word"] - 1 <= int(i.get("word", -9)) <= m["word"] + m["span"])
            errs = [i for i in r["issues"] if i.get("kind") in ERROR_KINDS]
            n += 1
            hit += any(ok(i) for i in errs)
            eng += any(ok(i) for i in errs if i.get("source") not in ("ayah_order", "similar_verse"))
            wrong += sum(1 for i in errs if i.get("source") in ("ayah_order", "similar_verse") and not ok(i))
        res[f"{fam}_recall"] = f"{hit}/{n} (engine {eng}/{n}, wrong rule flags {wrong})"
    # acted dev skip_ayah / substitution / skip_word caught by any error-kind issue (acted_eval.hits)
    from acted_eval import hits

    acted = {str(r["id"]): r for r in rows_by["help-acted"]}
    by = defaultdict(lambda: [0, 0, 0])
    for lab in labels:
        r = acted.get(str(lab["id"]))
        if r is None:
            continue
        errs = [i for i in r["issues"] if i.get("kind") in ERROR_KINDS]
        k = lab["label_kind"]
        by[k][1] += 1
        by[k][0] += any(hits(i, lab) for i in errs)
        by[k][2] += any(hits(i, lab) for i in errs if i.get("source") not in ("ayah_order", "similar_verse"))
    res["acted_dev"] = {k: f"{v[0]}/{v[1]} (engine {v[2]})" for k, v in sorted(by.items())}
    new = sum(1 for r in rows_by["help-acted"] for i in r["issues"]
              if i.get("source") in ("ayah_order", "similar_verse") and i.get("kind") in ERROR_KINDS)
    ok_new = 0
    for r in rows_by["help-acted"]:
        labs = [lab for lab in labels if str(lab["id"]) == str(r["id"])]
        for i in r["issues"]:
            if i.get("source") in ("ayah_order", "similar_verse") and i.get("kind") in ERROR_KINDS:
                ok_new += any(hits(i, lab) for lab in labs)
    res["acted_dev_new_flags_correct"] = f"{ok_new}/{new}"
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("cands", "stack"))
    ap.add_argument("--py", type=Path)
    ap.add_argument("--ts", type=Path)
    ap.add_argument("--cfg", default="nopass")
    ap.add_argument("--run")
    ap.add_argument("--sv")
    ap.add_argument("--ao")
    ap.add_argument("--sdk")
    ap.add_argument("--near-skip", action="store_true")
    ap.add_argument("--meta", default="/tmp/conf/syn/meta.json")
    ap.add_argument("--labels", default="/tmp/correction_eval/acted_located.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "cands":
        print(json.dumps(cands(args.py, args.ts, args.cfg), indent=1))
    else:
        print(json.dumps(stack(args), indent=1))


if __name__ == "__main__":
    main()
