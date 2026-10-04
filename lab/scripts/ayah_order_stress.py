"""Clean false-flag stress test and external eval for the combined structural stack.

Stack = engine rules (main defaults) + similar-verse (frozen) + ayah-order (frozen),
all label-free. Private data only (rows and clip lists under /tmp); prints aggregates.

    ../.venv/bin/python scripts/ayah_order_stress.py clips --out /tmp/aos/clips
    ../.venv/bin/python scripts/ayah_order_stress.py ff --run /tmp/aos/run/a0w-nopass --sv /tmp/aos/sv/a0w-nopass \\
        --ao /tmp/aos/ao/a0w-nopass --sv-all
    ../.venv/bin/python scripts/ayah_order_stress.py external --run ... --sv ... --ao ... --sv-all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acted_eval import load_jsonl  # noqa: E402
from ayah_order import RULE, flags_of  # noqa: E402
from unseen_scoreboard import ERROR_KINDS  # noqa: E402

SV_RULE = {"families": ["lookalike", "drop"], "e": {"default": 3.5, "drop": 10.0}, "max_span": 1,
           "loc": 0.25, "ayah_d": 0.5}


def clips(out: Path, everyayah: Path, ext: Path, verdicts: Path, tlog_audio: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((everyayah / "manifest.json").read_text(encoding="utf-8"))
    rows = man["samples"] if isinstance(man, dict) else man
    with (out / "everyayah.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "audio": str(everyayah / r["file"]), "split": None,
                                 "speaker": r.get("reciter"), "expected_verses": r.get("expected_verses") or []})
                     + "\n")
    labels = load_jsonl(ext / "labels.jsonl")
    seen = set()
    sob = open(out / "sobolev.jsonl", "w", encoding="utf-8")
    ret = open(out / "retasy.jsonl", "w", encoding="utf-8")
    for r in labels:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        h = r["id"].split("-", 1)[1]
        ev = [{"surah": int(r["surah"]), "ayah": int(r["ayah"])}] if r.get("surah") and r.get("ayah") else []
        if r["source"] == "sobolev":
            sob.write(json.dumps({"id": r["id"], "audio": str(ext / "sobolev" / "audio" / f"{h}.wav"), "split": None,
                                  "speaker": r.get("cluster"), "expected_verses": ev}) + "\n")
        else:
            ret.write(json.dumps({"id": r["id"], "audio": str(ext / "retasy" / "in_correct" / f"{h}.wav"),
                                  "split": None, "speaker": r.get("cluster"), "expected_verses": ev}) + "\n")
    sob.close()
    ret.close()
    with (out / "tlog-qc.jsonl").open("w", encoding="utf-8") as fh:
        for v in load_jsonl(verdicts):
            if v.get("pool") != "tlog_clean_dev" or v.get("bucket") != "usable":
                continue
            ev = [{"surah": int(v["surah"]), "ayah": a} for a in range(int(v["ayah"]), int(v["ayah_end"]) + 1)]
            fh.write(json.dumps({"id": v["id"], "audio": str(tlog_audio / f"{v['id']}.flac"), "split": None,
                                 "expected_verses": ev}) + "\n")
    for name in ("everyayah", "sobolev", "retasy", "tlog-qc"):
        print(name, sum(1 for _ in open(out / f"{name}.jsonl")))


def stack_rows(run: Path, sv: Path | None, ao: Path | None, name: str, sv_all: bool, ao_rule: dict) -> list[dict]:
    """Engine rows with similar-verse then ayah-order flags added, each deduplicated as in its own eval."""
    from similar_verse_eval import merged_rows

    rows = [r for r in load_jsonl(run / f"{name}.jsonl") if not r.get("error")]
    if sv is not None and (sv / f"{name}.jsonl").is_file():
        cc = {str(c["id"]): c["cands"] for c in load_jsonl(sv / f"{name}.jsonl")}
        rows, _ = merged_rows(rows, cc, {**SV_RULE, "all": sv_all})
    if ao is not None and (ao / f"{name}.jsonl").is_file():
        cand = {str(c["id"]): c["cand"] for c in load_jsonl(ao / f"{name}.jsonl")}
        out = []
        for r in rows:
            issues = list(r.get("issues") or [])
            for f in flags_of(cand.get(str(r["id"])), ao_rule):
                if any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                       and int(i.get("ayah", 0)) == f["ayah"] for i in issues):
                    continue
                issues.append(f)
            out.append({**r, "issues": issues})
        rows = out
    return rows


def poisson_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) interval for a Poisson count."""
    from scipy.stats import chi2

    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi


def source_of(name: str, r: dict) -> str | None:
    if name == "help-clean":
        return f"help clean {r.get('split')}"
    return {"v1": "v1", "everyayah": "EveryAyah dev", "tlog-qc": "TLOG clean dev, QC-usable"}.get(name)


def ff(run: Path, sv: Path | None, ao: Path | None, sv_all: bool, ao_rule: dict, ext_labels: Path) -> dict:
    agg: dict[str, dict] = defaultdict(lambda: {"clips": 0, "min": 0.0, "engine": 0, "sv": 0, "ao": 0,
                                                "clips_flagged": 0, "ao_clips": set()})
    hafs_clean = {r["id"] for r in load_jsonl(ext_labels) if r["source"] == "sobolev" and r.get("riwayah") == "hafs"
                  and r.get("role") == "clean"}
    for name in ("help-clean", "v1", "everyayah", "sobolev", "tlog-qc"):
        if not (run / f"{name}.jsonl").is_file():
            continue
        for r in stack_rows(run, sv, ao, name, sv_all, ao_rule):
            src = source_of(name, r)
            if name == "sobolev":
                if str(r["id"]) not in hafs_clean:
                    continue
                src = "external clean Hafs (sobolev)"
            if src is None:
                continue
            a = agg[src]
            a["clips"] += 1
            a["min"] += float(r.get("duration_s") or 0) / 60
            errs = [i for i in r["issues"] if i.get("kind") in ERROR_KINDS]
            a["clips_flagged"] += int(bool(errs))
            for i in errs:
                s = i.get("source")
                a["sv" if s == "similar_verse" else "ao" if s == "ayah_order" else "engine"] += 1
    out = {}
    for src, a in sorted(agg.items()):
        tot = a["engine"] + a["sv"] + a["ao"]
        new = a["sv"] + a["ao"]
        lo, hi = poisson_ci(tot)
        nlo, nhi = poisson_ci(new)
        m = max(a["min"], 1e-9)
        out[src] = {"clips": a["clips"], "minutes": round(a["min"], 1), "engine": a["engine"], "sv": a["sv"],
                    "ao": a["ao"], "total_per_min": round(tot / m, 4), "total_ci": [round(lo / m, 4), round(hi / m, 4)],
                    "new_per_min": round(new / m, 4), "new_ci": [round(nlo / m, 4), round(nhi / m, 4)]}
    return out


def external(run: Path, sv: Path | None, ao: Path | None, sv_all: bool, ao_rule: dict, ext_labels: Path) -> dict:
    """Read-only external harness metrics: sobolev Hafs word detection (same ayah, +-1 word),
    RetaSy in_correct clip hit (any error-kind issue)."""
    labels = load_jsonl(ext_labels)
    res: dict = {}
    sob = {str(r["id"]): r for r in stack_rows(run, sv, ao, "sobolev", sv_all, ao_rule)} \
        if (run / "sobolev.jsonl").is_file() else {}
    det = defaultdict(lambda: [0, 0])
    for lab in labels:
        if lab["source"] != "sobolev" or lab.get("riwayah") != "hafs" or lab.get("role") != "error" \
                or not lab.get("kind") or lab.get("word_index") is None:
            continue
        r = sob.get(str(lab["id"]))
        if r is None:
            continue
        hit = any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == int(lab["surah"])
                  and int(i.get("ayah", 0)) == int(lab["ayah"]) and abs(int(i.get("word", -9)) - int(lab["word_index"])) <= 1
                  for i in r["issues"])
        for k in (lab["kind"], "all"):
            det[k][0] += int(hit)
            det[k][1] += 1
    res["sobolev_hafs_word"] = {k: f"{c}/{n}" for k, (c, n) in sorted(det.items())}
    by_src = defaultdict(int)
    for r in sob.values():
        for i in r["issues"]:
            if i.get("kind") in ERROR_KINDS:
                by_src[i.get("source") or "engine"] += 1
    res["sobolev_issues_by_source"] = dict(by_src)
    if (run / "retasy.jsonl").is_file():
        ret = stack_rows(run, sv, ao, "retasy", sv_all, ao_rule)
        hit = defaultdict(int)
        for r in ret:
            srcs = {i.get("source") or "engine" for i in r["issues"] if i.get("kind") in ERROR_KINDS}
            hit["any"] += int(bool(srcs))
            hit["engine"] += int("engine" in srcs)
            hit["new_only"] += int(bool(srcs) and "engine" not in srcs)
        res["retasy_in_correct"] = {**{k: f"{v}/{len(ret)}" for k, v in hit.items()}}
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("clips", "ff", "external"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--everyayah", type=Path, default=Path("/tmp/phase0/dev_everyayah"))
    ap.add_argument("--ext", type=Path, default=Path("/tmp/external_eval"))
    ap.add_argument("--verdicts", type=Path, default=Path("/tmp/tlog_meta/verdicts.jsonl"))
    ap.add_argument("--tlog-audio", type=Path, default=Path("/tmp/correction_eval/audio/tlog"))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--sv", type=Path)
    ap.add_argument("--ao", type=Path)
    ap.add_argument("--sv-all", action="store_true")
    ap.add_argument("--ao-rule", help="JSON override of ayah_order.RULE")
    args = ap.parse_args(argv)
    if args.cmd == "clips":
        clips(args.out, args.everyayah, args.ext, args.verdicts, args.tlog_audio)
        return
    rule = {**RULE, **json.loads(args.ao_rule)} if args.ao_rule else RULE
    fn = ff if args.cmd == "ff" else external
    print(json.dumps(fn(args.run, args.sv, args.ao, args.sv_all, rule, args.ext / "labels.jsonl"), indent=1))


if __name__ == "__main__":
    main()
