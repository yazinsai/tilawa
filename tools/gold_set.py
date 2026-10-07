# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1", "pyarrow"]
# ///
"""Gold PER set: human-verified exact recitations, scored as canon PER vs the spliced reference (teacher-per-breakdown-v1).

    G="uv run --python cpython-3.12-macos-aarch64-none tools/gold_set.py"
    $G build                                  # docs/benchmarks/gold-per-v1.json + gold-per-v1-exclude.txt
    modal run modal/student_decode_cpu.py::gold --run e2-libri-full   # .scratch/gold/<run>/<id>.evidence.json
    $G score                                  # docs/benchmarks/gold-per-v1-scores.json

Gold: ticket 06 bucket-review clips judged `exact`, and ticket 12 help clips with audit verdict 1 (label right) and kind
`none`. Silver: the other clean, scored help eval clips, never audited. Audio stays in `.scratch`.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from per_breakdown import VOWELS, canon, spliced  # noqa: E402
from student import reference_phonemes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path(os.environ.get("MAIN_CHECKOUT", ROOT))
REVIEW = MAIN / ".scratch/v1-roadmap/bucket-review"
HELP = Path(os.environ.get("HELP_DIR", MAIN / ".scratch/help"))
TLOG = MAIN / ".scratch/tlog"
DECODES = MAIN / ".scratch/gold"
OUT = ROOT / "docs/benchmarks/gold-per-v1"
LISTENERS = ("teacher", "e5-top1-lite", "e5-soft5-lite", "e2-scratch-full", "e2-libri-full")
# Scored but not in the listener-bar fit, which stays on the five listeners it was calibrated on.
SCORED = LISTENERS + ("combo-libri-soft5-full", "combo-clean", "spliced-clean")

GROUPS = {
    "pharyngeals": set("حع"),
    "emphatics": set("صضطظ"),
    "hamza_ain": set("ءع"),
    "short_vowels": VOWELS,
}


def group_edits(ref: str, hyp: str) -> dict[str, int]:
    """Canon edits touching each group: the reference phoneme of a sub/del, the hypothesis phoneme of an ins or sub."""
    out = dict.fromkeys(GROUPS, 0)
    for op, i, j in Levenshtein.editops(ref, hyp):
        touched = {ref[i]} if op == "delete" else {hyp[j]} if op == "insert" else {ref[i], hyp[j]}
        for g, members in GROUPS.items():
            out[g] += bool(touched & members)
    return out


def score(rows: list[dict], hyps: dict[str, str]) -> dict:
    """Gold PER of one listener: canon (primary) and raw PER vs each row's spliced reference, by device and phoneme group."""
    rows = [r for r in rows if r["id"] in hyps]
    tot = defaultdict(lambda: [0, 0, 0, 0, 0])
    groups = {g: [0, 0] for g in GROUPS}
    for r in rows:
        h, ref = hyps[r["id"]], r["spliced"]
        ch, cr = canon(h), canon(ref)
        e = [Levenshtein.distance(ch, cr), len(cr), Levenshtein.distance(h, ref), len(ref), 1]
        for k in ("all", r["device"]):
            tot[k] = [a + b for a, b in zip(tot[k], e)]
        for g, n in group_edits(cr, ch).items():
            groups[g][0] += n
            groups[g][1] += sum(c in GROUPS[g] for c in cr)

    def per(t):
        return {"canon": t[0] / max(1, t[1]), "raw": t[2] / max(1, t[3]), "clips": t[4], "refPhonemes": t[1]}

    return {
        "clips": len(rows),
        "overall": per(tot["all"]),
        "byDevice": {k: per(v) for k, v in sorted(tot.items()) if k != "all"},
        "groups": {g: {"per": e / max(1, n), "edits": e, "phonemes": n} for g, (e, n) in groups.items()},
    }


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares slope, intercept and Pearson r of (PER, metric) points."""
    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    syy = sum((y - my) ** 2 for _, y in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    b = sxy / sxx
    return b, my - b * mx, sxy / (sxx * syy) ** 0.5 if syy else 0.0


def per_bar(points: list[tuple[float, float]], teacher_per: float, drop: float) -> float | None:
    """Gold PER at which the fitted end-to-end metric sits `drop` below its fitted value at the teacher's PER."""
    b, _, _ = fit(points)
    return teacher_per + drop / -b if b < 0 else None


def jsonl(p: Path) -> list[dict]:
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def tlog_teacher(keys: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    import pyarrow.parquet as pq

    out = {}
    for shard in sorted({s for s, _ in keys}):
        t = pq.read_table(TLOG / "pass1" / f"{shard}.parquet", columns=["row", "hyp"]).to_pydict()
        out.update({(shard, r): h for r, h in zip(t["row"], t["hyp"]) if (shard, r) in keys})
    return out


def build() -> None:
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    ayah = {f"{a['s']}:{a['a']}": a["p"] for a in ref["ayahs"]}

    verdict = {(a["shard"], a["row"]): a["verdict"] for a in jsonl(REVIEW / "answers.jsonl")}
    exact = [c for c in json.loads((REVIEW / "sample.json").read_text())["clips"] if verdict.get((c["shard"], c["row"])) == "exact"]
    split = {}
    for f in (TLOG / "build/tlog-v2").glob("*/*.jsonl"):
        for c in jsonl(f):
            split[(c["shard"], c["row"])] = f"{f.parent.name}/{f.stem}"
    teacher = tlog_teacher({(c["shard"], c["row"]) for c in exact})
    gold = []
    for c in exact:
        k = (c["shard"], c["row"])
        r = reference_phonemes(ref, c["label"])
        gold.append(dict(id=f"{c['shard']}:{c['row']}", source="tlog-bucket-review", device="tlog", speaker=f"tlog-cluster-{c['cluster']}",
                         tlogSplit=split.get(k), label=c["label"], audio=f"bucket-review/audio/{c['shard']}-{c['row']}.flac",
                         ref=r, teacher=teacher[k], spliced=spliced(r, teacher[k])))

    labels = {l["id"]: l for l in jsonl(HELP / "labels.jsonl")}
    audited = {a["id"]: a for a in jsonl(HELP / "answers.jsonl")}
    passage = json.loads((HELP / "passage.json").read_text())

    def help_row(l: dict, source: str) -> dict:
        ev = HELP / l["wav"].replace(".wav", ".evidence.json")
        t = "".join(p["p"] for p in json.loads(ev.read_text())["phonemes"]) if ev.exists() else passage[l["id"]]["hyp"]
        r = "".join(ayah[a] for a in l["ayahs"])
        return dict(id=l["id"], source=source, device=l.get("device", "unknown"), speaker=l["speaker"], helpSplit=l["split"],
                    ayahs=l["ayahs"], audio=l["wav"], ref=r, teacher=t, spliced=spliced(r, t))

    gold += [help_row(labels[a["id"]], "help-audited") for a in audited.values() if a["verdict"] == "1" and a["kind"] == "none"]
    silver = [help_row(l, "help-eval-clean") for l in labels.values()
              if l["kind"] == "none" and l["split"] == "eval" and l["scored"] and l["id"] not in audited]
    out = {
        "version": "v1",
        "metric": "canon PER vs spliced reference (teacher-per-breakdown-v1, tools/per_breakdown.py); raw PER vs spliced secondary",
        "devices": "tlog = tarteel-ai/tlog crowd app recordings (device unknown, mostly phones); help = ticket 12 phone / laptop",
        "slices": {"gold": gold, "silver": silver},
    }
    OUT.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    excl = sorted(r["id"] for r in gold if r["source"] == "tlog-bucket-review")
    (OUT.parent / "gold-per-v1-exclude.txt").write_text("\n".join(excl) + "\n")
    print(f"gold {len(gold)}, silver {len(silver)}, tlog exclusions {len(excl)}")


def decoded(listener: str, rows: list[dict]) -> dict[str, str]:
    if listener == "teacher":
        return {r["id"]: r["teacher"] for r in rows}
    d = DECODES / listener
    out = {}
    for r in rows:
        p = d / f"{r['id'].replace(':', '-')}.evidence.json"
        if p.exists():
            out[r["id"]] = "".join(x["p"] for x in json.loads(p.read_text())["phonemes"])
    return out


def score_all() -> None:
    m = json.loads(OUT.with_suffix(".json").read_text())
    gold, silver = m["slices"]["gold"], m["slices"]["silver"]
    heldout = [r for r in gold if r["source"] != "tlog-bucket-review" or not (r["tlogSplit"] or "").endswith("/train")]
    res = {}
    for name in (n for n in SCORED if n == "teacher" or (DECODES / n).is_dir()):
        res[name] = {"gold": score(gold, decoded(name, gold)), "goldHeldOut": score(heldout, decoded(name, heldout)),
                     "silver": score(silver, decoded(name, silver))}
        g = res[name]["gold"]
        print(f"{name:16s} gold {g['clips']} canon {100 * g['overall']['canon']:.2f}% raw {100 * g['overall']['raw']:.2f}%")
    (OUT.parent / "gold-per-v1-scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


CALIBRATED = {
    "declared.recall": ("checker.oracleLock.recall", "Declared-arm memorization recall"),
    "declared.precision": ("checker.oracleLock.precision", "Declared-arm memorization precision"),
    "short_vowel.recall": (None, "E1 help clips: short vowel greedy recall"),
}
DROP = 0.02


def calibrate(board: Path) -> None:
    """Fit each end-to-end metric against gold PER over teacher + students; the listener bar is the tightest crossing."""
    scores = json.loads((OUT.parent / "gold-per-v1-scores.json").read_text())
    rows = {}
    for r in json.loads(board.read_text())["rows"]:
        m = {x["id"]: x for x in r["metrics"]}
        if r["listener"] in LISTENERS and m.get("checker.oracleLock.recall", {}).get("value") is not None:
            rows[r["listener"]] = (r["sha"], m)
    points = {}
    for name in LISTENERS:
        sv = next(c for c in json.loads((DECODES / "e1" / f"{name}.json").read_text())["classes"] if c["name"] == "short vowel")
        sha, m = rows[name]
        points[name] = {"goldPer": scores[name]["gold"]["overall"]["canon"], "scoreboardSha": sha,
                        "declared.recall": m["checker.oracleLock.recall"]["value"], "declared.precision": m["checker.oracleLock.precision"]["value"],
                        "short_vowel.recall": sv["greedy_hit"] / sv["pos"], "short_vowel.n": sv["pos"]}
    teacher = points["teacher"]["goldPer"]
    fits = {}
    for k, (_, label) in CALIBRATED.items():
        pts = [(p["goldPer"], p[k]) for p in points.values()]
        b, a, r = fit(pts)
        fits[k] = {"name": label, "slopePerPerPoint": b / 100, "intercept": a, "r": r, "bar": per_bar(pts, teacher, DROP)}
    binding = min((f["bar"], k) for k, f in fits.items() if f["bar"] is not None and f["r"] <= -0.5)
    out = {"version": "v1", "drop": DROP, "teacherGoldPer": teacher, "points": points, "fits": fits,
           "bar": binding[0], "bindingMetric": binding[1]}
    (OUT.parent / "listener-target-v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "points"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "calibrate"])
    ap.add_argument("--board", default=str(ROOT / "docs/benchmarks/scoreboard.json"))
    a = ap.parse_args()
    build() if a.cmd == "build" else score_all() if a.cmd == "score" else calibrate(Path(a.board))
