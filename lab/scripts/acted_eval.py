"""Score correction rules on acted help mistakes plus the clean sets.

Private data only. Inputs are replay or live outputs under
``/tmp/correction_eval`` (one dir per rule set with ``help-acted.jsonl``,
``help-clean.jsonl``, ``v1.jsonl``, ``tlog-dev.jsonl``) and the labels from
``correction_eval.py locate-acted``. Acted takes may be used privately for
tuning and eval but never published: commit aggregate numbers only, never
ids, labels or per-clip rows.

Scoring
-------
A flag *hits* a label when it is on the labelled surah:ayah and its word is
within ±1 of the labelled word (``skip_ayah``: anywhere in that ayah).
Kind-correct: skip_word → possible_omission, substitution →
possible_substitution, vowel → possible_vowel, repeat → possible_repetition,
skip_ayah → possible_skipped_ayah / unclear_ayah. Tajweed is out of the
engine's scope (it never grades tajweed): its recall is reported, it is left
out of the headline recall, and a flag on a tajweed word is a hit.

Precision counts every error flag on the split's acted takes and clean help
takes. A flag that hits no label is a false alarm, even on an acted take.
Soft notes (``correction_note``) are counted apart from error flags.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

IN_SCOPE = ("skip_word", "substitution", "vowel", "skip_ayah", "repeat")
ALL_KINDS = IN_SCOPE + ("tajweed",)
KIND_OK = {
    "skip_word": {"possible_omission"},
    "substitution": {"possible_substitution"},
    "vowel": {"possible_vowel"},
    "repeat": {"possible_repetition"},
    "skip_ayah": {"possible_skipped_ayah", "unclear_ayah"},
    "tajweed": set(),
}
ISSUE_KINDS = ("possible_omission", "possible_substitution", "possible_vowel", "possible_repetition",
               "possible_skipped_ayah", "unclear_ayah")
CLEAN_SETS = ("help-clean", "tlog-dev", "v1")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def hits(issue: dict, label: dict) -> bool:
    try:
        if int(issue["surah"]) != int(label["surah"]) or int(issue["ayah"]) != int(label["ayah"]):
            return False
        if label["label_kind"] == "skip_ayah":
            return True
        return abs(int(issue["word"]) - int(label["word_index"])) <= 1
    except (KeyError, TypeError, ValueError):
        return False


def exact(issue: dict, label: dict) -> bool:
    if label["label_kind"] == "skip_ayah":
        return hits(issue, label)
    return hits(issue, label) and int(issue["word"]) == int(label["word_index"])


def clip_rows(run_dir: Path, name: str) -> dict[str, dict]:
    return {str(r["id"]): r for r in load_jsonl(run_dir / f"{name}.jsonl") if not r.get("error")}


def acted_clip_stats(label: dict, row: dict) -> dict:
    issues = list(row.get("issues") or [])
    notes = list(row.get("notes") or [])
    hit_issues = [i for i in issues if hits(i, label)]
    hit_notes = [n for n in notes if hits(n, label)]
    span = label.get("span_s")
    lat = None
    if hit_issues and span:
        times = [float(i["atSeconds"]) for i in hit_issues if i.get("atSeconds") is not None]
        lat = (min(times) - float(span[0])) if times else None
    return {
        "kind": label["label_kind"],
        "caught": bool(hit_issues),
        "kind_ok": any(i.get("kind") in KIND_OK[label["label_kind"]] for i in hit_issues),
        "exact": any(exact(i, label) for i in hit_issues),
        "note_caught": bool(hit_notes),
        "latency": lat,
        "flags": [(i.get("kind"), hits(i, label)) for i in issues],
        "notes": [(n.get("kind"), hits(n, label)) for n in notes],
    }


def collect(run_dir: Path, labels: list[dict], split: str, speakers: dict[str, str]) -> dict:
    """Per-clip units for one rule set and split, ready to aggregate or resample."""
    acted = clip_rows(run_dir, "help-acted")
    units = []
    for label in labels:
        if label.get("split") != split or not label.get("mapped") or label.get("error"):
            continue
        row = acted.get(str(label["id"]))
        if row is None:
            continue
        stat = acted_clip_stats(label, row)
        stat["speaker"] = speakers.get(str(label["id"]), str(label["id"]))
        stat["minutes"] = float(row.get("duration_s") or 0.0) / 60.0
        units.append(stat)
    clean = []
    for row in clip_rows(run_dir, "help-clean").values():
        if row.get("split") != split:
            continue
        clean.append({
            "speaker": speakers.get(str(row["id"]), str(row["id"])),
            "flags": [(i.get("kind"), False) for i in row.get("issues") or []],
            "notes": [(n.get("kind"), False) for n in row.get("notes") or []],
            "minutes": float(row.get("duration_s") or 0.0) / 60.0,
        })
    ff = {}
    for name in CLEAN_SETS:
        rows = [r for r in clip_rows(run_dir, name).values() if name != "help-clean" or r.get("split") == split]
        minutes = sum(float(r.get("duration_s") or 0.0) for r in rows) / 60.0
        n_issues = sum(len(r.get("issues") or []) for r in rows)
        n_notes = sum(len(r.get("notes") or []) for r in rows)
        ff[name] = {"clips": len(rows), "minutes": minutes, "issues": n_issues, "notes": n_notes,
                    "per_minute": (n_issues / minutes) if minutes else None}
    return {"acted": units, "clean": clean, "ff": ff}


def aggregate(units: list[dict], clean: list[dict]) -> dict:
    out: dict = {"by_kind": {}}
    for kind in ALL_KINDS:
        rows = [u for u in units if u["kind"] == kind]
        lats = [u["latency"] for u in rows if u["latency"] is not None]
        caught = sum(u["caught"] for u in rows)
        out["by_kind"][kind] = {
            "n": len(rows),
            "caught": caught,
            "kind_ok": sum(u["kind_ok"] for u in rows),
            "exact": sum(u["exact"] for u in rows),
            "note_caught": sum(u["note_caught"] for u in rows),
            "median_latency_s": statistics.median(lats) if lats else None,
        }
    flags = [f for u in units + clean for f in u["flags"]]
    notes = [f for u in units + clean for f in u["notes"]]
    by_issue = {}
    for kind in ISSUE_KINDS:
        chosen = [hit for k, hit in flags if k == kind]
        by_issue[kind] = {"flags": len(chosen), "tp": sum(chosen)}
    out["by_issue"] = by_issue
    out["notes"] = {"n": len(notes), "tp": sum(hit for _k, hit in notes)}
    scope = [u for u in units if u["kind"] in IN_SCOPE]
    tp = sum(hit for _k, hit in flags)
    out["all"] = {
        "labels": len(scope),
        "caught": sum(u["caught"] for u in scope),
        "kind_ok": sum(u["kind_ok"] for u in scope),
        "exact": sum(u["exact"] for u in scope),
        "flags": len(flags),
        "tp": tp,
        "acted_minutes": sum(u["minutes"] for u in units),
        "clean_minutes": sum(u["minutes"] for u in clean),
        "clean_flags": sum(len(u["flags"]) for u in clean),
    }
    lats = [u["latency"] for u in scope if u["latency"] is not None]
    out["all"]["median_latency_s"] = statistics.median(lats) if lats else None
    return out


def prf(tp_flags: int, flags: int, caught: int, n: int) -> dict:
    p = tp_flags / flags if flags else None
    r = caught / n if n else None
    f1 = (2 * p * r / (p + r)) if p is not None and r is not None and (p + r) > 0 else (0.0 if r is not None else None)
    return {"p": p, "r": r, "f1": f1}


def headline(agg: dict) -> dict:
    a = agg["all"]
    return prf(a["tp"], a["flags"], a["caught"], a["labels"])


def kind_prf(agg: dict, kind: str) -> dict:
    """P over flags of the label kind's issue kinds, R kind-correct."""
    row = agg["by_kind"][kind]
    issue_kinds = KIND_OK[kind]
    flags = sum(agg["by_issue"][k]["flags"] for k in issue_kinds)
    tp = sum(agg["by_issue"][k]["tp"] for k in issue_kinds)
    if kind == "repeat":
        flags += agg["notes"]["n"]
        tp += agg["notes"]["tp"]
        caught = max(row["kind_ok"], row["note_caught"])
    else:
        caught = row["kind_ok"]
    return prf(tp, flags, caught, row["n"])


def bootstrap(data: dict, fn, reps: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """Speaker-cluster bootstrap of ``fn(aggregate)`` (acted and clean units resampled together)."""
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for u in data["acted"]:
        groups[u["speaker"]].append(("a", u))
    for u in data["clean"]:
        groups[u["speaker"]].append(("c", u))
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        acted, clean = [], []
        for key in (rng.choice(keys) for _ in keys):
            for tag, u in groups[key]:
                (acted if tag == "a" else clean).append(u)
        v = fn(aggregate(acted, clean))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def paired_bootstrap(a: dict, b: dict, fn, reps: int = 2000, seed: int = 0) -> tuple[float, float, float] | None:
    """CI of fn(b) - fn(a) on the same speakers (same clips, two rule sets)."""
    def by_speaker(data):
        g: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for u in data["acted"]:
            g[u["speaker"]].append(("a", u))
        for u in data["clean"]:
            g[u["speaker"]].append(("c", u))
        return g

    ga, gb = by_speaker(a), by_speaker(b)
    keys = sorted(set(ga) | set(gb))
    rng = random.Random(seed)

    def agg(g, picks):
        acted, clean = [], []
        for key in picks:
            for tag, u in g.get(key, []):
                (acted if tag == "a" else clean).append(u)
        return fn(aggregate(acted, clean))

    point = agg(gb, keys) - agg(ga, keys)
    diffs = []
    for _ in range(reps):
        picks = [rng.choice(keys) for _ in keys]
        va, vb = agg(ga, picks), agg(gb, picks)
        if va is not None and vb is not None:
            diffs.append(vb - va)
    diffs.sort()
    return (point, diffs[int(0.025 * len(diffs))], diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))])


def summarize(data: dict, ci: bool = True) -> dict:
    agg = aggregate(data["acted"], data["clean"])
    out = {"agg": agg, "headline": headline(agg), "kinds": {k: kind_prf(agg, k) for k in ALL_KINDS}, "ff": data["ff"]}
    if ci:
        a = agg["all"]
        out["ci"] = {
            "p": wilson(a["tp"], a["flags"]),
            "r": wilson(a["caught"], a["labels"]),
            "f1": bootstrap(data, lambda g: headline(g)["f1"]),
            "kinds": {k: {"r": wilson(agg["by_kind"][k]["kind_ok"], agg["by_kind"][k]["n"]),
                          "f1": bootstrap(data, lambda g, k=k: kind_prf(g, k)["f1"], reps=500)} for k in ALL_KINDS},
        }
    return out


def load_labels(path: Path) -> list[dict]:
    return load_jsonl(path)


def load_speakers(manifest: Path) -> dict[str, str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {str(s["id"]): str(s["speaker"]) for s in data["samples"]}


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def render(rows: list[tuple[str, dict]]) -> str:
    lines = ["| rule set | P | R | F1 | kind-R | exact | lat s | flags | help clean FF | TLOG dev FF | v1 FF |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        a, h, ff = s["agg"]["all"], s["headline"], s["ff"]
        lines.append(
            f"| {name} | {_f(h['p'])} | {_f(h['r'])} | {_f(h['f1'])} | {a['kind_ok']}/{a['labels']} | {a['exact']} | "
            f"{_f(a['median_latency_s'], 2)} | {a['flags']} | {_f(ff['help-clean']['per_minute'])} ({ff['help-clean']['issues']}) | "
            f"{_f(ff['tlog-dev']['per_minute'])} ({ff['tlog-dev']['issues']}) | {_f(ff['v1']['per_minute'])} ({ff['v1']['issues']}) |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="name=dir,... (replay or live output dirs)")
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--out", type=Path, required=True, help="JSON summary (private)")
    ap.add_argument("--no-ci", action="store_true")
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    speakers = load_speakers(args.manifest)
    rows = []
    for part in args.runs.split(","):
        name, d = part.split("=", 1)
        rows.append((name, summarize(collect(Path(d), labels, args.split, speakers), ci=not args.no_ci)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows), indent=2, default=list) + "\n", encoding="utf-8")
    print(render(rows))


if __name__ == "__main__":
    main()
