"""Tracker accuracy (native engine via zipformer-ctc harness) on a v3-schema corpus dir.

Scores `predict()`'s emitted verse list against `expected_verses`:
ordered exact match (SeqAcc), set exact match, recall, and ayahs missing.
Per-clip JSONL goes to --out (outside the repo).

  ZIPFORMER_MODEL=/tmp/models/v3.onnx ZIPFORMER_ORT_DIR=../web/frontend/node_modules \\
    ../.venv/bin/python scripts/tracker_corpus_eval.py --corpus /tmp/phase0/heldout_multi --out /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))


def load_run():
    spec = importlib.util.spec_from_file_location("zipformer_run", LAB / "experiments" / "zipformer-ctc" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(expected: list[tuple[int, int]], got: list[tuple[int, int]]) -> dict:
    dedup = []
    for v in got:
        if v not in dedup:
            dedup.append(v)
    exp_set, got_set = set(expected), set(dedup)
    return {
        "seq_ok": dedup == expected,
        "set_ok": exp_set == got_set,
        "recall": len(exp_set & got_set) / max(len(exp_set), 1),
        "missing": sorted(exp_set - got_set),
        "extra": sorted(got_set - exp_set),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if str(out).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out must be outside the repo")
    run = load_run()
    corpus = Path(args.corpus)
    man = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    samples = man["samples"] if isinstance(man, dict) else man
    if args.limit:
        samples = samples[: args.limit]
    rows = []
    for smp in samples:
        exp = [(v["surah"], v["ayah"]) for v in smp["expected_verses"]]
        pred = run.predict(str(corpus / smp["file"]))
        got = [tuple(v) for v in pred.get("verses") or []]
        if not got and pred.get("surah"):
            end = pred.get("ayah_end") or pred["ayah"]
            got = [(pred["surah"], a) for a in range(pred["ayah"], end + 1)]
        rows.append({"id": smp["id"], "bridge": smp.get("bridge"), "expected": exp, "got": got, **score(exp, got)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    n = len(rows)
    summ = {
        "n": n,
        "seq_acc": sum(r["seq_ok"] for r in rows),
        "set_acc": sum(r["set_ok"] for r in rows),
        "recall": round(sum(r["recall"] for r in rows) / max(n, 1), 4),
        "ayahs_missing": sum(len(r["missing"]) for r in rows),
        "ayahs_extra": sum(len(r["extra"]) for r in rows),
    }
    print(json.dumps(summ))
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2) + "\n")


if __name__ == "__main__":
    main()
