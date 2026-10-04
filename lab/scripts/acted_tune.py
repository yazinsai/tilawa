"""Grid-tune correction thresholds on the dev split by paired trace replay.

Replays every rule set over one model's traces (``correction_eval.py run
--trace``) with ``replay_correction.ts --grid`` and scores it with
``acted_eval.py`` on one split. Writes one aggregate row per rule set to
--out (JSONL, private: no ids). Replay outputs are deleted after scoring.

    ../.venv/bin/python scripts/acted_tune.py --model shipped --split dev \
        --grid gop --out /tmp/correction_eval/tune/shipped_gop.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
INF = 1e309
OLD = {"gopFlag": -INF, "settle": False, "repetitionGain": INF, "omissionMaxHeard": 0}


def _acted_eval():
    spec = importlib.util.spec_from_file_location("acted_eval", LAB / "scripts" / "acted_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _name(th: dict) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float) and v in (INF, -INF):
            return "inf" if v > 0 else "-inf"
        return str(v)
    return ",".join(f"{k}={fmt(v)}" for k, v in sorted(th.items())) or "v2"


def grid(kind: str) -> list[dict]:
    if kind == "base":
        return [OLD, {}, {"gopFlag": -INF}, {"repetitionMode": "note"}, {"gopFlag": -INF, "repetitionMode": "note"}]
    if kind == "gop":
        out = [{"gopFlag": -INF}]
        for flag, anchor, states, kinds, persist, settle_gop, local in itertools.product(
            (-12, -9, -7, -5, -4, -3, -2.5),
            (-2, -1, -0.5, INF),
            ((True, True), (True, False), (False, True)),
            ((True, True), (True, False), (False, True)),
            (12, 24),
            (True, False),
            (True, False),
        ):
            out.append({
                "gopFlag": flag, "gopAnchor": anchor, "gopOnWrong": states[0], "gopOnSkipped": states[1],
                "gopOmission": kinds[0], "gopSubstitution": kinds[1], "gopPersistFrames": persist,
                "settleGop": settle_gop, "gopLocalMin": local,
            })
        return out
    if kind == "other":
        out = []
        for omh, settle, rep_mode, rep_gain, vm, vwm in itertools.product(
            (0, 0.34, 0.5, 1), (True, False), ("off", "note", "flag"), (4, 5, 7, 10),
            (0.05, 0.1, 0.2, 0.4), (0.5, 0.65, 0.8),
        ):
            if rep_mode == "off" and rep_gain != 5:
                continue
            out.append({"omissionMaxHeard": omh, "settle": settle, "repetitionMode": rep_mode,
                        "repetitionGain": rep_gain, "vowelMargin": vm, "vowelWordMargin": vwm})
        return out
    raise SystemExit(f"unknown grid {kind}")


def row_metrics(s: dict) -> dict:
    a, h = s["agg"]["all"], s["headline"]
    ff = {k: v["issues"] for k, v in s["ff"].items()}
    ff_notes = {k: v["notes"] for k, v in s["ff"].items()}
    return {
        "p": h["p"], "r": h["r"], "f1": h["f1"], "tp": a["tp"], "flags": a["flags"], "caught": a["caught"],
        "kind_ok": a["kind_ok"], "exact": a["exact"], "labels": a["labels"], "lat": a["median_latency_s"],
        "ff": ff, "ff_notes": ff_notes, "notes": s["agg"]["notes"],
        "by_kind": {k: [v["caught"], v["kind_ok"], v["n"]] for k, v in s["agg"]["by_kind"].items()},
        "by_issue": {k: [v["tp"], v["flags"]] for k, v in s["agg"]["by_issue"].items() if v["flags"]},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="trace tag under --traces")
    ap.add_argument("--traces", type=Path, default=Path("/tmp/correction_eval/traces"))
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--grid", default="gop", help="base | gop | other | path to a JSON list of thresholds")
    ap.add_argument("--fixed", default="{}", help="JSON thresholds merged under every grid point")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/tmp/correction_eval/tune_work"))
    args = ap.parse_args(argv)
    ev = _acted_eval()
    labels = ev.load_labels(Path("/tmp/correction_eval/acted_located.jsonl"))
    speakers = ev.load_speakers(Path("/tmp/help/manifest.json"))
    fixed = json.loads(args.fixed)
    points = json.loads(Path(args.grid).read_text()) if Path(args.grid).is_file() else grid(args.grid)
    points = [{**fixed, **p} for p in points]
    tsx = LAB.parent / "web" / "frontend" / "node_modules" / ".bin" / "tsx"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for start in range(0, len(points), args.chunk):
            chunk = points[start:start + args.chunk]
            work = args.work / args.model
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            spec = [{"name": f"g{start + i}", "thresholds": th} for i, th in enumerate(chunk)]
            # JSON has no Infinity; 1e309 parses to it in JS.
            (work / "grid.json").write_text(json.dumps(spec).replace("Infinity", "1e309"), encoding="utf-8")
            subprocess.run(
                [str(tsx), "--tsconfig", str(LAB.parent / "web" / "frontend" / "tsconfig.json"),
                 str(LAB / "experiments" / "zipformer-ctc" / "replay_correction.ts"),
                 "--in", str(args.traces / args.model), "--out", str(work), "--grid", str(work / "grid.json")],
                check=True, cwd=LAB,
            )
            for item in spec:
                s = ev.summarize(ev.collect(work / item["name"], labels, args.split, speakers), ci=False)
                sink.write(json.dumps({"name": _name(item["thresholds"]), "thresholds": item["thresholds"],
                                       **row_metrics(s)}, default=str).replace("Infinity", "1e309") + "\n")
            sink.flush()
            print(f"{args.model} {min(start + args.chunk, len(points))}/{len(points)}", file=sys.stderr, flush=True)
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
