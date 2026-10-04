"""Post-training routine for one arm: export 6 variants, Modal eval, gates,
local tracker on the best gate-passing blend, report with CIs.

Variants: epoch 1 and 2, raw and WiSE-FT blended with v3 at alpha 0.5 / 0.7.

  TILAWA_DATA_ROOT=... ../.venv/bin/python scripts/arm_pipeline.py --arm a0e --run a0e-w2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
PY = sys.executable
MODAL = str(Path(PY).parent / "modal")
V3_PT = "/vol/reference/zipformer_p_arabic_v3.pt"
WORK = Path(os.environ.get("PHASE0_WORK", "/tmp/phase0"))
VARIANTS = [(ep, a) for ep in (1, 2) for a in (1.0, 0.5, 0.7)]


def vname(arm: str, ep: int, a: float) -> str:
    return f"{arm}-ep{ep}" if a == 1.0 else f"{arm}-ep{ep}-a{a}"


def sh(cmd: list[str], log: Path, env: dict | None = None) -> int:
    with log.open("w") as f:
        return subprocess.run(cmd, cwd=LAB, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode


def export_and_eval(arm: str, run: str, ep: int, a: float) -> str:
    name = vname(arm, ep, a)
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rc = sh([MODAL, "run", "scripts/train_zipformer_ctc_modal.py", "--run-name", name,
             "--export-interp", f"{V3_PT}:/vol/exp/{run}/epoch-{ep}.pt:{a}"], logs / f"export-{name}.log")
    if rc:
        raise RuntimeError(f"export {name} failed (rc={rc})")
    rc = sh([MODAL, "run", "scripts/eval_modal.py", "--model-name", name, "--onnx", f"/vol/exports/{name}/model.onnx"],
            logs / f"eval-{name}.log")
    if rc:
        raise RuntimeError(f"eval {name} failed (rc={rc})")
    sh([MODAL, "volume", "get", "zipformer-ctc-training", f"phase0/eval/{name}/", str(WORK / "eval"), "--force"],
       logs / f"get-{name}.log")
    return name


def gates(name: str, extra: bool) -> dict:
    cmd = [PY, "scripts/promotion_gates.py", "--eval-dir", str(WORK / "eval"), "--cand", name, "--base", "v3",
           "--holdout-scores", str(WORK / "holdout_v3.jsonl"), "--qlab-text", "/tmp/qlab",
           "--corpus-manifests", f"heldout={WORK}/heldout_multi,dev={WORK}/dev_everyayah",
           "--json", str(WORK / f"gates_{name}.json")]
    if extra:
        cmd += ["--tracker-dir", str(WORK / "tracker")]
    sh(cmd, WORK / "logs" / f"gates-{name}.log")
    return json.loads((WORK / f"gates_{name}.json").read_text())


def local_checks(name: str) -> None:
    onnx = Path("/tmp/models") / f"{name}.onnx"
    if not onnx.is_file():
        sh([MODAL, "volume", "get", "zipformer-ctc-training", f"exports/{name}/model.onnx", str(onnx), "--force"],
           WORK / "logs" / f"get-onnx-{name}.log")
    env = {"ZIPFORMER_MODEL": str(onnx),
           "ZIPFORMER_CORPUS": os.environ.get("ZIPFORMER_CORPUS", str(LAB / "data" / "zipformer" / "quran.json")),
           "ZIPFORMER_ORT_DIR": str(LAB.parent / "web" / "frontend" / "node_modules")}
    for corpus, tag in ((WORK / "heldout_multi", "heldout_multi"), (WORK / "v1", "v1")):
        sh([PY, "scripts/tracker_corpus_eval.py", "--corpus", str(corpus), "--out", str(WORK / "tracker" / f"{name}_{tag}.jsonl")],
           WORK / "logs" / f"tracker-{name}-{tag}.log", env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run", required=True, help="/vol/exp/<run> with epoch-1.pt and epoch-2.pt")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    variants = [(ep, a) for ep, a in VARIANTS if ep <= args.epochs]
    names = [vname(args.arm, ep, a) for ep, a in variants]
    if not args.skip_export:
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(lambda v: export_and_eval(args.arm, args.run, *v), variants))
    results = {n: gates(n, extra=False) for n in names}
    ok = [n for n in names if results[n]["gates"]["heldout_multi_zero_drops"]]
    best = min(ok or names, key=lambda n: results[n]["metrics"]["cand"]["headline"]["per"])
    passing = [n for n in ok if results[n]["promote"]]
    picks = sorted({best, *(passing[:1])})
    print("local tracker on", picks, flush=True)
    for n in picks:
        local_checks(n)
        gates(n, extra=True)
    sh([PY, "scripts/arm_report.py", "--eval-dir", str(WORK / "eval"), "--gates-dir", str(WORK), "--arm", args.arm,
        "--qlab-text", "/tmp/qlab", "--json", str(WORK / f"report_{args.arm}.json")], WORK / "logs" / f"report-{args.arm}.log")
    print((WORK / "logs" / f"report-{args.arm}.log").read_text())


if __name__ == "__main__":
    main()
