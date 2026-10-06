"""One-shot continuation of the authorized eval after Modal evidence arrives.

Run alongside modal/decode_audio.py --help-eval. Does not start any GPU jobs.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / ".scratch/help"
labels = [json.loads(s) for s in (DIR / "labels.jsonl").read_text().splitlines()]
paths = [DIR / r["wav"].replace(".wav", ".evidence.json") for r in labels if r["split"] == "eval"]
deadline = time.monotonic() + 6 * 60 * 60
previous = -1
while True:
    remaining = sum(not p.exists() for p in paths)
    if remaining != previous:
        print(f"Waiting for teacher evidence: {len(paths) - remaining}/{len(paths)} ready", flush=True)
        previous = remaining
    if remaining == 0:
        break
    if time.monotonic() >= deadline:
        raise SystemExit("Teacher evidence still incomplete after six hours; rerun this continuation after the GPU queue clears.")
    time.sleep(10)
subprocess.run(["python3", str(ROOT / "tools/help_eval_labels.py")], cwd=ROOT, check=True)
subprocess.run(["pnpm", "typecheck"], cwd=ROOT, check=True)
subprocess.run(["node", str(ROOT / "packages/core/dist/bench/help-cli.js"), "--dir", str(DIR),
                "--report", str(ROOT / "docs/benchmarks/help-memorization-v1")], cwd=ROOT, check=True)
