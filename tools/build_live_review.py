"""Build the live page from main and package its exact registered listener for Modal.

    python3 tools/build_live_review.py
    modal deploy modal/lab_review.py

Only the browser UI is built locally; inference runs on the reviewing device.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main":
        raise SystemExit("Build from main so the live page matches the main-line engine")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "packages/core/src", "packages/sdk/src"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("Commit or isolate engine changes before building a main-line live page")
    subprocess.run(["pnpm", "--filter", "@tilawa/sample", "exec", "vite", "build", "--config", "vite.live.config.ts"], cwd=ROOT, check=True)
    dest = ROOT / "modal/live-dist"
    registry = json.loads((ROOT / "packages/sdk/src/models-v1.json").read_text())
    tier = next(t for t in registry["tiers"] if t["tier"] == "full")
    model = ROOT / ".scratch/models" / (tier["id"] + "-" + tier["version"].split("-", 1)[1]) / tier["manifest"]["file"]
    if hashlib.sha256(model.read_bytes()).hexdigest() != tier["sha256"]:
        raise SystemExit("Local model does not match main's registered SHA256")
    model_dest = dest / tier["base"]
    model_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, model_dest / tier["manifest"]["file"])
    ort = dest / "ort"
    ort.mkdir()
    for p in (ROOT / "packages/sample/node_modules/onnxruntime-web/dist").glob("ort-wasm-simd-threaded.jsep.*"):
        shutil.copy2(p, ort / p.name)
    if not list(ort.glob("*.wasm")):
        raise SystemExit("Missing ONNX Runtime browser assets")
    print(f"Live UI ready: {dest}")


if __name__ == "__main__":
    main()
