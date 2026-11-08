"""Serve the exact folded fp32 insertion teacher; primary inference stays in the browser."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "/lab/data/models/exp-listener-teacher-insert-marginal-gate-ad6eb8c0"
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    import lab_run
    image = lab_run.image
    image = image.add_local_file(ROOT / "packages/core/tsconfig.json", "/app/packages/core/tsconfig.json", copy=True)
    for directory in ("packages/core/src", "packages/core/assets", "packages/sdk/src"):
        image = image.add_local_dir(ROOT / directory, "/app/" + directory, copy=True)
    image = image.add_local_file(ROOT / "tools/live_teacher_worker.ts", "/app/tools/live_teacher_worker.ts", copy=True)
    image = image.run_commands("cd /app && pnpm --filter @tilawa/core build")
app = modal.App("tilawa-live-teacher", image=image)
vol = modal.Volume.from_name("tilawa-lab")


@app.cls(cpu=4, memory=12288, volumes={"/lab": vol}, timeout=600,
         max_containers=2, scaledown_window=300)
class Teacher:
    @modal.enter()
    def load(self):
        for filename, expected in (
            ("teacher.onnx", "5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18"),
            ("teacher.onnx.data", "d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4"),
        ):
            with open(Path(BUNDLE) / filename, "rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Folded teacher hash mismatch: {filename}")
        self.worker = subprocess.Popen(
            ["pnpm", "--filter", "@tilawa/sdk", "exec", "tsx", "../../tools/live_teacher_worker.ts", BUNDLE + "/teacher.onnx"],
            cwd="/app", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        if json.loads(self.worker.stdout.readline()) != {"ready": True}:
            raise RuntimeError("Teacher worker failed to start")

    @modal.method()
    def decode(self, audio: bytes, k: int):
        if not 1280 <= len(audio) <= 16000 * 4 * 60 or len(audio) % 4 or not 1 <= k <= 64:
            raise ValueError("Invalid teacher crop")
        self.worker.stdin.write(json.dumps({"audio": base64.b64encode(audio).decode(), "k": k}) + "\n")
        self.worker.stdin.flush()
        result = json.loads(self.worker.stdout.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["evidence"]

    @modal.exit()
    def close(self):
        self.worker.terminate()
