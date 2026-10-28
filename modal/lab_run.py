"""Run a bench command over shards on Modal against this worktree's source (.scratch/lab, ticket 09).

    modal run modal/lab_run.py --shards 4 --cmd 'pnpm --filter @tilawa/core exec vitest run --shard=$((LAB_SHARD+1))/$LAB_SHARDS'

Usually called through `tools/lab.py`, which sets LAB_SRC to the worktree to score (default: this file's checkout).
Each shard unpacks the worktree's tracked and untracked (non-ignored) files over
an image with the workspace dependencies installed, then runs `cmd` under bash in /app with:

    LAB_RUN     run id: <name>-<sha>[-dirty]-<utc time>
    LAB_SHARD   0-based shard index; LAB_SHARDS shard count
    LAB_DATA    /lab/data on the tilawa-lab volume (test sets, evidence; read-only by convention)
    LAB_OUT     /lab/runs/<run>/<shard>, where the command writes its results

Each shard's output tail and exit code are saved as LAB_OUT/log.txt and LAB_OUT/status.json.
"""

import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(os.environ.get("LAB_SRC") or Path(__file__).resolve().parent.parent)
PNPM = "pnpm@10.18.0"
FILTER = "--filter @tilawa/core... --filter @tilawa/sdk..."
MANIFESTS = ["package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", *(str(p.relative_to(ROOT)) for p in sorted(ROOT.glob("packages/*/package.json")))]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "git")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash -", "apt-get install -y nodejs", f"corepack enable && corepack prepare {PNPM} --activate")
)
for m in MANIFESTS:
    image = image.add_local_file(ROOT / m, f"/app/{m}", copy=True)
image = image.run_commands(f"cd /app && CI=1 pnpm install --frozen-lockfile {FILTER}")

app = modal.App("tilawa-lab-run", image=image)
vol = modal.Volume.from_name("tilawa-lab", create_if_missing=True)
lock_hash = lambda: hashlib.sha256(Path("/app/pnpm-lock.yaml").read_bytes()).hexdigest()


@app.function(cpu=2, memory=4096, volumes={"/lab": vol}, timeout=30 * 60)
def shard(src: bytes, cmd: str, run: str, i: int, n: int, out_dir: str = "") -> dict:
    baked = lock_hash()
    with tarfile.open(fileobj=io.BytesIO(src)) as t:
        t.extractall("/app", filter="data")
    if lock_hash() != baked:
        subprocess.run(f"cd /app && CI=1 pnpm install --frozen-lockfile {FILTER}", shell=True, check=True)
    vol.reload()
    out = Path(out_dir or f"/lab/runs/{run}/{i}")
    out.mkdir(parents=True, exist_ok=True)
    Path("/lab/data").mkdir(exist_ok=True)
    env = {**os.environ, "LAB_RUN": run, "LAB_SHARD": str(i), "LAB_SHARDS": str(n), "LAB_DATA": "/lab/data", "LAB_OUT": str(out)}
    t0 = time.time()
    p = subprocess.run(["bash", "-c", cmd], cwd="/app", env=env, capture_output=True, text=True)
    log = p.stdout + p.stderr
    status = {"shard": i, "code": p.returncode, "seconds": round(time.time() - t0, 1), "tail": log[-2000:]}
    (out / "log.txt").write_text(log)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    vol.commit()
    return status


def source() -> tuple[bytes, str]:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = [f for f in git("ls-files", "-co", "--exclude-standard").splitlines() if f in MANIFESTS or f.startswith(("packages/", "tools/")) and "/node_modules/" not in f]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in files:
            if (ROOT / f).is_file():
                t.add(ROOT / f, arcname=f)
    sha = git("rev-parse", "--short", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    return buf.getvalue(), sha + ("-dirty" if dirty else "")


@app.local_entrypoint()
def main(cmd: str, shards: int = 1, name: str = "adhoc", merge: str = "", run_file: str = ""):
    """`merge`: a command run once after every shard succeeds (LAB_OUT=/lab/runs/<run>/merge). `run_file`: local path that gets the run id."""
    src, sha = source()
    run = f"{name}-{sha}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if run_file:
        Path(run_file).write_text(run)
    print(f"run {run}: {shards} shard(s), source {len(src) / 1e6:.1f} MB")
    results = sorted(shard.starmap([(src, cmd, run, i, shards) for i in range(shards)]), key=lambda s: s["shard"])
    for s in results:
        print(f"shard {s['shard']}: exit {s['code']} in {s['seconds']} s")
        if s["code"]:
            print(s["tail"])
    if merge and not any(s["code"] for s in results):
        m = shard.remote(src, merge, run, 0, 1, f"/lab/runs/{run}/merge")
        print(f"merge: exit {m['code']} in {m['seconds']} s")
        if m["code"]:
            print(m["tail"])
        results.append(m)
    print(f"results: tilawa-lab volume /runs/{run}/")
    if any(s["code"] for s in results):
        raise SystemExit(1)
