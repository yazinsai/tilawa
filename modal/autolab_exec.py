"""Builders' Modal work, run for them by the supervisor (.scratch/autolab/spec.md decision 14).

    modal deploy modal/autolab_exec.py                       # from the main checkout, whenever this file changes
    python modal/autolab_exec.py run '<request json>'         # one builder command over shards (a tools/autolab.py job)
    python modal/autolab_exec.py sync '<spec json>'           # refresh tilawa-lab-exec, the practice-only data copy
    python modal/autolab_exec.py sync-help                     # replace only help-train/ (help recordings, train speakers)
    python modal/autolab_exec.py copy-model <exp> <path> <dst>   # tilawa-lab-scratch /<exp>/<path> -> tilawa-lab /data/<dst>

A builder's command runs in a Modal Sandbox, never in a function container: code in a function container talks to Modal
as the workspace (all it needs is MODAL_IS_REMOTE=1 and the server URL from /proc/1/environ), a Sandbox has no Modal
identity. A sandbox unpacks the worktree's files over /app (the lab_run image, or `train`: student_train's dependencies)
and mounts:

    /lab/data       tilawa-lab-exec, read-only: practice sessions of the current test set, their evidence, the main model
    /scratch        tilawa-lab-scratch: /scratch/<exp> persists across an experiment's runs; LAB_OUT=/scratch/<exp>/runs/<n>/<shard>
    /student        `train` only: tilawa-student's training data read-only (s1, aug, eval, prior-runs); /student/runs is in scratch

tilawa-lab is never mounted in a sandbox. `run` prints shard tails, copies LAB_OUT back to <worktree>/.autolab/modal-out-<n>/
(unless "download" is false) and ends with a JSON line {"result": {...}, "exitCode": n}.
"""

import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "modal"), str(ROOT / "tools")]
import autolab_isolation as I  # noqa: E402

APP, LAB, EXEC, SCRATCH, STUDENT = "tilawa-autolab-exec", "tilawa-lab", "tilawa-lab-exec", "tilawa-lab-scratch", "tilawa-student"
HELP = "tilawa-help"  # the help-page recordings; only their train speakers reach tilawa-lab-exec
MAX_FILE, MAX_MODEL, MAX_DOWNLOAD = 10 << 20, 4 << 30, 500 << 20

app = modal.App(APP)
lab_vol = modal.Volume.from_name(LAB)
exec_vol = modal.Volume.from_name(EXEC, create_if_missing=True)
scratch_vol = modal.Volume.from_name(SCRATCH, create_if_missing=True)
help_vol = modal.Volume.from_name(HELP)
fn_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("autolab_isolation")

ENTRY = r"""set -u
mkdir -p "$LAB_OUT"
[ -f /app/pnpm-lock.yaml ] && cp /app/pnpm-lock.yaml /tmp/baked.lock
mkdir -p /app && tar xzf "$LAB_SRC_TAR" -C /app || exit 96
cd /app
if command -v pnpm >/dev/null && ! cmp -s pnpm-lock.yaml /tmp/baked.lock; then
  CI=1 pnpm install --frozen-lockfile $LAB_FILTER >"$LAB_OUT/install.txt" 2>&1 || { tail -c 2000 "$LAB_OUT/install.txt"; exit 97; }
fi
if [ -d /student-data ]; then
  mkdir -p /student "/scratch/$AUTOLAB_EXP/student-runs"
  for f in /student-data/*; do [ "$(basename "$f")" = runs ] || ln -sfn "$f" /student/; done
  ln -sfn "/scratch/$AUTOLAB_EXP/student-runs" /student/runs
  ln -sfn /student-data/runs /student/prior-runs
  ln -sf /app/docs/benchmarks/gold-per-v1-exclude.txt /root/gold-per-v1-exclude.txt
fi
t0=$(date +%s)
bash -c "$LAB_CMD" >"$LAB_OUT/log.txt" 2>&1
code=$?
echo "{\"code\": $code, \"seconds\": $(( $(date +%s) - t0 ))}" >"$LAB_OUT/status.json"
tail -c 2000 "$LAB_OUT/log.txt"
exit $code
"""


def images() -> dict:
    """Built from the main checkout (cached by content); a sandbox whose lockfile differs reinstalls, as lab_run does."""
    import lab_run  # noqa: PLC0415 - builds from local files; only the local side needs it
    import student_train as T  # noqa: PLC0415
    train = (  # student_train's layers (so cached) without its baked tools: the worktree's own come from /app
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libsndfile1", "ffmpeg")
        .pip_install("torch==2.8.0", "torchaudio==2.8.0", T.K2)
        .pip_install("lhotse", "kaldialign", "sentencepiece", "tensorboard", "pypinyin==0.50.0", "typeguard", "dill",
                     "pyarrow", "soundfile", "numpy", "scipy", "huggingface_hub", "urllib3", "requests")
        .run_commands(f"git clone https://github.com/k2-fsa/icefall /icefall && cd /icefall && git checkout {T.ICEFALL}")
        .pip_install("rapidfuzz==3.14.1", "modal==1.3.4", "onnx==1.19.0", "onnxruntime==1.22.1")
    )
    return {"node": lab_run.image, "train": train}


def source(root: Path) -> bytes:
    """The worktree's tracked and untracked (non-ignored) files that code needs, each under MAX_FILE."""
    import lab_run  # noqa: PLC0415
    keep = lab_run.manifests(root)
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in files:
            p = root / f
            if (f in keep or f.startswith(("packages/", "tools/", "modal/", "docs/"))) and "/node_modules/" not in f \
                    and p.is_file() and not p.is_symlink() and p.stat().st_size <= MAX_FILE:
                t.add(p, arcname=f)
    return buf.getvalue()


def run(q: dict) -> dict:
    """One request: {exp, n, worktree, cmd, shards, gpu, image, timeout, download}."""
    import lab_run  # noqa: PLC0415
    exp, n, wt = q["exp"], str(q["n"]), Path(q["worktree"])
    out = f"/{exp}/runs/{n}"
    tar = f"/{exp}/.src/{n}.tgz"
    with scratch_vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(source(wt)), tar)
    gpu = q.get("gpu")
    volumes = {"/lab/data": exec_vol.read_only(), "/scratch": scratch_vol}
    if q["image"] == "train":
        volumes["/student-data"] = modal.Volume.from_name(STUDENT).read_only()
    env = {"LAB_CMD": q["cmd"], "LAB_SRC_TAR": f"/scratch{tar}", "LAB_DATA": "/lab/data", "LAB_FILTER": lab_run.FILTER,
           "LAB_RUN": f"{exp}-{n}", "LAB_SHARDS": str(q["shards"]), "AUTOLAB_EXP": exp, "HF_HOME": f"/scratch/{exp}/.hf"}
    if q["image"] == "train":
        env["PYTHONPATH"] = "/app/tools:/app/modal:/icefall:/icefall/egs/librispeech/ASR/zipformer"
    sandbox_app = modal.App.lookup(APP, create_if_missing=True)
    image = images()[q["image"]]
    boxes = []

    def stop(*_):
        for sb in boxes:
            with contextlib.suppress(Exception):
                sb.terminate()
        sys.exit(143)

    signal.signal(signal.SIGTERM, stop)
    t0 = time.time()
    for i in range(q["shards"]):
        boxes.append(modal.Sandbox.create("bash", "-c", ENTRY, app=sandbox_app, image=image, gpu=gpu, timeout=q["timeout"] * 60,
                                          cpu=8 if gpu else 2, memory=32768 if gpu else 4096, workdir="/", volumes=volumes,
                                          env={**env, "LAB_SHARD": str(i), "LAB_OUT": f"/scratch{out}/{i}"}))
        print(f"shard {i}: sandbox {boxes[-1].object_id}", file=sys.stderr, flush=True)
    shards = []
    for i, sb in enumerate(boxes):
        sb.wait(raise_on_termination=False)
        code = sb.returncode
        shards.append({"shard": i, "code": 124 if code is None else code, "seconds": round(time.time() - t0, 1),
                       "tail": (sb.stdout.read() or "")[-2000:] + (sb.stderr.read() or "")[-500:]})
    seconds = round(time.time() - t0, 1)
    local = None
    if q.get("download", True):
        local = wt / ".autolab" / f"modal-out-{n}"
        total = 0
        for e in scratch_vol.listdir(out, recursive=True):
            if e.type.name != "FILE" or total + e.size > MAX_DOWNLOAD:
                continue
            total += e.size
            dst = local / e.path.removeprefix(out.lstrip("/")).lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in scratch_vol.read_file(e.path):
                    f.write(chunk)
    return {"code": max((s["code"] for s in shards), default=0), "shards": shards, "seconds": seconds, "gpu": gpu,
            "out": f"/scratch{out}", "local": str(local) if local else None}


@app.function(image=fn_image, volumes={"/src": lab_vol.read_only(), "/help": help_vol.read_only(), "/dst": exec_vol}, cpu=2, memory=4096, timeout=60 * 60)
def sync(spec: dict) -> dict:
    """tilawa-lab-exec becomes the practice-only copy of one test set: {base, manifest, practice, listeners, models},
    plus the help recordings' train speakers in help-train/. Nothing else survives in it, so sessions that are exam in
    this version can't linger from an older one."""
    dst, stage = Path("/dst"), Path("/dst/.staging")
    shutil.rmtree(stage, ignore_errors=True)
    out = I.copy_practice(Path("/src/data"), stage, spec)
    out["help"] = I.copy_help_train(Path("/help"), stage)
    for old in dst.iterdir():
        if old != stage:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    for new in stage.iterdir():
        new.rename(dst / new.name)
    stage.rmdir()
    exec_vol.commit()
    return out


@app.function(image=fn_image, volumes={"/help": help_vol.read_only(), "/dst": exec_vol}, cpu=2, memory=4096, timeout=30 * 60)
def sync_help() -> dict:
    """Only help-train/ replaced, the practice data left as it is (builds may be reading it)."""
    stage = Path("/dst/.staging-help")
    shutil.rmtree(stage, ignore_errors=True)
    out = I.copy_help_train(Path("/help"), stage)
    shutil.rmtree(Path("/dst") / I.HELP_DIR, ignore_errors=True)
    (stage / I.HELP_DIR).rename(Path("/dst") / I.HELP_DIR)
    stage.rmdir()
    exec_vol.commit()
    return out


@app.function(image=fn_image, volumes={"/scratch": scratch_vol.read_only(), "/lab": lab_vol}, cpu=1, timeout=30 * 60)
def copy_model(exp: str, path: str, dst: str) -> dict:
    """A builder's exported model, regular files only, into tilawa-lab where the supervisor's decode reads it."""
    if not dst.startswith("models/exp-") or "/" in dst[len("models/"):]:
        raise ValueError(f"model destination {dst} is not models/exp-*")
    src = Path("/scratch") / exp / I.model_path({"path": path}, exp)
    if src.is_symlink() or not (src / "manifest.json").is_file():
        raise FileNotFoundError(f"no manifest.json in {src}")
    out = Path("/lab/data") / dst
    shutil.rmtree(out, ignore_errors=True)
    total, files = 0, []
    for d, dirs, names in os.walk(src):
        for name in dirs + names:
            if (Path(d) / name).is_symlink():
                raise ValueError(f"{Path(d, name).relative_to(src)} is a symlink")
        for name in names:
            f = Path(d) / name
            total += f.stat().st_size
            if total > MAX_MODEL:
                raise ValueError(f"model over {MAX_MODEL >> 30} GB")
            (out / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out / f.relative_to(src))
            files.append(str(f.relative_to(src)))
    lab_vol.commit()
    return {"dst": dst, "files": files, "bytes": total}


def main(argv: list[str]) -> int:
    op = argv[0] if argv else ""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            if op == "run":
                res = run(json.loads(argv[1]))
                for s in res["shards"]:
                    print(f"shard {s['shard']}: exit {s['code']} in {s['seconds']} s\n{s['tail']}")
                out = {"result": res, "exitCode": res["code"]}
            elif op == "sync":
                out = modal.Function.from_name(APP, "sync").remote(json.loads(argv[1]))
            elif op == "sync-help":
                out = modal.Function.from_name(APP, "sync_help").remote()
            elif op == "copy-model":
                out = modal.Function.from_name(APP, "copy_model").remote(*argv[1:4])
            else:
                print(__doc__)
                return 2
        except Exception as e:  # noqa: BLE001 - the supervisor reads the JSON line, not a traceback
            out = {"error": f"{type(e).__name__}: {e}"[-2000:]}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 1 if "error" in out or out.get("exitCode") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
