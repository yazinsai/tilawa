# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy", "huggingface_hub"]
# ///
"""Local side of the full tlog pass: pull pass1 outputs, relabel every shard, and gather split inputs.

    T="uv run --python cpython-3.12-macos-aarch64-none tools/tlog_corpus.py"
    $T pull --dir .scratch/tlog                       # pass1/*.parquet from the tilawa-tlog volume
    $T relabel --dir .scratch/tlog --jobs 8           # labels/<shard>.jsonl (decoded-only, ref-v2)
    $T holdout --dir .scratch/tlog                    # exclude-stems.txt: Quran-Lab tlog_holdout
    $T clusters --dir .scratch/tlog --file clusters/clusters-t0.3-c20.parquet
"""

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOLUME = "tilawa-tlog"


def pull(d: Path) -> None:
    (d / "pass1").mkdir(parents=True, exist_ok=True)
    names = subprocess.run(["modal", "volume", "ls", VOLUME, "pass1"], capture_output=True, text=True, check=True).stdout.split()
    todo = [n for n in names if n.endswith(".parquet") and not (d / n).exists()]
    print(f"{len(names)} on the volume, {len(todo)} to fetch")

    def get(n: str) -> None:
        for attempt in range(4):
            r = subprocess.run(["modal", "volume", "get", VOLUME, n, str(d / n), "--force"], capture_output=True, text=True)
            if r.returncode == 0:
                return
        raise RuntimeError(f"{n}: {r.stderr[-500:]}")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(get, todo))


def relabel_one(d: Path, parquet: Path) -> str:
    name = parquet.stem
    dest = d / "labels" / f"{name}.jsonl"
    if dest.exists():
        return f"{name} skipped"
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "pass1.jsonl"
        subprocess.run([sys.executable, str(ROOT / "tools/export_pass1.py"), "--parquet", str(parquet), "--out", str(jsonl), "--no-posteriors"], check=True, capture_output=True)
        subprocess.run(["pnpm", "--silent", "--filter", "@tilawa/core", "relabel", str(jsonl), "--out", tmp], check=True, capture_output=True, cwd=ROOT)
        (Path(tmp) / "proposed-labels.jsonl").replace(dest)
    return name


def relabel(d: Path, jobs: int) -> None:
    (d / "labels").mkdir(parents=True, exist_ok=True)
    shards = sorted((d / "pass1").glob("*.parquet"))
    with ThreadPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(lambda p: relabel_one(d, p), shards), 1):
            if i % 25 == 0 or i == len(shards):
                print(f"{i}/{len(shards)} {r}", flush=True)


def holdout(d: Path) -> None:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files("Quran-Lab/quranic-asr-benchmark", repo_type="dataset")
    stems = sorted(Path(f).stem for f in files if f.startswith("audio/tlog_holdout/") and f.endswith(".wav"))
    (d / "exclude-stems.txt").write_text("\n".join(stems) + "\n")
    print(f"{len(stems)} tlog_holdout stems -> {d / 'exclude-stems.txt'}")


def clusters(d: Path, file: str) -> None:
    import pyarrow.parquet as pq

    local = d / Path(file).name
    subprocess.run(["modal", "volume", "get", VOLUME, file, str(local), "--force"], check=True, capture_output=True)
    t = pq.read_table(local)
    with open(d / "clusters.tsv", "w") as f:
        for s, r, c in zip(t.column("shard").to_pylist(), t.column("row").to_pylist(), t.column("cluster").to_pylist()):
            f.write(f"{s}\t{r}\t{c}\n")
    print(f"{t.num_rows} clips -> {d / 'clusters.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "relabel", "holdout", "clusters"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--file")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pull":
        pull(d)
    elif a.cmd == "relabel":
        relabel(d, a.jobs)
    elif a.cmd == "holdout":
        holdout(d)
    else:
        clusters(d, a.file)


if __name__ == "__main__":
    main()
