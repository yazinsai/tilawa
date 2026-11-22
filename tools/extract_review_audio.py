# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "huggingface_hub"]
# ///
"""Write the audio of the sampled review clips to <dir>/audio/<row>.flac.

    uv run --python cpython-3.12-macos-aarch64-none tools/extract_review_audio.py \
        --dir .scratch/bookmark-tracker/review --shard clean-00000-of-00411

With --pack, clips come from many shards: <dir>/rows.tsv lists `shard<TAB>row`,
and audio goes to <dir>/audio/<shard>-<row>.flac.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--shard")
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()

    out = Path(args.dir) / "audio"
    out.mkdir(parents=True, exist_ok=True)
    if args.pack:
        by: dict[str, list[int]] = {}
        for line in (Path(args.dir) / "rows.tsv").read_text().split("\n"):
            if line:
                shard, row = line.split("\t")
                by.setdefault(shard, []).append(int(row))
        for shard, rows in sorted(by.items()):
            parquet = hf_hub_download("tarteel-ai/tlog", f"data/{shard}.parquet", repo_type="dataset")
            audio = pq.read_table(parquet, columns=["audio"]).column("audio")
            for r in rows:
                (out / f"{shard}-{r}.flac").write_bytes(audio[r].as_py()["bytes"])
        print(f"{sum(map(len, by.values()))} clips from {len(by)} shards -> {out}")
        return
    if not args.shard:
        ap.error("--shard is required without --pack")
    rows = [int(r) for r in (Path(args.dir) / "rows.txt").read_text().split()]
    parquet = hf_hub_download("tarteel-ai/tlog", f"data/{args.shard}.parquet", repo_type="dataset")
    audio = pq.read_table(parquet, columns=["audio"]).column("audio")
    for r in rows:
        (out / f"{r}.flac").write_bytes(audio[r].as_py()["bytes"])
    print(f"{len(rows)} clips -> {out}")


if __name__ == "__main__":
    main()
