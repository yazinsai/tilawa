# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy"]
# ///
"""Export tlog pass1 rows as JSONL tracker evidence for the relabel script.

    uv run --python cpython-3.12-macos-aarch64-none tools/export_pass1.py \
        --parquet /tmp/s0.parquet --out /tmp/s0.pass1.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixtures import FRAME_MS, INDEX, INVENTORY_ID, TOPK, decode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-posteriors", action="store_true", help="omit top-k posteriors to shrink the output")
    args = ap.parse_args()

    tokens = json.loads(INDEX.read_text())["inventory"]["tokens"]
    rows = sorted(pq.read_table(args.parquet).to_pylist(), key=lambda r: r["row"])
    mismatched = 0
    with open(args.out, "w") as f:
        for r in rows:
            evidence = None
            if r["status"] == "ok":
                n = r["n_frames"]
                ids = np.frombuffer(r["top5_ids"], dtype=np.uint8).reshape(n, TOPK)
                probs = np.frombuffer(r["top5_p"], dtype=np.float16).astype(np.float32).reshape(n, TOPK)
                phonemes = decode(ids, probs, tokens)
                # `max` and `topk` can break near-ties differently; keep the top-k decode, which has frame spans.
                mismatched += "".join(p["p"] for p in phonemes) != r["hyp"]
                evidence = {"inventory": INVENTORY_ID, "frameMs": FRAME_MS, "phonemes": phonemes}
                if not args.no_posteriors:
                    evidence["posteriors"] = {
                        "k": TOPK,
                        "startFrame": 0,
                        "nFrames": n,
                        "ids": ids.flatten().tolist(),
                        "probs": [round(float(p), 4) for p in probs.flatten()],
                    }
            clip = {
                "shard": r["shard"],
                "row": r["row"],
                "path": r["path"],
                "status": r["status"],
                "filenameLabel": {"surah": r["surah"], "ayah": r["ayah"]} if r["surah"] else None,
                "evidence": evidence,
                "teacher": {"meanConf": r["mean_conf"], "durationS": r["duration_s"]} if r["status"] == "ok" else None,
            }
            f.write(json.dumps(clip, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(rows)} rows -> {args.out} ({mismatched} top-k decodes differ from stored hyp)")


if __name__ == "__main__":
    main()
