# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy"]
# ///
"""Export QUD teacher-pass recordings as tracker benchmark inputs.

    modal volume get tilawa-bench qud .scratch/bench/
    uv run --python cpython-3.12-macos-aarch64-none tools/export_qud.py --dir .scratch/bench/qud
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixtures import INDEX, INVENTORY_ID, TOPK, decode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    tokens = json.loads(INDEX.read_text())["inventory"]["tokens"]
    for path in sorted(Path(args.dir).glob("*.parquet")):
        r = pq.read_table(path).to_pylist()[0]
        n = r["n_frames"]
        ids = np.frombuffer(r["top5_ids"], dtype=np.uint8).reshape(n, TOPK)
        probs = np.frombuffer(r["top5_p"], dtype=np.float16).astype(np.float32).reshape(n, TOPK)
        out = {
            "id": r["id"],
            "meta": {k: r[k] for k in ("reciter", "style", "content", "noisy", "passages", "multi_surah", "duration_s", "recited_words", "repeat_events")},
            "inventory": INVENTORY_ID,
            # The model's true stride, measured from audio length over frames: 40 ms.
            "frameMs": round(r["frame_ms"]),
            "nFrames": n,
            "phonemes": decode(ids, probs, tokens),
            "posteriors": {"k": TOPK, "ids": ids.flatten().tolist(), "probs": [round(float(p), 4) for p in probs.flatten()]},
            "truth": json.loads(r["truth"]),
            "segments": json.loads(r["segments"]),
        }
        dest = path.with_suffix(".json")
        dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
        print(f"{r['id']}: {len(out['phonemes'])} phonemes, {n} frames -> {dest}")


if __name__ == "__main__":
    main()
