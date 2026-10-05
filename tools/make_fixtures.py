# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy"]
# ///
"""Convert tlog teacher-pass rows into tracker replay fixtures.

    uv run --python cpython-3.12-macos-aarch64-none tools/make_fixtures.py \
        --parquet /tmp/s0.parquet --rows 816,926,303,483 --out packages/core/test/fixtures/tlog
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

INVENTORY_ID = "muaalem-v3_2-phonemes"
FRAME_MS = 20
TOPK = 5
INDEX = Path(__file__).resolve().parent.parent / "packages/core/assets/ref-v2.json"


def decode(top_ids: np.ndarray, top_p: np.ndarray, tokens: str) -> list[dict]:
    """Greedy CTC collapse with frame spans, matching Muaalem's phoneme decode."""
    best, conf = top_ids[:, 0], top_p[:, 0]
    out, prev = [], 0
    for f, (i, c) in enumerate(zip(best.tolist(), conf.tolist())):
        if i != 0 and i == prev:
            out[-1]["end"] = f
            out[-1]["conf"] = max(out[-1]["conf"], round(c, 4))
        elif i != 0:
            out.append({"p": tokens[i - 1], "start": f, "end": f, "conf": round(c, 4)})
        prev = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--rows", required=True, help="comma-separated `row` values from the pass1 output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", help="review dir: rows with a blind span in its answers.jsonl become audio-verified fixtures")
    args = ap.parse_args()

    spans = {}
    if args.review:
        for line in (Path(args.review) / "answers.jsonl").read_text().splitlines():
            a = json.loads(line)
            spans[a["row"]] = a["span"] if a["verdict"] == "span" else None

    tokens = json.loads(INDEX.read_text())["inventory"]["tokens"]
    wanted = {int(r) for r in args.rows.split(",")}
    table = pq.read_table(args.parquet).to_pylist()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for r in table:
        if r["row"] not in wanted:
            continue
        n = r["n_frames"]
        ids = np.frombuffer(r["top5_ids"], dtype=np.uint8).reshape(n, TOPK)
        probs = np.frombuffer(r["top5_p"], dtype=np.float16).astype(np.float32).reshape(n, TOPK)
        phonemes = decode(ids, probs, tokens)
        if "".join(p["p"] for p in phonemes) != r["hyp"]:
            print(f"skipping row {r['row']}: re-decoding its top-k doesn't reproduce the stored hyp", file=sys.stderr)
            continue
        fixture = {
            "id": f"tlog-{r['shard']}-{r['row']}",
            "provenance": {
                "kind": "tlog",
                "shard": r["shard"],
                "row": r["row"],
                "path": r["path"],
                "filenameLabel": {"surah": r["surah"], "ayah": r["ayah"]},
                "audioVerified": spans.get(r["row"]) is not None,
                **({"reviewNote": "blind review, ticket 07; edges within 1 word"} if spans.get(r["row"]) else {}),
            },
            "evidence": [
                {
                    "inventory": INVENTORY_ID,
                    "frameMs": FRAME_MS,
                    "phonemes": phonemes,
                    "posteriors": {
                        "k": TOPK,
                        "startFrame": 0,
                        "nFrames": n,
                        "ids": ids.flatten().tolist(),
                        "probs": [round(float(p), 4) for p in probs.flatten()],
                    },
                }
            ],
        }
        if spans.get(r["row"]):
            fixture["expect"] = {"state": "locked", "span": spans[r["row"]], "edgeWords": 1}
        (out / f"{fixture['id']}.json").write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")))
        print(fixture["id"], r["path"], len(phonemes), "phonemes")


if __name__ == "__main__":
    main()
