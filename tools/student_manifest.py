# /// script
# requires-python = ">=3.12"
# ///
"""Student training manifest: tlog-v2 clips with their reference phonemes, uploaded for `modal/student_prep.py`.

    uv run --python cpython-3.12-macos-aarch64-none tools/student_manifest.py --build .scratch/tlog/build/tlog-v2 --out .scratch/student/manifest-s1.jsonl
    modal volume put tilawa-student .scratch/student/manifest-s1.jsonl manifest-s1.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import reference_phonemes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ("clean-match", "deviant")
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for bucket in BUCKETS:
            for split in SPLITS:
                for line in open(Path(args.build) / bucket / f"{split}.jsonl"):
                    c = json.loads(line)
                    f.write(json.dumps({
                        "shard": c["shard"],
                        "row": c["row"],
                        "bucket": bucket,
                        "split": split,
                        "cluster": c["cluster"],
                        "per": c["per"],
                        "meanConf": c["meanConf"],
                        "ref": reference_phonemes(ref, c["label"]),
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} clips -> {out}")


if __name__ == "__main__":
    main()
