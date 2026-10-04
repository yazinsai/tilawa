"""Download a clip-id list of TLOG flacs from the private volume. Prints counts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--remote-prefix", default="audio/tlog")
    ap.add_argument("--suffix", default=".flac")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    ids = [ln.strip() for ln in args.ids.read_text(encoding="utf-8").splitlines() if ln.strip()]
    args.dest.mkdir(parents=True, exist_ok=True)
    missing = [cid for cid in ids if not (args.dest / f"{cid}{args.suffix}").is_file()]
    print(f"have={len(ids) - len(missing)} fetch={len(missing)}", flush=True)
    if not missing:
        return
    vol = modal.Volume.from_name("zipformer-ctc-training")

    def one(cid: str) -> None:
        data = bytearray()
        remote = f"{args.remote_prefix}/{cid}{args.suffix}"
        for chunk in vol.read_file(remote):
            data.extend(chunk)
        if not data:
            raise FileNotFoundError(cid)
        (args.dest / f"{cid}{args.suffix}").write_bytes(data)

    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, cid): cid for cid in missing}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception:
                fail += 1
            if done % 500 == 0 or done == len(futures):
                print(f"fetch {done}/{len(futures)} fail={fail}", flush=True)
    if fail:
        raise SystemExit(f"failed {fail}")


if __name__ == "__main__":
    main()
