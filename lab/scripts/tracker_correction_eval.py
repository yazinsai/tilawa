"""Parallel correction-mode runs for tracker work (acted + clean sets).

Private data only: rows (ids, issues, timelines) go under --out, outside the
repo. Commit aggregates only.

Spawns ``--workers`` harness processes (1 intra-op thread each) and writes one
JSONL per set in the ``acted_eval.py`` row format (``id``, ``duration_s``,
``split``, ``issues``, ``notes``) plus ``verses``, ``events`` and, with
``--diag``, the per-chunk tracker timeline.

``--lp record`` stores each clip's model log_probs under ``--lp-cache``;
``--lp replay`` re-runs the session on them with no ONNX, so engine variants
are compared on identical acoustic output. Both keep the encoder running across
idle / dismiss resets, which the live session (``--lp live``) restarts.

    ../.venv/bin/python scripts/tracker_correction_eval.py --sets help-acted,help-clean,tlog-dev,v1 \\
        --split dev --lp record --lp-cache /tmp/lp/shipped --out /tmp/tc/base --diag
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
REPO = LAB.parent
sys.path.insert(0, str(LAB))


def _correction_eval():
    spec = importlib.util.spec_from_file_location("correction_eval", LAB / "scripts" / "correction_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["correction_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def expected_of(clip: dict) -> dict | None:
    """One-surah passage spanning the clip's expected verses, else None."""
    verses = clip.get("expected_verses") or []
    surahs = {int(v["surah"]) for v in verses}
    if len(surahs) != 1:
        return None
    ayahs = [int(v["ayah"]) for v in verses]
    return {"surah": surahs.pop(), "ayah": min(ayahs), "ayahEnd": max(ayahs)}


class Harness:
    def __init__(self, env: dict[str, str]):
        tsx = Path(env["ZIPFORMER_ORT_DIR"]) / ".bin" / "tsx"
        self.proc = subprocess.Popen(
            [str(tsx), "--tsconfig", str(REPO / "web" / "frontend" / "tsconfig.json"),
             str(LAB / "experiments" / "zipformer-ctc" / "harness.ts")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, env=env, text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline()
        if not json.loads(ready or "{}").get("ready"):
            raise RuntimeError(f"harness failed to start: {ready!r}")
        self.n = 0

    def run(self, pcm_path: str, mode: str, expected: dict | None = None) -> dict:
        self.n += 1
        self.proc.stdin.write(json.dumps({"id": self.n, "pcm": pcm_path, "mode": mode, "expected": expected}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("harness died")
        res = json.loads(line)
        if "error" in res:
            raise RuntimeError(res["error"][:400])
        return res

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default="help-acted,help-clean,tlog-dev,v1")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="help sets only; TLOG clean dev and v1 have no split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("/tmp/models/shipped.onnx"))
    ap.add_argument("--mode", default="correction")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lp", choices=("live", "record", "replay"), default="live")
    ap.add_argument("--lp-cache", type=Path)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--expected", action="store_true",
                    help="pass each clip's expected passage (session.setExpected) when it has one")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the harness (engine config etc.)")
    ap.add_argument("--help-manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--help-audio", type=Path, default=Path("/tmp/help"))
    ap.add_argument("--v1", type=Path, default=Path("/tmp/phase0/v1"))
    ap.add_argument("--tlog-audio", type=Path, default=Path("/tmp/correction_eval/audio/tlog"))
    ap.add_argument("--dev-ids", type=Path, default=Path("/tmp/tlog_meta/dev_ids.txt"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", action="append", default=[],
                    help="NAME=clips.jsonl (id, audio, split, speaker, duration_s) added to --sets")
    ap.add_argument("--track", action="store_true",
                    help="record the post-stop tracker (encoder-frame word spans)")
    ap.add_argument("--slim", action="store_true", help="omit events and diag from the jsonl")
    ap.add_argument("--resume", action="store_true", help="skip clip ids already written without error")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if str(out).startswith(str(REPO.resolve())):
        raise SystemExit("--out must be outside the repo")
    if args.lp != "live" and not args.lp_cache:
        raise SystemExit("--lp record/replay needs --lp-cache")

    ce = _correction_eval()
    from shared.audio import load_audio

    builders = {
        "help-acted": lambda: ce.help_clips(args.help_manifest, args.help_audio, "acted"),
        "help-clean": lambda: ce.help_clips(args.help_manifest, args.help_audio, "clean"),
        "v1": lambda: ce.v1_clips(args.v1),
        "tlog-dev": lambda: ce.tlog_dev_clips(args.dev_ids, args.tlog_audio),
    }
    want = [s.strip() for s in args.sets.split(",") if s.strip()]
    if args.fetch and "tlog-dev" in want:
        ce._fetch_missing([c["id"] for c in builders["tlog-dev"]()], args.tlog_audio)

    env = dict(os.environ)
    env.update({
        "ZIPFORMER_MODEL": str(args.model.resolve()),
        "ZIPFORMER_CORPUS": env.get("ZIPFORMER_CORPUS", str(LAB / "data" / "zipformer" / "quran.json")),
        "ZIPFORMER_ORT_DIR": env.get("ZIPFORMER_ORT_DIR", str(REPO / "web" / "frontend" / "node_modules")),
        "TILAWA_DATA_ROOT": env.get("TILAWA_DATA_ROOT", str(LAB / "data")),
        "ZIPFORMER_INTRA_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    if args.lp != "live":
        env["ZIPFORMER_LP_CACHE"] = str(args.lp_cache.resolve())
        env["ZIPFORMER_LP_MODE"] = args.lp
    if args.diag:
        env["ZIPFORMER_DIAG"] = "1"
    if args.track:
        env["ZIPFORMER_DIAG_TRACK"] = "1"
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v

    for spec in args.extra:
        name, path = spec.split("=", 1)
        clips = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        builders[name] = lambda clips=clips: clips
        if name not in want:
            want.append(name)
    jobs: list[tuple[str, dict]] = []
    for name in want:
        clips = builders[name]()
        if name.startswith("help-") and args.split != "all":
            clips = [c for c in clips if c.get("split") == args.split]
        if args.limit:
            clips = clips[: args.limit]
        jobs.extend((name, c) for c in clips)
    out.mkdir(parents=True, exist_ok=True)
    done_ids: dict[str, set[str]] = {}
    for name in want:
        dest = out / f"{name}.jsonl"
        done_ids[name] = set()
        if args.resume and dest.is_file():
            for line in dest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "error" not in row:
                    done_ids[name].add(str(row["id"]))
        elif dest.is_file():
            dest.unlink()
    jobs = [(name, clip) for name, clip in jobs if str(clip["id"]) not in done_ids.get(name, set())]

    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def worker() -> None:
        h = Harness(env)
        try:
            while True:
                try:
                    name, clip = q.get_nowait()
                except queue.Empty:
                    return
                row = {"id": clip["id"], "duration_s": clip.get("duration_s"), "split": clip.get("split"),
                       "speaker": clip.get("speaker"), "set": name,
                       "expected_verses": clip.get("expected_verses") or [], "issues": [], "notes": []}
                try:
                    audio = load_audio(clip["audio"])
                    row["duration_s"] = row["duration_s"] or round(len(audio) / 16000, 3)
                    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
                        audio.astype("float32").tofile(f)
                        pcm = f.name
                    try:
                        res = h.run(pcm, args.mode, expected_of(clip) if args.expected else None)
                    finally:
                        os.unlink(pcm)
                    row["issues"] = [ce.slim_issue(i) for i in res.get("corrections") or []]
                    row["notes"] = res.get("notes") or []
                    row["verses"] = [[v["surah"], v["ayah"]] for v in res.get("verses") or []]
                    row["decode_ms"] = res.get("decodeMs")
                    if res.get("lpKey"):
                        row["lpKey"] = res["lpKey"]
                    if res.get("trackPost"):
                        row["trackPost"] = res["trackPost"]
                    if res.get("tokens"):
                        row["tokens"] = res["tokens"]
                    if not args.slim:
                        row["events"] = res.get("events") or []
                    if args.diag and not args.slim:
                        row["diag"] = res.get("diag") or []
                        row["seen"] = res.get("seen") or {}
                        if res.get("trace"):
                            row["trace"] = res["trace"]
                        if res.get("track"):
                            row["track"] = res["track"]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                with lock:
                    with (out / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[0] += 1
                    if done[0] % 25 == 0 or done[0] == len(jobs):
                        print(f"{done[0]}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        finally:
            h.close()

    threads = [threading.Thread(target=worker) for _ in range(max(1, args.workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
