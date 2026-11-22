# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy", "huggingface_hub"]
# ///
"""Plan stitched long sessions for the false-jump test (ticket 13) and export their evidence.

    uv run --python cpython-3.12-macos-aarch64-none tools/jumps_build.py

Writes `docs/benchmarks/jumps-v1-manifest.json`, tlog clip evidence under `.scratch/jumps/evidence/`, and stitched help
audio under `.scratch/jumps/audio/`. Score with `pnpm --filter @tilawa/core jumps`.
"""

import argparse
import json
import os
import random
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "packages/core/assets/ref-v2.json"
HELP = Path(os.environ.get("HELP_DIR", ROOT / ".scratch/help"))
TLOG = Path(os.environ.get("TLOG_DIR", ROOT / ".scratch/tlog"))
OUT = Path(os.environ.get("JUMPS_DIR", ROOT / ".scratch/jumps"))
MANIFEST = ROOT / "docs/benchmarks/jumps-v1-manifest.json"
VERSION = "jumps-v1"
# Within this many ayahs a step is local (acceptance-v3): one missing ayah keeps a session contiguous.
LOCAL = 2
GAP_MS = (300, 1500)
# Near-duplicates that aren't exact: 79:39 / 79:41, and al-Kafirun's alternating ayahs.
HARD_EXTRA = ["79:39", "79:41", "109:2", "109:3", "109:4", "109:5"]
SR = 16000
FRAME_MS = 40


@dataclass
class Clip:
    id: str
    ordinals: list[int]
    duration_s: float
    speaker: str


def chains(clips: list[Clip], rng: random.Random) -> list[list[Clip]]:
    """Maximal runs stepping 1 or 2 ayahs, one clip per step, keeping the previous clip's speaker when one is there."""
    by = defaultdict(list)
    for c in clips:
        by[c.ordinals[0]].append(c)
    out, cur = [], []
    for o in sorted(by):
        if cur and o <= cur[-1].ordinals[-1]:
            continue
        if cur and o - cur[-1].ordinals[-1] > LOCAL:
            out.append(cur)
            cur = []
        same = [c for c in by[o] if cur and c.speaker == cur[-1].speaker]
        cur.append(rng.choice(same or by[o]))
    if cur:
        out.append(cur)
    return out


def length(run: list[Clip]) -> float:
    return sum(c.duration_s for c in run)


def split(run: list[Clip], max_s: float) -> list[list[Clip]]:
    n = -(-int(length(run)) // int(max_s))
    target = length(run) / n
    pieces, cur = [], []
    for c in run:
        if cur and length(cur) + c.duration_s > target and len(pieces) < n - 1:
            pieces.append(cur)
            cur = []
        cur.append(c)
    return pieces + [cur]


def group(runs: list[list[Clip]], min_s: float, max_s: float, max_runs: int = 4) -> list[list[list[Clip]]]:
    """Runs joined by planted jumps: 2 to `max_runs` runs per session, between `min_s` and `max_s` long."""
    out, cur = [], []
    for r in runs:
        if cur and sum(map(length, cur)) + length(r) > max_s:
            continue
        cur.append(r)
        if sum(map(length, cur)) >= min_s and len(cur) >= 2:
            out.append(cur)
            cur = []
        elif len(cur) == max_runs:
            cur = []
    return out


def plan(clips: list[Clip], rng: random.Random, hard: set[int], min_s: float, max_s: float, min_run_s: float, hard_min_s: float = 60) -> list[dict]:
    """Contiguous sessions from long runs, jump sessions from shorter ones, hard sessions from runs over near-duplicates."""
    runs = chains(clips, rng)
    is_hard = lambda r: sum(o in hard for c in r for o in c.ordinals) >= 2  # noqa: E731
    sessions = []
    for r in runs:
        if length(r) >= min_s:
            sessions += [("contiguous", [p]) for p in split(r, max_s)]
    short = [r for r in runs if length(r) < min_s]
    hard_runs = [r for r in short if is_hard(r) and length(r) >= 15]
    rng.shuffle(hard_runs)
    sessions += [("hard", g) for g in group(hard_runs, hard_min_s, max_s)]
    rest = [r for r in short if not is_hard(r) and length(r) >= min_run_s]
    rng.shuffle(rest)
    sessions += [("jump", g) for g in group(rest, min_s, max_s)]
    return [session(kind, [c for r in g for c in r], rng, hard) for kind, g in sessions]


def session(kind: str, clips: list[Clip], rng: random.Random, hard: set[int]) -> dict:
    return {
        "kind": kind,
        "hardSegments": sum(any(o in hard for o in c.ordinals) for c in clips),
        "segments": [
            {"clip": c.id, "ordinals": c.ordinals, "duration_s": c.duration_s, "speaker": c.speaker,
             "gapBeforeMs": rng.randint(*GAP_MS) // FRAME_MS * FRAME_MS, "gapFill": rng.choice(["silence", "room"])}
            for c in clips
        ],
    }


def help_sessions(clips: list[Clip], rng: random.Random, hard: set[int], min_s: float, max_s: float) -> list[dict]:
    """One speaker's clean clips in recording order; most seams are planted jumps."""
    by = defaultdict(list)
    for c in clips:
        by[c.speaker].append(c)
    out = []
    for run in by.values():
        if length(run) >= min_s:
            out += [session("help", p, rng, hard) for p in split(run, max_s)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-export", action="store_true", help="write the manifest only")
    ap.add_argument("--audio", action="store_true", help="fetch 16 kHz audio of the manifest's tlog clips, for students, and stop")
    ap.add_argument("--part", default="0/1", help="with --audio: fetch only shard group i of n, for parallel runs")
    args = ap.parse_args()
    if args.audio:
        fetch_tlog_audio(json.loads(MANIFEST.read_text())["sessions"], *map(int, args.part.split("/")))
        return
    rng = random.Random(args.seed)

    ayahs = json.loads(INDEX.read_text())["ayahs"]
    keys = [f"{a['s']}:{a['a']}" for a in ayahs]
    ordinal = {k: i for i, k in enumerate(keys)}
    texts = defaultdict(list)
    for i, a in enumerate(ayahs):
        texts[a["p"]].append(i)
    hard = {i for same in texts.values() for i in same if any(abs(i - j) > LOCAL for j in same)}
    hard |= {ordinal[k] for k in HARD_EXTRA}

    tlog = []
    for split_name in ["test", "val"]:
        for line in (TLOG / f"build/tlog-v2/clean-match/{split_name}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["label"]["kind"] != "span":
                continue
            s = r["label"]["span"]
            a, b = ordinal[f"{s['start']['surah']}:{s['start']['ayah']}"], ordinal[f"{s['end']['surah']}:{s['end']['ayah']}"]
            tlog.append(Clip(id=f"{r['shard']}:{r['row']}", ordinals=list(range(a, b + 1)), duration_s=r["durationS"], speaker=f"tlog/{split_name}/{r['cluster']}"))
    durations = json.loads((HELP / "durations.json").read_text())
    helps = [
        Clip(id=l["id"], ordinals=[ordinal[a] for a in l["ayahs"]], duration_s=durations[l["id"]], speaker=l["speaker"])
        for l in map(json.loads, (HELP / "labels.jsonl").read_text().splitlines())
        if l["split"] == "eval" and l["scored"] and l["kind"] == "none"
    ]

    sessions = plan(tlog, rng, hard, min_s=180, max_s=600, min_run_s=45)
    for s in sessions:
        s["source"] = "tlog"
    for s in help_sessions(helps, rng, hard, min_s=120, max_s=600):
        s["source"] = "help"
        sessions.append(s)
    for i, s in enumerate(sessions):
        s["id"] = f"{s['source']}-{s['kind']}-{i:03d}"
        for g in s["segments"]:
            g["ayahs"] = [keys[o] for o in g.pop("ordinals")]
    manifest = {
        "version": VERSION,
        "seed": args.seed,
        "sources": {"tlog": "tlog-v2 clean-match test + val (held out by speaker cluster)", "help": "help eval split, clean clips"},
        "sessions": [{"id": s["id"], **{k: v for k, v in s.items() if k != "id"}} for s in sessions],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(sessions)} sessions -> {MANIFEST}")
    if not args.no_export:
        export_tlog(sessions)
        stitch_help_audio(sessions)


def export_tlog(sessions: list[dict]) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    from make_fixtures import INVENTORY_ID, TOPK, decode

    tokens = json.loads(INDEX.read_text())["inventory"]["tokens"]
    wanted = defaultdict(set)
    for s in sessions:
        if s["source"] == "tlog":
            for g in s["segments"]:
                shard, row = g["clip"].split(":")
                wanted[shard].add(int(row))
    out = OUT / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    for shard, rows in sorted(wanted.items()):
        for r in pq.read_table(TLOG / f"pass1/{shard}.parquet").to_pylist():
            if r["row"] not in rows:
                continue
            n = r["n_frames"]
            ids = np.frombuffer(r["top5_ids"], dtype=np.uint8).reshape(n, TOPK)
            probs = np.frombuffer(r["top5_p"], dtype=np.float16).astype(np.float32).reshape(n, TOPK)
            evidence = {
                "inventory": INVENTORY_ID, "frameMs": FRAME_MS, "nFrames": n, "phonemes": decode(ids, probs, tokens),
                "posteriors": {"k": TOPK, "ids": ids.flatten().tolist(), "probs": [round(float(p), 4) for p in probs.flatten()]},
            }
            (out / f"{shard}_{r['row']}.evidence.json").write_text(json.dumps(evidence, separators=(",", ":")))
    print(f"{sum(map(len, wanted.values()))} tlog clips -> {out}")


def fetch_tlog_audio(sessions: list[dict], part: int = 0, parts: int = 1) -> None:
    """16 kHz mono WAV of each tlog clip under `.scratch/jumps/tlog-audio/`, reading only the row groups that hold them."""
    import subprocess

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted = defaultdict(set)
    for s in sessions:
        if s["source"] == "tlog":
            for g in s["segments"]:
                shard, row = g["clip"].split(":")
                wanted[shard].add(int(row))
    out = OUT / "tlog-audio"
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    done = 0
    for shard, rows in sorted(wanted.items())[part::parts]:
        rows = {r for r in rows if not (out / f"{shard}_{r}.wav").exists()}
        if not rows:
            continue
        f = pq.ParquetFile(fs.open(f"datasets/tarteel-ai/tlog/data/{shard}.parquet", block_size=1 << 20))
        starts = [0]
        for g in range(f.metadata.num_row_groups):
            starts.append(starts[-1] + f.metadata.row_group(g).num_rows)
        for g in sorted({next(i for i in range(len(starts) - 1) if starts[i] <= r < starts[i + 1]) for r in rows}):
            audio = f.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
            for r in rows:
                if starts[g] <= r < starts[g + 1]:
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", str(out / f"{shard}_{r}.wav")],
                                   input=audio[r - starts[g]]["bytes"], check=True)
                    done += 1
        print(f"{shard}: {done} fetched", flush=True)
    print(f"tlog clip audio -> {out}")


def stitch_help_audio(sessions: list[dict]) -> None:
    """The same stitch as the evidence, on the help wavs: room tone repeats the clip's lead-in before its first phoneme."""
    import numpy as np

    out = OUT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for s in sessions:
        if s["source"] != "help":
            continue
        parts = []
        for g in s["segments"]:
            with wave.open(str(HELP / f"wav/{g['clip']}.wav")) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            phonemes = json.loads((HELP / f"wav/{g['clip']}.evidence.json").read_text())["phonemes"]
            gap = g["gapBeforeMs"] * SR // 1000
            lead = x[: (phonemes[0]["start"] if phonemes else 0) * FRAME_MS * SR // 1000]
            parts.append(np.resize(lead, gap) if g["gapFill"] == "room" and len(lead) >= 3 * FRAME_MS * SR // 1000 else np.zeros(gap, np.int16))
            parts.append(x)
        with wave.open(str(out / f"{s['id']}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.concatenate(parts).astype(np.int16).tobytes())
    print(f"help audio -> {out}")


if __name__ == "__main__":
    main()
