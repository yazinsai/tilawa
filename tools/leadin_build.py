# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "numpy", "huggingface_hub"]
# ///
"""Pick a lead-in and a lead-out for each help eval clip: 1–2 correctly recited ayahs just before it and just after it,
so its errors fall mid-recitation and later correct words can confirm them inside the session.

    uv run --python cpython-3.12-macos-aarch64-none tools/leadin_build.py            # manifest + teacher evidence
    uv run --python cpython-3.12-macos-aarch64-none tools/leadin_build.py audio      # tlog lead-in audio, for students

Writes `docs/benchmarks/leadin-v2-manifest.json`, teacher evidence of tlog lead-ins and lead-outs under `.scratch/leadin/evidence/teacher/`
and their 16 kHz audio under `.scratch/leadin/audio/` (ranged reads from the tarteel-ai/tlog parquet shards). Help
lead-ins reuse the help clip's own evidence. Scored by `pnpm scoreboard` (packages/core) with `--leadin`.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "packages/core/assets/ref-v2.json"
HELP = Path(os.environ.get("HELP_DIR", ROOT / ".scratch/help"))
TLOG = Path(os.environ.get("TLOG_DIR", ROOT / ".scratch/tlog"))
OUT = Path(os.environ.get("LEADIN_DIR", ROOT / ".scratch/leadin"))
MANIFEST = ROOT / "docs/benchmarks/leadin-v2-manifest.json"
VERSION = "leadin-v2"
GAP_MS = 600
MAX_LEAD_AYAHS = 2
HF_REPO = "datasets/tarteel-ai/tlog/data"
# Same speaker first, then the teacher decode of anyone else reciting the ayah, held-out tlog before tlog train.
PRIORITY = ["help-same-speaker", "help-other-speaker", "tlog-heldout", "tlog-train"]


@dataclass
class Source:
    kind: str
    id: str
    ayahs: list[str]
    speaker: str


def choose_lead(keys: list[str], ayahs: list[str], speaker: str, pool: list[Source], after: bool = False) -> Source | None:
    """The best clean recitation, at most two ayahs long in the same surah, ending on the ayah before `ayahs[0]`
    (or, `after`, starting on the ayah after `ayahs[-1]`)."""
    surah = ayahs[0].split(":")[0]
    same = lambda ks: len(ks) > 0 and all(k.split(":")[0] == surah for k in ks)
    if after:
        at = keys.index(ayahs[-1]) + 1
        wanted = [keys[at:at + n] for n in range(MAX_LEAD_AYAHS, 0, -1) if at + n <= len(keys) and same(keys[at:at + n])]
    else:
        at = keys.index(ayahs[0])
        wanted = [keys[at - n:at] for n in range(MAX_LEAD_AYAHS, 0, -1) if at - n >= 0 and same(keys[at - n:at])]
    best = None
    for s in pool:
        if s.ayahs not in wanted:
            continue
        kind = s.kind if s.kind != "help" else "help-same-speaker" if s.speaker == speaker else "help-other-speaker"
        rank = (PRIORITY.index(kind), -len(s.ayahs), s.id)
        if best is None or rank < best[0]:
            best = (rank, Source(kind=kind, id=s.id, ayahs=s.ayahs, speaker=s.speaker))
    return best[1] if best else None


def pool() -> list[Source]:
    ayahs = json.loads(INDEX.read_text())["ayahs"]
    words = {f"{a['s']}:{a['a']}": len(a["w"]) for a in ayahs}
    out = [
        Source("help", l["id"], l["ayahs"], l["speaker"])
        for l in map(json.loads, (HELP / "labels.jsonl").read_text().splitlines())
        if l["split"] == "eval" and l["scored"] and l["kind"] == "none" and (HELP / f"wav/{l['id']}.evidence.json").exists()
    ]
    for split, kind in [("test", "tlog-heldout"), ("val", "tlog-heldout"), ("train", "tlog-train")]:
        for line in (TLOG / f"build/tlog-v2/clean-match/{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["label"]["kind"] != "span":
                continue
            a, b = r["label"]["span"]["start"], r["label"]["span"]["end"]
            if a["surah"] != b["surah"] or a["word"] != 1 or b["word"] != words[f"{b['surah']}:{b['ayah']}"] or b["ayah"] - a["ayah"] >= MAX_LEAD_AYAHS:
                continue
            out.append(Source(kind, f"{r['shard']}:{r['row']}", [f"{a['surah']}:{n}" for n in range(a["ayah"], b["ayah"] + 1)], f"tlog/{split}/{r['cluster']}"))
    return out


def build() -> None:
    keys = [f"{a['s']}:{a['a']}" for a in json.loads(INDEX.read_text())["ayahs"]]
    sources = pool()
    by_end, by_start = defaultdict(list), defaultdict(list)
    for s in sources:
        by_end[s.ayahs[-1]].append(s)
        by_start[s.ayahs[0]].append(s)
    clips = []
    for l in map(json.loads, (HELP / "labels.jsonl").read_text().splitlines()):
        if l["split"] != "eval" or not l["scored"]:
            continue
        at = keys.index(l["ayahs"][0])
        lead = choose_lead(keys, l["ayahs"], l["speaker"], by_end[keys[at - 1]] if at else [])
        lead = lead if lead is None or lead.id != l["id"] else None
        reason = None if lead else "first ayah of the surah" if l["ayahs"][0].endswith(":1") else "no clean recitation of the preceding ayah"
        end = keys.index(l["ayahs"][-1]) + 1
        follow = choose_lead(keys, l["ayahs"], l["speaker"], by_start[keys[end]] if end < len(keys) else [], after=True)
        follow = follow if follow is None or follow.id != l["id"] else None
        last = end >= len(keys) or keys[end].split(":")[0] != l["ayahs"][-1].split(":")[0]
        follow_reason = None if follow else "last ayah of the surah" if last else "no clean recitation of the following ayah"
        clips.append({"clip": l["id"], "ayahs": l["ayahs"], "lead": asdict(lead) if lead else None, **({"reason": reason} if reason else {}),
                      "follow": asdict(follow) if follow else None, **({"followReason": follow_reason} if follow_reason else {})})
    counts, follow_counts = defaultdict(int), defaultdict(int)
    for c in clips:
        counts[c["lead"]["kind"] if c["lead"] else c["reason"]] += 1
        follow_counts[c["follow"]["kind"] if c["follow"] else c["followReason"]] += 1
    manifest = {
        "version": VERSION, "gapMs": GAP_MS, "gapFill": "silence",
        "sources": {"help-same-speaker": "the speaker's own clean help eval clip", "help-other-speaker": "another help eval speaker's clean clip (teacher decode)",
                    "tlog-heldout": "tlog-v2 clean-match test/val (held out from student training)", "tlog-train": "tlog-v2 clean-match train"},
        "counts": dict(sorted(counts.items())),
        "followCounts": dict(sorted(follow_counts.items())),
        "clips": clips,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(clips)} clips -> {MANIFEST}: lead-in {manifest['counts']}, lead-out {manifest['followCounts']}")
    export_teacher(tlog_ids(clips))


def tlog_ids(clips: list[dict]) -> list[str]:
    """The tlog recordings a lead-in or lead-out uses."""
    return sorted({s["id"] for c in clips for s in (c["lead"], c.get("follow")) if s and s["kind"].startswith("tlog")})


def export_teacher(ids: list[str]) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    from make_fixtures import INVENTORY_ID, TOPK, decode

    tokens = json.loads(INDEX.read_text())["inventory"]["tokens"]
    wanted = defaultdict(set)
    for i in ids:
        shard, row = i.split(":")
        wanted[shard].add(int(row))
    out = OUT / "evidence/teacher"
    out.mkdir(parents=True, exist_ok=True)
    for shard, rows in sorted(wanted.items()):
        for r in pq.read_table(TLOG / f"pass1/{shard}.parquet").to_pylist():
            if r["row"] not in rows:
                continue
            n = r["n_frames"]
            ids_ = np.frombuffer(r["top5_ids"], dtype=np.uint8).reshape(n, TOPK)
            probs = np.frombuffer(r["top5_p"], dtype=np.float16).astype(np.float32).reshape(n, TOPK)
            evidence = {
                "inventory": INVENTORY_ID, "frameMs": 40, "nFrames": n, "phonemes": decode(ids_, probs, tokens),
                "posteriors": {"k": TOPK, "ids": ids_.flatten().tolist(), "probs": [round(float(p), 4) for p in probs.flatten()]},
            }
            if (out / f"{shard}_{r['row']}.evidence.json").exists():
                continue
            (out / f"{shard}_{r['row']}.evidence.json").write_text(json.dumps(evidence, separators=(",", ":")))
    print(f"{sum(map(len, wanted.values()))} tlog lead-ins -> {out}")


def fetch_audio() -> None:
    """16 kHz mono WAV of each tlog lead-in, reading only the row group that holds it."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    manifest = json.loads(MANIFEST.read_text())
    wanted = defaultdict(set)
    for i in tlog_ids(manifest["clips"]):
        shard, row = i.split(":")
        wanted[shard].add(int(row))
    out = OUT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    done = 0
    for shard, rows in sorted(wanted.items()):
        rows = {r for r in rows if not (out / f"{shard}_{r}.wav").exists()}
        if not rows:
            continue
        f = pq.ParquetFile(fs.open(f"{HF_REPO}/{shard}.parquet", block_size=1 << 20))
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
    print(f"tlog lead-in audio -> {out}")


if __name__ == "__main__":
    {"build": build, "audio": fetch_audio}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
