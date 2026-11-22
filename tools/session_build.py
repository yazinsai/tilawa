"""Audit same-speaker consecutive help coverage; build continuous WAV sessions from a timed catalogue.

python3 tools/session_build.py audit --help-dir .scratch/help --out .scratch/session-bench/coverage.json
python3 tools/session_build.py build --catalog timed-clips.json --out .scratch/session-bench/build --seed 12

A timed catalogue is {"version":1,"clips":[...]} with each clip's speaker, split,
ayahs, audio, durationS, timing (reviewed/alignment), words (at,startS,endS), events,
and nonErrors. Error and non-error entries carry word references and intervals.
Audio must be 16 kHz mono PCM16. Outputs must be freshly decoded as continuous
audio using the SDK decode command, never by concatenating clip evidence.
"""
import argparse
import json
import random
import wave
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ayah_order():
    a = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())["ayahs"]
    return {f"{x['s']}:{x['a']}": i for i, x in enumerate(a)}


def chains(clips, order):
    """Strict consecutive runs within a speaker; choose duplicate clips by stable id."""
    by = defaultdict(dict)
    for c in sorted(clips, key=lambda c: c["id"]):
        if c.get("split") != "eval" or len(c["ayahs"]) != 1:
            continue
        o = order[c["ayahs"][0]]
        by[c["speaker"]].setdefault(o, c)
    runs = []
    for speaker in sorted(by):
        cur, prev = [], None
        for o in sorted(by[speaker]):
            if prev is not None and o != prev + 1:
                runs.append(cur); cur = []
            cur.append(by[speaker][o]); prev = o
        if cur:
            runs.append(cur)
    return runs


def audit(help_dir):
    labels = [json.loads(x) for x in (help_dir / "labels.jsonl").read_text().splitlines()]
    durations = json.loads((help_dir / "durations.json").read_text())
    clips = [{**x, "durationS": durations.get(x["id"]) or 0} for x in labels if x["split"] == "eval" and x["scored"]]
    runs = chains([c for c in clips if c["kind"] == "none"], ayah_order())
    minute_runs = [r for r in runs if sum(c["durationS"] for c in r) >= 60]
    clean = {(c["speaker"], c["ayahs"][0]) for c in clips if c["kind"] == "none" and len(c["ayahs"]) == 1}
    order = ayah_order(); reverse = {v: k for k, v in order.items()}
    both = []
    for c in clips:
        if c["kind"] == "none" or len(c["ayahs"]) != 1:
            continue
        n = order[c["ayahs"][0]]
        if (c["speaker"], reverse.get(n - 1)) in clean and (c["speaker"], reverse.get(n + 1)) in clean:
            both.append(c["id"])
    return {"version": 1, "heldOutClips": len(clips), "heldOutSpeakers": len({c['speaker'] for c in clips}),
            "sameSpeakerCleanRuns": len(runs), "longestRunClips": max(map(len, runs), default=0),
            "longestRunSeconds": max((sum(c['durationS'] for c in r) for r in runs), default=0),
            "minuteRuns": len(minute_runs), "minuteRunSpeakers": len({r[0]['speaker'] for r in minute_runs}),
            "errorClipsWithSameSpeakerBothNeighbours": both,
            "wordTimedHelpClips": sum(bool(c.get('words')) for c in clips),
            "timingNote": "Current help labels contain locations but no per-word intervals. A reviewed/aligned timed catalogue is required before stitching benchmark truth.",
            "coverageNote": "Clean clips only; strict consecutive ayahs, held-out speakers, no fallback voices. Does not infer speaker identity across tlog clusters."}


def build(catalog, out, seed=12, min_seconds=60):
    if catalog.get("version") != 1:
        raise ValueError("Invalid catalogue version")
    clips = catalog["clips"]
    for c in clips:
        if c.get("split") != "eval":
            raise ValueError("Catalogue must be held-out only")
        if c.get("timing") not in ("reviewed", "alignment") or not c.get("words"):
            raise ValueError(f"Missing word timing: {c['id']}")
        if c.get("events") and c.get("audit") != "verified":
            raise ValueError(f"Unreviewed planted error: {c['id']}")
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    sessions = []
    skipped = []
    for run in chains(clips, ayah_order()):
        if sum(c["durationS"] for c in run) < min_seconds:
            skipped.append({"speaker": run[0]["speaker"], "clips": len(run), "reason": "under minimum duration"}); continue
        # Require clean audio before and after every planted error.
        if any(c.get("events") and (i == 0 or i == len(run) - 1 or run[i - 1].get("events") or run[i + 1].get("events")) for i, c in enumerate(run)):
            skipped.append({"speaker": run[0]["speaker"], "reason": "error lacks clean continuations"}); continue
        ident = f"stitched-{len(sessions):03d}"
        destination = out / f"{ident}.wav"
        s = {"version": 1, "id": ident, "track": "memorization" if any(c.get('events') for c in run) else "recitation",
             "speaker": run[0]["speaker"], "source": "stitched", "audio": str(destination.resolve()),
             "context": {"kind": "none"}, "spans": [], "words": [], "events": [], "nonErrors": [], "segments": []}
        offset_frames = 0
        with wave.open(str(destination), "wb") as dst:
            dst.setnchannels(1); dst.setsampwidth(2); dst.setframerate(16000)
            for i, c in enumerate(run):
                if i:
                    gap = rng.randint(4800, 14400); dst.writeframes(b'\0\0' * gap); offset_frames += gap
                start = offset_frames / 16000
                with wave.open(c["audio"], "rb") as src:
                    if (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (1, 2, 16000):
                        raise ValueError("Audio must be 16 kHz mono PCM16")
                    frames = src.getnframes()
                    if abs(frames / 16000 - c["durationS"]) > 1 / 16000:
                        raise ValueError("Catalogue/audio duration mismatch")
                    dst.writeframes(src.readframes(frames)); offset_frames += frames
                end = offset_frames / 16000; span = f"clip-{i}"
                s["segments"].append({"id": c['id'], "speaker": c['speaker'], "audio": c['audio'], "split": "eval", "timing": c['timing'], "startS": start, "endS": end})
                s["spans"].append({"id": span, "kind": "recitation", "from": c['words'][0]['at'], "to": c['words'][-1]['at'], "startS": start, "endS": end})
                for field in ("words", "events", "nonErrors"):
                    for t in c.get(field, []):
                        if not 0 <= t['startS'] <= t['endS'] <= c['durationS']:
                            raise ValueError("Truth outside source clip")
                        s[field].append({**t, "span": span, "startS": start + t['startS'], "endS": start + t['endS']})
        s['durationS'] = offset_frames / 16000
        if s['track'] == 'memorization':
            s['context'] = {"kind": "assignment", "from": s['words'][0]['at'], "to": s['words'][-1]['at']}
        sessions.append({"session": s, "evidence": f"{ident}.evidence.json"})
    result = {"version": 1, "seed": seed, "evidenceStatus": "needs continuous-audio decode", "sessions": sessions, "skipped": skipped}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['audit', 'build']); ap.add_argument('--help-dir', type=Path, default=ROOT / '.scratch/help')
    ap.add_argument('--catalog', type=Path); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--seed', type=int, default=12)
    a = ap.parse_args()
    if a.command == 'audit':
        data = audit(a.help_dir); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(data, indent=2) + '\n'); print(json.dumps(data, indent=2))
    else:
        data = build(json.loads(a.catalog.read_text()), a.out, a.seed); print(f"{len(data['sessions'])} sessions; {len(data['skipped'])} skipped; continuous decode still required")
