"""Real-user clean set on Modal (.scratch/lab/issues/19-real-user-clean-set.md, docs/benchmarks/lab-realuser-v1.md).

    modal run modal/lab_realuser.py --action fetch     # clips of the three sources, their splits, a decodable clip set
    modal run modal/lab_decode.py --listener teacher --test-set realuser/clips --variants phone
    modal run modal/lab_realuser.py --action build     # verify each clip against its label, plan and join the sessions
    modal run modal/lab_decode.py --listener teacher --test-set realuser-v1 --variants phone
    modal run modal/lab_realuser.py --action label     # check each session, label it, write the manifest, the queue and the seam flags
    modal run modal/lab_decode.py --listener spliced-clean-int8 --model models/spliced-clean-int8 --test-set realuser-v1 --variants phone
    modal run modal/lab_realuser.py --action correct   # reviewed corrections: real omissions as mistakes, bad-audio windows flagged
    modal run modal/lab_realuser.py --action join --base lab-v1.4 --name lab-v1.4u   # a test set with the realuser slice

Pure logic is in tools/lab_realuser_core.py; the checks are tools/lab_recordings_core.check_take.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/app/tools"]

VOL = "/lab"
DATA = Path(VOL) / "data"
SOBOLEV = "sobolev210/quran-recitation-errors"
TLOG = "tarteel-ai/tlog"
RETASY_DIR = DATA / "public/retasy"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy", "rapidfuzz==3.14.1", "quran-transcript==0.6.4", "pyarrow", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_file(ROOT / ".scratch/tlog/build/tlog-v2/clean-match/test.jsonl", "/app/tlog/clean-match-test.jsonl")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__", "**/test_*.py"])
)
app = modal.App("tilawa-lab-realuser", image=image)
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
fn = dict(volumes={VOL: vol, "/cache": cache_vol}, secrets=[hf_secret], timeout=60 * 60)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(p: Path, data: bytes | str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    tmp.replace(p)


def read(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def sha256(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def to_wav(src: bytes | str, dst: Path) -> float:
    """ffmpeg (a path or bytes) to 16 kHz mono PCM16; the seconds of audio."""
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.wav")
    args = ["ffmpeg", "-v", "error", "-y", "-i", "pipe:0" if isinstance(src, bytes) else str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp)]
    subprocess.run(args, input=src if isinstance(src, bytes) else None, check=True, capture_output=True)
    tmp.replace(dst)
    return (dst.stat().st_size - 44) / 2 / 16000


def load_wav(p: Path):
    import wave

    import numpy as np

    with wave.open(str(p), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000), p
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


_REF = {}


def ref():
    import lab_sessions as S
    import lab_sources as L

    if "ref" not in _REF:
        _REF["asset"] = L.reference()
        _REF["ref"] = S.Ref(_REF["asset"])
    return _REF["ref"]


def quran():
    import lab_scripts as T

    if "quran" not in _REF:
        ref()
        _REF["quran"] = T.Quran(_REF["asset"])
    return _REF["quran"]


def word_index(r: dict) -> int:
    return ref().idx[(r["surah"], r["ayah"], r["word"])]


def ordinal(r: dict) -> int:
    return ref().ayah_first[(r["surah"], r["ayah"])]


# ---- fetch --------------------------------------------------------------------------------------------------------

@app.function(**fn, cpu=4, memory=8192)
def fetch_sobolev() -> dict:
    from huggingface_hub import HfApi, snapshot_download

    import lab_realuser_core as U

    rev = HfApi().dataset_info(SOBOLEV).sha
    d = snapshot_download(SOBOLEV, repo_type="dataset", revision=rev, allow_patterns=["metadata.jsonl", "README.md"])
    rows = [json.loads(x) for x in Path(d, "metadata.jsonl").read_text().splitlines() if x.strip()]
    clips, dropped = U.sobolev_clips(rows, ref().ayah_words)
    d = snapshot_download(SOBOLEV, repo_type="dataset", revision=rev, allow_patterns=[c["file"] for c in clips])
    out = DATA / U.ROOT / "clips/sobolev"
    for c in clips:
        c["durationS"] = round(to_wav(Path(d, c["file"]), out / f"{c['id']}.wav"), 3)
        c["wav"] = f"sobolev/{c['id']}.wav"
    vol.commit()
    return {"revision": rev, "rows": len(rows), "clips": clips, "dropped": dropped}


@app.function(**fn, cpu=4, memory=8192)
def fetch_retasy() -> dict:
    import shutil

    import lab_realuser_core as U

    m = json.loads((RETASY_DIR / "manifest.json").read_text())
    clips, dropped = U.retasy_clips(m["items"], ref().ayah_words)
    out = DATA / U.ROOT / "clips/retasy"
    out.mkdir(parents=True, exist_ok=True)
    for c in clips:
        shutil.copyfile(RETASY_DIR / c["file"], out / f"{c['id']}.wav")
        c["durationS"] = round((out / f"{c['id']}.wav").stat().st_size / 32000, 3)  # wav16k: 44-byte header, PCM16
        c["wav"] = f"retasy/{c['id']}.wav"
    vol.commit()
    return {"revision": m.get("revision"), "rows": len(m["items"]), "clips": clips, "dropped": dropped}


@app.function(**fn, cpu=2, memory=8192, max_containers=24, retries=2)
def fetch_tlog_shard(shard: str, rows: list[int]) -> dict[int, float]:
    """16 kHz WAVs of the given rows of one tlog shard, reading only the row groups that hold them."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    import lab_realuser_core as U

    f = pq.ParquetFile(HfFileSystem().open(f"datasets/{TLOG}/data/{shard}.parquet", block_size=1 << 20))
    starts = [0]
    for g in range(f.metadata.num_row_groups):
        starts.append(starts[-1] + f.metadata.row_group(g).num_rows)
    out, d = {}, DATA / U.ROOT / "clips/tlog"
    for g in sorted({next(i for i in range(len(starts) - 1) if starts[i] <= r < starts[i + 1]) for r in rows}):
        audio = f.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
        for r in rows:
            if starts[g] <= r < starts[g + 1]:
                out[r] = round(to_wav(audio[r - starts[g]]["bytes"], d / f"{U.clip_id('tlog', shard.replace('clean-', ''), r)}.wav"), 3)
    vol.commit()
    return out


@app.function(**fn, cpu=2, memory=8192)
def fetch_all() -> dict:
    from collections import defaultdict

    import lab_realuser_core as U

    vol.reload()
    sob, ret = fetch_sobolev.spawn(), fetch_retasy.spawn()
    rows = [json.loads(x) for x in Path("/app/tlog/clean-match-test.jsonl").read_text().splitlines() if x.strip()]
    tl = U.tlog_clips(rows)
    chains = U.tlog_sample(tl, word_index)
    picked = []
    for i, ch in enumerate(chains):
        for c in ch:
            picked.append({**c, "chain": i, "wav": f"tlog/{c['id']}.wav"})
    by = defaultdict(list)
    for c in picked:
        by[c["origin"]["shard"]].append(c["origin"]["row"])
    got = {}
    for shard, res in zip(sorted(by), fetch_tlog_shard.starmap([(s, sorted(by[s])) for s in sorted(by)])):
        got |= {(shard, r): d for r, d in res.items()}
    for c in picked:
        c["durationS"] = got[(c["origin"]["shard"], c["origin"]["row"])]
    s, r = sob.get(), ret.get()
    clips = s["clips"] + r["clips"] + picked
    found = {"sobolev": len(s["clips"]) + len(s["dropped"]), "retasy": len(r["clips"]) + len(r["dropped"]), "tlog": len(picked)}
    splits = U.splits(clips)
    root = DATA / U.ROOT
    write(root / "splits-v1.json", U.dumps({"version": 1, "seed": U.SEED, "rule": "per source, the seeded first half of its speakers is exam",
                                            "speakers": dict(sorted(splits.items()))}))
    doc = {"version": 1, "createdAt": now(), "sources": {"sobolev": {"repo": SOBOLEV, "revision": s["revision"], "rows": s["rows"]},
                                                          "retasy": {"dir": "public/retasy", "revision": r["revision"], "rows": r["rows"]},
                                                          "tlog": {"repo": TLOG, "split": "tlog-v2 clean-match test", "rows": len(rows), "spanClips": len(tl), "chains": len(chains)}},
           "found": found, "clips": clips, "dropped": [{**d, "stage": "fetch"} for d in s["dropped"] + r["dropped"]]}
    write(root / "clips-v1.json", U.dumps(doc))
    write(root / "clips/manifest-v1.json", U.dumps({"version": 1, "testSet": "realuser-clips", "evidencePattern": "{id}.{listener}.{variant}.evidence.json",
                                                    "sessions": [{"id": c["id"], "split": "practice", "audio": {"phone": c["wav"]}, "durationS": c["durationS"]} for c in clips]}))
    vol.commit()
    return {"found": found, "clips": dict(Counter(c["source"] for c in clips)), "minutes": {k: round(sum(c["durationS"] for c in clips if c["source"] == k) / 60, 1) for k in U.SOURCES},
            "speakers": {k: len({c["speaker"] for c in clips if c["source"] == k}) for k in U.SOURCES}}


# ---- build --------------------------------------------------------------------------------------------------------

def verify_all(clips: list[dict]) -> list[dict]:
    """Each clip's verification against its label from its teacher evidence (`verify_clip`): a whole-ayah label (sobolev,
    retasy) is checked against every ayah of the Quran too, so a mislabelled chunk is relabelled rather than dropped."""
    from per_breakdown import canon

    import lab_realuser_core as U
    import lab_scripts as T

    r, asset, q = ref(), _REF["asset"], quran()
    ist = canon(T.optional(q, "istiadha")["phonemes"])
    bas = canon(q.basmala["p"])
    prefixes = ["", ist, bas, ist + bas]
    ayahs = {f"{a['s']}:{a['a']}": canon(a["p"]) for a in asset["ayahs"]}
    words = lambda c: canon("".join(r.words[i]["cont"] for i in range(word_index(c["from"]), word_index(c["to"]) + 1)))
    surah_words = {}
    for i, w in enumerate(r.words):
        surah_words.setdefault(w["s"], []).append(i)
    # Word i's form said after a pause and before one, as clip_items says it; canon squeezes the difference anyway.
    said = lambda i: canon(r.words[i]["cont"])
    base = DATA / U.ROOT / "clips"
    for c in clips:
        ev = read(base / f"{c['id']}.teacher.phone.evidence.json")
        if not ev:
            c["verify"] = {"status": "drop", "why": "no teacher evidence"}
            continue
        heard = canon("".join(p["p"] for p in ev["phonemes"]))
        expected = words(c)
        span = None
        if c["source"] != "tlog" and U.verify_clip(heard, expected, prefixes)["status"] != "ok":
            ids = surah_words[c["from"]["surah"]]
            sp = U.best_span(heard, [said(i) for i in ids], prefixes)
            span = (ids[sp[0]], ids[sp[1]], sp[2]) if sp else None
        v = U.verify_clip(heard, expected, prefixes, ayahs if c["source"] != "tlog" else None, span)
        if v["status"] == "relabel":
            c["relabelledFrom"] = {"from": c["from"], "to": c["to"]}
            if "span" in v:
                c["from"], c["to"] = r.ref(v["span"][0]), r.ref(v["span"][1])
                v["to"] = f"{r.name(v['span'][0])}–{r.name(v['span'][1])}"
            else:
                s, a = map(int, v["to"].split(":"))
                c["from"], c["to"] = U.at(s, a, 1), U.at(s, a, len(r.ayah_words[v["to"]]))
        c["verify"] = {**v, "heard": heard[:400]}
    return clips


@app.function(**fn, cpu=8, memory=16384)
def build() -> dict:
    """Verify every clip, plan the sessions, write their audio and a pre-label manifest (enough for lab_decode)."""
    import lab_realuser_core as U
    import lab_sessions as S

    vol.reload()
    root = DATA / U.ROOT
    doc = read(root / "clips-v1.json")
    split_of = read(root / "splits-v1.json")["speakers"]
    r = ref()
    for c in doc["clips"]:  # a rebuild verifies against the source labels again
        old = c.pop("relabelledFrom", None)
        if isinstance(old, str):
            s, a = map(int, old.split(":"))
            c["from"], c["to"] = U.at(s, a, 1), U.at(s, a, len(r.ayah_words[old]))
        elif old:
            c["from"], c["to"] = old["from"], old["to"]
        c.pop("verify", None)
    clips = verify_all(doc["clips"])
    dropped = [d for d in doc["dropped"] if d["stage"] == "fetch"]
    kept = []
    for c in clips:
        v = c["verify"]
        if v["status"] == "drop":
            dropped.append({"id": c["id"], "source": c["source"], "kind": "not_label_text", "stage": "verify",
                            "why": v.get("why") or f"the teacher hears another text (PER {v['per']:.2f} to its label)"})
        elif not U.CLIP_S[0] <= c["durationS"] <= U.CLIP_S[1]:
            dropped.append({"id": c["id"], "source": c["source"], "kind": "audio", "stage": "verify", "why": f"{c['durationS']:.1f} s long"})
        else:
            kept.append(c)
    sessions, short = U.plan_sessions(kept, split_of, ordinal, word_index)
    dropped += [{**d, "stage": "plan"} for d in short]
    by = {c["id"]: c for c in kept}
    ts = DATA / U.TEST_SET
    ids = {s["id"] for s in sessions}
    for p in (ts / "sessions").glob("*") if (ts / "sessions").is_dir() else []:
        if p.name.split(".")[0] not in ids:
            p.unlink()
    for s in sessions:
        cs = [by[i] for i in s["clips"]]
        x, gains = U.join_audio([load_wav(root / "clips" / c["wav"]) for c in cs])
        p = ts / f"sessions/{s['id']}.phone.wav"
        write(p, S.pcm16(x))
        s["placed"] = U.join_plan([c["durationS"] for c in cs])
        s["gainsDb"] = gains
        s["durationS"] = round(len(x) / 16000, 3)
        write(ts / f"sessions/{s['id']}.checks.json", U.dumps({"version": 1, "id": s["id"], "phoneSha256": sha256(p), "source": "realuser"}))
    write(root / "clips-v1.json", U.dumps({**doc, "verifiedAt": now(), "clips": clips, "dropped": dropped}))
    write(ts / "plan-v1.json", U.dumps({"version": 1, "testSet": U.TEST_SET, "createdAt": now(), "sessions": sessions}))
    write(ts / "manifest-v1.json", U.dumps({"version": 1, "testSet": U.TEST_SET, "stage": "unlabelled", "evidencePattern": "sessions/{id}.{listener}.{variant}.evidence.json",
                                            "sessions": [{"id": s["id"], "split": s["split"], "audio": {"phone": f"sessions/{s['id']}.phone.wav"}, "durationS": s["durationS"]} for s in sessions]}))
    vol.commit()
    return {"verify": dict(Counter(f"{c['source']}/{c['verify']['status']}" for c in clips)), "dropped": dict(Counter(f"{d['source']}/{d['kind']}" for d in dropped)),
            "sessions": dict(Counter(f"{s['source']}/{s['split']}" for s in sessions))}


# ---- label --------------------------------------------------------------------------------------------------------

@app.function(**fn, cpu=2, memory=8192, max_containers=40)
def label_one(plan: dict, clips: list[dict]) -> dict:
    """check_take on one joined session's teacher evidence, then its BenchSession (written when it aligned and validates)."""
    import lab_realuser_core as U
    import lab_recordings_core as R
    import lab_scripts as T

    vol.reload()
    r, asset, q = ref(), _REF["asset"], quran()
    optional = {"istiadha": T.optional(q, "istiadha")}
    basmala = {"text": T.BASMALA, "phonemes": q.basmala["p"]}
    ts = DATA / U.TEST_SET
    ev = read(ts / f"sessions/{plan['id']}.teacher.phone.evidence.json")
    if not ev:
        return {"id": plan["id"], "ok": False, "why": "no teacher evidence"}
    sc = U.script(r, plan["id"], clips, optional, basmala)
    res = R.check_take(sc, ev, asset)
    if not res["aligned"]:
        return {"id": plan["id"], "ok": False, "why": "; ".join(res["why"]) or "alignment failed"}
    s = U.session(plan, clips, res, f"sessions/{plan['id']}.phone.wav", [tuple(p) for p in plan["placed"]], word_index)
    s["meta"]["gainsDb"] = plan["gainsDb"]
    problems = R.validate(s, {(w["s"], w["a"], w["w"]) for w in r.words})
    if problems:
        return {"id": plan["id"], "ok": False, "why": f"labels don't validate: {problems[0]}"}
    write(ts / f"sessions/{plan['id']}.json", U.dumps(s))
    vol.commit()
    return {"id": plan["id"], "ok": True, "coverage": res["coverage"], "flagged": len(s["meta"]["flagged"]), "dropped": res["dropped"]}


@app.function(**fn, cpu=2, memory=8192)
def label_all() -> dict:
    import lab_realuser_core as U

    vol.reload()
    root, ts = DATA / U.ROOT, DATA / U.TEST_SET
    plan = read(ts / "plan-v1.json")
    doc = read(root / "clips-v1.json")
    by = {c["id"]: c for c in doc["clips"]}
    outs = list(label_one.starmap([(p, [by[i] for i in p["clips"]]) for p in plan["sessions"]], return_exceptions=True, wrap_returned_exceptions=False))
    vol.reload()
    sessions, failed = [], []
    for p, o in zip(plan["sessions"], outs):
        if isinstance(o, Exception) or not o["ok"]:
            failed.append({"id": p["id"], "source": p["source"], "why": repr(o)[:300] if isinstance(o, Exception) else o["why"]})
            continue
        sessions.append(read(ts / f"sessions/{p['id']}.json"))
    dropped = doc["dropped"] + [{"id": c, "source": f["source"], "kind": "session_failed", "stage": "label", "why": f["why"]}
                                for f in failed for c in next(p for p in plan["sessions"] if p["id"] == f["id"])["clips"]]
    kept = {c for s in sessions for c in (x["id"] for x in s["meta"]["clips"])}
    m = U.manifest(sessions)
    write(ts / "manifest-v1.json", U.dumps(m))
    write(ts / "flags-v1.json", U.dumps(U.seam_exclusions(sessions)))
    write(ts / "review/flagged.json", U.dumps(U.flagged_queue(sessions)))
    summ = {"version": 1, "testSet": U.TEST_SET, "createdAt": now(), "sources": doc["sources"],
            "perSource": U.summary(doc["found"], [c for c in doc["clips"] if c["id"] in kept], dropped, sessions), "failedSessions": failed}
    write(ts / "summary-v1.json", U.dumps(summ))
    vol.commit()
    return {"sessions": len(sessions), "failed": failed[:10], "perSource": summ["perSource"]}


@app.function(**fn, cpu=2, memory=8192)
def seams() -> dict:
    """flags-v1.json of realuser-v1 from its labelled sessions: a window around every jump seam (`seam_exclusions`)."""
    import lab_realuser_core as U

    vol.reload()
    ts = DATA / U.TEST_SET
    sessions = [read(ts / e["label"]) for e in read(ts / "manifest-v1.json")["sessions"]]
    x = U.seam_exclusions(sessions)
    write(ts / "flags-v1.json", U.dumps(x))
    vol.commit()
    return {"windows": len(x["excluded"]), "sessions": len({e["session"] for e in x["excluded"]})}


@app.function(**fn, cpu=2, memory=8192)
def apply_corrections(doc: dict) -> dict:
    """Reviewed corrections (docs/benchmarks/lab-realuser-v1-corrections.json) into realuser-v1 before it is joined:
    omissions heard as real become mistakes in their sessions, and bad-audio windows join the seam flags."""
    import lab_realuser_core as U
    import lab_recordings_core as R

    vol.reload()
    ts = DATA / U.TEST_SET
    m = read(ts / "manifest-v1.json")
    sessions = [read(ts / e["label"]) for e in m["sessions"]]
    words = {(w["s"], w["a"], w["w"]) for w in ref().words}
    out = []
    for s in sessions:
        mine = [f for f in doc["fixes"] if f["session"] == s["id"]]
        if mine:
            s = U.correct({**s, "meta": {**s["meta"], "corrections": []}}, mine, word_index)
            problems = R.validate(s, words)
            if problems:
                raise ValueError(f"{s['id']}: {problems[0]}")
            write(ts / f"sessions/{s['id']}.json", U.dumps(s))
        out.append(s)
    write(ts / "manifest-v1.json", U.dumps(U.manifest(out)))
    x = U.seam_exclusions(out, extra=doc["excluded"])
    write(ts / "flags-v1.json", U.dumps(x))
    write(ts / "corrections-v1.json", U.dumps(doc))
    vol.commit()
    return {"events": sum(len(s["events"]) for s in out), "corrected": sorted({f["session"] for f in doc["fixes"]}), "flags": len(x["excluded"])}


# ---- join ---------------------------------------------------------------------------------------------------------

@app.function(**fn, cpu=2, memory=8192)
def join(base: str, name: str) -> dict:
    """/lab/data/<name>/: test set <base> plus the realuser sessions (files copied; <base> untouched). The base's flags
    (the latest flags-v<n>.json) are carried over as <name>/flags-v1.json with `base` set to the new name."""
    import shutil

    import lab_realuser_core as U

    vol.reload()
    src, dst, ru = DATA / base, DATA / name, DATA / U.TEST_SET
    if dst.exists():
        raise FileExistsError(f"/lab/data/{name} exists; test set versions are immutable")
    m, r = read(src / "manifest-v1.json"), read(ru / "manifest-v1.json")
    pattern, files = m["evidencePattern"], set()
    for e in m["sessions"]:
        files |= set(e["audio"].values())
        for p in src.glob(pattern.replace("{id}", e["id"]).replace("{listener}", "*").replace("{variant}", "*")):
            files.add(str(p.relative_to(src)))
        for extra in (f"sessions/{e['id']}.json", f"sessions/{e['id']}.checks.json"):
            if (src / extra).exists():
                files.add(extra)
    for extra in ("checks-v1.json",):
        if (src / extra).exists():
            files.add(extra)
    for f in sorted(files):
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)
    for e in r["sessions"]:
        for p in (ru / "sessions").glob(f"{e['id']}.*"):
            if not p.name.endswith(".tmp"):
                shutil.copy2(p, dst / "sessions" / p.name)
    out = U.join({**m, "testSet": name, "joinedFrom": base}, r)
    write(dst / "manifest-v1.json", U.dumps(out))
    flags = sorted(src.glob("flags-v*.json"), key=lambda p: int(p.stem.split("-v")[1]))
    carried = None
    if flags:
        f = read(flags[-1])
        write(dst / "flags-v1.json", U.dumps({**f, "testSet": name, "base": name, "carriedFrom": f"{base}/{flags[-1].name}"}))
        carried = flags[-1].name
    vol.commit()
    return {"testSet": name, "base": len(m["sessions"]), "realuser": len(r["sessions"]), "files": len(files), "flagsCarried": carried}


@app.local_entrypoint()
def main(action: str, base: str = "", name: str = ""):
    if action == "fetch":
        print(json.dumps(fetch_all.remote(), indent=1))
    elif action == "build":
        print(json.dumps(build.remote(), indent=1))
    elif action == "label":
        print(json.dumps(label_all.remote(), indent=1, ensure_ascii=False))
    elif action == "correct":
        print(json.dumps(apply_corrections.remote(json.loads((ROOT / "docs/benchmarks/lab-realuser-v1-corrections.json").read_text())), indent=1))
    elif action == "seams":
        print(json.dumps(seams.remote(), indent=1))
    elif action == "join":
        if not base or not name:
            raise SystemExit("--base <test set> and --name <new test set> are required")
        print(json.dumps(join.remote(base, name), indent=1))
    else:
        raise SystemExit(f"unknown action {action!r}")
