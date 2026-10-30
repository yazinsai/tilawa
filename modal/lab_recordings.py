"""Scripted-error recordings on Modal: process uploaded takes into recorded BenchSessions (.scratch/lab/issues/18-scripted-recordings.md,
docs/benchmarks/lab-recordings.md).

    modal deploy modal/lab_recordings.py                               # process_take, spawned by the /record page on upload
    modal run modal/lab_recordings.py --action scripts --file docs/benchmarks/lab-scripts-v1.json   # publish a script batch
    modal run modal/lab_recordings.py --action process --speaker ali [--script rec-v1-03] [--take 2]  # re-run checks on takes
    modal run modal/lab_recordings.py --action rebuild                 # review queue, manifest and index from every result
    modal run modal/lab_recordings.py --action join --base lab-v1.3 --name lab-v1.3r                 # a new test set with the takes
    modal run modal/lab_recordings.py --action test-audio --file /tmp/scripts-test.json --out /tmp/e2e  # synthetic takes (e2e)
    modal run modal/lab_recordings.py --action test-fix --file /tmp/scripts-test.json --out /tmp/e2e-fix  # a take and patches (fix e2e)
    modal run modal/lab_recordings.py --action cleanup --speaker test [--batch test]                  # delete a speaker's takes

A take is converted to 16 kHz mono WAV, decoded by the teacher (as modal/lab_decode.py decodes the sources), aligned to
what the script says was said, and every planted mistake is checked (tools/lab_recordings_core.py). A fix (a take
whose upload is only some ayahs recorded again) is checked on its own, joined into its base take at measured pauses,
and the composed take is then processed like any other. Files live under
/lab/data/recordings/ on the tilawa-lab volume (layout in the core module's docstring).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/app/tools"]

APP = os.environ.get("LAB_RECORDINGS_APP", "tilawa-lab-recordings")
TEACHER = "obadx/muaalem-model-v3_2"
TEACHER_GPUS = ["L4", "A10", "L40S"]
TOPK, BATCH = 5, 8
VOL = "/lab"
DATA = Path(VOL) / "data"
MIN_S, MAX_S = 5.0, 15 * 60.0

common = dict(python_version="3.12")
cpu_image = (
    modal.Image.debian_slim(**common)
    .apt_install("ffmpeg")
    .pip_install("numpy", "rapidfuzz==3.14.1", "quran-transcript==0.6.4", "pyarrow")
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__", "**/test_*.py"])
)
gpu_image = (
    modal.Image.debian_slim(**common)
    .pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "numpy", "rapidfuzz==3.14.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__", "**/test_*.py"])
)
app = modal.App(APP)
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(p: Path, data: bytes | str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    tmp.replace(p)


def read(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def scripts() -> dict[str, dict]:
    out = {}
    for p in sorted((DATA / "recordings/scripts").glob("*.json")):
        b = json.loads(p.read_text())
        for s in b["scripts"]:
            out[s["id"]] = {**s, "batch": b["batch"]}
    return out


@app.cls(image=gpu_image, gpu=TEACHER_GPUS, volumes={VOL: vol, "/cache": cache_vol}, secrets=[hf_secret], memory=16384, timeout=30 * 60,
         max_containers=4, scaledown_window=5 * 60)
class Teacher:
    """The teacher decode of modal/lab_decode.py: 30 s windows every 25 s, top-5 posteriors, greedy phonemes."""

    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        import lab_sources as L

        self.torch, self.L = torch, L
        m = Muaalem(TEACHER, device="cuda", dtype=torch.bfloat16)
        self.model, self.processor = m.model.eval(), m.processor
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.gpu = torch.cuda.get_device_name(0)
        vocab = m.multi_level_tokenizer.id_to_vocab["phonemes"]
        self.ref = L.reference()
        self.tokens = self.ref["inventory"]["tokens"]
        assert [vocab[i] for i in range(1, len(self.tokens) + 1)] == list(self.tokens), "teacher vocab differs from ref-v2 inventory"

    def infer(self, waves):
        torch = self.torch
        feats = self.processor(waves, sampling_rate=self.L.SR, return_tensors="pt", padding=True)
        mask = feats["attention_mask"]
        feats = {k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
        with torch.no_grad():
            lg = self.model(**feats, return_dict=False)[0]["phonemes"]
        probs = torch.softmax(lg.float(), dim=-1).cpu()
        ratio = probs.shape[1] / mask.shape[1]
        return [probs[i, : max(1, round(n * ratio))] for i, n in enumerate(mask.sum(dim=1).tolist())]

    @modal.method()
    def decode(self, wav: str, dst: str) -> dict:
        import wave

        import numpy as np

        L = self.L
        vol.reload()
        t0 = time.time()
        with wave.open(str(DATA / wav), "rb") as f:
            assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, L.SR), wav
            x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
        plan = L.windows(len(x))
        w = L.WINDOW_S * L.SR
        ids, probs = [], []
        for b in range(0, len(plan), BATCH):
            part = plan[b : b + BATCH]
            for (s, lo, hi), p in zip(part, self.infer([x[s : s + w] for s, _, _ in part])):
                if hi > p.shape[0]:
                    raise RuntimeError(f"window at {s} gave {p.shape[0]} frames, needs {hi}")
                top_p, top_i = p[lo : hi if hi >= 0 else None].topk(TOPK, dim=-1)
                ids.append(top_i.numpy())
                probs.append(top_p.numpy())
        ids_all, p_all = np.concatenate(ids), np.concatenate(probs)
        n = len(ids_all)
        frame_ms = L.FRAME * 1000 // L.SR
        if abs(n * frame_ms / 1000 - len(x) / L.SR) > 0.1:
            raise RuntimeError(f"{wav}: {n} frames for {len(x) / L.SR:.3f} s")
        phonemes = [{"p": self.tokens[i - 1], "start": s, "end": e, "conf": round(c, 4)} for i, s, e, c in L.greedy(ids_all[:, 0], p_all[:, 0])]
        ev = {"inventory": self.ref["inventory"]["id"], "frameMs": frame_ms, "nFrames": n, "phonemes": phonemes,
              "posteriors": {"k": TOPK, "ids": ids_all.astype(int).flatten().tolist(), "probs": [round(float(v), 4) for v in p_all.flatten()]},
              "listener": "teacher", "model": TEACHER, "revision": self.revision, "gpu": self.gpu}
        write_atomic(DATA / dst, json.dumps(ev, ensure_ascii=False, separators=(",", ":")))
        vol.commit()
        return {"seconds": round(time.time() - t0, 1), "audioS": round(len(x) / L.SR, 3), "nFrames": n, "revision": self.revision, "gpu": self.gpu}


def to_wav(src: Path, dst: Path) -> float:
    """ffmpeg to 16 kHz mono PCM16; the seconds of audio."""
    import subprocess

    import lab_sources as L

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(L.SR), "-sample_fmt", "s16", "-f", "wav", str(tmp)],
                   check=True, capture_output=True)
    tmp.replace(dst)
    return (dst.stat().st_size - 44) / 2 / L.SR


def load_wav(p: Path):
    import wave

    import numpy as np

    import lab_sources as L

    with wave.open(str(p), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, L.SR), p
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def compose_fix(script: dict, speaker: str, script_id: str, n: int, meta: dict) -> dict:
    """Take n is a fix: check its patch against the ayahs it records, join it into its base take at measured pauses
    and write the composed audio as take-<n>.wav (with its joins and resume points in take-<n>.json). The fix record
    for the result; raises lab_recordings_core.Refused when the patch can't be joined."""
    import lab_recordings_core as R
    import lab_sessions as S
    import lab_sources as L

    fx = meta["fix"]
    d = DATA / R.take_dir(speaker, script_id)
    bm, br = read(d / f"take-{fx['base']}.json"), read(d / f"take-{fx['base']}.result.json")
    R.fix_request(script, bm, br, fx["from"], fx["to"])
    ff = {k: DATA / v for k, v in R.fix_files(speaker, script_id, n, fx["patchExt"]).items()}
    patch_s = to_wav(ff["patch"], ff["patchWav"])
    info = {k: fx[k] for k in ("base", "from", "to", "patch")} | {"patchS": round(patch_s, 3)}
    if not R.PATCH_S[0] <= patch_s <= R.PATCH_S[1]:
        raise R.Refused([f"the fix is {patch_s:.1f} s; record {R.passage(fx['from'], fx['to'])} with a moment of silence either side"], info)
    vol.commit()
    info["teacher"] = Teacher().decode.remote(str(ff["patchWav"].relative_to(DATA)), str(ff["patchEvidence"].relative_to(DATA)))
    vol.reload()
    sub = R.fix_script(script, fx["from"], fx["to"])
    pc = R.check_take(sub, json.loads(ff["patchEvidence"].read_text()), L.reference())
    info["patchCheck"] = R.patch_summary(pc)
    if pc["verdict"] == "redo":
        raise R.Refused([f"the fix itself: {w}" for w in pc["why"]] + [f"record {R.passage(fx['from'], fx['to'])} again"], info | {"suggest": {"from": fx["from"], "to": fx["to"]}})
    x, y = load_wav(DATA / R.take_files(speaker, script_id, fx["base"], bm["ext"])["wav"]), load_wav(ff["patchWav"])
    plan = R.plan_fix(script, fx, br["said"], len(x) / L.SR, S.frame_db(x), pc["said"], len(y) / L.SR, S.frame_db(y))
    info |= {k: v for k, v in plan.items() if k not in ("ok", "why")}
    if not plan["ok"]:
        raise R.Refused(plan["why"], info)
    write_atomic(DATA / R.take_files(speaker, script_id, n, meta["ext"])["upload"], S.pcm16(R.stitch(x, y, plan, L.SR)))
    info["resumesS"] = R.fix_resumes(bm.get("resumesS") or [], fx.get("resumesS") or [], plan)
    write_atomic(d / f"take-{n}.json", R.dumps({**meta, "resumesS": info["resumesS"], "joinsS": plan["joinsS"],
                                                "fix": {**fx, "joins": plan["joins"], "gainDb": plan["gainDb"]}}))
    return info


@app.function(image=cpu_image, volumes={VOL: vol}, memory=8192, timeout=30 * 60)
def process_take(speaker: str, script_id: str, n: int) -> dict:
    """Convert, decode, check and label one take (a fix is composed first); then rebuild the queue, manifest and index."""
    import lab_recordings_core as R
    import lab_sources as L

    vol.reload()
    d = DATA / R.take_dir(speaker, script_id)
    meta = read(d / f"take-{n}.json")
    if not meta:
        raise FileNotFoundError(f"no take {n} of {script_id} by {speaker}")
    f = {k: DATA / v for k, v in R.take_files(speaker, script_id, n, meta["ext"]).items()}
    sid = R.session_id(speaker, script_id, n)
    base = {"version": 1, "speaker": speaker, "script": script_id, "take": n, "sessionId": sid, "uploadedAt": meta["uploadedAt"]}
    t0 = time.time()
    fix = None
    try:
        script = scripts().get(script_id)
        if not script:
            raise ValueError(f"script {script_id} is not published")
        if script["sha"] != meta["scriptSha"]:
            raise ValueError(f"script {script_id} changed since this take (take {meta['scriptSha']}, now {script['sha']})")
        if meta.get("fix"):
            fix = compose_fix(script, speaker, script_id, n, meta)
            meta = read(d / f"take-{n}.json")
        audio_s = to_wav(f["upload"], f["wav"])
        if not MIN_S <= audio_s <= MAX_S:
            raise ValueError(f"the take is {audio_s:.1f} s; takes are {MIN_S:.0f} s to {MAX_S / 60:.0f} min")
        write_atomic(f["checks"], R.dumps({"version": 1, "id": sid, "phoneSha256": sha256(f["wav"]), "source": "recorded", "upload": str(f["upload"].relative_to(DATA)),
                                           **({"fix": {k: fix[k] for k in ("base", "from", "to", "joinsS", "gainDb")}} if fix else {})}))
        vol.commit()
        dec = Teacher().decode.remote(str(f["wav"].relative_to(DATA)), str(f["evidence"].relative_to(DATA)))
        vol.reload()
        ev = json.loads(f["evidence"].read_text())
        result = R.check_take(script, ev, L.reference())
        upload_meta = {**meta, "upload": str(f["upload"].relative_to(DATA))}
        s = R.session(script, result, upload_meta, f"sessions/{sid}.phone.wav")
        problems = R.validate(s, {(w["s"], w["a"], w["w"]) for w in _ref_words()}) if result["aligned"] else []
        if problems:
            result["verdict"], result["why"] = "redo", [f"the labels don't validate: {problems[0]}"] + result["why"]
            s["meta"]["verdict"] = "redo"
        if result["verdict"] != "redo":
            write_atomic(f["session"], R.dumps(s))
        elif f["session"].exists():
            f["session"].unlink()
        out = {**base, **result, "audioS": round(audio_s, 3), "teacher": dec, "validation": problems, "processedAt": now(), "processS": round(time.time() - t0, 1),
               "meta": meta, **({"fix": fix} if fix else {})}
    except R.Refused as e:  # a fix that can't be joined: what to record instead
        out = {**base, "verdict": "refused", "why": e.why, "fix": {**{k: meta["fix"].get(k) for k in ("base", "from", "to", "patch")}, **e.fix}, "processedAt": now(), "processS": round(time.time() - t0, 1), "meta": meta}
    except Exception as e:  # the page shows this instead of a verdict
        out = {**base, "verdict": "error", "why": [f"{type(e).__name__}: {e}"[:500]], "processedAt": now(), "meta": meta}
    write_atomic(f["result"], R.dumps(out))
    vol.commit()
    rebuild.remote()
    return {k: out[k] for k in ("sessionId", "verdict", "why")}


_REF = {}


def _ref_words():
    import lab_sources as L
    import lab_sessions as S

    if "words" not in _REF:
        _REF["words"] = S.Ref(L.reference()).words
    return _REF["words"]


@app.function(image=cpu_image, volumes={VOL: vol}, memory=4096, timeout=20 * 60, max_containers=1)
def rebuild() -> dict:
    """review/takes.json, manifest-v1.json and index.json from every take's result (latest take per speaker and script
    that isn't a redo is the one used)."""
    import lab_recordings_core as R

    vol.reload()
    root = DATA / R.ROOT
    results = [r for p in sorted(root.glob("*/*/take-*.result.json")) if (r := read(p))]
    cur = R.current_takes([r for r in results if r["verdict"] in ("ok", "listen")])
    by_script = scripts()
    sessions, items = [], []
    for r in cur:
        s = read(root / "sessions" / f"{r['sessionId']}.json")
        script = by_script.get(r["script"])
        if not s or not script or script["sha"] != s["meta"]["scriptSha"]:
            continue
        sessions.append(s)
        items.append(R.review_items(s, r, script, f"{R.ROOT}/sessions/{r['sessionId']}.phone.wav"))
    splits = (read(root / "splits.json") or {}).get("speakers", {})
    write_atomic(root / "manifest-v1.json", R.dumps(R.manifest(sessions, splits)))
    if items:
        write_atomic(root / "review/takes.json", R.dumps(R.queue(items)))
    elif (root / "review/takes.json").exists():
        (root / "review/takes.json").unlink()
    keep = ("speaker", "script", "take", "sessionId", "verdict", "why", "uploadedAt", "processedAt", "audioS", "coverage", "heard", "dropped")
    index = {"version": 1, "updatedAt": now(), "current": [r["sessionId"] for r in cur],
             "takes": [{**{k: r.get(k) for k in keep}, "plants": [{k: p.get(k) for k in ("id", "n", "class", "ayah", "status", "reasons", "llr")} for p in r.get("plants", [])],
                        "flagged": [a["key"] for a in r.get("ayahs", []) if a.get("flag")],
                        **({"fix": {k: r["fix"].get(k) for k in ("base", "from", "to", "joinsS", "gainDb")}} if r.get("fix") else {})} for r in results]}
    write_atomic(root / "index.json", R.dumps(index))
    vol.commit()
    return {"takes": len(results), "current": len(cur), "sessions": len(sessions), "reviewItems": sum(map(len, items))}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def publish_scripts(batch: dict) -> str:
    import lab_recordings_core as R

    vol.reload()
    p = DATA / R.ROOT / "scripts" / f"{R.safe(batch['batch'], 'batch')}.json"
    old = read(p)
    if old and {s["id"]: s["sha"] for s in old["scripts"]} != {s["id"]: s["sha"] for s in batch["scripts"]}:
        changed = [s["id"] for s in batch["scripts"] if s["sha"] != {x["id"]: x["sha"] for x in old["scripts"]}.get(s["id"])]
        print(f"replacing {p.name}: {len(changed)} script(s) changed ({', '.join(changed[:5])}); their old takes keep the old sha and won't process")
    write_atomic(p, R.dumps(batch))
    vol.commit()
    return str(p.relative_to(DATA))


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def list_takes(speaker: str = "", script_id: str = "", take: int = 0) -> list[tuple[str, str, int]]:
    import lab_recordings_core as R

    vol.reload()
    out = []
    for p in sorted((DATA / R.ROOT).glob("*/*/take-*.json")):
        if p.name.endswith(".result.json") or p.parent.parent.name in R.RESERVED:
            continue
        m = read(p)
        if (not speaker or m["speaker"] == speaker) and (not script_id or m["script"] == script_id) and (not take or m["take"] == take):
            out.append((m["speaker"], m["script"], m["take"]))
    return out


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def cleanup(speaker: str, batch: str = "", reviewer: str = "") -> list[str]:
    """Delete a speaker's takes, sessions and results (and optionally a script batch and a reviewer's verdicts)."""
    import shutil

    import lab_recordings_core as R

    vol.reload()
    root, gone = DATA / R.ROOT, []
    d = root / R.safe(speaker, "speaker")
    if d.is_dir() and speaker not in R.RESERVED:
        gone += [str(p.relative_to(DATA)) for p in d.rglob("*") if p.is_file()]
        shutil.rmtree(d)
    for p in sorted((root / "sessions").glob(f"rec-{speaker}-*")):
        gone.append(str(p.relative_to(DATA)))
        p.unlink()
    if batch:
        p = root / "scripts" / f"{R.safe(batch, 'batch')}.json"
        if p.exists():
            gone.append(str(p.relative_to(DATA)))
            p.unlink()
    if reviewer:
        p = Path(VOL) / "reviews" / R.TEST_SET / f"{R.safe(reviewer, 'reviewer')}.json"
        if p.exists():
            gone.append(str(p.relative_to(VOL)))
            p.unlink()
    vol.commit()
    return gone


@app.function(image=cpu_image, volumes={VOL: vol}, memory=4096, timeout=60 * 60)
def join(base: str, name: str) -> dict:
    """/lab/data/<name>/: test set <base> plus the accepted takes as a `recorded` slice (files copied; <base> untouched)."""
    import shutil

    import lab_recordings_core as R

    vol.reload()
    src, dst, rec = DATA / R.safe(base, "base"), DATA / R.safe(name, "name"), DATA / R.ROOT
    if dst.exists():
        raise FileExistsError(f"/lab/data/{name} exists; test set versions are immutable")
    m = json.loads((src / "manifest-v1.json").read_text())
    r = json.loads((rec / "manifest-v1.json").read_text())
    if not r["sessions"]:
        raise ValueError("no accepted takes yet")
    pattern = m["evidencePattern"]
    files = set()
    for e in m["sessions"]:
        files |= set(e["audio"].values())
        for p in src.glob(pattern.replace("{id}", e["id"]).replace("{listener}", "*").replace("{variant}", "*")):
            files.add(str(p.relative_to(src)))
        for extra in (f"sessions/{e['id']}.json", f"sessions/{e['id']}.checks.json"):
            if (src / extra).exists():
                files.add(extra)
    for f in sorted(files):
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)
    # Recorded files go beside the base sessions (ids start rec-), so the evidence pattern finds their teacher evidence.
    for e in r["sessions"]:
        for p in (rec / "sessions").glob(f"{e['id']}.*"):
            if not p.name.endswith(".tmp"):
                (dst / "sessions").mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / "sessions" / p.name)
                files.add(f"sessions/{p.name}")
    out = R.join({**m, "testSet": name, "joinedFrom": base}, r, "sessions/")
    write_atomic(dst / "manifest-v1.json", R.dumps(out))
    vol.commit()
    return {"testSet": name, "base": len(m["sessions"]), "recorded": len(r["sessions"]), "files": len(files)}


@app.function(image=cpu_image, volumes={VOL: vol}, memory=4096, timeout=20 * 60)
def test_audio(script: dict, as_written: bool, read_id: str = "", written: list[str] = (), ayahs: list[str] | None = None,
               ayah_gaps: bool = False, linked: list[str] = (), lead_s: float = 1.0) -> bytes:
    """A synthetic take of `script` (a passage of Al-Mursalat) cut from an mp3quran source, as webm/opus: each run of
    consecutive words is one continuous cut; scripted substitutes, additions, slips and retakes are the reciter's own
    words from elsewhere in the surah; omitted words are cut out. `as_written` ignores the plants (a take that
    doesn't follow the script), `written` only those plants. `ayahs` [from, to]: only those ayahs (a fix's patch).
    `ayah_gaps`: a pause of silence at every ayah end but before the ayahs in `linked`, which keep the source's own
    pause; `lead_s` of silence first."""
    import subprocess
    import tempfile
    import wave

    import numpy as np

    import lab_recordings_core as R
    import lab_sources as L

    vol.reload()
    surah = script["from"]["surah"]
    srcs = sorted((DATA / "sources/mp3quran").glob(f"*/{surah:03d}.align.json"), key=lambda p: int(p.parent.name))
    p = next(p for p in srcs if not read_id or p.parent.name == read_id)
    al = json.loads(p.read_text())
    with wave.open(str(p.with_name(f"{surah:03d}.wav")), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, L.SR)
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    words = {(w["surah"], w["ayah"], w["word"]): w for w in al["words"]}
    order = {k: i for i, k in enumerate(sorted(words))}
    plants = {pl["id"]: pl for pl in script["plants"]}
    as_said = lambda pid: not as_written and pid not in written
    keep, unit = set(R.fix_items(script, *ayahs)) if ayahs else None, R.said_ayahs(script)
    segs = []  # (source word index or None, start, end, ayah)
    for i, it in enumerate(script["said"]):
        role = it["role"]
        if role in ("istiadha", "closing") or keep is not None and i not in keep:  # the source has neither; the take tests dropping them
            continue
        if role == "basmala":
            b = al.get("basmala")
            if b and b.get("present"):
                segs.append((None, b["startS"], b["endS"], unit[i]))
            continue
        if role in ("added", "slip") and not as_said(it["plant"]):
            continue
        if role == "retake" and not as_said(it["plant"]) and it["at"] != plants[it["plant"]]["location"]:
            continue
        at = plants[it["plant"]]["saidFrom"] if role in ("wrong", "added", "slip") and as_said(it["plant"]) else it["at"]
        k = (at["surah"], at["ayah"], at["word"])
        if k not in words:
            raise ValueError(f"reader {p.parent.name} has no timing for {k}")
        segs.append((order[k], words[k]["startS"], words[k]["endS"], unit[i]))
    for pl in script["plants"]:
        if pl["class"] == "omitted_word" and not as_said(pl["id"]) and (keep is None or pl["ayah"] in {unit[i] for i in keep}):
            loc = pl["location"]
            k = (loc["surah"], loc["ayah"], loc["word"])
            j = next(i for i, s in enumerate(segs) if s[0] == order[k] + 1)
            segs.insert(j, (order[k], words[k]["startS"], words[k]["endS"], pl["ayah"]))
    runs = []
    for s in segs:
        if runs and s[0] is not None and runs[-1][0] is not None and s[0] == runs[-1][0] + 1 and not (ayah_gaps and s[3] != runs[-1][3] and s[3] not in linked):
            runs[-1] = (s[0], runs[-1][1], s[2], s[3])
        else:
            runs.append(s)
    gap = np.zeros(int(0.35 * L.SR), dtype="<i2")
    parts = [np.zeros(int(lead_s * L.SR), dtype="<i2")]
    for _, a, b, _ in runs:
        parts += [x[int(a * L.SR) : int(b * L.SR)], gap]
    parts.append(np.zeros(int(1.0 * L.SR), dtype="<i2"))
    y = np.concatenate(parts)
    with tempfile.TemporaryDirectory() as t:
        w = Path(t) / "take.wav"
        with wave.open(str(w), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(L.SR)
            f.writeframes(y.tobytes())
        out = Path(t) / "take.webm"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(w), "-c:a", "libopus", "-b:a", "64k", str(out)], check=True)
        return out.read_bytes()


@app.local_entrypoint()
def main(action: str, file: str = "", speaker: str = "", script: str = "", take: int = 0, batch: str = "", reviewer: str = "",
         base: str = "", name: str = "", out: str = "", read_id: str = ""):
    if action == "scripts":
        b = json.loads(Path(file or ROOT / "docs/benchmarks/lab-scripts-v1.json").read_text())
        print(f"published /lab/data/{publish_scripts.remote(b)}: {len(b['scripts'])} scripts, {b['summary']['estimatedMinutes']} min")
    elif action == "process":
        todo = list_takes.remote(speaker, script, take)
        print(f"processing {len(todo)} take(s)")
        for (sp, sc, n), r in zip(todo, process_take.starmap(todo, return_exceptions=True)):
            print(f"{sp} {sc} take {n}: {r}")
    elif action == "rebuild":
        print(rebuild.remote())
    elif action == "join":
        if not base or not name:
            raise SystemExit("--base <test set> and --name <new test set> are required")
        print(join.remote(base, name))
    elif action == "test-audio":
        b = json.loads(Path(file).read_text())
        d = Path(out or "/tmp/lab-recordings-e2e")
        d.mkdir(parents=True, exist_ok=True)
        for s in b["scripts"]:
            for written in (False, True) if s["plants"] else (False,):
                p = d / f"{s['id']}{'-as-written' if written else ''}.webm"
                p.write_bytes(test_audio.remote(s, written, read_id))
                print(f"{p} ({p.stat().st_size // 1024} KB)")
    elif action == "test-fix":
        # Fixing a passage (rec-test-77): a take with mistake 1 (77:3) read as written and 77:13 run into 77:14, then patches.
        s = next(x for x in json.loads(Path(file).read_text())["scripts"] if x["id"] == "rec-test-77")
        d = Path(out or "/tmp/lab-recordings-fix")
        d.mkdir(parents=True, exist_ok=True)
        todo = {"take": dict(written=["m1"], linked=["77:14"]), "fix-77-3": dict(ayahs=["77:3", "77:3"]),
                "fix-77-3-no-lead": dict(ayahs=["77:3", "77:3"], lead_s=0.0), "fix-77-14": dict(ayahs=["77:14", "77:14"]),
                "fix-77-13-14": dict(ayahs=["77:13", "77:14"])}
        for k, kw in todo.items():
            p = d / f"{k}.webm"
            p.write_bytes(test_audio.remote(s, False, read_id, ayah_gaps=True, **kw))
            print(f"{p} ({p.stat().st_size // 1024} KB)")
    elif action == "cleanup":
        if not speaker:
            raise SystemExit("--speaker is required")
        gone = cleanup.remote(speaker, batch, reviewer)
        print(f"deleted {len(gone)} file(s)")
        print(rebuild.remote())
    else:
        raise SystemExit(f"unknown action {action!r}")
