"""Decode a lab test set with one listener on Modal, one container per session x variant (.scratch/lab, ticket 05).

    modal run modal/lab_decode.py --listener spliced-clean-int8 --model models/spliced-clean-int8   # student ONNX dir under /lab/data
    modal run modal/lab_decode.py --listener teacher                                                 # obadx/muaalem-model-v3_2 on L4
    ... --test-set lab-v1.4 --variants phone,studio --only lab-v1.4-p03 --force

Students run the SDK streaming decode (`pnpm --filter @tilawa/sdk decode`, onnxruntime-node, 20 ms pushes, posteriors
on) through modal/lab_run.py's shard against this worktree's source, so any exported checkpoint can be scored the same
way. The teacher decodes each session as the sources were decoded (lab-sources-v1.md): 30 s windows every 25 s, every
40 ms frame from the window whose centre is nearer, top-5 posteriors and greedy phonemes.

Evidence goes to /lab/data/<testSet>/<manifest evidencePattern> ({id}, {listener}, {variant}). The listener's model
hashes, flags and per-file timings (with the audio's sha256) are recorded in /lab/data/<testSet>/decode-v1.json. Existing
evidence is kept unless --force or its audio was rebuilt since (render's <id>.checks.json hash differs); a listener whose
recorded model hash differs from the one given refuses to mix files without --force.
"""

import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "modal"), str(ROOT / "tools"), "/app/tools"]
import lab_run  # noqa: E402

TEACHER = "obadx/muaalem-model-v3_2"
TEACHER_GPUS = ["L4", "A10", "L40S"]  # in order of preference; L4 capacity runs out
TOPK = 5
BATCH = 8
VOL = "/lab"
DATA = Path(VOL) / "data"
RECORD = "decode-v1.json"
SKIPPED = "LAB_DECODE_SKIPPED"

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "numpy", "rapidfuzz==3.14.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__"])
    .add_local_python_source("lab_run")
)
cpu_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("lab_run")
app = modal.App("tilawa-lab-decode")
app.include(lab_run.app)
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


def write_atomic(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)


@app.cls(image=gpu_image, gpu=TEACHER_GPUS, volumes={VOL: vol, "/cache": cache_vol}, secrets=[hf_secret], memory=16384, timeout=60 * 60, max_containers=40)
class Teacher:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        import lab_sources as L

        self.torch, self.L = torch, L
        m = Muaalem(TEACHER, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor
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
    def decode(self, wav: str, dst: str, force: bool) -> dict:
        import wave

        import numpy as np

        L = self.L
        out = DATA / dst
        vol.reload()
        if out.exists() and not force:
            return {"dst": dst, "skipped": True}
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
        phonemes = [{"p": self.tokens[i - 1], "start": s, "end": e, "conf": round(c, 4)}
                    for i, s, e, c in L.greedy(ids_all[:, 0], p_all[:, 0])]
        ev = {"inventory": self.ref["inventory"]["id"], "frameMs": frame_ms, "nFrames": n, "phonemes": phonemes,
              "posteriors": {"k": TOPK, "ids": ids_all.astype(int).flatten().tolist(), "probs": [round(float(v), 4) for v in p_all.flatten()]},
              "listener": "teacher", "model": TEACHER, "revision": self.revision, "gpu": self.gpu}
        write_atomic(out, json.dumps(ev, ensure_ascii=False, separators=(",", ":")).encode())
        vol.commit()
        return {"dst": dst, "seconds": round(time.time() - t0, 1), "audioS": round(len(x) / L.SR, 3), "nFrames": n, "phonemes": len(phonemes),
                "sha256": sha256(out), "revision": self.revision, "gpu": self.gpu}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def inspect(test_set: str, model: str | None, listener: str) -> dict:
    """The manifest, the listener's recorded entry, and the model files' hashes (students)."""
    vol.reload()
    base = DATA / test_set
    manifest = json.loads((base / "manifest-v1.json").read_text())
    rec = json.loads((base / RECORD).read_text()) if (base / RECORD).exists() else {}
    files = {}
    if model:
        d = DATA / model
        m = json.loads((d / "manifest.json").read_text())
        files = {p.name: sha256(p) for p in sorted(d.iterdir()) if p.is_file()}
        if m.get("sha256") and files.get(m["file"]) != m["sha256"]:
            raise RuntimeError(f"{m['file']} sha256 {files.get(m['file'])} != manifest {m['sha256']}")
    def audio_sha(e: dict) -> dict:
        p = base / f"sessions/{e['id']}.checks.json"
        c = json.loads(p.read_text()) if p.exists() else {}
        return {v: c.get(f"{v}Sha256") for v in ("studio", "phone")}

    pattern = manifest["evidencePattern"]
    have = lambda e, v: (base / pattern.replace("{id}", e["id"]).replace("{listener}", listener).replace("{variant}", v)).exists()
    return {"manifest": {k: v for k, v in manifest.items() if k != "sessions"},
            "sessions": [{"id": e["id"], "split": e["split"], "audio": e["audio"], "durationS": e["durationS"], "audioSha256": audio_sha(e),
                          "have": {v: have(e, v) for v in ("studio", "phone")}} for e in manifest["sessions"]],
            "recorded": rec.get("listeners", {}).get(listener), "files": files}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def finish(test_set: str, listener: str, entry: dict, jobs: list[dict]) -> dict:
    """Hash every evidence file and merge this listener's entry into decode-v1.json."""
    vol.reload()
    base = DATA / test_set
    p = base / RECORD
    rec = json.loads(p.read_text()) if p.exists() else {"version": 1, "testSet": test_set, "listeners": {}}
    old = rec["listeners"].get(listener, {}).get("files", {})
    files = dict(old)
    for j in jobs:
        f = DATA / j["dst"]
        if not f.exists():
            files.pop(j["key"], None)
            continue
        ev = json.loads(f.read_text())
        prev = old.get(j["key"], {})
        timing = j.get("timing") or {k: prev[k] for k in ("seconds", "audioS", "run", "gpu") if k in prev}
        audio = j.get("audioSha256") if j.get("timing") or j.get("fresh") else prev.get("audioSha256")
        files[j["key"]] = {"path": str(f.relative_to(DATA)), "sha256": sha256(f), "nFrames": ev["nFrames"], "frameMs": ev["frameMs"],
                           "phonemes": len(ev["phonemes"]), "posteriorsK": (ev.get("posteriors") or {}).get("k"), **timing,
                           **({"audioSha256": audio} if audio else {})}
    entry["files"] = dict(sorted(files.items()))
    t = [v for v in files.values() if v.get("seconds")]
    entry["totals"] = {"files": len(files), "audioS": round(sum(v.get("audioS") or 0 for v in files.values()), 1),
                       "containerS": round(sum(v["seconds"] for v in t), 1)}
    rec["listeners"][listener] = entry
    rec["updatedAt"] = now()
    write_atomic(p, (json.dumps(rec, ensure_ascii=False, indent=1) + "\n").encode())
    vol.commit()
    return entry["totals"]


@app.local_entrypoint()
def main(listener: str, model: str = "", test_set: str = "lab-v1.4", variants: str = "phone,studio", only: str = "", force: bool = False,
         push_ms: int = 20, evidence_ms: int = 0):
    t0 = time.time()
    teacher = listener == "teacher"
    if teacher == bool(model):
        raise SystemExit("--model <dir under /lab/data> is required for a student listener and not allowed for the teacher")
    info = inspect.remote(test_set, model or None, listener)
    pattern = info["manifest"]["evidencePattern"]
    rec = info["recorded"] or {}
    recorded = rec.get("files", {})
    jobs = []
    for s in info["sessions"]:
        if only and s["id"] not in only.split(","):
            continue
        for v in variants.split(","):
            if v not in s["audio"]:
                continue
            key, cur = f"{s['id']}.{v}", s["audioSha256"].get(v)
            dst = f"{test_set}/" + pattern.replace("{id}", s["id"]).replace("{listener}", listener).replace("{variant}", v)
            known = recorded.get(key, {}).get("audioSha256")
            # Evidence of audio that was rebuilt since is stale: decode it again. Files recorded before audio hashes were kept are trusted.
            fresh = s["have"][v] and (known == cur if known and cur else True)
            jobs.append({"key": key, "wav": f"{test_set}/{s['audio'][v]}", "dst": dst, "audioS": s["durationS"], "audioSha256": cur,
                         "fresh": fresh and not force, "force": force or (s["have"][v] and not fresh)})
    stale = sum(j["force"] and not force for j in jobs)
    if stale:
        print(f"{stale} evidence file(s) are of audio rebuilt since; decoding them again")
    src, sha = lab_run.source()
    model_id = {"hf": TEACHER} if teacher else {"path": model, "files": info["files"]}
    if rec and {k: rec.get("model", {}).get(k) for k in ("hf", "path", "files")} != {k: model_id.get(k) for k in ("hf", "path", "files")} and not force:
        raise SystemExit(f"{listener} evidence was decoded from {rec.get('model')}; pass --force to replace it with {model_id}")
    flags = {"teacher": {"windowS": 30, "hopS": 25, "frameMs": 40, "topK": TOPK, "dtype": "bfloat16", "gpu": TEACHER_GPUS}} if teacher else \
        {"student": {"cli": "pnpm --filter @tilawa/sdk decode", "runtime": "onnxruntime-node (cpu, 1 thread)", "pushMs": push_ms,
                     "evidenceMs": evidence_ms or "model frameMs", "posteriors": "top-k from the session evidence", "mode": "free recitation"}}
    print(f"{listener}: {len(jobs)} files ({test_set}, {variants}) from {'hf ' + TEACHER if teacher else '/lab/data/' + model}, source {sha}")
    results = {}
    if teacher:
        todo = [j for j in jobs if not j["fresh"]]
        for j in jobs:
            if j["fresh"]:
                results[j["key"]] = {"skipped": True}
        for j, r in zip(todo, list(Teacher().decode.starmap([(j["wav"], j["dst"], j["force"]) for j in todo], return_exceptions=True, wrap_returned_exceptions=False)) if todo else []):
            results[j["key"]] = r
    else:
        run = f"decode-{listener}-{sha}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        flag = f" --evidence-ms {evidence_ms}" if evidence_ms else ""
        cmds, todo = [], [j for j in jobs if not j["fresh"]]
        for j in jobs:
            if j["fresh"]:
                results[j["key"]] = {"skipped": True}
        for j in todo:
            stem = Path(j["dst"]).name.removesuffix(".evidence.json")
            dst, wav = shlex.quote(f"/lab/data/{j['dst']}"), shlex.quote(f"/lab/data/{j['wav']}")
            skip = "" if j["force"] else f"if [ -f {dst} ]; then echo {SKIPPED}; exit 0; fi; "
            # The SDK imports @tilawa/core from its dist.
            cmds.append(f"set -e; {skip}pnpm --filter @tilawa/core build; d=$(mktemp -d); ln -s {wav} $d/{shlex.quote(stem)}.wav; "
                        f"pnpm --filter @tilawa/sdk decode --model {shlex.quote(f'/lab/data/{model}')} --wavs $d --out $d/out --push {push_ms}{flag}; "
                        f"mkdir -p $(dirname {dst}); mv $d/out/{shlex.quote(stem)}.evidence.json {dst}.tmp; mv {dst}.tmp {dst}")
        print(f"run {run}")
        statuses = list(lab_run.shard.starmap([(src, c, run, i, len(todo)) for i, c in enumerate(cmds)], return_exceptions=True, wrap_returned_exceptions=False)) if todo else []
        for j, st in zip(todo, statuses):
            if isinstance(st, Exception) or st["code"]:
                results[j["key"]] = st if isinstance(st, Exception) else RuntimeError(st["tail"][-600:])
            else:
                results[j["key"]] = {"skipped": True} if SKIPPED in st["tail"] else {"seconds": st["seconds"], "audioS": j["audioS"], "run": run}
    failed = {k: r for k, r in results.items() if isinstance(r, Exception)}
    for k, r in failed.items():
        print(f"FAIL {k}: {r!r}"[:800])
    done = [{"key": j["key"], "dst": j["dst"], "audioSha256": j["audioSha256"], "fresh": j["fresh"],
             "timing": None if isinstance(results[j["key"]], Exception) or results[j["key"]].get("skipped")
             else {"seconds": results[j["key"]]["seconds"], "audioS": results[j["key"]]["audioS"], "run": results[j["key"]].get("run"),
                   **({"gpu": results[j["key"]]["gpu"]} if results[j["key"]].get("gpu") else {})}} for j in jobs]
    revision = next((r.get("revision") for r in results.values() if isinstance(r, dict) and r.get("revision")), None)
    if teacher and revision:
        model_id["revision"] = revision
    elif teacher and rec.get("model", {}).get("revision"):
        model_id["revision"] = rec["model"]["revision"]
    entry = {"kind": "teacher" if teacher else "student", "model": model_id, "flags": flags, "source": sha, "decodedAt": now(),
             "evidencePattern": pattern, "wallS": round(time.time() - t0, 1)}
    totals = finish.remote(test_set, listener, entry, done)
    print(f"{len(jobs) - len(failed)}/{len(jobs)} ok in {time.time() - t0:.0f} s wall; recorded {totals} in /lab/data/{test_set}/{RECORD}")
    if failed:
        raise SystemExit(1)
