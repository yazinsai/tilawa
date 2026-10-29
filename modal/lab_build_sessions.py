"""Build test set lab-v1.1: spliced sessions with planted mistakes, studio and phone copies (.scratch/lab, tickets 03, 04).

    modal run modal/lab_build_sessions.py --pilot          # bank, plan, then 2 sessions rendered and checked
    modal run modal/lab_build_sessions.py                  # every session
    modal run modal/lab_build_sessions.py --stage render   # re-render from the stored plan
    modal run modal/lab_build_sessions.py --stage loop     # build, seam gate, teacher decode, automated checks, refuse failures, repeat; then triage
    modal run modal/lab_build_sessions.py --stage gate     # the seam gate over the rendered set, report only

Stages (each skips finished outputs; the same seed and inputs give byte-identical files):
    aug     held-out noise (freesound-commercial-50k shards 100-103; the student uses 0-2) and simulated small and
            medium room RIRs not in the student bank, at 16 kHz -> <set>/aug/{noise,rir}.npz + bank-v1.json (copied
            byte for byte from lab-v1 when it has them)
    plan    windows, edits, labels from the catalog, alignments and refusals-v1.json (tools/lab_sessions.py) ->
            <set>/sessions/<id>.json, plan-v1.json, manifest-v1.json, summary-v1.json, review/v1-check.json
    render  per session, in parallel: <id>.studio.wav and <id>.phone.wav, plus <id>.checks.json (seam levels and
            clicks, every said word found at its labelled time in the output, phone/studio lag)
    refuse  the automated checks' studio failures (checks-v1.json, modal/lab_check.py) -> refusals-v1.json, cumulative:
            the next plan rebuilds each failed case elsewhere or drops it (spec decision 16); nothing is patched
    gate    the render's seam checks against lab_sessions.SEAM_LIMITS (levels, clicks, seam depth, words found, words
            cut short) -> seams-v1.json; in the loop, failing splices are refused and rebuilt before any decode
    loop    plan + render, gate, `lab_decode.py --listener teacher`, `lab_check.py`, refuse; until nothing fails. Then
            `lab_triage.py`: both review models score every case and write the listening queue review/triage.json
            (ticket 17); the set freezes once that queue is reviewed within its maxBad

All under /lab/data/<set> (lab_sessions.TEST_SET) on the tilawa-lab volume.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/app/tools"]
from lab_sessions import TEST_SET  # noqa: E402

VOL = "/lab"
DATA = Path(VOL) / "data"
OUT = DATA / TEST_SET
AUG_FROM = DATA / "lab-v1/aug"  # the phone-copy bank is shared across test set versions
REFUSALS = "refusals-v1.json"
NOISE_SHARDS = (100, 101, 102, 103)
NOISE_CLIPS = 400
NOISE_S = (4.0, 12.0)
RIR_ROOMS = {"smallroom": 120, "mediumroom": 80}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("numpy==2.3.3", "scipy==1.16.2", "soundfile==0.13.1", "soxr==1.0.0", "pyarrow==21.0.0", "huggingface_hub==0.35.3")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__"])
)
app = modal.App("tilawa-lab-build-sessions", image=image)
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache")
student_vol = modal.Volume.from_name("tilawa-student")


def write(p: Path, data: bytes) -> bool:
    """Atomic write; False (and no write) when the file already holds these bytes."""
    if p.exists() and p.read_bytes() == data:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)
    return True


def dumps(x) -> bytes:
    return (json.dumps(x, ensure_ascii=False, indent=1) + "\n").encode()


def read_wav(p: Path):
    import wave

    import numpy as np

    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def pack(waves):
    import numpy as np

    lens = np.array([len(w) for w in waves])
    return {"data": np.concatenate(waves).astype(np.float16), "offsets": np.concatenate([[0], np.cumsum(lens)])}


def unpack(p: Path):
    import numpy as np

    z = np.load(p)
    d, o = z["data"], z["offsets"]
    return [d[o[i]: o[i + 1]].astype(np.float32) for i in range(len(o) - 1)]


@app.function(volumes={VOL: vol, "/cache": cache_vol, "/student": student_vol}, cpu=8, memory=32768, timeout=2 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def aug(force: bool = False) -> dict:
    import glob
    import io
    import os
    import random
    import zipfile

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    import lab_sessions as S

    bank_json = OUT / "aug/bank-v1.json"
    if bank_json.exists() and not force:
        return json.loads(bank_json.read_text())
    if OUT / "aug" != AUG_FROM and (AUG_FROM / "bank-v1.json").exists():
        bank = json.loads((AUG_FROM / "bank-v1.json").read_text())
        (OUT / "aug").mkdir(parents=True, exist_ok=True)
        for k in ("noise", "rir"):
            data = (AUG_FROM / f"{k}.npz").read_bytes()
            assert hashlib.sha256(data).hexdigest() == bank["sha256"][k], k
            write(OUT / f"aug/{k}.npz", data)
        write(bank_json, dumps(bank))
        vol.commit()
        return bank

    def load(data: bytes):
        w, sr = sf.read(io.BytesIO(data), dtype="float32")
        if w.ndim > 1:
            w = w[:, 0]
        return soxr.resample(w, sr, S.SR) if sr != S.SR else w

    rng = random.Random(f"{S.SEED}:aug")
    rows = []
    for shard in NOISE_SHARDS:
        f = hf_hub_download("obadx/freesound-commercial-50k-noise-only", f"data/train-{shard:05d}-of-00104.parquet", repo_type="dataset")
        t = pq.read_table(f)
        cols = [c for c in t.column_names if c != "audio"]
        meta = t.select(cols).to_pylist() if cols else [{}] * t.num_rows
        for i, a in enumerate(t.column("audio").to_pylist()):
            rows.append((shard, i, a, meta[i]))
    rng.shuffle(rows)
    noises, noise_meta = [], []
    for shard, i, a, m in rows:
        if len(noises) == NOISE_CLIPS:
            break
        try:
            w = load(a["bytes"])
        except Exception:
            continue
        if len(w) < NOISE_S[0] * S.SR:
            continue
        w = w[: int(NOISE_S[1] * S.SR)].astype(np.float64)
        w -= w.mean()
        if np.sqrt(np.mean(w**2)) < 1e-3:
            continue
        noises.append((w / np.abs(w).max()).astype(np.float32))
        noise_meta.append({"name": f"freesound-commercial-50k/train-{shard:05d}#{i}", "durationS": round(len(w) / S.SR, 4),
                           **{k: v for k, v in m.items() if isinstance(v, (str, int, float)) and k in ("id", "license", "username", "name", "title")}})

    def h(w):
        return hashlib.sha256((w / max(1e-6, np.abs(w).max())).astype(np.float16).tobytes()).hexdigest()

    student = {hashlib.sha256(w.astype(np.float16).tobytes()).hexdigest() for w in unpack(Path("/student/aug/rir.npz"))}
    zpath = hf_hub_download("EaseZh/rirs_noises", "rirs_noises.zip", repo_type="dataset")
    os.makedirs("/tmp/rir", exist_ok=True)
    zipfile.ZipFile(zpath).extractall("/tmp/rir")
    rirs, rir_meta, dropped = [], [], 0
    for room, n in RIR_ROOMS.items():
        paths = sorted(glob.glob(f"/tmp/rir/**/simulated_rirs/{room}/**/*.wav", recursive=True))
        random.Random(f"{S.SEED}:rir:{room}").shuffle(paths)
        got = 0
        for p in paths:
            if got == n:
                break
            w = load(open(p, "rb").read())[: S.SR]
            if h(w) in student:
                dropped += 1
                continue
            rirs.append((w / max(1e-6, np.abs(w).max())).astype(np.float32))
            rir_meta.append({"name": "openslr-28/" + p.split("simulated_rirs/")[1], "room": room})
            got += 1
    (OUT / "aug").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "aug/noise.npz", **pack(noises))
    np.savez(OUT / "aug/rir.npz", **pack(rirs))
    bank = {"version": 1, "seed": S.SEED,
            "sources": {"noise": "obadx/freesound-commercial-50k-noise-only (commercial-use licences), shards " + ",".join(map(str, NOISE_SHARDS)) + "; the student bank uses shards 0-2",
                        "rir": "EaseZh/rirs_noises (OpenSLR-28) simulated small and medium rooms, minus every RIR in the student bank (tilawa-student/aug/rir.npz)"},
            "noises": noise_meta, "rirs": rir_meta, "rirsDroppedAsStudent": dropped,
            "sha256": {k: hashlib.sha256((OUT / f"aug/{k}.npz").read_bytes()).hexdigest() for k in ("noise", "rir")}}
    write(bank_json, dumps(bank))
    vol.commit()
    return bank


@app.function(volumes={VOL: vol}, cpu=4, memory=16384, timeout=60 * 60)
def plan(pilot: bool = False) -> dict:
    import lab_sessions as S
    from lab_review_core import validate_queue

    vol.reload()
    t0 = time.time()
    ref = S.Ref(json.loads(Path("/app/packages/core/assets/ref-v2.json").read_text()))
    dis = json.loads((DATA / "reference/disambiguation-v1.json").read_text())["words"]
    catalog = json.loads((DATA / "sources/mp3quran/catalog-v1.json").read_text())
    aligns = {it["align"]: json.loads((DATA / it["align"]).read_text()) for it in catalog["items"] if it["readId"] not in S.EXCLUDED_READS}
    bank = json.loads((OUT / "aug/bank-v1.json").read_text())
    rp = OUT / REFUSALS
    refusals = json.loads(rp.read_text()) if rp.exists() else None
    out = S.plan_set(catalog, aligns, ref, dis, bank, refusals=refusals)
    sessions = out["sessions"]
    queue = S.review_queue(sessions, ref)
    problems = validate_queue(queue)
    if problems:
        raise ValueError(f"review queue: {problems[:5]}")
    for s in sessions:
        problems = S.validate(s, ref)
        if problems:
            raise ValueError(f"{s['id']}: {problems[:3]}")
    refused = {**out["refused"], "rules": {"lenRatioMin": S.LEN_RATIO_MAX, "ayahPerMax": S.AYAH_PER_MAX, "weakOnsets": "".join(sorted(S.WEAK_ONSET))},
               "refusalsSha256": hashlib.sha256(rp.read_bytes()).hexdigest() if rp.exists() else None,
               "iterations": max((x["iteration"] for k in ("ayahs", "sites", "donors") for x in (refusals or {}).get(k, [])), default=0)}
    summary = S.summary(sessions, {"bank": {k: bank[k] for k in ("sources", "sha256")} | {"noises": len(bank["noises"]), "rirs": len(bank["rirs"])},
                                   "refused": refused})
    changed = 0
    for s in sessions:
        changed += write(OUT / f"sessions/{s['id']}.json", dumps(s))
    changed += write(OUT / "plan-v1.json", dumps({"version": 1, "testSet": S.TEST_SET, "seed": S.SEED, "plans": out["plans"]}))
    changed += write(OUT / "manifest-v1.json", dumps(S.manifest(sessions)))
    changed += write(OUT / "summary-v1.json", dumps(summary))
    changed += write(OUT / "review/v1-check.json", dumps(queue))
    vol.commit()
    ids = [p["id"] for p in out["plans"]]
    if pilot:
        ids = [next(s["id"] for s in sessions if not s["meta"]["clean"] and s["meta"]["mutashabihat"]), next(s["id"] for s in sessions if s["meta"]["clean"])]
    return {"ids": ids, "summary": summary, "changed": changed, "planS": round(time.time() - t0, 1),
            "example": next(s for s in sessions if s["id"] == ids[0])}


@app.function(volumes={VOL: vol}, timeout=10 * 60)
def refuse(iteration: int) -> dict:
    """checks-v1.json's studio failures -> refusals-v1.json (cumulative). Returns the new entries."""
    import lab_sessions as S

    vol.reload()
    checks = json.loads((OUT / "checks-v1.json").read_text())
    m = json.loads((OUT / "manifest-v1.json").read_text())
    sessions = {e["id"]: json.loads((OUT / f"sessions/{e['id']}.json").read_text()) for e in m["sessions"]}
    rp = OUT / REFUSALS
    doc, new = S.refusals_from_checks(checks, sessions, json.loads(rp.read_text()) if rp.exists() else None, iteration)
    write(rp, dumps(doc))
    vol.commit()
    return {"new": new, "totals": {k: len(doc[k]) for k in ("ayahs", "sites", "donors", "unmapped")}, "summary": checks["summary"]}


@app.function(volumes={VOL: vol}, cpu=2, memory=8192, timeout=20 * 60)
def gate(iteration: int, refuse_failed: bool = True) -> dict:
    """The render's seam checks against lab_sessions.SEAM_LIMITS -> seams-v1.json; with refuse_failed, failing splices
    join refusals-v1.json like automated-check failures (site and donor), so the next plan rebuilds them."""
    import lab_sessions as S

    vol.reload()
    m = json.loads((OUT / "manifest-v1.json").read_text())
    sessions = {e["id"]: json.loads((OUT / f"sessions/{e['id']}.json").read_text()) for e in m["sessions"]}
    rows, aligns = [], {}
    for sid, s in sessions.items():
        for audio in sorted({t["from"]["audio"] for t in s["said"]} - set(aligns)):
            a = json.loads((DATA / audio.replace(".wav", ".align.json")).read_text())
            aligns[audio] = {(w["surah"], w["ayah"], w["word"]): w for w in a["words"]}
        rows.append(S.seam_gate(s, json.loads((OUT / f"sessions/{sid}.checks.json").read_text()), aligns))
    doc = {**S.seam_gate_summary(rows, TEST_SET), "iteration": iteration}
    write(OUT / "seams-v1.json", dumps(doc))
    new = []
    if refuse_failed and doc["failed"]:
        rp = OUT / REFUSALS
        ref, new = S.refusals_from_checks({"testSet": TEST_SET, "failed": doc["failed"]}, sessions,
                                          json.loads(rp.read_text()) if rp.exists() else None, iteration)
        write(rp, dumps(ref))
    vol.commit()
    return {"summary": doc["summary"], "failed": doc["failed"], "unmapped": doc["unmapped"], "new": new}


def seam_checks(y, plan_pieces, session, load, gains):
    """Per seam: output level just before and after it against the same 20 ms of source on each side (gain applied),
    and the peak sample step inside the crossfade against the steps around it."""
    import numpy as np

    import lab_sessions as S

    def db(x):
        return 20 * np.log10(max(S.rms(x), 1e-6))

    n, out = 320, []
    h = S.XFADE // 2
    for i in range(1, len(plan_pieces)):
        a, b = plan_pieces[i - 1], plan_pieces[i]
        m = b["out"]
        xa, xb = load(a["audio"]), load(b["audio"])
        ga, gb = gains[i - 1], gains[i]
        src_before = xa[a["src"][1] - h - n: a["src"][1] - h] * ga
        src_after = xb[b["src"][0] + h: b["src"][0] + h + n] * gb
        o_before, o_after = y[m - h - n: m - h], y[m + h: m + h + n]
        d = np.abs(np.diff(y[max(0, m - 4 * n): m + 4 * n].astype(np.float64)))
        local = d[: len(d) // 2 - S.XFADE].tolist() + d[len(d) // 2 + S.XFADE:].tolist()
        xf = np.abs(np.diff(y[m - h - 1: m + h + 1].astype(np.float64)))
        out.append({"outS": round(m / S.SR, 4), "beforeDb": round(db(o_before), 2), "afterDb": round(db(o_after), 2),
                    "srcBeforeDb": round(db(src_before), 2), "srcAfterDb": round(db(src_after), 2),
                    "levelErrDb": round(max(abs(db(o_before) - db(src_before)), abs(db(o_after) - db(src_after))), 3),
                    "seamDb": round(db(y[m - h: m + h]) - 20 * np.log10(max(S.active_rms(y), 1e-6)), 2),
                    "clickRatio": round(float(xf.max() / max(np.percentile(local, 99), 1e-6)), 3) if local else None})
    return out


def word_checks(y, session, load, aligns):
    """Every said word found at its labelled time: the 60 ms around the middle of its source audio, cross-correlated
    with the output over +-50 ms around the labelled middle. Words clipped by a cut (shorter in the output) are counted
    apart."""
    import numpy as np

    import lab_sessions as S

    W, L = int(0.03 * S.SR), int(0.05 * S.SR)
    res = {"words": 0, "found": 0, "clipped": 0, "worst": []}
    for t in session["said"]:
        f = t["from"]
        w = aligns[f["audio"]][(f["surah"], f["ayah"], f["word"])]
        out_len, src_len = t["endS"] - t["startS"], w["endS"] - w["startS"]
        res["words"] += 1
        if abs(out_len - src_len) > 0.0015:
            res["clipped"] += 1
            continue
        x = load(f["audio"])
        cs, co = round((w["startS"] + w["endS"]) / 2 * S.SR), round((t["startS"] + t["endS"]) / 2 * S.SR)
        ref = x[cs - W: cs + W].astype(np.float64)
        seg = y[co - W - L: co + W + L].astype(np.float64)
        if len(ref) < 2 * W or len(seg) < 2 * (W + L):
            res["clipped"] += 1
            continue
        best, lag = -2.0, None
        for k in range(0, 2 * L + 1):
            s = seg[k: k + 2 * W]
            c = float(np.dot(ref, s) / max(np.linalg.norm(ref) * np.linalg.norm(s), 1e-9))
            if c > best:
                best, lag = c, k - L
        ok = lag == 0 and best > 0.999
        res["found"] += ok
        if not ok:
            res["worst"].append({"at": t["at"], "role": t["role"], "lagSamples": lag, "corr": round(best, 4)})
    res["worst"] = res["worst"][:10]
    return res


@app.function(volumes={VOL: vol}, cpu=2, memory=16384, timeout=30 * 60, max_containers=40)
def render(sid: str, force: bool = False) -> dict:
    import numpy as np

    import lab_sessions as S

    vol.reload()
    session = json.loads((OUT / f"sessions/{sid}.json").read_text())
    plans = {p["id"]: p for p in json.loads((OUT / "plan-v1.json").read_text())["plans"]}
    p = plans[sid]
    key = hashlib.sha256(json.dumps({"plan": p, "phone": session["meta"]["phone"], "bank": json.loads((OUT / "aug/bank-v1.json").read_text())["sha256"]},
                                    sort_keys=True).encode()).hexdigest()
    checks_path = OUT / f"sessions/{sid}.checks.json"
    studio_p, phone_p = OUT / session["meta"]["audio"]["studio"], OUT / session["meta"]["audio"]["phone"]
    if not force and checks_path.exists() and studio_p.exists() and phone_p.exists() and json.loads(checks_path.read_text()).get("planSha256") == key:
        return {**json.loads(checks_path.read_text()), "skipped": True}
    t0 = time.time()
    cache = {}

    def load(audio):
        if audio not in cache:
            cache[audio] = read_wav(DATA / audio)
        return cache[audio]

    y = S.render(p["pieces"], load)
    assert len(y) == p["samples"], (len(y), p["samples"])
    noises, rirs = unpack(OUT / "aug/noise.npz"), unpack(OUT / "aug/rir.npz")
    ph = session["meta"]["phone"]
    z = S.degrade(y, rirs[ph["rir"]], [noises[i] for i in ph["noise"]], ph["snrDb"])
    studio, phone = S.pcm16(y), S.pcm16(z)
    write(studio_p, studio)
    write(phone_p, phone)

    aligns = {}
    for audio in sorted({t["from"]["audio"] for t in session["said"]}):
        a = json.loads((DATA / audio.replace(".wav", ".align.json")).read_text())
        aligns[audio] = {(w["surah"], w["ayah"], w["word"]): w for w in a["words"]}
    ys = read_wav(studio_p)
    gains = [10 ** (q["gainDb"] / 20) for q in p["pieces"]]
    seams = seam_checks(ys, p["pieces"], session, load, gains)
    words = word_checks(ys, session, load, aligns)
    exact = []
    for q, g in zip(p["pieces"], gains):
        a, b = q["out"] + S.EDGE_FADE, q["out"] + q["len"] - S.EDGE_FADE
        src = load(q["audio"])[q["src"][0] + S.EDGE_FADE: q["src"][1] - S.EDGE_FADE] * g
        exact.append(float(np.max(np.abs(ys[a:b] - np.clip(src, -1, 32767 / 32768)))) if b > a else 0.0)
    zp = read_wav(phone_p)
    lags = []
    for c in np.linspace(5, len(ys) / S.SR - 5, 6):
        i = int(c * S.SR)
        a, b = ys[i - S.SR: i + S.SR].astype(np.float64), zp[i - S.SR - 400: i + S.SR + 400].astype(np.float64)
        cc = [float(np.dot(a, b[k: k + len(a)])) for k in range(0, 801, 4)]
        lags.append(int(np.argmax(cc)) * 4 - 400)
    speech = S.active_rms(ys)
    checks = {"id": sid, "planSha256": key, "samples": len(ys), "durationS": round(len(ys) / S.SR, 4),
              "labelDurationS": session["durationS"], "studioSha256": hashlib.sha256(studio).hexdigest(), "phoneSha256": hashlib.sha256(phone).hexdigest(),
              "pieceExactMaxErr": round(max(exact), 6), "seams": seams,
              "seamLevelErrDbMax": max((s["levelErrDb"] for s in seams), default=0), "clickRatioMax": max((s["clickRatio"] or 0 for s in seams), default=0),
              "words": words, "phoneLagSamples": lags, "studioActiveDbfs": round(20 * np.log10(max(speech, 1e-6)), 2),
              "phonePeak": round(float(np.max(np.abs(zp))), 4), "renderS": round(time.time() - t0, 1)}
    write(checks_path, dumps(checks))
    vol.commit()
    return checks


@app.local_entrypoint()
def main(stage: str = "all", pilot: bool = False, force: bool = False, iterations: int = 6):
    t0 = time.time()
    if stage in ("all", "aug", "loop"):
        bank = aug.remote(force)
        print(f"bank: {len(bank['noises'])} noises, {len(bank['rirs'])} rirs ({bank['rirsDroppedAsStudent']} dropped as student) [{time.time() - t0:.0f} s]")
    if stage == "loop":
        for it in range(1, iterations + 1):
            print(f"\n== iteration {it}: build [{time.time() - t0:.0f} s]", flush=True)
            if build(False, False, t0):
                raise SystemExit("render failed")
            g = gate.remote(it)
            print_gate(g, f"== iteration {it}: seam gate", t0)
            if g["new"]:
                continue
            if g["failed"]:
                raise SystemExit("seam gate failures that refusals didn't remove")
            for cmd in (["modal/lab_decode.py", "--listener", "teacher", "--test-set", TEST_SET],
                        ["modal/lab_check.py", "--test-set", TEST_SET]):
                print(f"== iteration {it}: {' '.join(cmd)} [{time.time() - t0:.0f} s]", flush=True)
                subprocess.run(["modal", "run", *cmd], cwd=ROOT, check=True)
            r = refuse.remote(it)
            st = r["summary"]["studio"]
            print(f"== iteration {it}: studio mistakes {st['mistakes']['pass']}/{st['mistakes']['n']}, non-errors {st['nonErrors']['pass']}/{st['nonErrors']['n']}, "
                  f"clean {st['clean']['pass']}/{st['clean']['n']}, timing {st['timing']['pass']}/{st['timing']['sessions']}; "
                  f"{len(r['new'])} new refusals, totals {r['totals']}", flush=True)
            for x in r["new"]:
                print(f"  {x['kind']}: {json.dumps({k: v for k, v in x.items() if k not in ('kind', 'iteration')}, ensure_ascii=False)}")
            if not [x for x in r["new"] if x["kind"] != "unmapped"]:
                print(f"converged after {it} iteration(s)")
                break
        else:
            raise SystemExit(f"not converged after {iterations} iterations; no triage")
        stage = "triage"
    if stage == "gate":
        print_gate(gate.remote(0, refuse_failed=False), "seam gate (report only)", t0)
    if stage == "triage":
        cmd = ["modal/lab_triage.py", "--test-set", TEST_SET]
        print(f"== triage: {' '.join(cmd)} [{time.time() - t0:.0f} s]", flush=True)
        subprocess.run(["modal", "run", *cmd], cwd=ROOT, check=True)
    if stage in ("all", "plan", "render"):
        build(pilot, force, t0, render_too=stage != "plan")
    print(f"total {time.time() - t0:.0f} s")


def print_gate(g: dict, head: str, t0: float) -> None:
    s = g["summary"]
    print(f"{head}: seams {s['seamsPass']}/{s['seams']} pass (level {s['byCheck']['level']}, click {s['byCheck']['click']}, depth {s['byCheck']['depth']}), "
          f"words {s['wordsFound']}/{s['wordsChecked']} found, {s['clipsOver']}/{s['clipped']} clipped words over the limit, "
          f"{s['splicesFailed']} splices failed, {len(g['new'])} new refusals [{time.time() - t0:.0f} s]", flush=True)
    for f in g["failed"]:
        print(f"  {f['id']}: {'; '.join(f['reasons'])}")
    for u in g["unmapped"]:
        print(f"  unmapped: {json.dumps(u, ensure_ascii=False)}")


def build(pilot: bool, force: bool, t0: float, render_too: bool = True) -> int:
    """Plan (writes the docs summary unless a pilot), then render; returns the number of failed renders."""
    res = plan.remote(pilot)
    print(f"plan: {len(res['ids'])} to render, {res['changed']} files changed, {res['planS']} s [{time.time() - t0:.0f} s]")
    print(json.dumps(res["summary"]["all"], ensure_ascii=False), json.dumps(res["summary"]["refused"], ensure_ascii=False))
    if not pilot:
        s = res["summary"]
        (ROOT / f"docs/benchmarks/{TEST_SET}.json").write_text(json.dumps({k: s[k] for k in ("version", "testSet", "seed", "all", "splits", "surahs", "bank", "refused")}
                                                                         | {"reciters": len(s["reciters"])}, ensure_ascii=False, indent=1) + "\n")
    (ROOT / ".scratch/lab/cache").mkdir(parents=True, exist_ok=True)
    (ROOT / f".scratch/lab/cache/{TEST_SET}-example.json").write_text(json.dumps(res["example"], ensure_ascii=False, indent=1))
    if not render_too:
        return 0
    results = list(render.map(res["ids"], kwargs={"force": force}, return_exceptions=True, wrap_returned_exceptions=False))
    bad = [r for r in results if isinstance(r, Exception)]
    ok = [r for r in results if not isinstance(r, Exception)]
    for r in ok:
        w = r["words"]
        if not r.get("skipped"):
            print(f"{r['id']}: {r['durationS']} s (label {r['labelDurationS']}), exact {r['pieceExactMaxErr']}, seam level err max {r['seamLevelErrDbMax']} dB, "
                  f"click max {r['clickRatioMax']}, words {w['found']}/{w['words'] - w['clipped']} found ({w['clipped']} clipped), phone lag {r['phoneLagSamples']}")
    print(f"render: {sum(not r.get('skipped') for r in ok)} rendered, {sum(bool(r.get('skipped')) for r in ok)} unchanged, {len(bad)} failed [{time.time() - t0:.0f} s]")
    (ROOT / f".scratch/lab/cache/{TEST_SET}-checks.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1))
    for e in bad:
        print("FAILED:", repr(e)[:500])
    return len(bad)
