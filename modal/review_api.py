"""Commercial ASR APIs as the second independent model of two-model triage (.scratch/lab/issues/12-review-asr-calibration.md,
17-two-model-triage.md; docs/benchmarks/lab-review-asr-v1.md, section "Commercial APIs").

    modal run modal/review_api.py::smoke                                   # one window outside every queue, both vendors
    modal run modal/review_api.py::bakeoff_main                            # both vendors on your verdicts + ticket 12 controls
    modal run modal/review_api.py::triage_main --vendor soniox             # lab-v1.3 re-triage with Nemotron + the winner

Vendors, called only from Modal functions that mount their own secret (the keys are read from the environment and never
printed, logged or saved):
- Soniox async: POST /v1/files, POST /v1/transcriptions {model stt-async-v5, language_hints ['ar']}, poll, GET the
  transcript, then delete the transcription and the file.
- Speechmatics batch: POST /v2/jobs {language ar, model enhanced}, poll, GET the json-v2 transcript, then delete the job.

Each request carries only the window's 16 kHz mono PCM16 WAV (review_asr.clip_bytes: the bytes every other model heard),
named by its sha256, and the language. Never the text, the label, a context or a vocabulary. Per test set,
/lab/reviews/<testSet>/asr-<vendor>.json keeps the vendor, model, request config, and per item the audio sha256, raw
responses, transcript, timing and estimated cost; transcripts are reused by sha256 and config unless --force. The
readings (tools/review_api_core.py) are added to the same records.
"""

import hashlib
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "modal"), "/root"]
import review_asr as RA  # noqa: E402

VOL = "/lab"
DATA, REVIEWS = Path(VOL) / "data", Path(VOL) / "reviews"
TEST_SETS = ("lab-v1.1", "lab-v1.2", "lab-v1.3")
CONTROLS = "lab-v1.1/review/v1-check.json"  # ticket 12's 40 windows, for the synthetic controls
SUMMARY = "lab-v1.1/asr-calibration.api-bakeoff.json"
SONIOX_API = "https://api.soniox.com"
SONIOX_CONFIG = {"model": "stt-async-v5", "language_hints": ["ar"]}  # + language_hints_strict when GET /v1/models says it's supported
SPEECHMATICS_API = "https://eu1.asr.api.speechmatics.com/v2"
SPEECHMATICS_CONFIG = {"type": "transcription", "transcription_config": {"language": "ar", "model": "enhanced"}}
PRICE_PER_HOUR = {"soniox": 0.10, "speechmatics": 0.75}  # USD list prices on 2026-10-29: Soniox async, Speechmatics Enhanced pre-recorded
SMOKE = {"id": "smoke", "path": "lab-v1.1/sessions/lab-v1.1-e20.studio.wav", "startS": 0.0, "endS": 8.0}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("requests")
    .add_local_file(ROOT / "modal/review_asr.py", "/root/review_asr.py")
)
for f in ("lab_review_core", "review_asr_core", "review_api_core", "lab_triage_core", "lab_sessions"):
    image = image.add_local_file(ROOT / f"tools/{f}.py", f"/root/{f}.py")
app = modal.App("tilawa-review-api")
vol = modal.Volume.from_name("tilawa-lab")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.replace(p)


def _wav_seconds(data: bytes) -> float:
    import io
    import wave

    with wave.open(io.BytesIO(data)) as w:
        return round(w.getnframes() / w.getframerate(), 3)


class _Http:
    """requests with retries on 429, 5xx and connection errors; one session per thread. Error text never carries the key."""

    def __init__(self, key: str):
        self.key, self.local = key, threading.local()

    def __call__(self, method: str, url: str, ok=(200, 201, 202, 204), tries: int = 7, **kw):
        import requests

        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers["Authorization"] = f"Bearer {self.key}"
        for k in range(tries):
            try:
                r = self.local.s.request(method, url, timeout=120, **kw)
            except requests.RequestException as e:
                err = f"{type(e).__name__}"
            else:
                if r.status_code in ok:
                    return r
                err = f"HTTP {r.status_code}: {r.text[:300]}".replace(self.key, "***")
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"{method} {url.split('?')[0]}: {err}")
            time.sleep(min(60, 2 ** k))
        raise RuntimeError(f"{method} {url.split('?')[0]}: gave up after {tries} tries: {err}")


def _pool(fn, clips: list[dict], workers: int) -> list[dict]:
    def safe(c):
        try:
            return fn(c)
        except Exception as e:  # recorded per item; the item gets no transcript and stays unsure
            return {"id": c["id"], "error": str(e)[:500]}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(safe, clips))


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("soniox")], timeout=2 * 60 * 60, cpu=2,
              max_containers=4)
def soniox(clips: list[dict], workers: int = 8) -> dict:
    """clips: [{id, path, startS, endS}] only."""
    http = _Http(os.environ["SONIOX_API_KEY"])
    models = http("GET", f"{SONIOX_API}/v1/models").json().get("models", [])
    info = next((m for m in models if m.get("id") == SONIOX_CONFIG["model"]), None)
    if not info:
        raise RuntimeError(f"{SONIOX_CONFIG['model']} not in GET /v1/models")
    config = {**SONIOX_CONFIG, **({"language_hints_strict": True} if info.get("supports_language_hints_strict") else {})}

    def one(c: dict) -> dict:
        data = RA.clip_bytes(c)
        sha = hashlib.sha256(data).hexdigest()
        t0 = time.time()
        f = http("POST", f"{SONIOX_API}/v1/files", files={"file": (f"{sha[:16]}.wav", data, "audio/wav")}).json()
        t1 = time.time()
        try:
            tr = http("POST", f"{SONIOX_API}/v1/transcriptions", json={**config, "file_id": f["id"]}).json()
            try:
                while (st := http("GET", f"{SONIOX_API}/v1/transcriptions/{tr['id']}").json())["status"] not in ("completed", "error"):
                    time.sleep(0.5)
                t2 = time.time()
                if st["status"] == "error":
                    raise RuntimeError(f"transcription error: {st.get('error_type')}: {st.get('error_message')}")
                doc = http("GET", f"{SONIOX_API}/v1/transcriptions/{tr['id']}/transcript").json()
            finally:
                http("DELETE", f"{SONIOX_API}/v1/transcriptions/{tr['id']}", ok=(200, 204, 404))
        finally:
            http("DELETE", f"{SONIOX_API}/v1/files/{f['id']}", ok=(200, 204, 404))
        t3 = time.time()
        import review_api_core as A

        dur = _wav_seconds(data)
        return {"id": c["id"], "audioSha256": sha, "durationS": dur, "transcript": A.soniox_transcript(doc),
                "raw": {"transcript": doc, "transcription": st, "file": f},
                "timing": {"uploadS": round(t1 - t0, 2), "submitToDoneS": round(t2 - t1, 2), "totalS": round(t3 - t0, 2)},
                "costUsd": round(dur / 3600 * PRICE_PER_HOUR["soniox"], 6)}

    t0 = time.time()
    res = _pool(one, clips, workers)
    return {"vendor": "soniox", "endpoint": SONIOX_API, "model": config["model"], "modelInfo": info, "config": config,
            "pricePerHour": PRICE_PER_HOUR["soniox"], "wallS": round(time.time() - t0, 1), "results": res}


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("speechmatics")], timeout=2 * 60 * 60, cpu=2,
              max_containers=4)
def speechmatics(clips: list[dict], workers: int = 8) -> dict:
    """clips: [{id, path, startS, endS}] only."""
    http = _Http(os.environ["SPEECHMATICS_API_KEY"])
    cfg = json.dumps(SPEECHMATICS_CONFIG)

    def one(c: dict) -> dict:
        data = RA.clip_bytes(c)
        sha = hashlib.sha256(data).hexdigest()
        t0 = time.time()
        jid = http("POST", f"{SPEECHMATICS_API}/jobs", files={"data_file": (f"{sha[:16]}.wav", data, "audio/wav")}, data={"config": cfg}).json()["id"]
        t1 = time.time()
        try:
            while (job := http("GET", f"{SPEECHMATICS_API}/jobs/{jid}").json()["job"])["status"] == "running":
                time.sleep(1.0)
            t2 = time.time()
            if job["status"] != "done":
                raise RuntimeError(f"job {job['status']}: {job.get('errors')}")
            doc = http("GET", f"{SPEECHMATICS_API}/jobs/{jid}/transcript", params={"format": "json-v2"}).json()
        finally:
            http("DELETE", f"{SPEECHMATICS_API}/jobs/{jid}", ok=(200, 204, 404, 410))
        t3 = time.time()
        import review_api_core as A

        dur = _wav_seconds(data)
        billed = job.get("duration") or dur
        return {"id": c["id"], "audioSha256": sha, "durationS": dur, "transcript": A.speechmatics_transcript(doc),
                "raw": {"transcript": doc, "job": job}, "timing": {"uploadS": round(t1 - t0, 2), "submitToDoneS": round(t2 - t1, 2), "totalS": round(t3 - t0, 2)},
                "costUsd": round(billed / 3600 * PRICE_PER_HOUR["speechmatics"], 6)}

    t0 = time.time()
    res = _pool(one, clips, workers)
    model = next((r["raw"]["transcript"].get("metadata", {}) for r in res if "raw" in r), {})
    return {"vendor": "speechmatics", "endpoint": SPEECHMATICS_API, "model": SPEECHMATICS_CONFIG["transcription_config"]["model"],
            "modelInfo": {k: model.get(k) for k in ("transcription_config", "language_pack_info", "orchestrator_version") if k in model},
            "config": SPEECHMATICS_CONFIG, "pricePerHour": PRICE_PER_HOUR["speechmatics"], "wallS": round(time.time() - t0, 1), "results": res}


FUNCS = {"soniox": soniox, "speechmatics": speechmatics}
CONFIGS = {"soniox": SONIOX_CONFIG, "speechmatics": SPEECHMATICS_CONFIG}


def _record(vendor: str, test_set: str) -> Path:
    return REVIEWS / test_set / f"asr-{vendor}.json"


def transcribe(vendor: str, test_set: str, items: list[dict], force: bool, chunk: int = 50, workers: int = 8) -> dict:
    """{id: record} for items [{id, audio}] from asr-<vendor>.json, transcribing what isn't there yet (same audio sha256,
    same request config, no error). New results are merged into the record after every chunk, so a rerun resumes."""
    p = _record(vendor, test_set)
    doc = json.loads(p.read_text()) if p.is_file() else {"version": 1, "testSet": test_set, "vendor": vendor, "createdAt": now(), "model": None, "items": {}}
    by_sha = {} if force else {r["audioSha256"]: r for r in doc["items"].values() if r.get("transcript") is not None and not r.get("error")
                               and (r.get("request") or {}).get("base") == CONFIGS[vendor]}
    shas = {it["id"]: hashlib.sha256(RA.clip_bytes(it["audio"])).hexdigest() for it in items}
    have = {i: {**by_sha[s], "audio": it["audio"]} for it in items if (s := shas[(i := it["id"])]) in by_sha}
    need = [{"id": it["id"], **it["audio"]} for it in items if it["id"] not in have]
    print(f"{vendor} {test_set}: {len(have)} reused, {len(need)} to transcribe", flush=True)
    calls = [FUNCS[vendor].spawn(need[k: k + chunk], workers) for k in range(0, len(need), chunk)]
    audio = {it["id"]: it["audio"] for it in items}
    for n, call in enumerate(calls):
        res = call.get()
        meta = {k: v for k, v in res.items() if k not in ("results", "wallS")}
        doc["model"] = doc["model"] or meta
        for r in res["results"]:
            rec = {"audio": audio[r["id"]], **{k: v for k, v in r.items() if k != "id"}, "request": {"base": CONFIGS[vendor], "sent": meta["config"]},
                   "transcribedAt": now(), "chunkWallS": res["wallS"]}
            if r.get("error"):
                rec |= {"transcript": None, "audioSha256": shas[r["id"]]}
            assert rec["audioSha256"] == shas[r["id"]], r["id"]
            have[r["id"]] = doc["items"][r["id"]] = rec
        _write(p, doc)
        vol.commit()
        print(f"  chunk {n + 1}/{len(calls)}: {sum(1 for r in res['results'] if r.get('error'))} errors, {res['wallS']} s", flush=True)
    if any(i not in doc["items"] for i in have):
        for i, r in have.items():
            doc["items"].setdefault(i, r)
        _write(p, doc)
        vol.commit()
    return have


def _json(p: Path):
    return json.loads(p.read_text())


def _verdicts(test_set: str) -> dict[str, dict]:
    """Latest verdict per item id across the test set's reviewer files (not the model records that share the folder)."""
    out = {}
    for p in sorted((REVIEWS / test_set).glob("*.json")):
        doc = _json(p)
        if not doc.get("reviewer") or not isinstance(doc.get("items"), dict):
            continue
        for i, r in doc["items"].items():
            if isinstance(r, dict) and "verdict" in r and (i not in out or (r.get("at") or "") > (out[i].get("at") or "")):
                out[i] = {**r, "reviewer": doc["reviewer"]}
    return out


def _session(test_set: str, sid: str, cache: dict) -> dict | None:
    if sid not in cache:
        p = DATA / test_set / "sessions" / f"{sid}.json"
        cache[sid] = _json(p) if p.is_file() else None
    return cache[sid]


def user_rows(test_sets: tuple[str, ...] = TEST_SETS) -> tuple[list[dict], list[dict], dict]:
    """Every listening verdict of yours on the test sets, on the item exactly as it was heard (from the queue the verdict
    names), with Nemotron's and FastConformer's triage decisions on the same window and text. ok = the label fits;
    bad_* = it doesn't. Byte-identical clips with the same text count once (review_api_core.dedupe)."""
    import lab_review_core as R
    import review_api_core as A
    import review_asr_core as core

    rows, skipped, sessions = [], [], {}
    for ts in test_sets:
        tri = {c["id"]: c for c in _json(DATA / ts / "triage-v1.json")["cases"]}
        queues = {}
        for iid, v in sorted(_json(REVIEWS / ts / "me.json")["items"].items(), key=lambda kv: (kv[1].get("at") or "", kv[0])):
            q = v.get("queue") or ""
            if q not in queues:
                queues[q] = {it["id"]: it for it in _json(DATA / q)["items"]} if q and (DATA / q).is_file() else {}
            it = queues[q].get(iid)
            if not it or (v["verdict"] != "ok" and v["verdict"] not in R.BAD):
                skipped.append({"id": iid, "testSet": ts, "verdict": v["verdict"], "queue": q, "why": "not in its queue" if not it else "not a data verdict"})
                continue
            s = _session(ts, core.session_id(it), sessions)
            c = tri.get(iid)
            same = bool(c) and c["audio"] == it["audio"] and c["claimed"] == it["text"]
            x = next((x for x in (s or {}).get("splices", []) if f"{s['id']}-{x['id']}" == iid), None)
            rev = core.revert_text(it, s) if s and x else None
            a, b = it["audio"]["startS"], it["audio"]["endS"]
            words = [w for w in (s or {}).get("said", []) if w["endS"] > a and w["startS"] < b]
            rows.append({"id": iid, "testSet": ts, "queue": q, "verdict": v["verdict"], "note": v.get("note", ""), "truth": v["verdict"] == "ok",
                         "kind": it["kind"], "label": it["label"]["type"], "class": c["class"] if c else it["label"]["type"],
                         "phone": ".phone." in it["audio"]["path"], "audio": it["audio"], "claimed": it["text"],
                         "alternative": c["alternative"] if same else rev and rev["text"], "alternativeOp": c["alternativeOp"] if same else rev and rev["op"],
                         "sameAsTriage": same, "nemotron": c["nemotron"]["decision"] if same else "unsure",
                         "nemotronDiff": c["nemotron"]["diff"] if same else None, "fastconformer": c["fastconformer"]["decision"] if same else "unsure",
                         "fastconformerDiff": c["fastconformer"]["diff"] if same else None, "oldOutcome": c["outcome"] if same else None,
                         "revert": rev and rev["text"], "revertEdit": rev and rev["edit"],
                         "zone": sorted(A.edit_zone(words, x)) if " ".join(w["text"] for w in words) == it["text"] else [],
                         "audioSha256": hashlib.sha256(RA.clip_bytes(it["audio"])).hexdigest()})
    kept, dropped = A.dedupe(rows)
    return kept, dropped + skipped, sessions


def _stats_doc(vendor: str, rows: list[dict], heard: dict, ctrl_rows: list[dict], rule: str | None = None) -> dict:
    import review_api_core as A
    import review_asr_core as core

    R = {r["id"]: A.readings(heard[r["id"]].get("transcript"), r["claimed"], r["alternative"], rule) for r in rows}
    rev = {r["id"]: (core.compare(r["revert"], t)["decision"] if (t := heard[r["id"]].get("transcript")) else "unsure",
                     A.judge(t or "", r["revert"], r["claimed"], rule)["decision"]) for r in rows if r["revert"]}
    out = {"readings": {k: A.reading_stats(rows, lambda r, k=k: R[r["id"]][k]["decision"]) for k in A.READINGS}}
    ok_rows = [r for r in rows if r["truth"]]
    rep = [r for r in ok_rows if r["revertEdit"] in ("repeat", "restart")]
    edited = [r for r in ok_rows if r["revert"]]
    out["revertControls"] = {
        "note": "every edited window you heard as ok, with the edit undone in the text (the audio has it): flagged = not accepted",
        "opinion": core.rate(sum(rev[r["id"]][0] != "match" for r in edited), len(edited)),
        "judge": core.rate(sum(rev[r["id"]][1] != "match" for r in edited), len(edited)),
        "repeatsAndRestarts": len(rep),
        "repeatMisses": {"opinion": sorted(r["id"] for r in rep if rev[r["id"]][0] == "match"),
                         "judge": sorted(r["id"] for r in rep if rev[r["id"]][1] == "match")}}
    ctl = {}
    for k in ("judge", "opinion"):
        get = (lambda r, c: A.judge(heard[r["id"]].get("transcript") or "", r[c], r["labelled"], rule)["decision"]) if k == "judge" else \
              (lambda r, c: core.compare(r[c], heard[r["id"]]["transcript"])["decision"] if heard[r["id"]].get("transcript") else "unsure")
        ctl[k] = A.control_stats(ctrl_rows, get) | {"agreement": core.rate(sum(R[r["id"]][k]["decision"] == "match" for r in ctrl_rows), len(ctrl_rows))}
    out["controls"] = ctl
    out["pairing"] = {k: A.pairing(rows, lambda r, k=k: R[r["id"]][k]["decision"]) for k in ("judge", "opinion")}
    tax = {}
    for r in ok_rows:
        if R[r["id"]]["opinion"]["decision"] != "match" and heard[r["id"]].get("transcript") is not None:
            tax[r["id"]] = A.bucket(r["claimed"], heard[r["id"]]["transcript"], R[r["id"]]["opinion"], r["revert"], set(r["zone"]), r["phone"])
    out["taxonomy"] = {"opinionRejections": len(tax), "byBucket": {A.BUCKETS[b]: sorted(i for i, t in tax.items() if t["bucket"] == b) for b in A.BUCKETS},
                       "atEdit": sorted(i for i, t in tax.items() if t["atEdit"]), "phone": sorted(i for i, t in tax.items() if t["phone"]), "items": tax}
    jr = {r["id"]: R[r["id"]]["judge"] for r in ok_rows if R[r["id"]]["judge"]["decision"] != "match"}
    out["judgeRejections"] = {i: {k: j.get(k) for k in ("decision", "reason", "claimed", "alternative", "diff", "need")} for i, j in jr.items()}
    t = [heard[r["id"]] for r in rows]
    lat = sorted(x["timing"]["submitToDoneS"] for x in t if x.get("timing"))
    tot = sorted(x["timing"]["totalS"] for x in t if x.get("timing"))
    out["cost"] = {"items": len(t), "errors": sorted(r["id"] for r in rows if heard[r["id"]].get("error")),
                   "audioS": round(sum(x.get("durationS") or 0 for x in t), 1), "costUsd": round(sum(x.get("costUsd") or 0 for x in t), 4),
                   "pricePerHour": PRICE_PER_HOUR[vendor],
                   "submitToDoneS": {"median": statistics.median(lat) if lat else None, "p90": lat[int(0.9 * (len(lat) - 1))] if lat else None, "max": lat[-1] if lat else None},
                   "totalS": {"median": statistics.median(tot) if tot else None, "p90": tot[int(0.9 * (len(tot) - 1))] if tot else None}}
    out["perItem"] = {r["id"]: {"transcript": heard[r["id"]].get("transcript"), **{k: R[r["id"]][k]["decision"] for k in A.READINGS},
                                "judgeDiff": R[r["id"]]["judge"].get("diff"), "judgeReason": R[r["id"]]["judge"].get("reason"),
                                **({"revertOpinion": rev[r["id"]][0], "revertJudge": rev[r["id"]][1]} if r["id"] in rev else {})} for r in rows}
    return out


@app.function(image=image, volumes={VOL: vol}, timeout=4 * 60 * 60, cpu=2, memory=8192)
def bakeoff(vendors: list[str], force: bool = False) -> dict:
    import review_api_core as A
    import review_asr_core as core

    vol.reload()
    rows, dropped, sessions = user_rows()
    print(f"{len(rows)} verdicts ({sum(r['truth'] for r in rows)} ok, {sum(not r['truth'] for r in rows)} bad), {len(dropped)} dropped", flush=True)
    vq = _json(DATA / CONTROLS)
    ss = {sid: s for sid in {core.session_id(it) for it in vq["items"]} if (s := _session("lab-v1.1", sid, sessions))}
    ctrl = core.controls(vq, ss)
    ctrl_rows = [{"id": it["id"], "labelled": it["text"], "oneEdit": ctrl[it["id"]]["oneEdit"]["text"], "shifted": ctrl[it["id"]]["shifted"]["text"],
                  "oneEditEdit": ctrl[it["id"]]["oneEdit"]["edit"], "oneEditWords": ctrl[it["id"]]["oneEdit"].get("words", 1)} for it in vq["items"]]
    by_id = {r["id"]: r for r in rows}
    assert all(r["id"] in by_id and by_id[r["id"]]["claimed"] == r["labelled"] for r in ctrl_rows), "the 40 control windows are among your verdicts"
    heard = {}
    for v in vendors:
        heard[v] = {}
        for ts in TEST_SETS:
            heard[v] |= transcribe(v, ts, [r for r in rows if r["testSet"] == ts], force)
    stats = {v: _stats_doc(v, rows, heard[v], ctrl_rows) for v in vendors}
    old = A.pairing(rows, lambda r: r["fastconformer"])
    cands = {v: {"badAccepted": len(st["pairing"]["judge"]["badAccepted"]), "okAccepted": st["pairing"]["judge"]["okAccepted"]["k"],
                 "badFlagged": st["readings"]["judge"]["badFlagged"]["k"], "oneEditFlagged": st["controls"]["judge"]["oneEdit"]["k"],
                 "costPerHour": PRICE_PER_HOUR[v]} for v, st in stats.items()}
    winner = A.pick_winner(cands)
    for v in vendors:
        for ts in TEST_SETS:
            p = _record(v, ts)
            doc = _json(p)
            for r in rows:
                if r["testSet"] == ts:
                    doc["items"][r["id"]]["bakeoff"] = {"human": r["verdict"], "claimed": r["claimed"], "alternative": r["alternative"],
                                                        "revert": r["revert"], **stats[v]["perItem"][r["id"]]}
            _write(p, doc)
    models = {v: _json(_record(v, "lab-v1.1"))["model"] for v in vendors}
    doc = {"version": 1, "kind": "api-bakeoff", "createdAt": now(), "vendors": vendors, "models": models,
           "rule": {"judge": {"min": A.JUDGE_MIN, "share": A.JUDGE_SHARE, "maxDist": A.JUDGE_MAX_DIST,
                              "source": "tools/review_api_core.py judge: fit_dist on space-free rasm skeletons; match when the transcript is on "
                                        "the labelled side of the midpoint by max(min, share x distance between the texts) letters"},
                    "opinion": "tools/review_asr_core.py compare (round-1 rule) against the labelled text alone",
                    "opinionRasm": "tools/review_asr_core.py compare_rasm (no edge context)",
                    "pairing": "Nemotron's lab triage likelihood decision (triage-v1.json, same window and text) with the API's judge: "
                               "both match -> auto-accepted, else to you",
                    "winner": "review_api_core.pick_winner: fewest bad auto-accepted, most ok auto-accepted, bad flagged, one-edit flagged, cost"},
           "set": {"items": len(rows), "ok": sum(r["truth"] for r in rows), "bad": sum(not r["truth"] for r in rows),
                   "byTestSet": {ts: sum(r["testSet"] == ts for r in rows) for ts in TEST_SETS}, "dropped": dropped,
                   "badItems": [{k: r[k] for k in ("id", "verdict", "class", "nemotron", "fastconformer", "oldOutcome")} for r in rows if not r["truth"]]},
           "stats": stats, "oldPairing": old, "nemotronAlone": A.reading_stats(rows, lambda r: r["nemotron"]),
           "fastconformerAlone": A.reading_stats(rows, lambda r: r["fastconformer"]), "candidates": cands, "winner": winner,
           "rows": [{k: r[k] for k in r if k not in ("zone",)} for r in rows]}
    _write(REVIEWS / SUMMARY, doc)
    vol.commit()
    return doc


def _cached(vendor: str, test_set: str, items: list[dict]) -> dict:
    """{id: record} from asr-<vendor>.json for items [{id, audio}], by audio sha256 and request config; never calls the
    vendor. Every item must be there."""
    p = _record(vendor, test_set)
    doc = _json(p)
    by_sha = {r["audioSha256"]: r for r in doc["items"].values() if r.get("transcript") is not None and not r.get("error")
              and (r.get("request") or {}).get("base") == CONFIGS[vendor]}
    out, missing = {}, []
    for it in items:
        sha = hashlib.sha256(RA.clip_bytes(it["audio"])).hexdigest()
        (missing.append(it["id"]) if sha not in by_sha else out.__setitem__(it["id"], {**by_sha[sha], "audio": it["audio"]}))
    if missing:
        raise RuntimeError(f"{vendor} {test_set}: {len(missing)} items have no cached transcript, e.g. {missing[:3]}")
    return out


def _triage_cases(test_set: str, seed: int) -> tuple[list[dict], dict, dict]:
    """The test set's triage cases (lab_triage_core.session_cases + its v1 check's clean windows), asserted identical to
    triage-v1.json; returns (cases, triage-v1.json cases by id, sessions)."""
    import lab_triage_core as T

    base = DATA / test_set
    old = _json(base / "triage-v1.json")
    ids = [e["id"] for e in _json(base / "manifest-v1.json")["sessions"]]
    sessions = {i: _json(base / f"sessions/{i}.json") for i in ids}
    cases = [c for i in ids for c in T.session_cases(sessions[i], test_set, seed)]
    v1 = _json(base / "review/v1-check.json") if (base / "review/v1-check.json").is_file() else {"items": []}
    cases += T.reviewed_clean_cases(v1, sessions, {c["id"] for c in cases}, seed)
    oldc = {c["id"]: c for c in old["cases"]}
    assert set(oldc) == {c["id"] for c in cases}, "the case list differs from triage-v1.json"
    for c in cases:
        o = oldc[c["id"]]
        assert (o["claimed"], o["alternative"], o["audio"]) == (c["claimed"], c["alternative"], c["item"]["audio"]), c["id"]
    return cases, oldc, sessions


MARGIN_CALIBRATION = ("lab-v1.1", "lab-v1.2")  # the verdicts the new margin is calibrated on; lab-v1.3 is reported, not used


@app.function(image=image, volumes={VOL: vol}, timeout=60 * 60, cpu=4, memory=16384)
def margin(vendor: str = "soniox", rules: tuple[str, str] = ("half", "reach"), report_set: str = "lab-v1.3", seed: int = 17) -> dict:
    """The judge margin re-preregistered (docs/benchmarks/lab-triage-v1.md, "Soniox margin v2"), from cached
    transcripts only: both rules on your lab-v1.1 and lab-v1.2 verdicts and the ticket 12 controls, the preregistered
    adoption rule (review_api_core.adopt_margin), then both rules on every case of `report_set` and on your verdicts there."""
    import lab_review_core as LR
    import lab_triage_core as T
    import review_api_core as A
    import review_asr_core as core

    vol.reload()
    old, new = rules
    rows, dropped, sessions = user_rows(MARGIN_CALIBRATION)
    heard = {}
    for ts in MARGIN_CALIBRATION:
        heard |= _cached(vendor, ts, [r for r in rows if r["testSet"] == ts])
    vq = _json(DATA / CONTROLS)
    ss = {sid: s for sid in {core.session_id(it) for it in vq["items"]} if (s := _session("lab-v1.1", sid, sessions))}
    ctrl = core.controls(vq, ss)
    ctrl_rows = [{"id": it["id"], "labelled": it["text"], "oneEdit": ctrl[it["id"]]["oneEdit"]["text"], "shifted": ctrl[it["id"]]["shifted"]["text"],
                  "oneEditEdit": ctrl[it["id"]]["oneEdit"]["edit"], "oneEditWords": ctrl[it["id"]]["oneEdit"].get("words", 1)} for it in vq["items"]]
    stats = {r: _stats_doc(vendor, rows, heard, ctrl_rows, r) for r in rules}
    decision = A.adopt_margin(stats[old], stats[new])
    slim = lambda st: {k: v for k, v in st.items() if k not in ("perItem", "taxonomy", "judgeRejections")}
    calib = {"set": {"items": len(rows), "ok": sum(r["truth"] for r in rows), "bad": sum(not r["truth"] for r in rows), "dropped": len(dropped),
                     "badItems": sorted(r["id"] for r in rows if not r["truth"])},
             "stats": {r: slim(s) for r, s in stats.items()}, "decision": decision,
             "byClass": {r: {cls: dict(Counter(stats[r]["perItem"][x["id"]]["judge"] for x in rows if x["class"] == cls and x["truth"]))
                             for cls in sorted({x["class"] for x in rows})} for r in rules},
             "changed": {x["id"]: {r: stats[r]["perItem"][x["id"]]["judge"] for r in rules} | {"truth": x["truth"], "class": x["class"]}
                         for x in rows if len({stats[r]["perItem"][x["id"]]["judge"] for r in rules}) > 1}}
    # the report set: every triage case and your verdicts on it, under both rules (Nemotron's decisions from triage-v1.json)
    cases, oldc, _ = _triage_cases(report_set, seed)
    heard3 = _cached(vendor, report_set, [{"id": c["id"], "audio": c["item"]["audio"]} for c in cases])
    nem = {i: {"decision": o["nemotron"]["decision"], "diff": o["nemotron"]["diff"]} for i, o in oldc.items()}
    reviews = _verdicts(report_set)
    rep = {}
    for r in rules:
        api = {c["id"]: A.judge(heard3[c["id"]]["transcript"] or "", c["claimed"], c["alternative"], r) for c in cases}
        res = {pol: T.triage(cases, {"nemotron": nem, vendor: api}, report_set, (T.SPOT_EDITS, T.SPOT_CLEAN), seam, seed, reviews, {},
                             ("nemotron", vendor)) for pol, seam in (("seam10", (T.SEAM_SPOT, T.SEAM_CAP)), ("seamPools", T.SEAM_POOLS))}
        by = {c["id"]: c for c in res["seam10"]["cases"]}
        heard_ids = [i for i, v in reviews.items() if i in by and (v["verdict"] == "ok" or v["verdict"] in LR.BAD)]
        ok = [i for i in heard_ids if reviews[i]["verdict"] == "ok"]
        bad = [i for i in heard_ids if reviews[i]["verdict"] != "ok"]
        rep[r] = {"counts": {k: res["seam10"]["counts"][k] for k in ("outcomes", "byClass", "perModel")},
                  "queue": {pol: x["counts"]["queue"] for pol, x in res.items()},
                  "verdicts": {"items": len(heard_ids), "ok": len(ok), "bad": len(bad),
                               "okAccepted": sum(by[i]["outcome"] == "auto-accepted" for i in ok),
                               "okOutcomes": dict(Counter(by[i]["outcome"] for i in ok)),
                               "badAccepted": sorted(i for i in bad if by[i]["outcome"] == "auto-accepted"),
                               "badOutcomes": {i: {"verdict": reviews[i]["verdict"], "outcome": by[i]["outcome"], "class": by[i]["class"],
                                                   "nemotron": by[i]["nemotron"]["decision"], "nemotronDiff": by[i]["nemotron"]["diff"],
                                                   vendor: by[i][vendor]["decision"], f"{vendor}Diff": by[i][vendor].get("diff"),
                                                   "oldOutcome": oldc[i]["outcome"], "transcript": heard3[i]["transcript"]} for i in bad}},
                  "oldPairingOnVerdicts": {"okAccepted": sum(oldc[i]["outcome"] == "auto-accepted" for i in ok),
                                           "badAccepted": sorted(i for i in bad if oldc[i]["outcome"] == "auto-accepted")}}
    doc = {"version": 1, "kind": "judge-margin", "createdAt": now(), "vendor": vendor, "rules": list(rules),
           "calibration": calib, "report": {"testSet": report_set, **rep}}
    _write(REVIEWS / "lab-v1.1" / f"asr-{vendor}.margin-v2.json", doc)
    vol.commit()
    return doc


@app.local_entrypoint()
def margin_main(vendor: str = "soniox", out: str = "docs/benchmarks/lab-triage-soniox-margin-v2.json"):
    d = margin.remote(vendor)
    (ROOT / out).write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    c = d["calibration"]
    print(json.dumps(c["set"], ensure_ascii=False))
    for r, st in c["stats"].items():
        p, j = st["pairing"]["judge"], st["readings"]["judge"]
        k = st["controls"]["judge"]
        print(f"{r:6} judge agree {j['agreement']['k']}/{j['agreement']['n']}, bad flagged {j['badFlagged']['k']}/{j['badFlagged']['n']} "
              f"(mismatch {j['badMismatch']['k']}); pairing ok accepted {p['okAccepted']['k']}/{p['okAccepted']['n']}, bad accepted {p['badAccepted']}, "
              f"queued {p['queued']}/{p['items']}; controls one-edit {k['oneEdit']['k']}/{k['oneEdit']['n']} one-word {k['oneWord']['k']}/{k['oneWord']['n']} "
              f"shifted {k['shifted']['k']}/{k['shifted']['n']} repeat misses {k['repeatMisses']}; revert {st['revertControls']['judge']['k']}/{st['revertControls']['judge']['n']} "
              f"{st['revertControls']['repeatMisses']['judge']}")
    print("decision:", json.dumps(c["decision"], ensure_ascii=False))
    for r, x in d["report"].items():
        if r == "testSet":
            continue
        print(f"{d['report']['testSet']} {r}: {json.dumps(x['counts']['outcomes'])} queue {[(p, q['items']) for p, q in x['queue'].items()]}")
        v = x["verdicts"]
        print(f"   verdicts {v['items']} ({v['ok']} ok, {v['bad']} bad): ok accepted {v['okAccepted']}, bad accepted {v['badAccepted']}; "
              f"old pairing ok {x['oldPairingOnVerdicts']['okAccepted']} bad {x['oldPairingOnVerdicts']['badAccepted']}")
    print(f"wrote {out}")


@app.function(image=image, volumes={VOL: vol}, timeout=4 * 60 * 60, cpu=4, memory=16384)
def triage(test_set: str, vendor: str, spot: tuple[int, int], seam_spot: tuple[float, int] | dict, seed: int, force: bool = False,
           rule: str = "half") -> dict:
    """Nemotron (its lab triage decisions, triage-v1.json, reused as they are) + the vendor's judge on every case of the
    test set, by the lab triage rule. Writes review/triage-<vendor>.json, review/triage-<vendor>-confirm.json and
    triage-<vendor>-v1.json; never touches triage.json."""
    import lab_review_core as R
    import lab_triage_core as T
    import review_api_core as A

    t0 = time.time()
    vol.reload()
    base = DATA / test_set
    old = _json(base / "triage-v1.json")
    cases, oldc, _ = _triage_cases(test_set, seed)
    nem = {i: {"claimed": o["nemotron"]["claimed"], "alternative": o["nemotron"]["alternative"]} for i, o in oldc.items()}
    heard = transcribe(vendor, test_set, [{"id": c["id"], "audio": c["item"]["audio"]} for c in cases], force)
    api = {c["id"]: A.judge(heard[c["id"]].get("transcript") or "", c["claimed"], c["alternative"], rule) for c in cases}
    reviews = _verdicts(test_set)
    names = ("triage.json", "triage-confirm.json", "v1-check.json")
    earlier = {it["id"]: it for n in names if (base / "review" / n).is_file() for it in _json(base / "review" / n)["items"]}
    models = ("nemotron", vendor)
    seam_spot = seam_spot if isinstance(seam_spot, dict) else tuple(seam_spot)
    res = T.triage(cases, {"nemotron": nem, vendor: api}, test_set, tuple(spot), seam_spot, seed, reviews, earlier, models)
    for c in res["cases"]:
        assert c["nemotron"]["decision"] == oldc[c["id"]]["nemotron"]["decision"], c["id"]
    problems = R.validate_queue(res["queue"])
    if problems:
        raise ValueError(f"queue: {problems[:5]}")
    confirm = T.confirm_queue(res["queue"])
    qrel, crel = f"review/triage-{vendor}.json", f"review/triage-{vendor}-confirm.json"
    old_q = _json(base / "review/triage.json")
    old_ids = {it["id"]: it["triage"]["reasons"] for it in old_q["items"]}
    new_ids = {it["id"]: it["triage"]["reasons"] for it in res["queue"]["items"]}
    old_confirm = [it["id"] for it in _json(base / "review/triage-confirm.json")["items"]]
    by = {c["id"]: c for c in res["cases"]}
    reviewed = T.compare_reviewed(res["cases"], {"items": list(earlier.values())}, reviews, models)
    cmp = {"oldQueue": len(old_ids), "newQueue": len(new_ids), "both": len(set(old_ids) & set(new_ids)),
           "onlyOld": len(set(old_ids) - set(new_ids)), "onlyNew": len(set(new_ids) - set(old_ids)),
           "oldOutcomesOfNewItems": dict(Counter(oldc[i]["outcome"] for i in new_ids)),
           "newOutcomesOfOldModelQueued": dict(Counter(by[i]["outcome"] for i, r in old_ids.items() if r[0] not in ("seam", "spot"))),
           "oldConfirm": {i: {"newOutcome": by[i]["outcome"], "nemotron": by[i]["nemotron"]["diff"], vendor: {k: by[i][vendor].get(k) for k in ("decision", "diff", "reason")},
                              "transcript": heard[i].get("transcript"), "verdict": reviews.get(i, {}).get("verdict"), "inNewQueue": i in new_ids} for i in old_confirm}}
    doc = {"version": 1, "kind": "lab-triage", "testSet": test_set, "createdAt": now(), "variant": "studio", "models": {
        "nemotron": {**old["models"]["nemotron"], "reused": "decisions from triage-v1.json (same cases, windows and texts)"},
        vendor: {**_json(_record(vendor, test_set))["model"], "reading": "review_api_core.judge on its label-blind transcript",
                 "rule": {"min": A.JUDGE_MIN, "share": A.JUDGE_SHARE, "maxDist": A.JUDGE_MAX_DIST, "margin": rule}}},
        "rule": {**old["rule"], "pairwise": f"Nemotron: as triage-v1.json; {vendor}: review_api_core.judge rule {rule!r} (transcript closer to the "
                                            "claimed text than to the alternative by the margin, letters of rasm skeleton)"},
        "construction": old["construction"], "spot": {"edits": spot[0], "clean": spot[1]},
        "seamSpot": seam_spot if isinstance(seam_spot, dict) else {"share": seam_spot[0], "cap": seam_spot[1]},
        "seed": seed, "queue": f"{test_set}/{qrel}", "confirm": f"{test_set}/{crel}", "counts": res["counts"], "reviewed": reviewed,
        "comparison": cmp, "cost": {"audioS": round(sum(h.get("durationS") or 0 for h in heard.values()), 1),
                                    "costUsd": round(sum(h.get("costUsd") or 0 for h in heard.values()), 4),
                                    "errors": sorted(i for i, h in heard.items() if h.get("error")), "wallS": round(time.time() - t0, 1)},
        "cases": [{k: c[k] for k in ("id", "session", "group", "class", "linked", "claimed", "alternative", "alternativeOp", "nemotron", vendor, "outcome")}
                  | {"audio": c["item"]["audio"], "transcript": heard[c["id"]].get("transcript"), "reviewedOnly": bool(c.get("reviewedOnly")),
                     **({} if c.get("reviewedOnly") else {k: c[k] for k in ("spotCheck", "seamSpotCheck", "queued")})} for c in res["cases"]]}
    _write(base / f"triage-{vendor}-v1.json", doc)
    _write(base / qrel, res["queue"])
    _write(base / crel, confirm)
    p = _record(vendor, test_set)
    rec = _json(p)
    for c in res["cases"]:
        rec["items"][c["id"]]["triage"] = {"claimed": c["claimed"], "alternative": c["alternative"], **{k: c[vendor].get(k) for k in ("decision", "diff", "reason", "need")}}
    _write(p, rec)
    vol.commit()
    return {k: doc[k] for k in ("testSet", "models", "counts", "reviewed", "comparison", "cost", "queue", "confirm")}


@app.local_entrypoint()
def smoke(vendors: str = "soniox,speechmatics"):
    """One window outside every listening queue (lab-v1.1 e20, 0-8 s), to check each vendor end to end."""
    for v in vendors.split(","):
        r = FUNCS[v].remote([SMOKE], 1)
        x = r["results"][0]
        print(json.dumps({k: r[k] for k in ("vendor", "endpoint", "model", "config", "modelInfo", "wallS")}, ensure_ascii=False)[:1500])
        print(v, "|", x.get("transcript"), "|", x.get("timing"), x.get("costUsd"), x.get("error"))
        print(json.dumps(x.get("raw"), ensure_ascii=False)[:1200])


@app.local_entrypoint()
def bakeoff_main(vendors: str = "soniox,speechmatics", force: bool = False, out: str = "docs/benchmarks/lab-review-asr-v1-api-bakeoff.json"):
    d = bakeoff.remote(vendors.split(","), force)
    slim = {**d, "stats": {v: {k: x for k, x in s.items() if k != "perItem"} for v, s in d["stats"].items()},
            "perItem": {v: s["perItem"] for v, s in d["stats"].items()}}
    (ROOT / out).write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(d["set"] | {"dropped": len(d["set"]["dropped"])}, ensure_ascii=False))
    for v, s in d["stats"].items():
        for k, st in s["readings"].items():
            print(f"{v:13} {k:12} agree {st['agreement']['k']}/{st['agreement']['n']} {st['agreement']['ci95']}  bad flagged "
                  f"{st['badFlagged']['k']}/{st['badFlagged']['n']} (mismatch {st['badMismatch']['k']})")
        for k, c in s["controls"].items():
            print(f"{v:13} controls {k:8} agree40 {c['agreement']['k']}/40 one-edit {c['oneEdit']['k']}/{c['oneEdit']['n']} one-word "
                  f"{c['oneWord']['k']}/{c['oneWord']['n']} shifted {c['shifted']['k']}/{c['shifted']['n']} repeat misses {c['repeatMisses']}")
        for k, p in s["pairing"].items():
            print(f"{v:13} pairing {k:8} ok accepted {p['okAccepted']['k']}/{p['okAccepted']['n']} bad accepted {p['badAccepted']} queued {p['queued']}/{p['items']}")
        print(f"{v:13} revert {json.dumps(s['revertControls'], ensure_ascii=False)}")
        print(f"{v:13} cost {json.dumps(s['cost'])}")
    o = d["oldPairing"]
    print(f"old Nemotron+FastConformer: ok accepted {o['okAccepted']['k']}/{o['okAccepted']['n']} bad accepted {o['badAccepted']} queued {o['queued']}")
    print("winner:", d["winner"], json.dumps(d["candidates"]))
    print(f"wrote {out}")


@app.local_entrypoint()
def triage_main(test_set: str = "lab-v1.3", vendor: str = "", spot_edits: int = 15, spot_clean: int = 15, seam_spot: float = 0.10,
                seam_cap: int = 20, seed: int = 17, force: bool = False, out: str = ""):
    if not vendor:
        raise SystemExit("--vendor: the bake-off winner")
    r = triage.remote(test_set, vendor, (spot_edits, spot_clean), (seam_spot, seam_cap), seed, force)
    out = out or f"docs/benchmarks/lab-triage-{vendor}-{test_set}.json"
    (ROOT / out).write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    c = r["counts"]
    print(json.dumps({k: c[k] for k in ("cases", "outcomes", "perModel", "queue")}, ensure_ascii=False, indent=1))
    for cls, row in c["byClass"].items():
        print(f"{cls:16} {row}")
    print("comparison:", json.dumps(r["comparison"], ensure_ascii=False, indent=1))
    print("cost:", json.dumps(r["cost"]))
    for q in (r["confirm"], r["queue"]):
        print(f"queue https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue={q}")
    print(f"wrote {out}")
