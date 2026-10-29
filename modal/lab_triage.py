"""Two-model triage of a lab test set (.scratch/lab/issues/17-two-model-triage.md, spec decisions 16-17).

    modal run modal/lab_triage.py --test-set lab-v1.1
    modal run modal/lab_triage.py --test-set lab-v1.1 --spot 0.1 --seam-spot 0.1 --seed 17 --nemotron-gpus 4

Every case of the test set's studio copy (tools/lab_triage_core.py: each planted edit, repeat and restart, and each
clean window) is scored by both likelihood arms, in parallel:

- Nemotron 3.5 Transcribe on H100 (modal/review_asr.py `Nemotron`): label-blind transcript first (used only to spell
  the candidates, reused by audio sha256 from earlier runs), then the LM log-likelihood of the labelled and the
  alternative text, plain spellings (core.plain_candidates), pairwise at 2 nats;
- FastConformer on T4 (modal/review_fastconformer.py `FastConformer.score`): CTC log-likelihood of both texts,
  diacritised spellings against its own label-blind transcripts, pairwise at 2 nats.

Writes /lab/data/<testSet>/triage-v1.json (every case: both models' scores and decisions, outcome, spot-check flags,
counts, the comparison with earlier verdicts), the listening queue /lab/data/<testSet>/review/triage.json (ticket 06
format), and the per-spelling model records under /lab/data/<testSet>/triage/. Verdicts already given on the same ids
(e.g. the v1 listening check) carry over; the queue marks them.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "modal"), "/root"]
import review_asr as RA  # noqa: E402
import review_fastconformer as RF  # noqa: E402

VOL = "/lab"
DATA, REVIEWS = Path(VOL) / "data", Path(VOL) / "reviews"
NAME = "triage-v1"
QUEUE = "review/triage.json"
EARLIER = "review/v1-check.json"
TRANSCRIBER = "nemotron-3.5-transcribe"
PROMPT = "verbatim"  # the decision prompt of the calibrated likelihood arm

cpu_image = modal.Image.debian_slim(python_version="3.12")
for f in ("lab_review_core", "review_asr_core", "lab_sessions", "lab_triage_core"):
    cpu_image = cpu_image.add_local_file(ROOT / f"tools/{f}.py", f"/root/{f}.py")
for f in ("review_asr", "review_fastconformer"):
    cpu_image = cpu_image.add_local_file(ROOT / f"modal/{f}.py", f"/root/{f}.py")
app = modal.App("tilawa-lab-triage")
app.include(RA.app)
app.include(RF.app)
vol = modal.Volume.from_name("tilawa-lab")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.replace(p)


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def _verdicts(test_set: str) -> dict[str, dict]:
    """Latest verdict per item id across every reviewer file of the test set (not the model records that share the folder)."""
    out = {}
    for p in sorted((REVIEWS / test_set).glob("*.json")):
        doc = json.loads(p.read_text())
        if not doc.get("reviewer") or not isinstance(doc.get("items"), dict):
            continue
        for i, r in doc["items"].items():
            if isinstance(r, dict) and "verdict" in r and (i not in out or (r.get("at") or "") > (out[i].get("at") or "")):
                out[i] = {**r, "reviewer": doc["reviewer"]}
    return out


def _saved_transcripts(test_set: str, own: Path, force: bool) -> tuple[dict | None, dict[str, dict]]:
    """Nemotron transcripts with the decision prompt, by audio sha256, from this stage's record and the calibration runs."""
    if force:
        return None, {}
    files = sorted((REVIEWS / test_set).glob(f"asr-{TRANSCRIBER}*.json")) + ([own] if own.is_file() else [])
    meta, by_sha = None, {}
    for p in sorted(files, key=lambda p: p != own):
        doc = json.loads(p.read_text())
        m = doc.get("model") or {}
        if m.get("promptKey") != PROMPT or (meta and m.get("revision") != meta["revision"]):
            continue
        meta = meta or m
        for r in doc.get("items", {}).values():
            if "raw" in r and "transcript" in r:
                by_sha.setdefault(r["audioSha256"], r)
    return meta, by_sha


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=6 * 60 * 60, cpu=4, memory=16384)
def run(test_set: str, spot: float, seam_spot: float, seed: int, chunk: int, nemotron_gpus: int, fc_gpus: int, force: bool,
        only: list[str] | None = None, no_write: bool = False) -> dict:
    """only: score just these sessions (a smoke run); no_write: keep triage-v1.json and the queue as they are."""
    import lab_review_core as R
    import lab_triage_core as T
    import review_asr_core as core

    t0 = time.time()
    vol.reload()
    base = DATA / test_set
    ids = [e["id"] for e in json.loads((base / "manifest-v1.json").read_text())["sessions"] if not only or e["id"] in only]
    sessions = {i: json.loads((base / f"sessions/{i}.json").read_text()) for i in ids}
    cases = [c for i in ids for c in T.session_cases(sessions[i], test_set, seed)]
    earlier = json.loads((base / EARLIER).read_text()) if (base / EARLIER).is_file() else {"items": []}
    cases += T.reviewed_clean_cases(earlier, sessions, {c["id"] for c in cases}, seed)
    by_id = {c["id"]: c for c in cases}
    assert len(by_id) == len(cases), "duplicate case ids"
    shas = {c["id"]: hashlib.sha256(RA.clip_bytes(c["item"]["audio"])).hexdigest() for c in cases}
    print(f"{len(cases)} cases ({sum(not c.get('reviewedOnly') for c in cases)} in the test set) [{time.time() - t0:.0f} s]", flush=True)

    clip = lambda c: {"id": c["id"], **c["item"]["audio"]}
    scored = [c for c in cases if c["alternative"] is not None]
    fc = RF.FastConformer.with_options(max_containers=fc_gpus)()
    fc_calls = [fc.score.spawn([{**clip(c), "texts": {"labelled": c["claimed"], "original": c["alternative"]}} for c in part])
                for part in _chunks(scored, chunk)]
    own = base / "triage" / f"asr-{TRANSCRIBER}.json"
    meta, by_sha = _saved_transcripts(test_set, own, force)
    heard = {c["id"]: {**by_sha[shas[c["id"]]], "audioSha256": shas[c["id"]]} for c in cases if shas[c["id"]] in by_sha}
    reused = len(heard)
    nem = RA.Nemotron.with_options(max_containers=nemotron_gpus)()
    need = [clip(c) for c in cases if c["id"] not in heard]
    tr_calls = [nem.transcribe.spawn(part, PROMPT) for part in _chunks(need, chunk)]
    gpu = {"nemotronTranscribeS": 0.0, "nemotronScoreS": 0.0, "nemotronLoadS": [], "fastconformerS": 0.0, "fastconformerLoadS": []}
    for call in tr_calls:
        res = call.get()
        if meta and res["revision"] != meta["revision"]:
            raise RuntimeError(f"Nemotron transcripts from revision {res['revision']}, saved ones from {meta['revision']}; rerun with --force")
        meta = meta or {k: v for k, v in res.items() if k != "results"}
        gpu["nemotronLoadS"].append(res["loadSeconds"])
        for r in res["results"]:
            heard[r["id"]] = {k: v for k, v in r.items() if k != "id"}
            gpu["nemotronTranscribeS"] += r["seconds"]
    print(f"transcripts: {reused} reused, {len(need)} new [{time.time() - t0:.0f} s]", flush=True)
    _write(own, {"version": 1, "testSet": test_set, "createdAt": now(), "model": meta,
                 "items": {i: {"audio": by_id[i]["item"]["audio"], **{k: heard[i][k] for k in ("audioSha256", "transcript", "raw") if k in heard[i]},
                               "seconds": heard[i].get("seconds")} for i in sorted(heard)}})
    vol.commit()
    jobs = [{**clip(c), "candidates": {"labelled": core.plain_candidates(c["claimed"], heard[c["id"]]["transcript"]),
                                      "original": core.plain_candidates(c["alternative"], heard[c["id"]]["transcript"])}} for c in scored]
    sc_calls = [nem.score.spawn(part, PROMPT) for part in _chunks(jobs, chunk)]
    nem_res, fc_res, nem_meta, fc_meta = {}, {}, None, None
    for call in sc_calls:
        res = call.get()
        if res["revision"] != meta["revision"]:
            raise RuntimeError(f"scores from revision {res['revision']}, transcripts from {meta['revision']}")
        nem_meta = nem_meta or {k: v for k, v in res.items() if k != "results"}
        gpu["nemotronLoadS"].append(res["loadSeconds"])
        for r in res["results"]:
            nem_res[r["id"]] = r
            gpu["nemotronScoreS"] += r["seconds"]
    print(f"nemotron scores: {len(nem_res)} [{time.time() - t0:.0f} s]", flush=True)
    for call in fc_calls:
        res = call.get()
        fc_meta = fc_meta or {k: v for k, v in res.items() if k != "results"}
        gpu["fastconformerS"] += res["gpuSeconds"]
        gpu["fastconformerLoadS"].append(res["loadSeconds"])
        for r in res["results"]:
            fc_res[r["id"]] = r
    print(f"fastconformer scores: {len(fc_res)} [{time.time() - t0:.0f} s]", flush=True)
    for i, r in nem_res.items():
        assert r["audioSha256"] == shas[i], i
    for i, r in fc_res.items():
        assert r["audioSha256"] == shas[i], i

    scores = {"nemotron": {i: {"claimed": r["scores"]["labelled"]["llm"], "alternative": r["scores"]["original"]["llm"]} for i, r in nem_res.items()},
              "fastconformer": {i: {"claimed": r["scores"]["labelled"]["diac"], "alternative": r["scores"]["original"]["diac"]} for i, r in fc_res.items()}}
    reviews = _verdicts(test_set)
    res = T.triage(cases, scores, test_set, spot, seam_spot, seed, reviews, {it["id"]: it for it in earlier["items"]})
    problems = R.validate_queue(res["queue"])
    if problems:
        raise ValueError(f"triage queue: {problems[:5]}")
    reviewed = T.compare_reviewed(res["cases"], earlier, reviews)
    queue_rel = f"{test_set}/{QUEUE}"
    models = {
        "nemotron": {"id": nem_meta["id"], "revision": nem_meta["revision"], "device": nem_meta["device"], "prompt": nem_meta["prompt"],
                     "margin": T.MARGINS["nemotron"], "reading": "LM log-likelihood of the text as the reply to the transcription request, "
                     "plain spellings (core.plain_candidates against its own label-blind transcript), best spelling per text",
                     "transcriptsReused": reused, "transcriptsNew": len(need)},
        "fastconformer": {"id": fc_meta["id"], "revision": fc_meta["revision"], "device": fc_meta["device"], "margin": T.MARGINS["fastconformer"],
                          "reading": "CTC forced-alignment log-likelihood, diacritised spellings (core.diacritised_candidates against its own "
                                     "label-blind CTC greedy text and RNNT transcript), best spelling per text"},
    }
    loads = {m: sorted(set(gpu.pop(f"{m}LoadS"))) for m in ("nemotron", "fastconformer")}  # one load time per container
    gpu = {**{k: round(v, 1) for k, v in gpu.items()}, "containers": {m: len(v) for m, v in loads.items()},
           "loadS": {m: round(sum(v), 1) for m, v in loads.items()}, "wallS": round(time.time() - t0, 1)}
    doc = {
        "version": T.VERSION, "kind": "lab-triage", "testSet": test_set, "createdAt": now(), "variant": "studio", "models": models,
        "rule": {"pairwise": "claimed - alternative >= margin -> match, <= -margin -> mismatch, else unsure; a side with no score "
                             "(no CTC path) counts as infinitely worse",
                 "outcome": "both match -> auto-accepted; both mismatch -> both-mismatch; one match and one mismatch -> disagree; else unsure",
                 "queue": "every case that is not auto-accepted, a seeded spot check of the auto-accepted cases and a seeded seam spot check "
                          "of the planted edits (splices, repeats, restarts); one item per case, reasons merged",
                 "maxBad": f"at most 1 bad per {T.BAD_PER} queue items, at least 1", "source": "tools/lab_triage_core.py"},
        "construction": {"edit": "the review_splice window (2.5 s either side of the edit, whole words); alternative = the original text at "
                                 "that spot (review_asr_core.revert_text)",
                         "repeat": "the review_repeat window (3 s either side, whole words); alternative = the text without the repeated words",
                         "clean": f"each clean run (lab_checks stretches: unplanted words with no seam inside) cut at word boundaries into "
                                  f"windows of about {T.CLEAN_TARGET_S:g} s (at least {T.CLEAN_MIN_WORDS} words); alternative = one interior "
                                  "word substituted by a clearly different word of the session, or a foreign word inserted "
                                  "(review_asr_core.perturb_text, ops lab_triage_core.CLEAN_OPS: no deletions, CTC drops a heard "
                                  "word cheaply), operation and position seeded by the case id"},
        "spot": spot, "seamSpot": seam_spot, "seed": seed, "queue": queue_rel, "counts": res["counts"], "reviewed": reviewed, "gpu": gpu,
        "cases": [{k: c[k] for k in ("id", "session", "group", "class", "linked", "claimed", "alternative", "alternativeOp", "nemotron",
                                     "fastconformer", "outcome")}
                  | {"audio": c["item"]["audio"], "reviewedOnly": bool(c.get("reviewedOnly")),
                     **({} if c.get("reviewedOnly") else {k: c[k] for k in ("spotCheck", "seamSpotCheck", "queued")})} for c in res["cases"]],
    }
    if no_write:
        return {k: doc[k] for k in ("testSet", "models", "counts", "reviewed", "gpu", "queue", "cases")} | {"fastconformer": fc_res, "nemotron": nem_res}
    _write(base / f"{NAME}.json", doc)
    _write(base / QUEUE, res["queue"])
    _write(base / "triage" / f"{NAME}.nemotron.json", {"version": 1, "testSet": test_set, "createdAt": now(), "model": nem_meta,
                                                         "items": {i: {"candidates": r["scores"], "ctcGreedy": r["ctcGreedy"], "seconds": r["seconds"],
                                                                       "transcript": heard[i]["transcript"]} for i, r in nem_res.items()}})
    _write(base / "triage" / f"{NAME}.fastconformer.json", {"version": 1, "testSet": test_set, "createdAt": now(), "model": fc_meta,
                                                              "items": {i: {k: r[k] for k in ("transcript", "scores", "ctcGreedy")} for i, r in fc_res.items()}})
    vol.commit()
    return {k: doc[k] for k in ("testSet", "models", "counts", "reviewed", "gpu", "queue")}


@app.local_entrypoint()
def main(test_set: str = "lab-v1.1", spot: float = 0.10, seam_spot: float = 0.10, seed: int = 17, chunk: int = 100, nemotron_gpus: int = 4,
         fc_gpus: int = 1, force: bool = False, out: str = "", only: str = "", no_write: bool = False):
    t0 = time.time()
    r = run.remote(test_set, spot, seam_spot, seed, chunk, nemotron_gpus, fc_gpus, force, only.split(",") if only else None, no_write)
    if no_write:
        Path(out or "/tmp/lab-triage-smoke.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps({k: r[k] for k in ("counts", "reviewed", "gpu")}, ensure_ascii=False, indent=1))
        return
    out = out or f"docs/benchmarks/lab-triage-v1-{test_set}.json"
    (ROOT / out).write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    c = r["counts"]
    print(json.dumps({k: c[k] for k in ("cases", "outcomes", "perModel", "queue")}, ensure_ascii=False, indent=1))
    for cls, row in c["byClass"].items():
        print(f"{cls:16} {row}")
    print("reviewed:", json.dumps({k: v for k, v in r["reviewed"].items() if k != "items"}, ensure_ascii=False))
    print("gpu:", json.dumps(r["gpu"]))
    print(f"queue https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue={r['queue']}  wrote {out} [{time.time() - t0:.0f} s]")
