"""Automated checks of a lab test set from the teacher's evidence of its built audio (.scratch/lab, ticket 04).

    modal run modal/lab_check.py                       # lab-v1: studio (acceptance) and phone
    modal run modal/lab_check.py --test-set lab-v1 --variants studio --only lab-v1-p03 --no-write

Needs teacher evidence (`modal run modal/lab_decode.py --listener teacher`). Logic in tools/lab_checks.py. Writes, under
/lab/data/<testSet>/:

    checks-v1.json         per-case pass/fail with reasons, per variant; summary; `failed` (acceptance variant)
    review/v1-check.json   the listening queue with every failed planted edit in it (replacing passing items of its kind)
    review/v1-check.base.json  the builder's queue (`lab_build_sessions.py --stage plan` rewrites v1-check.json; re-run this after it)
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/app/tools"]

VOL = "/lab"
DATA = Path(VOL) / "data"
ACCEPT = "studio"
RECITERS = "docs/benchmarks/lab-reciters-v1.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "rapidfuzz==3.14.1", "quran-transcript==0.6.4", "pyarrow")
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_file(ROOT / "docs/benchmarks/listener-target-v1.json", "/app/docs/benchmarks/listener-target-v1.json")
    .add_local_file(ROOT / RECITERS, f"/app/{RECITERS}")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__"])
)
app = modal.App("tilawa-lab-check", image=image)
vol = modal.Volume.from_name("tilawa-lab")


@app.function(volumes={VOL: vol}, cpu=2, memory=8192, timeout=30 * 60, max_containers=80)
def check(test_set: str, sid: str, variant: str, pattern: str) -> dict:
    import lab_checks as C
    import lab_sources as L

    t0 = time.time()
    base = DATA / test_set
    session = json.loads((base / f"sessions/{sid}.json").read_text())
    ev = json.loads((base / pattern.replace("{id}", sid).replace("{listener}", "teacher").replace("{variant}", variant)).read_text())
    out = C.check_session(session, ev, L.reference(), variant)
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.function(volumes={VOL: vol}, timeout=10 * 60)
def load(test_set: str) -> dict:
    base = DATA / test_set
    m = json.loads((base / "manifest-v1.json").read_text())
    return {"pattern": m["evidencePattern"], "ids": [e["id"] for e in m["sessions"]]}


@app.function(volumes={VOL: vol}, timeout=10 * 60)
def finish(test_set: str, ids: list[str], results: list[dict], errors: list[str], write: bool) -> dict:
    """Reciter checks, the report, and the review queue with every failed planted edit in it."""
    import lab_checks as C
    import lab_sessions as S
    import lab_sources as L
    from lab_review_core import validate_queue

    vol.reload()
    base = DATA / test_set
    every = [e["id"] for e in json.loads((base / "manifest-v1.json").read_text())["sessions"]]
    sessions = {i: json.loads((base / f"sessions/{i}.json").read_text()) for i in every}
    reciters = json.loads(Path("/app", RECITERS).read_text())
    rec = {i: C.reciter_check(sessions[i], reciters) for i in ids}
    accept = [r for r in results if r["variant"] == ACCEPT]
    failed = [e for r in accept for e in r["edits"] if not e["pass"]]
    report = {
        "version": C.VERSION, "kind": "lab-checks", "testSet": test_set, "teacher": "obadx/muaalem-model-v3_2", "acceptanceVariant": ACCEPT,
        "checkedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "thresholds": C.THRESHOLDS,
        "definitions": C.__doc__.split("\n\n", 1)[1].strip(),
        "summary": {**C.summarize(results), "reciters": {"n": len(rec), "pass": sum(r["pass"] for r in rec.values())}},
        "failed": [{k: e.get(k) for k in ("id", "session", "splice", "class", "type", "location", "startS", "endS", "reasons", "llr", "localPer")} for e in failed],
        "failedClean": [{k: c[k] for k in ("id", "session", "from", "to", "startS", "endS", "per", "phonemes", "reasons")} for r in accept for c in r["clean"] if not c["pass"]],
        "failedTiming": [{"session": r["session"], **r["timing"]} for r in accept if not r["timing"]["pass"]],
        "failedReciters": [{"session": i, **r} for i, r in rec.items() if not r["pass"]],
        "errors": errors,
        "sessions": [{"id": i, "reciter": rec[i], "variants": {r["variant"]: r for r in results if r["session"] == i}} for i in ids],
    }
    q_path, keep = base / "review/v1-check.json", base / "review/v1-check.base.json"
    original = S.review_queue([sessions[k] for k in every], S.Ref(L.reference()))
    if original != json.loads(q_path.read_text()) and not (keep.exists() and original == json.loads(keep.read_text())):
        raise RuntimeError("the builder's review queue no longer reproduces review/v1-check.json; not rewriting it")
    queue, changes = C.review_with_failures(original, sessions, report["failed"], test_set)
    problems = validate_queue(queue)
    if problems:
        raise RuntimeError(f"review queue: {problems[:5]}")
    report["review"] = {"queue": str(q_path.relative_to(DATA)), **changes}
    if write:
        keep.write_text(json.dumps(original, ensure_ascii=False, indent=1) + "\n")
        q_path.write_text(json.dumps(queue, ensure_ascii=False, indent=1) + "\n")
        (base / "checks-v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        vol.commit()
    return {k: report[k] for k in ("summary", "failed", "failedClean", "failedTiming", "failedReciters", "errors", "review")}


@app.local_entrypoint()
def main(test_set: str = "lab-v1", variants: str = "studio,phone", only: str = "", no_write: bool = False):
    t0 = time.time()
    info = load.remote(test_set)
    ids = [i for i in info["ids"] if not only or i in only.split(",")]
    jobs = [(test_set, i, v, info["pattern"]) for i in ids for v in variants.split(",")]
    results = list(check.starmap(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    errors = [f"{j[1]} {j[2]}: {r!r}"[:400] for j, r in zip(jobs, results) if isinstance(r, Exception)]
    out = finish.remote(test_set, ids, [r for r in results if not isinstance(r, Exception)], errors, not no_write and not only)
    print(json.dumps(out["summary"], indent=1))
    print(f"failed edits ({ACCEPT}): {len(out['failed'])}; clean stretches: {len(out['failedClean'])}; timing: {len(out['failedTiming'])}; reciters: {len(out['failedReciters'])}")
    for e in out["failed"]:
        print(f"  {e['id']} {e['class']}: {'; '.join(e['reasons'])}")
    for c in out["failedClean"]:
        print(f"  {c['id']} {c['from']}-{c['to']}: {'; '.join(c['reasons'])}")
    for t in out["failedTiming"]:
        print(f"  timing {t['session']}: {t['off'][:5]}")
    print(f"review queue: {out['review']}")
    if errors:
        print("errors:", *errors, sep="\n  ")
    print(f"done in {time.time() - t0:.0f} s")
