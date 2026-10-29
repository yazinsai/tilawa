"""Lab dashboard and listening review page (.scratch/lab/issues/06-review-ui.md and 11-dashboard.md,
docs/benchmarks/lab-review.md).

    modal deploy modal/lab_review.py      # https://<workspace>--tilawa-lab-review-web.modal.run/ (dashboard)
                                          #   .../review?queue=<path under /lab/data> (review page; /?queue= still works)
    LAB_REVIEW_APP=tilawa-lab-review-preview modal deploy modal/lab_review.py
                                          # a preview copy at https://<workspace>--tilawa-lab-review-preview-web.modal.run/
    modal run modal/lab_review.py         # (re)writes the synthetic demo test set to /lab/data/fixture-review/

Queues and audio are read from /lab/data on the tilawa-lab volume; verdicts go to /lab/reviews/<testSet>/<reviewer>.json.
The dashboard reads /lab/ledger.json and the latest main-line run's /lab/runs/<run>/score.json.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# A preview app shares the volume; it writes to the same review files as the live one, so test with a throwaway reviewer.
APP = os.environ.get("LAB_REVIEW_APP", "tilawa-lab-review")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("fastapi[standard]==0.115.4")
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_file(ROOT / "tools/lab_ledger.py", "/root/lab_ledger.py")
    .add_local_file(ROOT / "tools/lab_dashboard_core.py", "/root/lab_dashboard_core.py")
    .add_local_file(ROOT / "docs/benchmarks/scoreboard.md", "/root/scoreboard.md")
    .add_local_dir(ROOT / "modal/lab_web", "/root/lab_web")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name("tilawa-lab", create_if_missing=True)
LAB, DATA, REVIEWS, WEB = Path("/lab"), Path("/lab/data"), Path("/lab/reviews"), Path("/root/lab_web")


@app.function(volumes={"/lab": vol}, max_containers=1, scaledown_window=20 * 60, timeout=10 * 60)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    import io
    import json
    import re
    import subprocess
    import threading
    import time
    from datetime import datetime, timezone
    from functools import lru_cache

    import lab_dashboard_core as D
    import lab_ledger as L
    import lab_review_core as core
    from fastapi import Body, FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI(title="Tilawa listening review")
    api.mount("/static", StaticFiles(directory=WEB), name="static")

    @api.middleware("http")
    async def revalidate_static(request: Request, call_next):
        static = request.url.path.startswith("/static/")
        if static:
            # Modal mounts files with mtime 0: Last-Modified never changes, so a conditional request would always get a 304.
            request.scope["headers"] = [(k, v) for k, v in request.scope["headers"] if k not in (b"if-modified-since", b"if-none-match")]
        resp = await call_next(request)
        if static:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable" if request.url.path.startswith("/static/fonts/") else "no-cache"
        return resp

    # This container is the only writer of /lab/reviews (max_containers=1). Saves commit in the background, and a reload
    # (to see new queues and audio) never runs while a save is uncommitted.
    cond, st = threading.Condition(), {"dirty": False, "committing": False, "reloaded": 0.0}

    def committer():
        while True:
            with cond:
                while not st["dirty"]:
                    cond.wait()
                st["dirty"], st["committing"] = False, True
            try:
                vol.commit()
            except Exception as e:
                print(f"commit failed, retrying: {e!r}")
                time.sleep(1)
                with cond:
                    st["dirty"] = True
            finally:
                with cond:
                    st["committing"] = False

    threading.Thread(target=committer, daemon=True).start()

    def reload(force=False):
        with cond:
            if st["dirty"] or st["committing"] or not force and time.time() - st["reloaded"] < 5:
                return
            try:
                vol.reload()
                st["reloaded"] = time.time()
            except Exception as e:
                print(f"reload failed: {e!r}")

    def data_file(rel: str) -> Path:
        try:
            p = DATA / core.safe_rel(rel)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not p.is_file():
            reload(force=True)
        if not p.is_file():
            raise HTTPException(404, f"not on the volume: {rel}")
        return p

    def name(v: str, what: str) -> str:
        if not core.SAFE.match(v or ""):
            raise HTTPException(400, f"{what} must be letters, digits, '.', '_' or '-'")
        return v

    def load_queue(rel: str) -> dict:
        try:
            q = json.loads(data_file(rel).read_text())
        except json.JSONDecodeError as e:
            raise HTTPException(422, {"problems": [f"not valid JSON: {e}"]})
        if problems := core.validate_queue(q):
            raise HTTPException(422, {"problems": problems})
        return q

    def reviews_path(test_set: str, reviewer: str) -> Path:
        return REVIEWS / test_set / f"{reviewer}.json"

    def load_reviews(test_set: str, reviewer: str) -> dict:
        p = reviews_path(test_set, reviewer)
        return json.loads(p.read_text()) if p.is_file() else core.empty_reviews(test_set, reviewer)

    @lru_cache(maxsize=128)
    def clip(path: str, mtime: float, start: float, end: float) -> bytes:
        if path.lower().endswith(".wav"):
            with open(path, "rb") as f:
                return core.wav_slice(f, start, end)
        cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", "-"]
        return core.wav_slice(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout))

    @api.get("/")
    def page(request: Request):
        html = "review.html" if "queue" in request.query_params else "dashboard.html"
        return FileResponse(WEB / html, headers={"Cache-Control": "no-cache"})

    @api.get("/review")
    def review_page():
        return FileResponse(WEB / "review.html", headers={"Cache-Control": "no-cache"})

    @api.get("/api/queues")
    def queues():
        reload()
        return {"queues": sorted(str(p.relative_to(DATA)) for p in DATA.glob("*/review/*.json"))}

    @api.get("/api/queue")
    def queue(path: str, reviewer: str = "me"):
        reload()
        q = load_queue(path)
        rv = load_reviews(q["testSet"], name(reviewer, "reviewer"))["items"]
        return {"path": path, "reviewer": reviewer, "queue": q, "reviews": rv, "tally": core.tally(q, rv), "verdicts": core.VERDICTS}

    @api.post("/api/verdict")
    def verdict(body: dict = Body(...)):
        path, reviewer, item = body.get("queue"), name(body.get("reviewer", "me"), "reviewer"), body.get("id")
        q = load_queue(path)
        if item not in {it["id"] for it in q["items"]}:
            raise HTTPException(404, f"no item {item!r} in {path}")
        with cond:
            doc = load_reviews(q["testSet"], reviewer)
            at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                rec = core.upsert(doc, item, body.get("verdict"), body.get("note", ""), at, path)
            except ValueError as e:
                raise HTTPException(400, str(e))
            p = reviews_path(q["testSet"], reviewer)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            tmp.replace(p)
            st["dirty"] = True
            cond.notify()
        return {"id": item, "review": rec, "tally": core.tally(q, doc["items"]), "file": str(p)}

    @api.get("/audio")
    def audio(path: str, start: float, end: float, request: Request):
        if not (0 <= start < end) or end - start > 600:
            raise HTTPException(400, "need 0 <= start < end, at most 10 minutes")
        p = data_file(path)
        try:
            body = clip(str(p), p.stat().st_mtime, round(start, 3), round(end, 3))
        except (ValueError, subprocess.CalledProcessError) as e:
            raise HTTPException(422, f"cannot read audio: {e}")
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=600", "Content-Type": "audio/wav"}
        try:
            r = core.byte_range(request.headers.get("range"), len(body))
        except ValueError:
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{len(body)}"})
        if r is None:
            return Response(body, headers=headers)
        a, b = r
        return Response(body[a : b + 1], status_code=206, headers={**headers, "Content-Range": f"bytes {a}-{b}/{len(body)}"})

    # Dashboard (ticket 11). Every file is parsed once per (mtime, size); requests only shape cached JSON.

    parsed: dict[Path, tuple] = {}
    RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
    gates = D.release(Path("/root/scoreboard.md").read_text())

    def read_json(p: Path):
        try:
            s = p.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        hit = parsed.get(p)
        if hit and hit[0] == (s.st_mtime, s.st_size):
            return hit[1]
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"cannot read {p}: {e!r}")
            return None
        parsed[p] = ((s.st_mtime, s.st_size), obj)
        return obj

    def run_score(run: str | None):
        if not run:
            return None
        if not RUN.match(run) or ".." in run:
            raise HTTPException(400, "bad run name")
        sc = read_json(LAB / "runs" / run / "score.json")
        return sc if sc and sc.get("kind") == "lab-score" else None

    def lab_state(run: str | None = None, ledger: str | None = None):
        """(ledger or None, test set entry or None, run shown, its score.json or None). `ledger` is a throwaway ledger at
        the volume root, as `lab.py --ledger` writes; the default is /lab/ledger.json."""
        reload()
        led = read_json(LAB / (name(ledger, "ledger") if ledger else "ledger.json"))
        led = led if led and led.get("kind") == "lab-ledger" else None
        r = run or D.main_run(led)
        sc = run_score(r)
        tsd = (led["testSets"].get(led["testSet"]) if led and led.get("testSet") else None) if not run else None
        return led, tsd, r, sc

    def base_for(tsd, sc) -> str:
        return tsd["base"] if tsd else D.base_of((sc or {}).get("testSet", ""))

    def all_reviews(base: str) -> list[dict]:
        d = REVIEWS / base
        return [x for x in (read_json(p) for p in sorted(d.glob("*.json"))) if x] if base and d.is_dir() else []

    def flags_and_check(tsd, sc):
        base = base_for(tsd, sc)
        reviews = all_reviews(base)
        queues = {}
        for q in {r.get("queue") for rv in reviews for r in rv.get("items", {}).values()} - {None}:
            try:
                queues[q] = read_json(DATA / core.safe_rel(q))
            except ValueError:
                pass
        sessions = set(tsd["sessions"]["practice"] + tsd["sessions"]["exam"]) if tsd else {r["id"] for r in (sc or {}).get("results", [])}
        applied = ((read_json(DATA / tsd["exclude"]) or {}).get("excluded", [])) if tsd and tsd.get("exclude") else []
        pend, warn = D.pending_flags(reviews, {k: v for k, v in queues.items() if v}, sessions, applied)
        path = D.check_queue(base, lambda n: (DATA / base / "review" / n).is_file())
        check = D.check_status(read_json(DATA / core.safe_rel(path)), reviews, path)
        return base, reviews, pend, warn, check

    def everywhere(led, sc) -> set:
        """Practice cases failing on the shown run and in every open experiment's latest run on the current test set."""
        if not led:
            return set()
        runs = [r["run"] for e in led["experiments"].values() if e["state"] == "open" and (r := L.latest(e, led["testSet"]))]
        sets = [D.failing_keys(s) for s in (run_score(r) for r in runs) if s]
        return D.fails_everywhere(D.failing_keys(sc), sets)

    def find_case(sc, key: str) -> dict:
        c = next((c for c in sc.get("cases", []) if c["key"] == key and c.get("split") == "practice"), None)
        if not c:
            raise HTTPException(404, f"no practice case {key!r} in run {sc.get('run')}")
        return c

    def session_doc(c: dict):
        p = D.session_doc_path(c)
        try:
            return read_json(DATA / core.safe_rel(p)) if p else None
        except ValueError:
            return None

    def need_score(sc, r):
        if not sc:
            raise HTTPException(404, f"no score.json for run {r}" if r else "no baseline yet: run `python3 tools/lab.py baseline`")
        return sc

    @api.get("/api/dashboard")
    def dashboard(run: str | None = None, ledger: str | None = None):
        led, tsd, r, sc = lab_state(run, ledger)
        _, _, pend, warn, check = flags_and_check(tsd, sc)
        out = {"header": D.header(sc, led=None if run else led, pending=len(pend), check=check, run=r), "run": r, "asked": run,
               "overview": D.overview(sc) if sc else None, "experiments": D.experiment_rows(led) if led else [],
               "history": D.history(led) if led else [], "flagWarnings": warn, "runs": []}
        if not led or not sc:
            found = sorted((LAB / "runs").glob("*/score.json"), key=lambda p: p.stat().st_mtime, reverse=True) if (LAB / "runs").is_dir() else []
            out["runs"] = [p.parent.name for p in found[:12]]
        return out

    @api.get("/api/cases")
    def cases(component: str, variant: str = "phone", run: str | None = None, ledger: str | None = None):
        if component not in D.COMPONENTS or variant not in ("phone", "studio"):
            raise HTTPException(400, f"component is one of {', '.join(D.COMPONENTS)}; variant phone or studio")
        led, tsd, r, sc = lab_state(run, ledger)
        need_score(sc, r)
        _, reviews, *_ = flags_and_check(tsd, sc)
        rows = D.order_cases(D.failing(sc, component, variant), everywhere(led, sc), L.latest_verdicts(reviews))
        other = "studio" if variant == "phone" else "phone"
        return {"component": component, "variant": variant, "run": r, "testSet": sc["testSet"], "cases": rows,
                "other": {"variant": other, "n": len(D.failing(sc, component, other))},
                "card": next(c for c in D.cards(sc["splits"]["practice"]) if c["key"] == component) if sc["splits"]["practice"]["sessions"] else None}

    @api.get("/api/case")
    def case(key: str, run: str | None = None, reviewer: str = "me", ledger: str | None = None):
        led, tsd, r, sc = lab_state(run, ledger)
        c = find_case(need_score(sc, r), key)
        base = base_for(tsd, sc)
        return {"item": D.case_item(c, session_doc(c)), "story": D.engine_story(c), "engine": D.engine_facts(c), "suspects": D.suspects(c),
                "component": c["component"],
                "review": load_reviews(base, name(reviewer, "reviewer"))["items"].get(key), "testSet": base}

    @api.post("/api/case-verdict")
    def case_verdict(body: dict = Body(...)):
        reviewer, key = name(body.get("reviewer", "me"), "reviewer"), body.get("key")
        led, tsd, r, sc = lab_state(body.get("run"), body.get("ledger"))
        c = find_case(need_score(sc, r), key)
        base = base_for(tsd, sc)
        qrel = f"{base}/review/{D.DASH_QUEUE}"
        item = D.case_item(c, session_doc(c))
        with cond:
            doc = load_reviews(base, reviewer)
            at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                rec = core.upsert(doc, key, body.get("verdict"), body.get("note", ""), at, qrel)
            except ValueError as e:
                raise HTTPException(400, str(e))
            qp = DATA / qrel
            qd = json.loads(qp.read_text()) if qp.is_file() else None
            for p, obj in ((qp, D.upsert_queue_item(qd, base, item)), (reviews_path(base, reviewer), doc)):
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
                tmp.replace(p)
            st["dirty"] = True
            cond.notify()
        _, _, pend, _, _ = flags_and_check(tsd, sc)
        return {"key": key, "review": rec, "pending": len(pend), "file": str(reviews_path(base, reviewer)), "queue": qrel}

    @api.get("/api/release")
    def release():
        return {"release": gates, "source": "docs/benchmarks/scoreboard.md"}

    return api


@app.function(volumes={"/lab": vol}, timeout=10 * 60)
def make_fixture() -> list[str]:
    import json

    import lab_review_core as core

    out = []
    for rel, v in core.fixture().items():
        p = DATA / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v if isinstance(v, bytes) else json.dumps(v, ensure_ascii=False, indent=1).encode())
        out.append(f"{rel} ({p.stat().st_size // 1024} KB)")
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    for line in make_fixture.remote():
        print(line)
    print("open: <web url>/?queue=fixture-review/review/demo.json")
