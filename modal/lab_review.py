"""Listening review page for lab test sets (.scratch/lab/issues/06-review-ui.md, docs/benchmarks/lab-review.md).

    modal deploy modal/lab_review.py      # https://<workspace>--tilawa-lab-review-web.modal.run/?queue=<path under /lab/data>
    modal run modal/lab_review.py         # (re)writes the synthetic demo test set to /lab/data/fixture-review/

Queues and audio are read from /lab/data on the tilawa-lab volume; verdicts go to /lab/reviews/<testSet>/<reviewer>.json.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("fastapi[standard]==0.115.4")
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_dir(ROOT / "modal/lab_web", "/root/lab_web")
)
app = modal.App("tilawa-lab-review", image=image)
vol = modal.Volume.from_name("tilawa-lab", create_if_missing=True)
DATA, REVIEWS, WEB = Path("/lab/data"), Path("/lab/reviews"), Path("/root/lab_web")


@app.function(volumes={"/lab": vol}, max_containers=1, scaledown_window=20 * 60, timeout=10 * 60)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    import io
    import json
    import subprocess
    import threading
    import time
    from datetime import datetime, timezone
    from functools import lru_cache

    import lab_review_core as core
    from fastapi import Body, FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI(title="Tilawa listening review")
    api.mount("/static", StaticFiles(directory=WEB), name="static")

    @api.middleware("http")
    async def revalidate_static(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
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
    def page():
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
