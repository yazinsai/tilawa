"""Fetch mp3quran full-surah recordings of the held-out lab reciters and teacher-align their words (.scratch/lab, ticket 02).

    modal run modal/lab_fetch_align.py --pilot              # 2 reciters x 3 surahs
    modal run modal/lab_fetch_align.py                      # the whole pool, then the report
    modal run modal/lab_fetch_align.py --stage align --force-align   # re-align from stored evidence (CPU only)

The pool is chosen once from the timed-surah lists (rule in tools/lab_sources.py) and kept in
docs/benchmarks/lab-sources-v1.json; --repool re-fetches the lists and re-picks. Every stage skips finished items.
On the tilawa-lab volume, under data/sources/mp3quran/<readId>/<sss> (sss = zero-padded surah):

    .wav            16 kHz mono PCM16, decoded by ffmpeg (bitexact) from the mp3
    .source.json    mp3 URL, sha256, bytes, fetch time, wav hash, and the raw ayat_timing JSON with its hash and fetch time
    .evidence.npz   teacher pass over the whole surah: top-5 ids/probs per 40 ms frame, greedy phonemes, window plan
    .align.json     ayah windows, aligned ayah spans, word timings, consistency verdicts (schema in lab-sources-v1.md)

plus report-v1.json and catalog-v1.json beside them.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/app/tools"]
import lab_sources as L  # noqa: E402

MODEL = "obadx/muaalem-model-v3_2"
MP3Q = "https://www.mp3quran.net/api/v3"
UA = "tilawa-lab/1"
TOPK = 5
BATCH = 8
VOL = "/lab"
BASE = "sources/mp3quran"
ALIGN_VERSION = "lab-align-v2"
PAUSE_DB = -12
LOUD_DB = -6
RECITERS = ROOT / "docs/benchmarks/lab-reciters-v1.json"
POOL_FILE = ROOT / "docs/benchmarks/lab-sources-v1.json"

base = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "curl")
cpu_image = (
    base.pip_install("numpy", "rapidfuzz==3.14.1", "quran-transcript==0.6.4", "pyarrow")
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__"])
)
gpu_image = (
    base.pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "numpy", "rapidfuzz==3.14.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .add_local_file(ROOT / "packages/core/assets/ref-v2.json", "/app/packages/core/assets/ref-v2.json")
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["**/__pycache__"])
)
app = modal.App("tilawa-lab-fetch-align")
vol = modal.Volume.from_name("tilawa-lab", create_if_missing=True)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def path(read: int, surah: int, ext: str) -> Path:
    return Path(VOL) / "data" / BASE / str(read) / f"{surah:03d}.{ext}"


def rel(p: Path) -> str:
    return str(p.relative_to(Path(VOL) / "data"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get(url: str) -> bytes:
    import time
    import urllib.request

    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=600).read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * 2**attempt)


def write_atomic(p: Path, data: bytes) -> None:
    import os

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def read_wav(p: Path):
    import wave

    import numpy as np

    with wave.open(str(p), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, L.SR), p
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


@app.function(image=cpu_image, timeout=10 * 60)
def timed(reads: list[int]) -> dict:
    return {"fetchedAt": now(), "reads": {r: sorted(int(x["id"]) for x in json.loads(get(f"{MP3Q}/ayat_timing/soar?read={r}"))) for r in reads}}


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=60 * 60, max_containers=32, retries=1)
def fetch(read: int, server: str, surah: int) -> dict:
    import hashlib
    import subprocess
    import tempfile

    src = path(read, surah, "source.json")
    if src.exists():
        return {"item": f"{read}/{surah}", "skipped": True}
    t_url = f"{MP3Q}/ayat_timing?surah={surah}&read={read}"
    t_raw, t_at = get(t_url), now()
    url = f"{server.rstrip('/')}/{surah:03d}.mp3"
    # curl drops a stalled CDN transfer (under 50 kB/s for 30 s) and retries; urllib would hang on it.
    mp3 = subprocess.run(["curl", "-fsSL", "--http1.1", "-A", UA, "--retry", "8", "--retry-all-errors", "--retry-delay", "5", "--speed-limit", "50000",
                          "--speed-time", "30", "--max-time", "1800", url], capture_output=True, check=True).stdout
    at = now()
    wav = path(read, surah, "wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(mp3)
        f.flush()
        tmp = wav.with_name(wav.name + ".tmp.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f.name, "-ac", "1", "-ar", str(L.SR), "-c:a", "pcm_s16le", "-map_metadata", "-1",
                        "-fflags", "+bitexact", "-flags:a", "+bitexact", str(tmp)], check=True)
        tmp.replace(wav)
    data = wav.read_bytes()
    n = len(read_wav(wav))
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    meta = {
        "version": 1, "readId": read, "surah": surah, "url": url, "bytes": len(mp3), "sha256": hashlib.sha256(mp3).hexdigest(), "fetchedAt": at,
        "wav": {"path": rel(wav), "sha256": hashlib.sha256(data).hexdigest(), "samples": n, "durationS": n / L.SR, "sampleRate": L.SR, "ffmpeg": ffmpeg},
        # mp3quran has no API revision; the timing's hash and fetch time stand in for one.
        "timingUrl": t_url, "timingFetchedAt": t_at, "timingSha256": hashlib.sha256(t_raw).hexdigest(), "timing": json.loads(t_raw),
    }
    write_atomic(src, json.dumps(meta, ensure_ascii=False).encode())
    vol.commit()
    return {"item": f"{read}/{surah}", "mb": round(len(mp3) / 1e6, 1), "durationS": round(n / L.SR, 1)}


@app.cls(image=gpu_image, gpu="L4", volumes={VOL: vol, "/cache": cache_vol}, secrets=[hf_secret], memory=16384, timeout=2 * 60 * 60, max_containers=32)
class Teacher:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        self.torch = torch
        m = Muaalem(MODEL, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor
        vocab = m.multi_level_tokenizer.id_to_vocab["phonemes"]
        tokens = L.reference()["inventory"]["tokens"]
        assert [vocab[i] for i in range(1, len(tokens) + 1)] == list(tokens), "teacher vocab differs from ref-v2 inventory"

    def infer(self, waves):
        torch = self.torch
        feats = self.processor(waves, sampling_rate=L.SR, return_tensors="pt", padding=True)
        mask = feats["attention_mask"]
        feats = {k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
        with torch.no_grad():
            lg = self.model(**feats, return_dict=False)[0]["phonemes"]
        probs = torch.softmax(lg.float(), dim=-1).cpu()
        ratio = probs.shape[1] / mask.shape[1]
        return [probs[i, : max(1, round(n * ratio))] for i, n in enumerate(mask.sum(dim=1).tolist())]

    @modal.method()
    def decode(self, read: int, surah: int) -> dict:
        import io
        import time

        import numpy as np

        dest = path(read, surah, "evidence.npz")
        vol.reload()
        if dest.exists():
            return {"item": f"{read}/{surah}", "skipped": True}
        t0 = time.time()
        x = read_wav(path(read, surah, "wav"))
        plan = L.windows(len(x))
        w = L.WINDOW_S * L.SR
        ids, probs = [], []
        for b in range(0, len(plan), BATCH):
            part = plan[b : b + BATCH]
            for (s, lo, hi), p in zip(part, self.infer([x[s : s + w] for s, _, _ in part])):
                if hi > p.shape[0]:
                    raise RuntimeError(f"window at {s} gave {p.shape[0]} frames, needs {hi}")
                top_p, top_i = p[lo : hi if hi >= 0 else None].topk(TOPK, dim=-1)
                ids.append(top_i.numpy().astype(np.uint8))
                probs.append(top_p.numpy().astype(np.float16))
        ids_all, p_all = np.concatenate(ids), np.concatenate(probs)
        g = L.greedy(ids_all[:, 0], p_all[:, 0].astype(np.float32))
        buf = io.BytesIO()
        np.savez_compressed(buf, ids=ids_all, p=p_all, greedy=np.array([t[:3] for t in g], dtype=np.int32).reshape(-1, 3),
                            greedy_conf=np.array([t[3] for t in g], dtype=np.float16), windows=np.array(plan, dtype=np.int64),
                            frame_ms=np.int32(L.FRAME * 1000 // L.SR), model=np.str_(MODEL))
        write_atomic(dest, buf.getvalue())
        vol.commit()
        return {"item": f"{read}/{surah}", "frames": int(len(ids_all)), "expected": len(x) // L.FRAME, "seconds": round(time.time() - t0, 1),
                "audioS": round(len(x) / L.SR, 1)}


def align_surah(read: int, surah: int, meta: dict) -> dict:
    """Word timings and ayah verdicts for one surah from its stored evidence, audio and mp3quran timing."""
    import numpy as np
    from e1_report import TOKEN, log_probs, viterbi

    ref = L.reference()
    tokens = ref["inventory"]["tokens"]
    rows = [a for a in ref["ayahs"] if a["s"] == surah]
    src = json.loads(path(read, surah, "source.json").read_text())
    with np.load(path(read, surah, "evidence.npz")) as ev:
        ids, p, frame_ms = ev["ids"], ev["p"], int(ev["frame_ms"])
    T, fs = len(ids), frame_ms / 1000
    lp = log_probs({"n_frames": T, "top5_ids": ids.tobytes(), "top5_p": p.tobytes()})
    x = read_wav(path(read, surah, "wav"))
    db = L.rms_db(x)
    ayat, pre, bad = L.parse_timing(src["timing"], len(rows), len(x) / L.SR)
    out = {"version": ALIGN_VERSION, "readId": read, "surah": surah, "reciter": meta["nameEn"], "split": meta["split"],
           "audio": rel(path(read, surah, "wav")), "evidence": rel(path(read, surah, "evidence.npz")), "durationS": len(x) / L.SR,
           "frameMs": frame_ms, "teacher": MODEL, "reference": ref["version"], "timingSha256": src["timingSha256"],
           "timingProblems": bad, "preRoll": pre, "basmala": None, "ayahs": [], "words": []}
    if bad:
        out["ayahs"] = [{"ayah": a["ayah"], "windowS": [a["startS"], a["endS"]], "excluded": True, "reasons": ["timing"]} for a in ayat]
        return out

    def hyp(f0: int, f1: int) -> str:
        f0, f1 = max(0, f0), min(T, f1)
        return "".join(tokens[i - 1] for i, *_ in L.greedy(ids[f0:f1, 0], p[f0:f1, 0].astype(np.float32)))

    top1 = ids[:, 0].astype(np.int64)

    def force(parts: list[tuple[str, list[int]]], s0: float, s1: float):
        """Align the parts' text in sequence over [s0, s1]; per part, its word spans and PER.

        Neighbours' text claims their audio, so a rhyming ayah ending (fawasil) can't pull the focus ayah into them.
        """
        segs = [L.collapse(pp, oo) for pp, oo in parts]
        seq = "".join(c for c, _ in segs)
        f0, f1 = max(0, int(s0 / fs)), min(T, int(np.ceil(s1 / fs)))
        _, spans = viterbi(L.garbage(lp[f0:f1]), seq)
        if spans is None:
            return None
        spans = [(s + f0, e + f0) for s, e in spans]
        tp = [float(np.exp(lp[s:e, TOKEN[ch]]).mean()) for (s, e), ch in zip(spans, seq)]
        ext = L.extend(spans, [TOKEN[ch] for ch in seq], top1, f1)
        out, k0 = [], 0
        for (pp, _), (c, offs) in zip(parts, segs):
            mine = ext[k0 : k0 + len(c)]
            h = hyp(mine[0][0] - 1, mine[-1][1] + 1)
            out.append((L.word_spans(offs, mine, tp[k0 : k0 + len(c)]), L.per(pp, h), len(L.squeeze(h)) / max(1, len(L.squeeze(pp)))))
            k0 += len(c)
        return out

    parts = [(r["p"], r["o"]) for r in rows]
    aligned, words = [], []
    for k, a in enumerate(ayat):
        lo, hi = max(0, k - 1), min(len(ayat), k + 2)
        res = force(parts[lo:hi], ayat[lo]["startS"] - L.PAD_S, ayat[hi - 1]["endS"] + L.PAD_S)
        res = res and res[k - lo]
        if k == 0 and surah not in (1, 9):
            # The basmala, when recited, ends like many first ayahs (ar-rahman); give it its own tokens and keep the better fit.
            b = ref["ayahs"][0]
            alt = force([(b["p"], b["o"]), *parts[:hi]], max(pre["startS"] if pre else 0.0, a["startS"] - L.BASMALA_MAX_S), ayat[hi - 1]["endS"] + L.PAD_S)
            if alt:
                (bws, be, _), alt = alt[0], alt[1]
                out["basmala"] = {"startS": round(bws[0][0] * fs, 3), "endS": round(bws[-1][1] * fs, 3), "per": round(be, 4), "present": be <= L.MAX_PER}
                if res is None or alt[1] < res[1]:
                    res = alt
        if res is None:
            aligned.append(None)
            words.append(None)
            continue
        ws, e, ratio = res
        aligned.append((ws[0][0] * fs, ws[-1][1] * fs))
        words.append((ws, e, ratio))
    mism = L.mismatches(ayat, aligned)
    flat = []
    for k, (a, sp, wv, m) in enumerate(zip(ayat, aligned, words, mism)):
        row = {"ayah": a["ayah"], "windowS": [a["startS"], a["endS"]]}
        if sp is None:
            out["ayahs"].append({**row, "startS": None, "endS": None, "per": None, "score": None, "minWordScore": None, "mismatchS": None,
                                 "excluded": True, "reasons": L.verdict(m, None, None)})
            continue
        ws, e, ratio = wv
        scores = [s for *_, s in ws if s is not None]
        ref_db = float(np.median(db[int(sp[0] * 100) : max(int(sp[0] * 100) + 1, int(sp[1] * 100))]))
        reasons = L.verdict(m, e, min(scores) if scores else None)
        out["ayahs"].append({**row, "startS": round(sp[0], 3), "endS": round(sp[1], 3), "per": round(e, 4), "lenRatio": round(ratio, 3),
                             "score": round(float(np.mean(scores)), 4) if scores else None, "minWordScore": round(min(scores), 4) if scores else None,
                             "mismatchS": round(m, 3), "levelDb": round(ref_db, 1), "excluded": bool(reasons), "reasons": reasons})
        for i, (f0, f1, s) in enumerate(ws):
            flat.append({"surah": surah, "ayah": a["ayah"], "word": i + 1, "startS": round(f0 * fs, 3), "endS": round(f1 * fs, 3),
                         "score": None if s is None else round(s, 4), "_db": ref_db})
    for w, nxt in zip(flat, flat[1:] + [None]):
        ref_db = w.pop("_db")
        if nxt is None or nxt["ayah"] - w["ayah"] > 1:
            w["cutAfterS"] = w["cutDb"] = None
        else:
            t, d = L.cut(db, w["endS"], nxt["startS"])
            w["cutAfterS"], w["cutDb"] = round(t, 3), round(d - ref_db, 1)
    out["words"] = flat

    out["checks"] = spot_checks(out, db)
    return out


def spot_checks(out: dict, db) -> dict:
    """Numeric alignment sanity from the audio energy (10 ms dB, relative to the ayah's median speech level).

    wordCutDb / wordMidMinDb: the quietest 10 ms frame in the 80 ms around each in-ayah word boundary vs the same search
    at each word's middle (the control). At each ayah boundary whose quietest frame within 0.3 s is PAUSE_DB below the
    level: whether that dip lies in the aligned gap (40 ms slack), and the lag of the aligned ayah end/start behind
    the energy offset/onset (LOUD_DB crossing; a reverb tail counts as sound). Also whether the mp3quran boundary lies
    in the aligned gap.
    """
    import numpy as np

    ay = [a for a in out["ayahs"] if a.get("startS") is not None]
    level = {a["ayah"]: a["levelDb"] for a in ay}
    ws = out["words"]
    inner = [w["cutDb"] for w, n in zip(ws, ws[1:]) if w["ayah"] == n["ayah"] and w["cutDb"] is not None]
    mids = []
    for w in ws:
        if w["endS"] - w["startS"] >= 0.2:
            m = int((w["startS"] + w["endS"]) * 50)
            mids.append(float(np.min(db[m - 4 : m + 5])) - level[w["ayah"]])

    def pause(end: float, start: float, lvl: float, acc: dict) -> None:
        lo, hi = max(0, int((end - 0.3) * 100)), min(len(db), int((start + 0.3) * 100))
        if hi - lo < 2:
            return
        q = lo + int(np.argmin(db[lo:hi]))
        if db[q] - lvl > PAUSE_DB:
            return
        acc["inGap"].append(end - 0.04 <= q / 100 <= start + 0.04)
        loud = np.nonzero(db[max(0, q - 100) : q] > lvl + LOUD_DB)[0]
        if len(loud):
            acc["offset"].append(end - (max(0, q - 100) + int(loud[-1]) + 1) / 100)
        loud = np.nonzero(db[q : q + 100] > lvl + LOUD_DB)[0]
        if len(loud):
            acc["onset"].append(start - (q + int(loud[0])) / 100)

    A = {"inGap": [], "offset": [], "onset": []}
    mp3q = []
    for a, b in zip(ay, ay[1:]):
        if b["ayah"] == a["ayah"] + 1:
            pause(a["endS"], b["startS"], (a["levelDb"] + b["levelDb"]) / 2, A)
            mp3q.append(a["endS"] - L.MISMATCH_S <= b["windowS"][0] <= b["startS"] + L.MISMATCH_S)
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
    frac = lambda v: round(float(np.mean(v)), 3) if v else None
    return {"wordCutDb": q(inner), "wordMidMinDb": q(mids), "mp3quranBoundaryInGap": frac(mp3q), "ayahPauses": len(A["inGap"]), "ayahDipInGap": frac(A["inGap"]),
            "ayahOffsetLagS": q(A["offset"]), "ayahOnsetLagS": q(A["onset"])}


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=60 * 60, max_containers=64)
def align(read: int, surah: int, meta: dict, force: bool = False) -> dict:
    dest = path(read, surah, "align.json")
    vol.reload()
    if dest.exists() and not force:
        out = json.loads(dest.read_text())
        if out.get("version") == ALIGN_VERSION:
            return summary(out)
    out = align_surah(read, surah, meta)
    write_atomic(dest, json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode())
    vol.commit()
    return summary(out)


def clean_runs(ay: list[dict]) -> list[float]:
    """Seconds of each maximal run of consecutive included ayahs, first aligned start to last aligned end."""
    runs, cur = [], []
    for a in ay + [{"excluded": True}]:
        if a["excluded"]:
            if cur:
                runs.append(round(cur[-1]["endS"] - cur[0]["startS"], 2))
            cur = []
        else:
            cur.append(a)
    return runs


def summary(out: dict) -> dict:
    ay = out["ayahs"]
    return {"readId": out["readId"], "surah": out["surah"], "split": out["split"], "reciter": out["reciter"], "durationS": round(out["durationS"], 1),
            "ayahs": len(ay), "words": len(out["words"]), "excluded": [{"ayah": a["ayah"], "reasons": a["reasons"]} for a in ay if a["excluded"]],
            "per": [a.get("per") for a in ay], "lenRatio": [a.get("lenRatio") for a in ay], "minWordScore": [a.get("minWordScore") for a in ay], "mismatchS": [a.get("mismatchS") for a in ay],
            "basmala": out["basmala"], "timingProblems": out["timingProblems"], "checks": out.get("checks"), "cleanRunsS": clean_runs(ay)}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=10 * 60)
def write_report(report: dict, catalog: dict) -> None:
    root = Path(VOL) / "data" / BASE
    write_atomic(root / "report-v1.json", json.dumps(report, ensure_ascii=False, indent=1).encode())
    write_atomic(root / "catalog-v1.json", json.dumps(catalog, ensure_ascii=False, indent=1).encode())
    vol.commit()


def choose_pool(reciters: list[dict]) -> dict:
    ref = L.reference()
    words = L.surah_words(ref)
    t = timed.remote([r["readId"] for r in reciters])
    pool = [{"readId": r["readId"], "nameEn": r["nameEn"], "split": r["split"], "server": r["server"],
             "surahs": L.pool(r["readId"], t["reads"][r["readId"]], words)} for r in sorted(reciters, key=lambda r: r["readId"])]
    return {"version": 1, "seed": L.SEED, "timedFetchedAt": t["fetchedAt"],
            "rule": f"required {list(L.REQUIRED)} when timed; then per (min words, max words, count) stratum {[list(s) for s in L.STRATA]} a seeded "
                    f"(seed:{L.SEED}:readId) shuffle taking the least-covered juz first; filled to {L.POOL} from timed surahs of >= {L.MIN_WORDS} words",
            "pool": pool}


def pctl(v: list[float], qs=(10, 50, 90, 99)) -> list[float]:
    import numpy as np

    return [round(float(x), 3) for x in np.percentile(v, qs)] if v else []


def report(results: list[dict], failures: list[str], pool: dict) -> dict:
    from collections import Counter

    per = [x for r in results for x in r["per"] if x is not None]
    mws = [x for r in results for x in r["minWordScore"] if x is not None]
    mis = [x for r in results for x in r["mismatchS"] if x is not None]
    reasons = Counter(rr.split(" ")[0] for r in results for e in r["excluded"] for rr in e["reasons"])
    # PER exclusions where the teacher heard at least 10% more than the text: a repeat or restart in the recording.
    per_long = sum(1 for r in results for p_, q_ in zip(r["per"], r["lenRatio"]) if p_ is not None and p_ > L.MAX_PER and q_ >= 1.1)
    by = {}
    for r in results:
        b = by.setdefault(r["readId"], {"readId": r["readId"], "reciter": r["reciter"], "split": r["split"], "surahs": 0, "hours": 0.0, "ayahs": 0,
                                        "excludedAyahs": 0, "words": 0})
        b["surahs"] += 1
        b["hours"] += r["durationS"] / 3600
        b["ayahs"] += r["ayahs"]
        b["excludedAyahs"] += len(r["excluded"])
        b["words"] += r["words"]
    for r in results:
        b = by[r["readId"]]
        b["cleanHours"] = b.get("cleanHours", 0.0) + sum(r["cleanRunsS"]) / 3600
        b["cleanRunsAtLeast3min"] = b.get("cleanRunsAtLeast3min", 0) + sum(180 <= x for x in r["cleanRunsS"])
        b["cleanRunsOver8min"] = b.get("cleanRunsOver8min", 0) + sum(480 < x for x in r["cleanRunsS"])
    for b in by.values():
        b["hours"], b["cleanHours"] = round(b["hours"], 2), round(b["cleanHours"], 2)
    inside = [r["checks"]["mp3quranBoundaryInGap"] for r in results if r["checks"] and r["checks"]["mp3quranBoundaryInGap"] is not None]
    ck = lambda k: pctl([r["checks"][k][1] for r in results if r["checks"] and r["checks"].get(k)], (10, 50, 90))
    pooled = lambda k, n: round(sum(r["checks"][k] * r["checks"][n] for r in results if r["checks"] and r["checks"][k] is not None)
                                / max(1, sum(r["checks"][n] for r in results if r["checks"] and r["checks"][k] is not None)), 4)
    mismatched = lambda r: sum(any(x.startswith("window-mismatch") for x in e["reasons"]) for e in r["excluded"])
    offset = [f"{r['readId']}/{r['surah']}" for r in results if not r["timingProblems"] and mismatched(r) >= r["ayahs"] / 2]
    return {
        "version": 1, "alignVersion": ALIGN_VERSION, "teacher": MODEL, "generatedAt": now(),
        "timingProblemSurahs": {f"{r['readId']}/{r['surah']}": r["timingProblems"] for r in results if r["timingProblems"]},
        # Half or more of the ayah windows off by over MISMATCH_S: the timing was made for another edit of the audio.
        "timingOffsetSurahs": offset,
        "thresholds": {"padS": L.PAD_S, "mismatchS": L.MISMATCH_S, "maxPer": L.MAX_PER, "minWordScore": L.MIN_WORD_SCORE},
        "counts": {"reciters": len(by), "surahs": len(results), "hours": round(sum(r["durationS"] for r in results) / 3600, 2),
                   "ayahs": sum(r["ayahs"] for r in results), "excludedAyahs": sum(len(r["excluded"]) for r in results),
                   "words": sum(r["words"] for r in results), "failedItems": len(failures),
                   "cleanHours": round(sum(sum(r["cleanRunsS"]) for r in results) / 3600, 2),
                   "cleanRunsAtLeast3min": sum(180 <= x for r in results for x in r["cleanRunsS"]),
                   "timingProblemSurahs": sum(bool(r["timingProblems"]) for r in results),
                   "basmalaFound": sum(bool(r["basmala"] and r["basmala"]["present"]) for r in results)},
        "excludedReasons": dict(reasons), "perExcludedLonger": per_long,
        "distributions": {"ayahPer": pctl(per), "ayahMinWordScore": pctl(mws, (1, 10, 50, 90)), "ayahMismatchS": pctl(mis),
                          "surahMedians": {k: ck(k) for k in ("wordCutDb", "wordMidMinDb", "ayahOffsetLagS", "ayahOnsetLagS")},
                          "ayahDipInGap": pooled("ayahDipInGap", "ayahPauses"),
                          "mp3quranBoundaryInGapMeanOfSurahs": round(sum(inside) / len(inside), 4) if inside else None},
        "reciters": sorted(by.values(), key=lambda b: (b["split"], b["readId"])),
        "surahs": [{k: r[k] for k in ("readId", "surah", "split", "durationS", "ayahs", "words", "excluded", "basmala", "timingProblems", "checks", "cleanRunsS")}
                   for r in sorted(results, key=lambda r: (r["readId"], r["surah"]))],
        "failures": failures,
    }


@app.local_entrypoint()
def main(stage: str = "all", pilot: bool = False, repool: bool = False, force_align: bool = False, limit: int = 0):
    import time

    t0 = time.time()
    reciters = json.loads(RECITERS.read_text())["heldOut"]
    if repool or not POOL_FILE.exists():
        POOL_FILE.write_text(json.dumps(choose_pool(reciters), ensure_ascii=False, indent=1) + "\n")
    pool = json.loads(POOL_FILE.read_text())
    meta = {r["readId"]: r for r in pool["pool"]}
    items = [(r["readId"], s) for r in pool["pool"] for s in r["surahs"]]
    if pilot:
        ex = next(r for r in pool["pool"] if r["split"] == "exam")
        pr = next(r for r in pool["pool"] if r["split"] == "practice")
        host = lambda r: next(x for x in r["surahs"] if x not in L.REQUIRED and x >= 67)
        items = [(r["readId"], s) for r in (ex, pr) for s in (55, 77, host(r))]
    if limit:
        items = items[:limit]
    print(f"{len(items)} items")
    failures = []

    def run(name, it):
        for k, res in enumerate(it):
            args = stage_args[k]
            if isinstance(res, Exception):
                failures.append(f"{name} {args[0]}/{args[-1] if name != 'align' else args[1]}: {res!r}"[:400])
                print("FAIL", failures[-1])
            else:
                yield res

    if stage in ("all", "fetch"):
        stage_args = [(r, meta[r]["server"], s) for r, s in items]
        for res in run("fetch", fetch.starmap(stage_args, return_exceptions=True, wrap_returned_exceptions=False)):
            print(res)
    if stage in ("all", "decode"):
        stage_args = items
        for res in run("decode", Teacher().decode.starmap(stage_args, return_exceptions=True, wrap_returned_exceptions=False)):
            print(res)
    results = []
    if stage in ("all", "align", "report"):
        stage_args = [(r, s, meta[r], force_align) for r, s in items]
        for res in run("align", align.starmap(stage_args, return_exceptions=True, wrap_returned_exceptions=False)):
            results.append(res)
            print(f"{res['readId']}/{res['surah']}: {res['ayahs']} ayahs, {len(res['excluded'])} excluded, checks {res['checks']}")
            for e in res["excluded"]:
                print("   ", e)
    if results and pilot:
        rep = report(results, failures, pool)
        print(json.dumps({k: rep[k] for k in ("counts", "excludedReasons", "distributions")}, indent=1))
    if results and not pilot:
        rep = report(results, failures, pool)
        catalog = {"version": 1, "alignVersion": ALIGN_VERSION, "root": f"data/{BASE}",
                   "items": [{"readId": r["readId"], "surah": r["surah"], "split": r["split"], "reciter": r["reciter"], "durationS": r["durationS"],
                              "audio": f"{BASE}/{r['readId']}/{r['surah']:03d}.wav", "align": f"{BASE}/{r['readId']}/{r['surah']:03d}.align.json",
                              "evidence": f"{BASE}/{r['readId']}/{r['surah']:03d}.evidence.npz", "ayahs": r["ayahs"],
                              "excludedAyahs": [e["ayah"] for e in r["excluded"]], "cleanRunsS": r["cleanRunsS"],
                              "basmala": r["basmala"]} for r in sorted(results, key=lambda r: (r["readId"], r["surah"]))]}
        write_report.remote(rep, catalog)
        pool["result"] = {k: rep[k] for k in ("alignVersion", "generatedAt", "thresholds", "counts", "excludedReasons", "perExcludedLonger", "distributions", "reciters",
                                              "timingProblemSurahs", "timingOffsetSurahs", "failures")}
        POOL_FILE.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(rep["counts"]), json.dumps(rep["excludedReasons"]))
    print(f"done in {time.time() - t0:.0f} s; {len(failures)} failures")
