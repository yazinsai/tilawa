"""zipformer-ctc -- native MIT recitation engine + Zipformer2-CTC, benchmark wrapper.

Pipeline (packages/core/src/recitation + ZipformerHost):
  16 kHz PCM -> Kaldi fbank (80 mel) -> streaming Zipformer2-CTC ONNX (251
  tajweed-phoneme tokens; default is shipped interp-gentle-a0.5 int8)
  -> greedy CTC -> whole-Quran 5-gram search + per-surah online DP tracker
  -> per-word verdicts.

This file only: decodes audio with shared.audio, ships raw float32 to a
long-lived `tsx harness.ts` over stdin/stdout, and turns the harness's
per-ayah verdict tallies into {surah, ayah, ayah_end}.

Registered as `zipformer-ctc`.

Model + corpus are fetched on first use into data/zipformer/ from GitHub
release yazinsai/tilawa v0.3.0 (env override: ZIPFORMER_DATA_DIR) unless
ZIPFORMER_MODEL already points at an existing file. I/O manifest is
experiments/zipformer-ctc/zipformer-io.json.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # lab/
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ZIPFORMER_DATA_DIR", PROJECT_ROOT / "data" / "zipformer"))
MODEL_NAME = "zipformer_interp_gentle_a05.int8.onnx"
CORPUS_NAME = "zipformer_quran.json"
MODEL_PATH = DATA_DIR / MODEL_NAME
CORPUS_PATH = DATA_DIR / CORPUS_NAME
ORT_DIR = Path(
    os.environ.get("ZIPFORMER_ORT_DIR", REPO_ROOT / "web" / "frontend" / "node_modules")
)

RELEASE = "https://github.com/yazinsai/tilawa/releases/download/v0.3.0"
MODEL_URL = f"{RELEASE}/{MODEL_NAME}"
CORPUS_URL = f"{RELEASE}/{CORPUS_NAME}"

_proc: subprocess.Popen | None = None
_req_id = 0
_model_sha_cache: dict[str, str] = {}


def resolved_model_path() -> Path:
    override = os.environ.get("ZIPFORMER_MODEL")
    if override:
        return Path(override)
    return MODEL_PATH


def benchmark_name(base: str = "zipformer-ctc") -> str:
    """Result JSON `name`; suffix only when ZIPFORMER_MODEL is set."""
    override = os.environ.get("ZIPFORMER_MODEL")
    if override:
        return f"{base}[{Path(override).name}]"
    return base


def model_sha256_prefix(path: Path | None = None) -> str:
    """First 8 hex chars of the ONNX file sha256; cached per resolved path."""
    p = path if path is not None else resolved_model_path()
    key = str(p.resolve()) if p.is_file() else str(p)
    cached = _model_sha_cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    prefix = h.hexdigest()[:8]
    _model_sha_cache[key] = prefix
    return prefix


def _provenance(path: Path | None = None) -> dict:
    p = path if path is not None else resolved_model_path()
    out = {"model": p.name, "model_sha256_prefix": ""}
    try:
        out["model_sha256_prefix"] = model_sha256_prefix(p)
    except OSError:
        pass
    return out


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[zipformer-ctc] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _ensure_assets() -> None:
    override = os.environ.get("ZIPFORMER_MODEL")
    if override:
        model = Path(override)
        if not model.is_file():
            raise FileNotFoundError(
                f"ZIPFORMER_MODEL={override!r} is not an existing file"
            )
    elif not MODEL_PATH.exists():
        _fetch(MODEL_URL, MODEL_PATH)
    if not CORPUS_PATH.exists():
        _fetch(CORPUS_URL, CORPUS_PATH)
    if not (ORT_DIR / "onnxruntime-node").exists():
        raise FileNotFoundError(
            f"onnxruntime-node not found under {ORT_DIR}; run `npm install` in web/frontend "
            "or set ZIPFORMER_ORT_DIR"
        )


def _ensure_proc() -> subprocess.Popen:
    global _proc, _HARNESS_GAP_MAX_WORDS
    if _proc is not None and _proc.poll() is None:
        return _proc
    _ensure_assets()
    env = dict(os.environ)
    env.setdefault("ZIPFORMER_MODEL", str(MODEL_PATH))
    env.setdefault("ZIPFORMER_CORPUS", str(CORPUS_PATH))
    env.setdefault("ZIPFORMER_ORT_DIR", str(ORT_DIR))
    env.setdefault("ZIPFORMER_GAP_MAX_WORDS", str(_gap_max_words()))
    tsx = Path(env["ZIPFORMER_ORT_DIR"]) / ".bin" / "tsx"
    if not tsx.is_file():
        raise FileNotFoundError(
            f"tsx not found at {tsx}; set ZIPFORMER_ORT_DIR to web/frontend/node_modules"
        )
    # `@tilawa/core` resolves through web/frontend's tsconfig `paths`; tsx picks
    # its tsconfig from cwd, which is lab/ here, so point it at the frontend's.
    frontend_tsconfig = REPO_ROOT / "web" / "frontend" / "tsconfig.json"
    tsx_argv = [str(tsx)]
    if frontend_tsconfig.is_file():
        tsx_argv += ["--tsconfig", str(frontend_tsconfig)]
    _proc = subprocess.Popen(
        [*tsx_argv, str(HERE / "harness.ts")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        text=True,
        bufsize=1,
    )
    ready = _proc.stdout.readline()
    parsed = json.loads(ready) if ready else {}
    if not parsed.get("ready"):
        raise RuntimeError(f"harness failed to start: {ready!r}")
    if "gapMaxWords" in parsed:
        _HARNESS_GAP_MAX_WORDS = int(parsed["gapMaxWords"])
    atexit.register(_shutdown)
    return _proc


def _shutdown() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None


def recognize(audio_path: str, mode: str = "tracking") -> dict:
    """Raw harness result: verses (accepted ayahs), all tallies, transcript, events.

    ``mode="correction"`` also returns ``corrections`` (issues, dismissed on sight)."""
    global _req_id
    proc = _ensure_proc()
    audio = load_audio(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        audio.astype("float32").tofile(f)
        pcm_path = f.name
    try:
        _req_id += 1
        proc.stdin.write(json.dumps({"id": _req_id, "pcm": pcm_path, "mode": mode}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    finally:
        os.unlink(pcm_path)
    if not line:
        _shutdown()
        raise RuntimeError("harness died")
    res = json.loads(line)
    if "error" in res:
        raise RuntimeError(res["error"])
    return res


GAP_MAX_WORDS = 3
_HARNESS_GAP_MAX_WORDS: int | None = None
_ayah_words_cache: dict[tuple[int, int], int] | None = None


def _allow_gaps() -> bool:
    return os.environ.get("ZIPFORMER_ALLOW_GAPS") == "1"


def _gap_max_words() -> int:
    raw = os.environ.get("ZIPFORMER_GAP_MAX_WORDS")
    if raw not in (None, ""):
        return int(raw)
    if _HARNESS_GAP_MAX_WORDS is not None:
        return _HARNESS_GAP_MAX_WORDS
    return GAP_MAX_WORDS


def _ayah_word_count(surah: int, ayah: int) -> int:
    """Word count from the zipformer corpus; 99 (do not bridge) if unknown."""
    global _ayah_words_cache
    if _ayah_words_cache is None:
        _ayah_words_cache = {}
        path = Path(os.environ.get("ZIPFORMER_CORPUS", CORPUS_PATH))
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for s in data.get("surahs", []):
                n = int(s["n"])
                for i, a in enumerate(s.get("ayahs", [])):
                    _ayah_words_cache[(n, i + 1)] = len(a.get("w", []))
    return _ayah_words_cache.get((surah, ayah), 99)


def _bridge_extras(
    accepted: list[dict],
    tallies: list[dict],
    *,
    gap_max_words: int | None = None,
) -> list[dict]:
    """Inject a below-threshold short ayah only when both neighbours already emit.

    Prefix fill is forbidden: a tally for ayah 3 with accepted [4, 5] stays out.
    Requires ok+unsure >= 1.
    """
    limit = _gap_max_words() if gap_max_words is None else gap_max_words
    have = {(t["surah"], t["ayah"]) for t in accepted}
    extra = []
    for t in tallies:
        key = (t["surah"], t["ayah"])
        if key in have:
            continue
        if t.get("words", 99) > limit:
            continue
        if t.get("ok", 0) + t.get("unsure", 0) < 1:
            continue
        if t.get("wrong", 0) > t.get("ok", 0) + t.get("unsure", 0):
            continue
        if (t["surah"], t["ayah"] - 1) not in have:
            continue
        if (t["surah"], t["ayah"] + 1) not in have:
            continue
        extra.append({**t, "bridged": True})
        have.add(key)
    if not extra:
        return accepted
    out = list(accepted) + extra
    out.sort(key=lambda t: t.get("firstSeen", 0))
    return out


def _contiguous_head(
    verses: list[dict],
    *,
    allow_gaps: bool | None = None,
    word_count=None,
) -> tuple[int, int, int | None]:
    """First verse plus the longest run of consecutive ayahs in the same surah.

    When allow_gaps (ZIPFORMER_ALLOW_GAPS=1), skip a single missing ayah of
    ≤ gap-max words *only if that ayah is already in `verses`* (harness-bridged).
    Does not invent a hole from the corpus, and does not walk backward: [4, 5]
    stays start=4.
    """
    if allow_gaps is None:
        allow_gaps = _allow_gaps()
    count_fn = word_count or _ayah_word_count
    present = {(v["surah"], v["ayah"]) for v in verses}
    first = verses[0]
    surah, ayah = first["surah"], first["ayah"]
    end = ayah
    for v in verses[1:]:
        if v["surah"] != surah:
            break
        if v["ayah"] == end + 1:
            end = v["ayah"]
            continue
        if allow_gaps and v["ayah"] == end + 2:
            skipped = end + 1
            if (surah, skipped) in present and count_fn(surah, skipped) <= _gap_max_words():
                end = v["ayah"]
                continue
        break
    return surah, ayah, (end if end != ayah else None)


def predict(audio_path: str) -> dict:
    res = recognize(audio_path)
    provenance = _provenance()
    verses = res["verses"]
    if _allow_gaps():
        accepted = [v for v in verses if not v.get("bridged")]
        tallies = res.get("all") or verses
        verses = _bridge_extras(accepted, tallies)
    if not verses:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": res["transcript"],
            **provenance,
        }
    surah, ayah, ayah_end = _contiguous_head(verses)
    ok = sum(v["ok"] for v in verses)
    words = sum(v["words"] for v in verses)
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": ayah_end,
        "score": ok / max(1, words),
        "transcript": res["transcript"],
        "verses": [(v["surah"], v["ayah"]) for v in verses],
        **provenance,
    }


def transcribe(audio_path: str) -> str:
    return recognize(audio_path)["transcript"]


def model_size() -> int:
    p = resolved_model_path()
    try:
        return p.stat().st_size
    except OSError:
        return 69_245_985


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = recognize(p)
        print(
            Path(p).name,
            f"{r['decodeMs']}ms",
            r["state"],
            [(v["surah"], v["ayah"], v["ok"], v["unsure"], v["words"]) for v in r["verses"]],
            [e["type"] for e in r["events"]],
        )
        print("   ", r["transcript"])
