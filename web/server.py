"""
WebSocket backend for real-time Quran verse recognition.

Uses NVIDIA FastConformer (NeMo) for Arabic ASR with a rolling
audio window. Matches transcripts against QuranDB and sends
verse_match or raw_transcript messages to the frontend.
"""

import asyncio
import logging
import os
import sys
import tempfile
import types
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Levenshtein import ratio as lev_ratio

from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB, partial_ratio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 2.0
TRIGGER_SAMPLES = int(SAMPLE_RATE * TRIGGER_SECONDS)
MAX_WINDOW_SECONDS = 10.0
MAX_WINDOW_SAMPLES = int(SAMPLE_RATE * MAX_WINDOW_SECONDS)
SILENCE_RMS_THRESHOLD = 0.005
PORT = 8000

# Matching thresholds
VERSE_MATCH_THRESHOLD = 0.45
FIRST_MATCH_THRESHOLD = 0.75  # higher bar before any verse is locked on
RAW_TRANSCRIPT_THRESHOLD = 0.25
SURROUNDING_CONTEXT = 2  # verses before/after current
CONTINUATION_STRONG_THRESHOLD = 0.65
LONG_VERSE_CONTINUATION_THRESHOLD = 0.80
AMBIGUOUS_MATCH_GAP = 0.05
LEXICAL_RERANK_MIN_WORDS = 5
LEXICAL_RERANK_SCORE_GAP = 0.15
LEXICAL_RERANK_SWITCH_MARGIN = 0.03
WORD_MATCH_THRESHOLD = 0.72
LONG_TRACKING_WORD_THRESHOLD = 20
LONG_TRACKING_MIN_MATCHED_WORDS = 3

# Tracking mode (word-level): faster cycle once a verse is locked on
TRACKING_TRIGGER_SECONDS = 0.5
TRACKING_TRIGGER_SAMPLES = int(SAMPLE_RATE * TRACKING_TRIGGER_SECONDS)
TRACKING_SILENCE_TIMEOUT = 4.0  # seconds of silence before exiting tracking
TRACKING_SILENCE_SAMPLES = int(SAMPLE_RATE * TRACKING_SILENCE_TIMEOUT)
TRACKING_MAX_WINDOW_SECONDS = 5.0  # shorter window to flush old-audio residue
TRACKING_MAX_WINDOW_SAMPLES = int(SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS)

# Model config
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
DECODER_TYPE = os.getenv("NVIDIA_FASTCONFORMER_DECODER", "ctc")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tarteel-ws")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
quran_db: QuranDB | None = None
_nemo_model = None


# ---------------------------------------------------------------------------
# FastConformer model loading
# ---------------------------------------------------------------------------
def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback when package is absent.

    NeMo imports kaldialign in context-biasing utilities, even for inference
    flows that do not use those codepaths. This fallback unblocks model import.
    """
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _load_fastconformer():
    global _nemo_model
    if _nemo_model is not None:
        return

    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from nemo.utils import logging as nemo_logging
    except ImportError as exc:
        raise ImportError(
            "NeMo ASR dependencies required. Install with: "
            "pip install 'nemo_toolkit[asr]'"
        ) from exc

    nemo_logging.set_verbosity(nemo_logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    log.info("Loading FastConformer from %s on %s...", source, device)

    try:
        _nemo_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=source,
            map_location=device,
        )
    except Exception:
        if LOCAL_MODEL_DIR.exists():
            nemo_files = sorted(LOCAL_MODEL_DIR.glob("*.nemo"))
            if not nemo_files:
                raise
            _nemo_model = EncDecHybridRNNTCTCBPEModel.restore_from(
                str(nemo_files[0]),
                map_location=device,
            )
        else:
            raise

    _nemo_model.eval()
    try:
        _nemo_model.change_decoding_strategy(decoder_type=DECODER_TYPE)
    except Exception:
        pass

    log.info("FastConformer loaded successfully")


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
def _transcribe(audio: np.ndarray) -> str:
    """Transcribe audio array using FastConformer."""
    _load_fastconformer()

    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), audio, SAMPLE_RATE)

        try:
            outputs = _nemo_model.transcribe(
                audio=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
                verbose=False,
            )
        except TypeError:
            outputs = _nemo_model.transcribe(
                paths2audio_files=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
            )

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list) and outputs:
            transcript = _extract_text(outputs[0])
        else:
            transcript = _extract_text(outputs)

        return normalize_arabic(transcript)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _is_silence(audio: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms < SILENCE_RMS_THRESHOLD


def _get_surrounding_verses(db: QuranDB, surah: int, ayah: int) -> list[dict]:
    """Get surrounding verses for context display."""
    verses = db.get_surah(surah)
    result = []
    for v in verses:
        if abs(v["ayah"] - ayah) <= SURROUNDING_CONTEXT:
            result.append(
                {
                    "surah": v["surah"],
                    "ayah": v["ayah"],
                    "text": v["text_uthmani"],
                    "is_current": v["ayah"] == ayah,
                }
            )
    return result


def _match_key(match: dict) -> str:
    ayah_end = match.get("ayah_end")
    if ayah_end and ayah_end != match["ayah"]:
        return f"{match['surah']}:{match['ayah']}-{ayah_end}"
    return f"{match['surah']}:{match['ayah']}"


def _runner_up_gap(match: dict) -> float:
    current_key = _match_key(match)
    for runner in match.get("runners_up", []):
        runner_key = f"{runner['surah']}:{runner['ayah']}"
        if runner_key != current_key:
            return match["score"] - runner["score"]
    return 1.0


def _is_expected_followup(
    db: QuranDB,
    last_ref: tuple[int, int] | None,
    match: dict | None,
    lookahead: int = 3,
) -> bool:
    if not last_ref or not match:
        return False

    next_ref = last_ref
    for _ in range(lookahead):
        next_verse = db.get_next_verse(*next_ref)
        if not next_verse:
            return False
        candidate = (next_verse["surah"], next_verse["ayah"])
        if candidate == (match["surah"], match["ayah"]):
            return True
        next_ref = candidate
    return False


def _is_long_verse_ref(db: QuranDB, ref: tuple[int, int] | None) -> bool:
    if not ref:
        return False
    verse = db.get_verse(*ref)
    if not verse:
        return False
    return len(verse["text_clean"].split()) >= LONG_TRACKING_WORD_THRESHOLD


def _candidate_clean_text(
    db: QuranDB,
    candidate: dict,
    fallback_text: str | None = None,
) -> str:
    if fallback_text:
        return normalize_arabic(fallback_text)

    ayah_start = candidate["ayah"]
    ayah_end = candidate.get("ayah_end") or ayah_start
    parts: list[str] = []
    for ayah in range(ayah_start, ayah_end + 1):
        verse = db.get_verse(candidate["surah"], ayah)
        if not verse:
            break
        if ayah == ayah_start:
            parts.append(verse.get("text_clean_no_bsm") or verse["text_clean"])
        else:
            parts.append(verse["text_clean"])
    return normalize_arabic(" ".join(parts).strip())


def _candidate_display_text(db: QuranDB, candidate: dict) -> str:
    ayah_start = candidate["ayah"]
    ayah_end = candidate.get("ayah_end") or ayah_start
    parts: list[str] = []
    for ayah in range(ayah_start, ayah_end + 1):
        verse = db.get_verse(candidate["surah"], ayah)
        if not verse:
            break
        parts.append(verse["text_uthmani"])
    return " ".join(parts).strip()


def _words_similar(w1: str, w2: str) -> bool:
    return lev_ratio(w1, w2) >= WORD_MATCH_THRESHOLD


def _lexical_candidate_score(query_text: str, candidate_text: str) -> float:
    query_words = normalize_arabic(query_text).split()
    candidate_words = normalize_arabic(candidate_text).split()
    if not query_words or not candidate_words:
        return 0.0

    matched = 0
    candidate_pos = 0
    for query_word in query_words:
        found = False
        for idx in range(candidate_pos, len(candidate_words)):
            if _words_similar(query_word, candidate_words[idx]):
                matched += 1
                candidate_pos = idx + 1
                found = True
                break
        if not found:
            continue

    prefix_matches = 0
    for query_word, candidate_word in zip(query_words, candidate_words):
        if _words_similar(query_word, candidate_word):
            prefix_matches += 1
        else:
            break

    coverage = matched / max(min(len(query_words), len(candidate_words)), 1)
    prefix_score = prefix_matches / max(min(3, len(query_words), len(candidate_words)), 1)
    length_fit = min(len(query_words), len(candidate_words)) / max(len(query_words), len(candidate_words))
    leftover_penalty = max(len(query_words) - matched, 0) / max(len(query_words), 1)

    lexical = (
        0.60 * coverage
        + 0.25 * prefix_score
        + 0.15 * length_fit
        - 0.15 * leftover_penalty
    )
    return max(0.0, min(1.0, lexical))


def _rerank_ambiguous_match(
    db: QuranDB,
    query_text: str,
    match: dict | None,
) -> dict | None:
    if not match or len(query_text.split()) < LEXICAL_RERANK_MIN_WORDS:
        return match

    runners_up = match.get("runners_up", [])
    if not runners_up:
        return match

    current_key = _match_key(match)
    candidates = [dict(match)]
    for runner in runners_up:
        if match["score"] - runner["score"] > LEXICAL_RERANK_SCORE_GAP:
            continue
        candidates.append(
            {
                "surah": runner["surah"],
                "ayah": runner["ayah"],
                "ayah_end": runner.get("ayah_end"),
                "score": runner["score"],
                "raw_score": runner["raw_score"],
                "bonus": runner["bonus"],
            }
        )

    if len(candidates) < 2:
        return match

    ranked: list[tuple[float, float, dict]] = []
    for candidate in candidates:
        fallback_text = match.get("text_clean") if _match_key(candidate) == current_key else None
        candidate_text = _candidate_clean_text(db, candidate, fallback_text)
        lexical_score = _lexical_candidate_score(query_text, candidate_text)
        combined = 0.55 * candidate["score"] + 0.45 * lexical_score
        ranked.append((combined, lexical_score, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    current_combined = next(
        combined for combined, _lexical, candidate in ranked
        if _match_key(candidate) == current_key
    )
    best_combined, _best_lexical, best_candidate = ranked[0]
    if _match_key(best_candidate) == current_key:
        return match
    if best_combined < current_combined + LEXICAL_RERANK_SWITCH_MARGIN:
        return match

    reranked = dict(best_candidate)
    reranked["text_clean"] = _candidate_clean_text(db, best_candidate)
    reranked["text"] = _candidate_display_text(db, best_candidate)
    reranked["runners_up"] = runners_up
    log.info(
        "  (lexical rerank %s -> %s, combined %.3f -> %.3f)",
        current_key,
        _match_key(best_candidate),
        current_combined,
        best_combined,
    )
    return reranked


# ---------------------------------------------------------------------------
# Word-level alignment (for tracking mode)
# ---------------------------------------------------------------------------
def _words_match(w1: str, w2: str, threshold: float = 0.7) -> bool:
    """Check if two Arabic words match, tolerating ASR errors."""
    if w1 == w2:
        return True
    if len(w1) <= 2 or len(w2) <= 2:
        return w1 == w2
    return lev_ratio(w1, w2) >= threshold


def _align_position(
    recognized_words: list[str],
    verse_words: list[str],
    start_from: int = 0,
) -> tuple[int, list[int]]:
    """Find how far into the verse the recognized words reach.

    Uses greedy forward alignment: scans recognized words left-to-right
    and matches each to the earliest available verse word from the
    current position forward. This prevents jumping to later occurrences
    of repeated words (e.g. "من تشاء" × 4 in 3:26).

    The start_from parameter allows resuming alignment from the last
    known position, so the rolling audio window (which may not contain
    the verse beginning) still works for long verses.

    Returns (position, matched_indices) where position is the furthest
    verse word index reached + 1, and matched_indices lists which verse
    word indices were matched.
    """
    if not recognized_words or not verse_words:
        return 0, []

    LOOKAHEAD = 5  # max words to skip in verse (handles ASR deletions)

    matched_indices = []
    verse_ptr = start_from

    for rec in recognized_words:
        if verse_ptr >= len(verse_words):
            break
        # Search forward from current position with limited lookahead
        limit = min(verse_ptr + LOOKAHEAD, len(verse_words))
        for j in range(verse_ptr, limit):
            if _words_match(rec, verse_words[j]):
                matched_indices.append(j)
                verse_ptr = j + 1
                break

    if matched_indices:
        return matched_indices[-1] + 1, matched_indices
    return start_from, []


def _strip_leading_residual(
    text: str,
    previous_text: str,
    min_score: float = 0.82,
) -> tuple[str, float]:
    """Remove a leading copy of the previously emitted verse if present.

    Streaming windows often contain the full previous verse plus the start of
    the next one. Skipping the whole transcript in that case drops recall on
    multi-ayah recitations; trimming the residual prefix preserves the new tail.
    """
    text_words = text.split()
    prev_words = previous_text.split()
    if len(text_words) < 2 or len(prev_words) < 2:
        return "", 0.0

    best_score = 0.0
    best_k = 0
    min_k = max(1, len(prev_words) - 1)
    max_k = min(len(text_words) - 1, len(prev_words) + 1)
    for k in range(min_k, max_k + 1):
        prefix = " ".join(text_words[:k])
        score = lev_ratio(prefix, previous_text)
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < min_score:
        return "", best_score

    remainder = " ".join(text_words[best_k:]).strip()
    return remainder, best_score


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    global quran_db
    quran_db = QuranDB()
    application.state.transcribe_lock = asyncio.Lock()
    log.info("QuranDB loaded: %d verses", quran_db.total_verses)
    _load_fastconformer()
    yield


app = FastAPI(
    title="Offline Tarteel",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected: %s", ws.client)

    full_audio = np.empty(0, dtype=np.float32)
    new_audio_count = 0
    last_emitted_ref: tuple[int, int] | None = None
    last_emitted_text: str = ""
    prev_emitted_ref: tuple[int, int] | None = None
    prev_emitted_text: str = ""
    pending_match_key: str | None = None
    pending_match_count = 0

    # Tracking mode state
    tracking_verse: dict | None = None  # the verse we're tracking within
    tracking_verse_words: list[str] = []  # normalized words of tracked verse
    tracking_last_word_idx = -1  # last word_index we sent
    silence_samples = 0  # consecutive silence samples in tracking mode
    stale_cycles = 0  # consecutive tracking cycles with no progress
    STALE_CYCLE_LIMIT = 4  # exit tracking after this many no-progress cycles

    def _enter_tracking(verse: dict, ref: tuple[int, int]) -> None:
        nonlocal tracking_verse, tracking_verse_words, tracking_last_word_idx
        nonlocal silence_samples, stale_cycles
        tracking_verse = verse
        tracking_verse_words = verse["text_clean"].split()
        tracking_last_word_idx = -1
        silence_samples = 0
        stale_cycles = 0
        log.info(
            "TRACKING enter %s:%s (%d words)",
            ref[0], ref[1], len(tracking_verse_words),
        )

    def _exit_tracking(reason: str) -> None:
        nonlocal tracking_verse, tracking_verse_words, tracking_last_word_idx
        nonlocal silence_samples, stale_cycles
        nonlocal last_emitted_ref, last_emitted_text
        nonlocal prev_emitted_ref, prev_emitted_text
        log.info("TRACKING exit: %s", reason)

        verse_len = len(tracking_verse_words)
        progress = (tracking_last_word_idx + 1) / verse_len if verse_len > 0 else 0
        long_tracking = (
            reason.startswith("stale")
            and verse_len >= LONG_TRACKING_WORD_THRESHOLD
        )
        long_partial_tracking = (
            long_tracking
            and tracking_last_word_idx + 1 >= LONG_TRACKING_MIN_MATCHED_WORDS
        )

        if reason == "verse complete":
            pass  # caller already updated last_emitted_ref/text
        elif reason.startswith("stale") and progress < 0.5 and not long_tracking:
            # Low progress + stale = likely misidentification (e.g. two
            # verses share a prefix but diverge). Revert to pre-tracking
            # state so discovery isn't blocked by the residual overlap check.
            log.info("  (misidentification detected, progress=%.0f%%, reverting state)", progress * 100)
            last_emitted_ref = prev_emitted_ref
            last_emitted_text = prev_emitted_text
        elif reason.startswith("stale") and long_tracking and tracking_last_word_idx < 0:
            log.info("  (long-verse lock retained after stale cycle)")
        elif reason.startswith("stale") and tracking_verse_words and tracking_last_word_idx >= 0:
            # Good progress + stale = was tracking correctly but user
            # paused or diverged. Trim residual text to tracked portion.
            tracked_portion = " ".join(tracking_verse_words[:tracking_last_word_idx + 1])
            last_emitted_text = tracked_portion
            if long_partial_tracking:
                log.info(
                    "  (long-verse partial progress %.0f%%, keeping state with %d words)",
                    progress * 100,
                    tracking_last_word_idx + 1,
                )
            else:
                log.info(
                    "  (updated residual text to tracked portion: %d words)",
                    tracking_last_word_idx + 1,
                )

        tracking_verse = None
        tracking_verse_words = []
        tracking_last_word_idx = -1
        silence_samples = 0
        stale_cycles = 0

    try:
        while True:
            data = await ws.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)
            full_audio = np.concatenate([full_audio, samples])
            new_audio_count += len(samples)

            # Trim to max window (shorter in tracking mode to flush
            # old-audio residue that causes false matches in verses
            # with repeated phrases like 3:26 "من تشاء" × 4)
            max_samples = (
                TRACKING_MAX_WINDOW_SAMPLES
                if tracking_verse is not None
                else MAX_WINDOW_SAMPLES
            )
            if len(full_audio) > max_samples:
                full_audio = full_audio[-max_samples:]

            # ---------------------------------------------------------------
            # TRACKING MODE: fast cycle for word-level progress
            # ---------------------------------------------------------------
            if tracking_verse is not None:
                # Check silence accumulation
                chunk_rms = float(np.sqrt(np.mean(samples**2)))
                if chunk_rms < SILENCE_RMS_THRESHOLD:
                    silence_samples += len(samples)
                    if silence_samples >= TRACKING_SILENCE_SAMPLES:
                        _exit_tracking("extended silence")
                        new_audio_count = 0
                        continue
                else:
                    silence_samples = 0

                # Faster trigger in tracking mode
                if new_audio_count < TRACKING_TRIGGER_SAMPLES:
                    continue
                new_audio_count = 0

                # Transcribe
                async with ws.app.state.transcribe_lock:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, _transcribe, full_audio.copy()
                    )

                if not text or len(text.strip()) < 3:
                    continue

                recognized_words = text.split()

                # Align against known verse, starting from last
                # known position (handles rolling window that may
                # not contain the verse beginning)
                resume_from = max(tracking_last_word_idx, 0)
                word_pos, matched_indices = _align_position(
                    recognized_words, tracking_verse_words,
                    start_from=resume_from,
                )

                # Check for stale tracking (no progress)
                advanced = (
                    matched_indices
                    and matched_indices[-1] > tracking_last_word_idx
                )
                if not advanced:
                    stale_cycles += 1
                    if stale_cycles >= STALE_CYCLE_LIMIT:
                        _exit_tracking(
                            f"stale ({stale_cycles} cycles, no progress)"
                        )
                        new_audio_count = 0
                        continue
                else:
                    stale_cycles = 0

                # Send word_progress if we've advanced
                if advanced:
                    tracking_last_word_idx = matched_indices[-1]
                    await ws.send_json(
                        {
                            "type": "word_progress",
                            "surah": tracking_verse["surah"],
                            "ayah": tracking_verse["ayah"],
                            "word_index": word_pos,
                            "total_words": len(tracking_verse_words),
                            "matched_indices": matched_indices,
                        }
                    )
                    log.info(
                        "TRACKING %s:%s word %d/%d  indices=%s",
                        tracking_verse["surah"],
                        tracking_verse["ayah"],
                        word_pos,
                        len(tracking_verse_words),
                        matched_indices[-5:],  # last few for brevity
                    )

                # Check if verse is complete (always check, not just
                # when advancing — the verse may already be complete
                # on the first tracking cycle)
                if matched_indices:
                    coverage = len(matched_indices) / len(tracking_verse_words)
                    near_end = matched_indices[-1] >= len(tracking_verse_words) - 2
                    if coverage >= 0.8 and near_end:
                        log.info(
                            "TRACKING verse complete %s:%s (coverage=%.0f%%)",
                            tracking_verse["surah"],
                            tracking_verse["ayah"],
                            coverage * 100,
                        )
                        # Mark the current verse complete, then return to
                        # discovery mode. Blindly auto-advancing to the next
                        # verse produces extra false verse_match events at
                        # recitation boundaries and on single-ayah samples.
                        cur_ref = (
                            tracking_verse["surah"],
                            tracking_verse["ayah"],
                        )
                        last_emitted_ref = cur_ref
                        last_emitted_text = normalize_arabic(
                            tracking_verse["text_clean"]
                        )
                        _exit_tracking("verse complete")

                        # Reset audio window — keep more context (last 2s)
                        # so next verse tracking has something to work with
                        keep_samples = min(
                            len(full_audio), TRIGGER_SAMPLES
                        )
                        full_audio = full_audio[-keep_samples:].copy()

                continue  # stay in tracking loop

            # ---------------------------------------------------------------
            # DISCOVERY MODE: normal 2-second cycle
            # ---------------------------------------------------------------
            if new_audio_count < TRIGGER_SAMPLES:
                continue
            new_audio_count = 0

            # Skip silent chunks
            tail = full_audio[-TRIGGER_SAMPLES:]
            if _is_silence(tail):
                continue

            # Transcribe the full audio window
            async with ws.app.state.transcribe_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, _transcribe, full_audio.copy()
                )

            if not text or len(text.strip()) < 5:
                continue

            audio_len = len(full_audio) / SAMPLE_RATE
            log.info(
                "Transcribed (%.1fs): %s",
                audio_len,
                text[:120],
            )

            query_text = text

            # If the window starts with the previously emitted verse but also
            # contains new words, trim the old prefix instead of skipping.
            if last_emitted_text:
                residual = partial_ratio(text, last_emitted_text)
                stripped_text, strip_score = _strip_leading_residual(
                    text, last_emitted_text
                )
                if stripped_text and len(stripped_text.split()) >= 2:
                    log.info(
                        "  (trimmed residual prefix %.2f -> %s)",
                        strip_score,
                        stripped_text[:120],
                    )
                    query_text = stripped_text
                elif residual > 0.70:
                    log.info(
                        "  (residual overlap %.2f with last emitted, skipping)",
                        residual,
                    )
                    continue

            # Match against QuranDB (span-aware, with continuation bias)
            match = quran_db.match_verse(
                query_text,
                threshold=RAW_TRANSCRIPT_THRESHOLD,
                max_span=4,
                hint=last_emitted_ref,
                return_top_k=5,
            )
            match = _rerank_ambiguous_match(quran_db, query_text, match)

            # --- Debug log: full prediction table ---
            hint_str = (
                f"{last_emitted_ref[0]}:{last_emitted_ref[1]}"
                if last_emitted_ref
                else "none"
            )
            if match:
                match_key = _match_key(match)
                if pending_match_key == match_key:
                    pending_match_count += 1
                else:
                    pending_match_key = match_key
                    pending_match_count = 1

                ayah_end = match.get("ayah_end", "")
                end_str = f"-{ayah_end}" if ayah_end else ""
                log.info(
                    "MATCH  %s:%s%s  score=%.3f (raw=%.3f +bonus=%.3f)  hint=%s",
                    match["surah"],
                    match["ayah"],
                    end_str,
                    match["score"],
                    match.get("raw_score", match["score"]),
                    match.get("bonus", 0.0),
                    hint_str,
                )
                for i, r in enumerate(match.get("runners_up", []), 1):
                    tag = " <<<" if r.get("bonus", 0) > 0 else ""
                    log.info(
                        "  #%d  %s:%s  score=%.3f (raw=%.3f +%.3f)  %s%s",
                        i,
                        r["surah"],
                        r["ayah"],
                        r["score"],
                        r["raw_score"],
                        r["bonus"],
                        r["text_clean"][:40],
                        tag,
                    )
            else:
                pending_match_key = None
                pending_match_count = 0
                log.info("NO MATCH (below %.2f)  hint=%s", RAW_TRANSCRIPT_THRESHOLD, hint_str)

            effective_threshold = FIRST_MATCH_THRESHOLD if last_emitted_ref is None else VERSE_MATCH_THRESHOLD
            match_gap = _runner_up_gap(match) if match else 0.0
            repeat_confirmed = pending_match_count >= 2
            is_expected_followup = _is_expected_followup(
                quran_db, last_emitted_ref, match
            )
            continuation_threshold = (
                LONG_VERSE_CONTINUATION_THRESHOLD
                if _is_long_verse_ref(quran_db, last_emitted_ref)
                else CONTINUATION_STRONG_THRESHOLD
            )
            strong_continuation = (
                match is not None
                and is_expected_followup
                and match["score"] >= continuation_threshold
                and match_gap >= AMBIGUOUS_MATCH_GAP
            )
            strong_first_match = (
                match is not None
                and last_emitted_ref is None
                and match_gap >= AMBIGUOUS_MATCH_GAP
            )

            should_commit = (
                match is not None
                and match["score"] >= effective_threshold
                and (
                    repeat_confirmed
                    or strong_first_match
                    or (last_emitted_ref is not None and strong_continuation)
                )
            )

            if should_commit:
                ref = (match["surah"], match["ayah"])

                # Dedup: skip if same verse was just sent
                if ref == last_emitted_ref:
                    log.info("  (dedup — same as last emitted, skipping)")
                    continue

                verse = quran_db.get_verse(match["surah"], match["ayah"])
                surrounding = _get_surrounding_verses(
                    quran_db, match["surah"], match["ayah"]
                )

                await ws.send_json(
                    {
                        "type": "verse_match",
                        "surah": match["surah"],
                        "ayah": match["ayah"],
                        "ayah_end": match.get("ayah_end"),
                        "verse_text": (
                            verse["text_uthmani"] if verse else match.get("text", "")
                        ),
                        "surah_name": verse["surah_name"] if verse else "",
                        "confidence": round(match["score"], 2),
                        "surrounding_verses": surrounding,
                    }
                )

                # For multi-verse spans, advance hint to the last verse
                ayah_end = match.get("ayah_end")
                effective_ref = (match["surah"], ayah_end) if ayah_end else ref
                log.info(
                    ">>> EMITTED verse_match %s:%s%s (was %s)",
                    match["surah"],
                    match["ayah"],
                    f"-{ayah_end}" if ayah_end else "",
                    hint_str,
                )
                # Save pre-match state for recovery if tracking determines misidentification
                prev_emitted_ref = last_emitted_ref
                prev_emitted_text = last_emitted_text
                last_emitted_ref = effective_ref
                last_emitted_text = normalize_arabic(
                    match.get("text_clean", "")
                    or (verse["text_clean"] if verse else "")
                )
                pending_match_key = None
                pending_match_count = 0

                # Enter tracking mode for this verse
                if verse:
                    _enter_tracking(verse, ref)
                    # Keep full audio buffer — tracking needs it to
                    # detect if the verse is already complete (reader
                    # may be ahead of the system for short verses)
                else:
                    # No tracking — reset window for next discovery
                    full_audio = tail.copy()
            else:
                score = round(match["score"], 2) if match else 0.0
                if match and match["score"] >= effective_threshold:
                    log.info(
                        "  (held candidate key=%s repeats=%d gap=%.3f score=%.2f)",
                        pending_match_key,
                        pending_match_count,
                        match_gap,
                        score,
                    )
                else:
                    log.info(
                        "  (below threshold %.2f — sending raw_transcript, score=%.2f)",
                        effective_threshold,
                        score,
                    )
                await ws.send_json(
                    {
                        "type": "raw_transcript",
                        "text": text,
                        "confidence": score,
                    }
                )

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", ws.client)
    except Exception:
        log.exception("WebSocket error")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    return {"model_loaded": _nemo_model is not None}


@app.get("/api/surah/{surah_num}")
async def get_surah(surah_num: int):
    verses = quran_db.get_surah(surah_num)
    if not verses:
        raise HTTPException(status_code=404, detail="Surah not found")
    return {
        "surah": surah_num,
        "surah_name": verses[0]["surah_name"],
        "surah_name_en": verses[0]["surah_name_en"],
        "verses": [
            {
                "ayah": v["ayah"],
                "text_uthmani": v["text_uthmani"],
            }
            for v in verses
        ],
    }


# ---------------------------------------------------------------------------
# Serve frontend (if built)
# ---------------------------------------------------------------------------
_frontend_dist = PROJECT_ROOT / "web" / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dist), html=True),
        name="frontend",
    )
    log.info("Serving frontend from %s", _frontend_dist)
else:

    @app.get("/")
    async def _root():
        return {
            "status": "ok",
            "message": "Offline Tarteel backend. Frontend not built yet.",
        }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
