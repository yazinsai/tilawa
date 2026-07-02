"""FastConformer N-Best + CTC Brute-Force Fallback.

Pipeline:
  1. FastConformer CTC logits → pyctcdecode beam search → N-best hypotheses
  2. For each hypothesis: QuranDB.match_verse(text, max_span=6)
  3. If any match score >= CONFIDENCE_THRESHOLD → return best (fast path)
  4. Collect top-K surahs across ALL N-best matches
  5. CTC brute-force: score ALL verses + spans in those surahs
  6. Return best CTC-scored result

This recovers failures where the greedy transcript is wrong but:
  - A different beam hypothesis is close enough for QuranDB, OR
  - The correct surah appears in any hypothesis's matches, so CTC can find it
"""

import math
import os
import sys
import types
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

# Lazy imports (loaded in _ensure_loaded to avoid heavy imports during module discovery)
score_candidates = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BEST = 5
CONFIDENCE_THRESHOLD = float(os.getenv("NBEST_CONFIDENCE", "0.7"))
TOP_SURAHS_BRUTEFORCE = 10
MAX_SPAN = 6
SPAN_THRESHOLD = float(os.getenv("NVIDIA_FASTCONFORMER_SPAN_THRESHOLD", "0.25"))

# Stage 1 — FastConformer
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
FASTCONFORMER_SIZE_BYTES = 115 * 1024 * 1024

# Stage 2 — CTC model fallback chain
STAGE2_CANDIDATES = [
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8l-first_n-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8l-evenly_spaced-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-12l-evenly_spaced-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8",
    PROJECT_ROOT / "data" / "ctc-base-distilled",
    PROJECT_ROOT / "data" / "ctc-base-finetuned",
    PROJECT_ROOT / "data" / "ctc-model",
]
STAGE2_HF_FALLBACK = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_fastconformer_model = None
_beam_decoder = None  # pyctcdecode BeamSearchDecoderCTC
_ctc_model = None
_ctc_processor = None
_ctc_device = None
_db = None
_stage2_size_bytes = 0


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# kaldialign fallback (needed for NeMo import)
# ---------------------------------------------------------------------------

def _install_kaldialign_fallback() -> None:
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


# ---------------------------------------------------------------------------
# Stage 2 model resolution
# ---------------------------------------------------------------------------

def _resolve_stage2_model() -> tuple[str, int]:
    for candidate in STAGE2_CANDIDATES:
        if candidate.exists():
            return str(candidate), _dir_size_bytes(candidate)
    return STAGE2_HF_FALLBACK, 1_200 * 1024 * 1024


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

def _ensure_loaded():
    global _fastconformer_model, _beam_decoder
    global _ctc_model, _ctc_processor, _ctc_device
    global _db, _stage2_size_bytes, score_candidates

    if _fastconformer_model is not None:
        return

    # Lazy-import heavy dependencies (avoids loading during benchmark discovery)
    if score_candidates is None:
        _scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
        _spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
        _scorer_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_scorer_mod)
        score_candidates = _scorer_mod.score_candidates

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    # --- Stage 1: FastConformer ---
    _install_kaldialign_fallback()

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    except Exception as exc:
        raise ImportError(
            "NeMo ASR dependencies are required. "
            "Install with: pip install 'nemo_toolkit[asr]'"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    print(f"[Stage 1] Loading NVIDIA FastConformer from {source} on {device}...")

    try:
        _fastconformer_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=source,
            map_location=device,
        )
    except Exception:
        if LOCAL_MODEL_DIR.exists():
            nemo_files = sorted(LOCAL_MODEL_DIR.glob("*.nemo"))
            if not nemo_files:
                raise
            _fastconformer_model = EncDecHybridRNNTCTCBPEModel.restore_from(
                str(nemo_files[0]),
                map_location=device,
            )
        else:
            raise
    _fastconformer_model.eval()

    # Use greedy CTC for the base model (stable, fast)
    try:
        _fastconformer_model.change_decoding_strategy(decoder_type="ctc")
    except Exception:
        pass

    # Build pyctcdecode beam search decoder from the model's vocabulary.
    # This bypasses NeMo's beam search (which requires KenLM) and gives
    # us N-best hypotheses directly from the CTC log-probabilities.
    try:
        from pyctcdecode import build_ctcdecoder

        vocab = _fastconformer_model.tokenizer.vocab  # list of 1024 BPE tokens
        # NeMo CTC blank is at index len(vocab); pyctcdecode expects "" for blank
        labels = list(vocab) + [""]
        _beam_decoder = build_ctcdecoder(labels)
        print(f"[Stage 1] Built pyctcdecode beam decoder (vocab={len(vocab)}, beam_width={N_BEST})")
    except Exception as e:
        print(f"[Stage 1] pyctcdecode setup failed ({e}), N-best disabled")
        _beam_decoder = None

    # --- Stage 2: CTC model ---
    stage2_source, _stage2_size_bytes = _resolve_stage2_model()
    print(f"[Stage 2] Loading CTC model from {stage2_source}...")
    _ctc_processor = Wav2Vec2Processor.from_pretrained(stage2_source)  # noqa: F821
    _ctc_model = Wav2Vec2ForCTC.from_pretrained(stage2_source)  # noqa: F821
    _ctc_model.eval()

    _ctc_device = "mps" if torch.backends.mps.is_available() else "cpu"
    _ctc_model.to(_ctc_device)

    _db = QuranDB()


# ---------------------------------------------------------------------------
# FastConformer N-best transcription
# ---------------------------------------------------------------------------

def _fastconformer_transcribe_nbest(audio_path: str) -> list[dict]:
    """Return N-best transcriptions: [{"text": str, "score": float}, ...]

    Uses model internals (preprocessor → encoder → ctc_decoder) to get
    CTC log-probs, then pyctcdecode for beam search. Falls back to greedy
    if pyctcdecode is unavailable.
    """
    _ensure_loaded()

    audio = load_audio(audio_path)
    audio_signal = torch.tensor(audio).unsqueeze(0)
    audio_length = torch.tensor([len(audio)])

    with torch.no_grad():
        processed_signal, processed_signal_length = _fastconformer_model.preprocessor(
            input_signal=audio_signal, length=audio_length
        )
        encoded, encoded_len = _fastconformer_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        log_probs = _fastconformer_model.ctc_decoder(encoder_output=encoded)

    if _beam_decoder is not None:
        # pyctcdecode beam search on CTC log-probs
        lp = F.log_softmax(log_probs[0], dim=-1).numpy()
        beam_results = _beam_decoder.decode_beams(lp, beam_width=N_BEST)

        hypotheses = []
        for text, _lm_state, _indices, logit_score, _lm_score in beam_results:
            text = normalize_arabic(text)
            hypotheses.append({"text": text, "score": float(logit_score)})

        if hypotheses:
            return hypotheses

    # Fallback: greedy decode from log_probs
    greedy_ids = torch.argmax(log_probs, dim=-1)[0]
    # Simple CTC decode: collapse repeated tokens, remove blanks
    blank_id = log_probs.shape[-1] - 1
    prev = -1
    tokens = []
    for idx in greedy_ids.tolist():
        if idx != prev and idx != blank_id:
            tokens.append(idx)
        prev = idx

    vocab = _fastconformer_model.tokenizer.vocab
    text = "".join(vocab[t] for t in tokens if t < len(vocab))
    text = text.replace("▁", " ").strip()
    text = normalize_arabic(text)
    return [{"text": text, "score": 0.0}]


# ---------------------------------------------------------------------------
# CTC helpers (Stage 2)
# ---------------------------------------------------------------------------

def _tokenize_for_ctc(text: str) -> list[int]:
    encoded = _ctc_processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _stage2_logits(audio_path: str) -> tuple[torch.Tensor, int]:
    audio = load_audio(audio_path)
    inputs = _ctc_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_ctc_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _ctc_model(**inputs).logits

    return logits.cpu(), _ctc_processor.tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

def _collect_candidate_surahs(nbest_matches: list[dict]) -> list[int]:
    """Union of top surahs from all N-best QuranDB matches."""
    seen = set()
    ordered = []

    for match in nbest_matches:
        if match is None:
            continue
        surah = match["surah"]
        if surah not in seen:
            seen.add(surah)
            ordered.append(surah)

    return ordered[:TOP_SURAHS_BRUTEFORCE]


def _build_bruteforce_candidates(surahs: list[int]) -> list[dict]:
    """Build all single-verse + multi-verse span candidates for the given surahs."""
    candidates = []
    for surah_num in surahs:
        verses = _db.get_surah(surah_num)
        # Single verses
        for v in verses:
            candidates.append({
                "surah": surah_num,
                "ayah": v["ayah"],
                "ayah_end": None,
                "text_clean": v["text_clean"],
                "text_uthmani": v["text_uthmani"],
                "text": v["text_uthmani"],
            })
        # Multi-verse spans
        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "text_clean": " ".join(v["text_clean"] for v in chunk),
                    "text_uthmani": " ".join(v["text_uthmani"] for v in chunk),
                    "text": " ".join(v["text_uthmani"] for v in chunk),
                })
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str) -> str:
    """Return the best (top-1) transcript."""
    hypotheses = _fastconformer_transcribe_nbest(audio_path)
    return hypotheses[0]["text"] if hypotheses else ""


def predict(audio_path: str) -> dict:
    """N-best + CTC brute-force prediction.

    1. N-best transcribe via pyctcdecode beam search on FastConformer CTC logits
    2. QuranDB match each hypothesis
    3. If any match score >= CONFIDENCE_THRESHOLD → return best (fast path)
    4. Collect top surahs across all matches + QuranDB.search()
    5. CTC brute-force all verses + spans in those surahs
    6. Return best CTC-scored result
    """
    _ensure_loaded()

    hypotheses = _fastconformer_transcribe_nbest(audio_path)

    # Match each hypothesis against QuranDB
    nbest_matches = []
    best_fast_match = None
    best_fast_score = 0.0

    for hyp in hypotheses:
        text = hyp["text"]
        if not text.strip():
            nbest_matches.append(None)
            continue

        match = _db.match_verse(text, threshold=SPAN_THRESHOLD, max_span=MAX_SPAN)
        nbest_matches.append(match)

        if match and match["score"] > best_fast_score:
            best_fast_score = match["score"]
            best_fast_match = match

    # Fast path: if any N-best match is confident enough, return it
    if best_fast_match and best_fast_score >= CONFIDENCE_THRESHOLD:
        return {
            "surah": best_fast_match["surah"],
            "ayah": best_fast_match["ayah"],
            "ayah_end": best_fast_match.get("ayah_end"),
            "score": best_fast_score,
            "transcript": hypotheses[0]["text"],
        }

    # Widen the surah candidate pool using QuranDB.search() on each hypothesis
    all_search_surahs = set()
    for hyp in hypotheses:
        text = hyp["text"]
        if not text.strip():
            continue
        search_results = _db.search(text, top_k=10)
        for r in search_results:
            all_search_surahs.add(r["surah"])

    # Collect surahs from direct matches
    match_surahs = _collect_candidate_surahs(nbest_matches)

    # Merge: match surahs first (higher priority), then search surahs
    seen = set(match_surahs)
    merged_surahs = list(match_surahs)
    for s in all_search_surahs:
        if s not in seen:
            seen.add(s)
            merged_surahs.append(s)
    candidate_surahs = merged_surahs[:TOP_SURAHS_BRUTEFORCE]

    if not candidate_surahs:
        if best_fast_match:
            return {
                "surah": best_fast_match["surah"],
                "ayah": best_fast_match["ayah"],
                "ayah_end": best_fast_match.get("ayah_end"),
                "score": best_fast_score,
                "transcript": hypotheses[0]["text"],
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": hypotheses[0]["text"] if hypotheses else "",
        }

    # CTC brute-force: score all verses + spans in candidate surahs
    candidates = _build_bruteforce_candidates(candidate_surahs)
    print(f"  [Brute-force] {len(candidate_surahs)} surahs, {len(candidates)} candidates")

    logits_cpu, blank_id = _stage2_logits(audio_path)
    scored = score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)

    if not scored:
        if best_fast_match:
            return {
                "surah": best_fast_match["surah"],
                "ayah": best_fast_match["ayah"],
                "ayah_end": best_fast_match.get("ayah_end"),
                "score": best_fast_score,
                "transcript": hypotheses[0]["text"],
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": hypotheses[0]["text"] if hypotheses else "",
        }

    best_candidate, best_loss = scored[0]
    ctc_confidence = math.exp(-best_loss) if math.isfinite(best_loss) else 0.0

    ctc_result = {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(ctc_confidence, 4),
        "transcript": hypotheses[0]["text"],
    }

    # Compare CTC result with best fast match — return whichever is more confident
    if best_fast_match and best_fast_score > ctc_confidence:
        return {
            "surah": best_fast_match["surah"],
            "ayah": best_fast_match["ayah"],
            "ayah_end": best_fast_match.get("ayah_end"),
            "score": best_fast_score,
            "transcript": hypotheses[0]["text"],
        }

    return ctc_result


def model_size() -> int:
    _ensure_loaded()
    if LOCAL_MODEL_DIR.exists():
        fc_size = sum(
            f.stat().st_size for f in LOCAL_MODEL_DIR.rglob("*") if f.is_file()
        )
    else:
        fc_size = FASTCONFORMER_SIZE_BYTES
    return fc_size + _stage2_size_bytes
