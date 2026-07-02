"""Two-stage experiment: FastConformer Stage 1 + CTC re-scoring Stage 2.

Stage 1:
  NVIDIA FastConformer (stt_ar_fastconformer_hybrid_large_pcd_v1.0)
  -> transcript -> QuranDB match
  -> if score >= CONFIDENCE_THRESHOLD, return immediately (skip Stage 2)

Stage 2 (fallback):
  Pruned Rabah CTC forced-alignment re-scoring on top-N candidates.
  Only runs when FastConformer confidence is low.

This combines the best ASR model (FastConformer, 85% SeqAcc) with the
best alignment model (pruned CTC) to recover errors on different samples.
"""

import math
import io
import os
import sys
import tempfile
import types
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

# Import ctc_scorer from ctc-alignment experiment
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(os.getenv("FASTCONFORMER_CTC_CONFIDENCE", "0.7"))
TOP_N = 50
TOP_SURAHS_SPAN = 5
MAX_SPAN = 4
SKIP_SPAN_THRESHOLD = 0.15

# Stage 1 — FastConformer
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
FASTCONFORMER_SIZE_BYTES = 115 * 1024 * 1024
DECODER_TYPE = os.getenv("NVIDIA_FASTCONFORMER_DECODER", "ctc")
SPAN_THRESHOLD = float(os.getenv("NVIDIA_FASTCONFORMER_SPAN_THRESHOLD", "0.25"))

# Stage 2 — CTC re-scoring fallback chain
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
STAGE2_DYNAMIC_INT8 = os.getenv("FASTCONFORMER_CTC_STAGE2_DYNAMIC_INT8", "1") == "1"

# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_fastconformer_model = None
_ctc_model = None
_ctc_processor = None
_ctc_device = None
_db = None
_stage2_size_bytes = 0


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _estimate_state_dict_size(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def _can_use_dynamic_int8() -> bool:
    supported = set(torch.backends.quantized.supported_engines)
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    engine = torch.backends.quantized.engine
    return engine in {"qnnpack", "fbgemm", "x86"} and engine in supported


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
    global _fastconformer_model, _ctc_model, _ctc_processor
    global _ctc_device, _db, _stage2_size_bytes

    if _fastconformer_model is not None:
        return

    # --- Stage 1: FastConformer ---
    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from nemo.utils import logging as nemo_logging
    except Exception as exc:
        raise ImportError(
            "NeMo ASR dependencies are required for fastconformer-ctc-rescore. "
            "Install with: pip install 'nemo_toolkit[asr]'"
        ) from exc

    nemo_logging.set_verbosity(nemo_logging.ERROR)

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

    try:
        _fastconformer_model.change_decoding_strategy(decoder_type=DECODER_TYPE)
    except Exception:
        pass

    # --- Stage 2: CTC model ---
    stage2_source, _stage2_size_bytes = _resolve_stage2_model()
    print(f"[Stage 2] Loading CTC model from {stage2_source}...")
    _ctc_processor = Wav2Vec2Processor.from_pretrained(stage2_source)
    _ctc_model = Wav2Vec2ForCTC.from_pretrained(stage2_source)

    if STAGE2_DYNAMIC_INT8 and _can_use_dynamic_int8():
        _ctc_model = torch.ao.quantization.quantize_dynamic(
            _ctc_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        _ctc_device = "cpu"
        _stage2_size_bytes = _estimate_state_dict_size(_ctc_model)
    else:
        if STAGE2_DYNAMIC_INT8:
            print("Warning: dynamic int8 unavailable; using fp32 Stage 2 CTC model.")
        _ctc_device = "mps" if torch.backends.mps.is_available() else "cpu"

    _ctc_model.eval()
    _ctc_model.to(_ctc_device)

    _db = QuranDB()


# ---------------------------------------------------------------------------
# FastConformer helpers
# ---------------------------------------------------------------------------

def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _fastconformer_transcribe(audio_path: str) -> str:
    _ensure_loaded()

    audio = load_audio(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(tmp_path, audio, 16000)
        try:
            outputs = _fastconformer_model.transcribe(
                audio=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
                verbose=False,
            )
        except TypeError:
            outputs = _fastconformer_model.transcribe(
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


# ---------------------------------------------------------------------------
# CTC helpers
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


def _stage2_rescore(
    logits_cpu: torch.Tensor,
    blank_id: int,
    candidates: list[dict],
) -> list[tuple[dict, float]]:
    return score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)


def _build_span_candidates(single_results: list[tuple[dict, float]]) -> list[dict]:
    seen = set()
    top_surahs = []
    for cand, _ in single_results:
        surah = cand["surah"]
        if surah not in seen:
            seen.add(surah)
            top_surahs.append(surah)
            if len(top_surahs) >= TOP_SURAHS_SPAN:
                break

    spans = []
    for surah_num in top_surahs:
        verses = _db.get_surah(surah_num)
        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                spans.append(
                    {
                        "surah": surah_num,
                        "ayah": chunk[0]["ayah"],
                        "ayah_end": chunk[-1]["ayah"],
                        "text_clean": " ".join(v["text_clean"] for v in chunk),
                        "text_uthmani": " ".join(v["text_uthmani"] for v in chunk),
                        "text": " ".join(v["text_uthmani"] for v in chunk),
                    }
                )

    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str) -> str:
    """FastConformer transcription only."""
    return _fastconformer_transcribe(audio_path)


def predict(audio_path: str) -> dict:
    """Two-stage prediction with confidence-gated fallback.

    1. FastConformer transcribe + QuranDB span-aware match
    2. If score >= CONFIDENCE_THRESHOLD, return immediately
    3. Otherwise, run CTC re-scoring on top-N candidates
    4. Compare CTC best vs FastConformer best, return winner
    """
    _ensure_loaded()

    transcript = _fastconformer_transcribe(audio_path)
    if not transcript.strip():
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }

    # Stage 1: QuranDB match
    stage1_result = _db.match_verse(
        transcript, threshold=SPAN_THRESHOLD, max_span=MAX_SPAN,
    )

    if stage1_result and stage1_result["score"] >= CONFIDENCE_THRESHOLD:
        # High confidence — skip Stage 2
        return {
            "surah": stage1_result["surah"],
            "ayah": stage1_result["ayah"],
            "ayah_end": stage1_result.get("ayah_end"),
            "score": stage1_result["score"],
            "transcript": transcript,
        }

    # Stage 2: CTC re-scoring fallback
    candidates = _db.search(transcript, top_k=TOP_N)
    if not candidates:
        if stage1_result:
            return {
                "surah": stage1_result["surah"],
                "ayah": stage1_result["ayah"],
                "ayah_end": stage1_result.get("ayah_end"),
                "score": stage1_result["score"],
                "transcript": transcript,
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    logits_cpu, blank_id = _stage2_logits(audio_path)
    single_results = _stage2_rescore(logits_cpu, blank_id, candidates)

    if not single_results:
        if stage1_result:
            return {
                "surah": stage1_result["surah"],
                "ayah": stage1_result["ayah"],
                "ayah_end": stage1_result.get("ayah_end"),
                "score": stage1_result["score"],
                "transcript": transcript,
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    # Try multi-verse spans around top candidates
    span_results = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            span_results = _stage2_rescore(logits_cpu, blank_id, span_candidates)
            span_results.sort(key=lambda x: x[1])

    best_candidate, best_loss = single_results[0]
    if span_results and span_results[0][1] < best_loss:
        best_candidate, best_loss = span_results[0]

    ctc_confidence = math.exp(-best_loss) if math.isfinite(best_loss) else 0.0

    # Compare CTC result with Stage 1 result — return whichever is more confident
    ctc_result = {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(ctc_confidence, 4),
        "transcript": transcript,
    }

    if stage1_result and stage1_result["score"] > ctc_confidence:
        return {
            "surah": stage1_result["surah"],
            "ayah": stage1_result["ayah"],
            "ayah_end": stage1_result.get("ayah_end"),
            "score": stage1_result["score"],
            "transcript": transcript,
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
