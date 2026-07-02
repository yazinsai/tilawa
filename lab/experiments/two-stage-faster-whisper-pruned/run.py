"""Two-stage experiment: faster-whisper Quran Stage 1 + pruned CTC Stage 2.

Stage 1:
  OdyAsh/faster-whisper-base-ar-quran (CTranslate2)
  -> transcript -> QuranDB top-N candidates

Stage 2:
  CTC forced alignment re-scoring on candidates, preferring pruned/distilled
  student checkpoints over large fallback.
"""

import math
import io
import os
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# Stage 1 (faster-whisper)
STAGE1_MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"
STAGE1_SIZE_BYTES = 147 * 1024 * 1024

# Stage 2 (CTC rescoring) fallback chain — fine-tuned pruned models preferred
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
STAGE2_DYNAMIC_INT8 = os.getenv("TWO_STAGE_STAGE2_DYNAMIC_INT8", "1") == "1"

TOP_N = 50
TOP_SURAHS_SPAN = 3
MAX_SPAN = 4
SKIP_SPAN_THRESHOLD = 0.15

_stage1_model = None
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


def _resolve_stage2_model() -> tuple[str, int]:
    for candidate in STAGE2_CANDIDATES:
        if candidate.exists():
            return str(candidate), _dir_size_bytes(candidate)
    return STAGE2_HF_FALLBACK, 1_200 * 1024 * 1024


def _ensure_loaded():
    global _stage1_model, _ctc_model, _ctc_processor
    global _ctc_device, _db, _stage2_size_bytes

    if _stage1_model is not None:
        return

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise ImportError(
            "faster-whisper is required for two-stage-faster-whisper-pruned. "
            "Install with: pip install faster-whisper"
        ) from exc

    # faster-whisper runs on CPU/CUDA (no MPS backend).
    print(f"Loading faster-whisper Stage 1 from {STAGE1_MODEL_ID}...")
    _stage1_model = WhisperModel(
        STAGE1_MODEL_ID,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )

    stage2_source, _stage2_size_bytes = _resolve_stage2_model()
    print(f"Loading Stage 2 CTC from {stage2_source}...")
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


def _stage1_transcribe(audio_path: str) -> str:
    _ensure_loaded()

    audio = load_audio(audio_path)
    segments, _ = _stage1_model.transcribe(
        audio,
        language="ar",
        beam_size=3,
        best_of=3,
        vad_filter=True,
    )

    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return normalize_arabic(transcript)


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


def transcribe(audio_path: str) -> str:
    return _stage1_transcribe(audio_path)


def predict(audio_path: str) -> dict:
    _ensure_loaded()

    transcript = _stage1_transcribe(audio_path)
    if not transcript.strip():
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }

    candidates = _db.search(transcript, top_k=TOP_N)
    if not candidates:
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
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    span_results = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            span_results = _stage2_rescore(logits_cpu, blank_id, span_candidates)
            span_results.sort(key=lambda x: x[1])

    best_candidate, best_loss = single_results[0]
    if span_results and span_results[0][1] < best_loss:
        best_candidate, best_loss = span_results[0]

    confidence = math.exp(-best_loss) if math.isfinite(best_loss) else 0.0

    return {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(confidence, 4),
        "transcript": transcript,
    }


def model_size() -> int:
    _ensure_loaded()
    return STAGE1_SIZE_BYTES + _stage2_size_bytes
