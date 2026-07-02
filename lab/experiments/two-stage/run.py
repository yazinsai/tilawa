"""Two-stage retrieval experiment.

Stage 1: Moonshine Tiny Arabic (27M) → fast ASR transcript
         → QuranDB.search() to get top-N verse candidates
Stage 2: CTC forced-alignment re-score on just those N candidates
         (using wav2vec2-base fine-tuned, or fallback to large model)

This bounds CTC computation to N candidates instead of 6,236,
enabling use of accurate CTC scoring with manageable latency.
"""
import sys
import math
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# Import ctc_scorer from sibling experiment
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

# ── Config ──
TOP_N = 50                  # candidates from Stage 1 for CTC re-scoring
CTC_CHUNK_SIZE = 50         # batch size for CTC scoring
TOP_SURAHS_SPAN = 3         # surahs to try multi-verse spans
MAX_SPAN = 4                # max consecutive verses in a span

# Model paths
MOONSHINE_MODEL = "UsefulSensors/moonshine-tiny-ar"
CTC_MODEL_SMALL = PROJECT_ROOT / "data" / "ctc-base-finetuned"  # after Modal training
CTC_MODEL_LARGE = PROJECT_ROOT / "data" / "ctc-model"            # fallback
CTC_PRETRAINED = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"  # HF fallback

# ── Lazy-loaded globals ──
_moonshine_model = None
_moonshine_processor = None
_ctc_model = None
_ctc_processor = None
_db = None
_device = None
_ctc_size_bytes = 0


def _ensure_loaded():
    global _moonshine_model, _moonshine_processor
    global _ctc_model, _ctc_processor
    global _db, _device, _ctc_size_bytes

    if _moonshine_model is not None:
        return

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Stage 1: Moonshine Tiny Arabic
    print(f"Loading Moonshine Tiny Arabic from {MOONSHINE_MODEL}...")
    _moonshine_processor = AutoProcessor.from_pretrained(MOONSHINE_MODEL)
    _moonshine_model = MoonshineForConditionalGeneration.from_pretrained(MOONSHINE_MODEL)
    _moonshine_model.eval()
    _moonshine_model.to(_device)

    # Stage 2: CTC model (prefer small fine-tuned, fall back to large)
    if CTC_MODEL_SMALL.exists():
        ctc_path = str(CTC_MODEL_SMALL)
        print(f"Loading fine-tuned CTC-base from {ctc_path}...")
        _ctc_size_bytes = sum(f.stat().st_size for f in CTC_MODEL_SMALL.rglob("*") if f.is_file())
    elif CTC_MODEL_LARGE.exists():
        ctc_path = str(CTC_MODEL_LARGE)
        print(f"Loading large CTC model from {ctc_path} (fallback)...")
        _ctc_size_bytes = 1_200 * 1024 * 1024
    else:
        ctc_path = CTC_PRETRAINED
        print(f"Loading CTC model from HuggingFace: {ctc_path}...")
        _ctc_size_bytes = 1_200 * 1024 * 1024

    _ctc_processor = Wav2Vec2Processor.from_pretrained(ctc_path)
    _ctc_model = Wav2Vec2ForCTC.from_pretrained(ctc_path)
    _ctc_model.eval()
    _ctc_model.to(_device)

    _db = QuranDB()


def _stage1_transcribe(audio: np.ndarray) -> str:
    """Moonshine Tiny Arabic → raw transcript."""
    inputs = _moonshine_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = _moonshine_model.generate(**inputs, max_new_tokens=256)

    text = _moonshine_processor.decode(generated[0], skip_special_tokens=True)
    return normalize_arabic(text)


def _stage1_candidates(transcript: str) -> list[dict]:
    """Get top-N verse candidates from transcript via text matching."""
    results = _db.search(transcript, top_k=TOP_N)
    return results


def _tokenize_for_ctc(text: str) -> list[int]:
    """Convert Arabic text to token IDs for CTC scoring."""
    encoded = _ctc_processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _stage2_ctc_rescore(audio: np.ndarray, candidates: list[dict]) -> list[tuple[dict, float]]:
    """CTC re-score candidates against audio logits."""
    inputs = _ctc_processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _ctc_model(**inputs).logits  # (1, T, V)

    logits_cpu = logits.cpu()
    blank_id = _ctc_processor.tokenizer.pad_token_id

    return score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)


def _build_span_candidates(single_results: list[tuple[dict, float]]) -> list[dict]:
    """Build multi-verse span candidates from top-scoring surahs."""
    seen = set()
    top_surahs = []
    for cand, _ in single_results:
        s = cand["surah"]
        if s not in seen:
            seen.add(s)
            top_surahs.append(s)
            if len(top_surahs) >= TOP_SURAHS_SPAN:
                break

    span_candidates = []
    for surah_num in top_surahs:
        verses = _db.get_surah(surah_num)
        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                combined_clean = " ".join(v["text_clean"] for v in chunk)
                combined_uthmani = " ".join(v["text_uthmani"] for v in chunk)
                span_candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "text_clean": combined_clean,
                    "text_uthmani": combined_uthmani,
                    "text": combined_uthmani,
                })
    return span_candidates


def transcribe(audio_path: str) -> str:
    """Return raw Moonshine transcript (for streaming pipeline)."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    return _stage1_transcribe(audio)


def predict(audio_path: str) -> dict:
    """Two-stage prediction: Moonshine ASR → CTC re-score."""
    _ensure_loaded()
    audio = load_audio(audio_path)

    # Stage 1: Fast ASR + text matching
    transcript = _stage1_transcribe(audio)
    if not transcript.strip():
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    candidates = _stage1_candidates(transcript)
    if not candidates:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Stage 2: CTC re-score top-N candidates
    single_results = _stage2_ctc_rescore(audio, candidates)

    if not single_results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Multi-verse span scoring for top surahs
    span_results = []
    if single_results[0][1] >= 0.15:  # skip if already confident
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            span_results = _stage2_ctc_rescore(audio, span_candidates)

    # Pick best from single + span
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
    """Combined size: Moonshine Tiny (~103MB) + CTC model."""
    _ensure_loaded()
    moonshine_size = 103 * 1024 * 1024
    return moonshine_size + _ctc_size_bytes
