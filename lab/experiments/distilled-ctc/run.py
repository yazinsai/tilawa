"""Distilled CTC experiment.

Uses wav2vec2-base (95M params, ~380MB) fine-tuned for Arabic CTC,
optionally knowledge-distilled from the large wav2vec2-xlsr-53-arabic.

Same CTC forced-alignment approach as ctc-alignment/ but 3x smaller model.
After int8 quantization, targets ~95MB on disk.
"""
import sys
import math
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from Levenshtein import ratio as lev_ratio

# Import ctc_scorer
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

# ── Config ──
TOP_K = 100           # Levenshtein pre-filter candidates
CHUNK_SIZE = 50       # CTC scoring batch size
TOP_SURAHS = 5        # surahs for multi-verse spans
MAX_SPAN = 4          # max consecutive verses
SKIP_SPAN_THRESHOLD = 0.15

# Model paths — prefer distilled > fine-tuned > none
MODEL_DISTILLED = PROJECT_ROOT / "data" / "ctc-base-distilled"
MODEL_FINETUNED = PROJECT_ROOT / "data" / "ctc-base-finetuned"

_model = None
_processor = None
_db = None
_device = None
_model_size_bytes = 0


def _ensure_loaded():
    global _model, _processor, _db, _device, _model_size_bytes
    if _model is not None:
        return

    # Find best available model
    if MODEL_DISTILLED.exists():
        model_path = str(MODEL_DISTILLED)
        _model_size_bytes = sum(f.stat().st_size for f in MODEL_DISTILLED.rglob("*") if f.is_file())
        print(f"Loading distilled CTC-base from {model_path}...")
    elif MODEL_FINETUNED.exists():
        model_path = str(MODEL_FINETUNED)
        _model_size_bytes = sum(f.stat().st_size for f in MODEL_FINETUNED.rglob("*") if f.is_file())
        print(f"Loading fine-tuned CTC-base from {model_path}...")
    else:
        print("ERROR: No trained CTC-base model found!")
        print("  Expected: data/ctc-base-distilled/ or data/ctc-base-finetuned/")
        print("  Run Modal training first: modal run scripts/train_ctc_base_modal.py")
        raise FileNotFoundError("No trained CTC-base model. Run Modal training first.")

    _processor = Wav2Vec2Processor.from_pretrained(model_path)
    _model = Wav2Vec2ForCTC.from_pretrained(model_path)
    _model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(_device)
    _db = QuranDB()


def _tokenize_for_ctc(text: str) -> list[int]:
    encoded = _processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _levenshtein_candidates(text: str, top_k: int = TOP_K) -> list[dict]:
    normalized = normalize_arabic(text)
    spaceless = normalized.replace(" ", "")

    scored = []
    for v in _db.verses:
        score_regular = lev_ratio(normalized, v["text_clean"])
        score_spaceless = lev_ratio(spaceless, v["text_clean"].replace(" ", ""))
        best = max(score_regular, score_spaceless)
        scored.append({**v, "score": best, "text": v["text_uthmani"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _build_span_candidates(single_results):
    seen = set()
    top_surahs = []
    for cand, _ in single_results:
        s = cand["surah"]
        if s not in seen:
            seen.add(s)
            top_surahs.append(s)
            if len(top_surahs) >= TOP_SURAHS:
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
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(pred_ids.cpu().numpy())
    return normalize_arabic(text)


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)

    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    logits_cpu = logits.cpu()
    blank_id = _processor.tokenizer.pad_token_id

    # Greedy decode for Levenshtein pre-filtering
    pred_ids = torch.argmax(logits_cpu, dim=-1)[0]
    transcript = normalize_arabic(_processor.decode(pred_ids.numpy()))

    if not transcript.strip():
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    # Levenshtein pre-filter → CTC re-score
    candidates = _levenshtein_candidates(transcript, top_k=TOP_K)
    single_results = score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)

    if not single_results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Multi-verse span scoring
    span_results = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            for i in range(0, len(span_candidates), CHUNK_SIZE):
                chunk = span_candidates[i:i + CHUNK_SIZE]
                scored = score_candidates(logits_cpu, chunk, _tokenize_for_ctc, blank_id=blank_id)
                span_results.extend(scored)
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
    return _model_size_bytes
