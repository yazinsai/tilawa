"""CTC forced-alignment experiment.

Uses a pre-trained Arabic wav2vec2 CTC model to produce frame-level
character logits, then scores candidate Quran verses via the CTC
forward algorithm.

Two-pass approach:
  1. Encode audio -> frame-level logits
  2. Greedy decode -> rough text for fast Levenshtein pre-filtering
  3. CTC-score top-200 candidates (length-normalized, short-verse safe)
  4. For top-5 surahs, score multi-verse spans (2-4 consecutive)
  5. Best score wins (single or multi-verse)
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

# Import ctc_scorer from sibling file (hyphenated directory)
_scorer_path = Path(__file__).parent / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

TOP_K = 100            # Levenshtein candidates for CTC re-scoring
CHUNK_SIZE = 50        # verses per CTC scoring batch
TOP_SURAHS = 5         # surahs to try multi-verse spans for
MAX_SPAN = 4           # max consecutive verses in a span
SKIP_SPAN_THRESHOLD = 0.15  # skip span scoring if best single score < this

LOCAL_MODEL = PROJECT_ROOT / "data" / "ctc-model"
PRETRAINED_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

_model = None
_processor = None
_db = None
_device = None


def _ensure_loaded():
    global _model, _processor, _db, _device
    if _model is not None:
        return
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else PRETRAINED_MODEL
    print(f"Loading CTC model from {model_path}...")
    _processor = Wav2Vec2Processor.from_pretrained(model_path)
    _model = Wav2Vec2ForCTC.from_pretrained(model_path)
    _model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(_device)
    _db = QuranDB()


def transcribe(audio_path: str) -> str:
    """Return raw greedy-decoded transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(pred_ids.cpu().numpy())
    return normalize_arabic(text)


def _tokenize_for_ctc(text: str) -> list[int]:
    """Convert Arabic text to token IDs using model's tokenizer."""
    encoded = _processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _levenshtein_candidates(text: str, top_k: int = TOP_K) -> list[dict]:
    """Fast Levenshtein pre-filter against all verses.
    Uses both regular and spaceless matching, merging results.
    """
    normalized = normalize_arabic(text)
    spaceless = normalized.replace(" ", "")

    scored = []
    for v in _db.verses:
        # Best of regular and spaceless Levenshtein
        score_regular = lev_ratio(normalized, v["text_clean"])
        score_spaceless = lev_ratio(spaceless, v["text_clean"].replace(" ", ""))
        best = max(score_regular, score_spaceless)
        scored.append({**v, "score": best, "text": v["text_uthmani"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _build_span_candidates(
    single_results: list[tuple[dict, float]],
) -> list[dict]:
    """Build multi-verse span candidates from top-scoring surahs."""
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


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)

    # 1. Encode audio -> frame-level logits
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits  # (1, T, V)

    logits_cpu = logits.cpu()
    blank_id = _processor.tokenizer.pad_token_id

    # 2. Greedy decode -> rough text for Levenshtein pre-filtering
    pred_ids = torch.argmax(logits_cpu, dim=-1)[0]
    transcript = normalize_arabic(_processor.decode(pred_ids.numpy()))

    if not transcript.strip():
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": "",
        }

    # 3. Levenshtein pre-filter -> top-K candidates
    candidates = _levenshtein_candidates(transcript, top_k=TOP_K)

    # 4. CTC re-score candidates (length-normalized)
    single_results = score_candidates(
        logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id,
    )

    if not single_results:
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": transcript,
        }

    # 5. Multi-verse span scoring for top surahs
    #    Skip if best single-verse score is already confident (saves 10-40s)
    span_candidates = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
    if span_candidates:
        span_results = []
        for i in range(0, len(span_candidates), CHUNK_SIZE):
            chunk = span_candidates[i:i + CHUNK_SIZE]
            scored = score_candidates(
                logits_cpu, chunk, _tokenize_for_ctc, blank_id=blank_id,
            )
            span_results.extend(scored)
        span_results.sort(key=lambda x: x[1])
    else:
        span_results = []

    # 6. Pick best from single + span results
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
    """wav2vec2-large-xlsr-53-arabic ~1.2GB."""
    return 1_200 * 1024 * 1024
