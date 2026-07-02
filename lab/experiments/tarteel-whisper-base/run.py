import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import re
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from shared.audio import load_audio
from shared.quran_db import QuranDB

_SPECIAL_PREFIX = re.compile(r"^(<\|[^|]+\|>)+\s*")

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"

_model = None
_processor = None
_db = None


def _ensure_loaded():
    global _model, _processor, _db
    if _model is not None:
        return
    _processor = WhisperProcessor.from_pretrained(MODEL_ID)
    _model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    _model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(device)
    _db = QuranDB()


def transcribe(audio_path: str) -> str:
    """Return raw Whisper transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = _model.generate(inputs["input_features"], max_new_tokens=225)
    text = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    return _SPECIAL_PREFIX.sub("", text).strip()


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = _model.generate(inputs["input_features"], max_new_tokens=225)
    transcript = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    # Model bakes <|ar|><|transcribe|><|notimestamps|> into output text — strip them
    transcript = _SPECIAL_PREFIX.sub("", transcript).strip()
    match = _db.match_verse(transcript)

    if match is None:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    return {
        "surah": match["surah"],
        "ayah": match["ayah"],
        "ayah_end": match.get("ayah_end"),
        "score": match["score"],
        "transcript": transcript,
    }


def model_size() -> int:
    """Whisper-base fine-tuned (~290MB)."""
    return 290 * 1024 * 1024
