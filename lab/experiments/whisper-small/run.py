import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_ID = "openai/whisper-small"

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
        ids = _model.generate(
            inputs["input_features"],
            max_new_tokens=225,
            language="arabic",
        )
    return _processor.batch_decode(ids, skip_special_tokens=True)[0]


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    transcript = transcribe(audio_path)
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
    """Whisper-small ~461MB."""
    return 461 * 1024 * 1024
