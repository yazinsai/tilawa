import sys
from pathlib import Path

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_ID = "openai/whisper-small"
ADAPTER_PATH = PROJECT_ROOT / "data" / "lora-adapter-small"

_model = None
_processor = None
_db = None


def _ensure_loaded():
    global _model, _processor, _db
    if _model is not None:
        return
    _processor = WhisperProcessor.from_pretrained(MODEL_ID, language="arabic", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
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
            repetition_penalty=1.2,
            language="ar",
            task="transcribe",
        )
    return _processor.batch_decode(ids, skip_special_tokens=True)[0]


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = _model.generate(
            inputs["input_features"],
            max_new_tokens=225,
            repetition_penalty=1.2,
            language="ar",
            task="transcribe",
        )
    transcript = _processor.batch_decode(ids, skip_special_tokens=True)[0]
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
    """Whisper-small base + LoRA adapter."""
    total = 0
    # HF cache for whisper-small (~461MB)
    total += 461 * 1024 * 1024
    # LoRA adapter on disk
    adapter = PROJECT_ROOT / "data" / "lora-adapter-small"
    if adapter.exists():
        total += sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file())
    return total
