import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODELS = {
    "whisper-large-v3-turbo": {
        "hf_id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "size_mb": 3000,
    },
    "tarteel-whisper-base": {
        "hf_id": "tarteel-ai/whisper-base-ar-quran",
        "type": "whisper",
        "size_mb": 277,
    },
    "distil-whisper-large-v3": {
        "hf_id": "distil-whisper/distil-large-v3",
        "type": "whisper",
        "size_mb": 1500,
    },
    "moonshine-tiny-arabic": {
        "hf_id": "UsefulSensors/moonshine-tiny-ar",
        "type": "moonshine",
        "size_mb": 103,
    },
    "mms-1b-all": {
        "hf_id": "facebook/mms-1b-all",
        "type": "mms",
        "size_mb": 3700,
    },
    "wav2vec2-quran-hamza": {
        "hf_id": "HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr",
        "type": "wav2vec2",
        "size_mb": 360,
    },
    "tadabur-whisper-small": {
        "hf_id": "FaisaI/tadabur-Whisper-Small",
        "type": "whisper",
        "size_mb": 461,
    },
}

_loaded_models = {}
_db = None


def list_models() -> list[str]:
    return list(MODELS.keys())


def _ensure_db():
    global _db
    if _db is None:
        _db = QuranDB()


def _load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    cfg = MODELS[model_name]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if cfg["type"] == "whisper":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(cfg["hf_id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("whisper", processor, model, device)

    elif cfg["type"] == "moonshine":
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        processor = AutoProcessor.from_pretrained(cfg["hf_id"])
        model = MoonshineForConditionalGeneration.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("moonshine", processor, model, device)

    elif cfg["type"] == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["hf_id"], target_lang="ara")
        model = Wav2Vec2ForCTC.from_pretrained(cfg["hf_id"], target_lang="ara").to(device)
        model.eval()
        _loaded_models[model_name] = ("mms", processor, model, device)

    elif cfg["type"] == "wav2vec2":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(cfg["hf_id"])
        model = Wav2Vec2ForCTC.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("wav2vec2", processor, model, device)

    return _loaded_models[model_name]


def transcribe(audio_path: str, model_name: str = "tarteel-whisper-base") -> str:
    """Return raw transcript text."""
    model_type, processor, model, device = _load_model(model_name)
    audio = load_audio(audio_path)

    with torch.no_grad():
        if model_type == "whisper":
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            ids = model.generate(inputs["input_features"], max_new_tokens=225)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        elif model_type == "moonshine":
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            audio_len = len(audio) / 16000
            max_tokens = int(audio_len * 13) + 10
            ids = model.generate(**inputs, max_new_tokens=max_tokens, repetition_penalty=1.2)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        elif model_type in ("mms", "wav2vec2"):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(ids)[0]


def predict(audio_path: str, model_name: str = "tarteel-whisper-base") -> dict:
    _ensure_db()
    model_type, processor, model, device = _load_model(model_name)
    audio = load_audio(audio_path)

    with torch.no_grad():
        if model_type == "whisper":
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            ids = model.generate(inputs["input_features"], max_new_tokens=225)
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]

        elif model_type == "moonshine":
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Token limit to prevent hallucination
            audio_len = len(audio) / 16000
            max_tokens = int(audio_len * 13) + 10
            ids = model.generate(**inputs, max_new_tokens=max_tokens, repetition_penalty=1.2)
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]

        elif model_type in ("mms", "wav2vec2"):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            transcript = processor.batch_decode(ids)[0]

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


def model_size(model_name: str = "tarteel-whisper-base") -> int:
    return MODELS[model_name]["size_mb"] * 1024 * 1024
