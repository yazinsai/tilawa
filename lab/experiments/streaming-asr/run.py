import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from streaming_transcriber import StreamingTranscriber
from shared.quran_db import QuranDB

_transcriber = None
_db = None


def _ensure_loaded():
    global _transcriber, _db
    if _transcriber is not None:
        return
    _transcriber = StreamingTranscriber(chunk_seconds=3.0, overlap_seconds=0.0)
    _db = QuranDB()


def transcribe(audio_path: str) -> dict:
    """Return transcript with confidence info for streaming pipeline gating."""
    _ensure_loaded()
    result = _transcriber.transcribe_batch(audio_path)
    return {"text": result.get("text", ""), "avg_logprob": result.get("avg_logprob", 0.0)}


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    result = _transcriber.transcribe_batch(audio_path)
    transcript = result.get("text", "")

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
    """mlx-whisper base model."""
    return 145 * 1024 * 1024  # whisper-base ~145MB
