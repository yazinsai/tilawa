import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from existing search module in same directory
sys.path.insert(0, str(Path(__file__).parent))
from search import load_search_model, load_index, search_verse

DEFAULT_MODEL = "hubert"
DEFAULT_RECITER = "alafasy"

_model = None
_extractor = None
_device = None
_index = None
_metadata = None


def _ensure_loaded():
    global _model, _extractor, _device, _index, _metadata
    if _model is not None:
        return
    _model, _extractor, _device = load_search_model(DEFAULT_MODEL)
    _index, _metadata = load_index(DEFAULT_MODEL, DEFAULT_RECITER)


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    results, _, _ = search_verse(
        audio_path, _index, _metadata, _model, _extractor, _device, top_k=1
    )

    if not results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    best = results[0]
    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": None,
        "score": best["score"],
        "transcript": "",
    }


def model_size() -> int:
    """HuBERT-base model + FAISS index."""
    total = 360 * 1024 * 1024  # HuBERT-base ~360MB
    # FAISS index on disk
    index_dir = Path(__file__).parent / "indices"
    if index_dir.exists():
        total += sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    return total
