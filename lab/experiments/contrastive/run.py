import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model import QuranCLAP, get_tokenizer
from evaluate import precompute_text_embeddings, encode_audio_file, find_nearest_verses
from dataset import load_quran_verses

EXPERIMENT_DIR = Path(__file__).parent
CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"

_model = None
_text_embeddings = None
_verse_keys = None


def _ensure_loaded():
    global _model, _text_embeddings, _verse_keys
    if _model is not None:
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _model = QuranCLAP()

    if CHECKPOINT.exists():
        state = torch.load(str(CHECKPOINT), map_location=device, weights_only=True)
        _model.load_state_dict(state)

    _model.to(device)
    _model.eval()

    tokenizer = get_tokenizer()
    verses = load_quran_verses()
    _text_embeddings, _verse_keys = precompute_text_embeddings(
        _model, tokenizer, verses, device
    )


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    device = next(_model.parameters()).device
    audio_emb = encode_audio_file(_model, audio_path, device)
    results = find_nearest_verses(audio_emb, _text_embeddings, _verse_keys, top_k=1)

    if not results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    surah, ayah, score = results[0]
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": None,
        "score": score,
        "transcript": "",
    }


def model_size() -> int:
    """HuBERT-base + AraBERT + projection heads + checkpoint."""
    total = 360 * 1024 * 1024  # HuBERT-base
    total += 540 * 1024 * 1024  # AraBERT
    if CHECKPOINT.exists():
        total += CHECKPOINT.stat().st_size
    return total
