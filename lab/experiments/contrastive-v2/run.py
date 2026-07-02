"""Contrastive audio fingerprinting v2.

Encodes audio → 256-dim speaker-invariant embedding → nearest neighbor
search against pre-computed FAISS index of all 6,236 verse embeddings.

No ASR needed. One forward pass + nearest neighbor = verse ID.
Requires trained model from Modal: data/contrastive-v2-model/
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for model.py import

import torch
import numpy as np
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_DIR = PROJECT_ROOT / "data" / "contrastive-v2-model"
INDEX_PATH = MODEL_DIR / "verse_index.faiss"
METADATA_PATH = MODEL_DIR / "verse_metadata.json"

_model = None
_index = None
_metadata = None
_db = None
_device = None


def _ensure_loaded():
    global _model, _index, _metadata, _db, _device
    if _model is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"No trained contrastive model at {MODEL_DIR}. "
            "Run Modal training first: modal run scripts/train_contrastive_v2_modal.py"
        )

    import faiss
    from model import QuranCLAPv2

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load trained model (audio encoder + projection only needed at inference)
    print(f"Loading contrastive-v2 model from {MODEL_DIR}...")
    _model = QuranCLAPv2(freeze_audio=True, freeze_text=True)
    state_dict = torch.load(MODEL_DIR / "best_model.pt", map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.eval()
    _model.to(_device)

    # Load FAISS index
    print(f"Loading FAISS index from {INDEX_PATH}...")
    _index = faiss.read_index(str(INDEX_PATH))

    # Load metadata (maps index position → surah/ayah)
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)

    _db = QuranDB()
    print(f"Contrastive-v2 loaded: {_index.ntotal} verses indexed")


def transcribe(audio_path: str) -> str:
    """No transcript for embedding-based approach."""
    return ""


def predict(audio_path: str) -> dict:
    _ensure_loaded()

    audio = load_audio(audio_path)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        embedding = _model.encode_audio(audio_tensor)  # (1, 256)

    query = embedding.cpu().numpy().astype(np.float32)

    # Search top-5 nearest verses
    distances, indices = _index.search(query, 5)

    if indices[0][0] == -1:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    best_idx = int(indices[0][0])
    best_distance = float(distances[0][0])
    meta = _metadata[best_idx]

    # Convert L2 distance to confidence (cosine similarity for normalized vectors)
    # For normalized vectors: L2^2 = 2 - 2*cos_sim → cos_sim = 1 - L2^2/2
    confidence = max(0.0, 1.0 - best_distance / 2.0)

    return {
        "surah": meta["surah"],
        "ayah": meta["ayah"],
        "ayah_end": None,
        "score": round(confidence, 4),
        "transcript": "",
    }


def model_size() -> int:
    """Audio encoder (~360MB) + projection (~1MB) + FAISS index (~6MB)."""
    if MODEL_DIR.exists():
        return sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    return 367 * 1024 * 1024  # estimate
