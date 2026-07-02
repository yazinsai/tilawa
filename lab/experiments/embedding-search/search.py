#!/usr/bin/env python3
"""Query the FAISS index with audio to find matching Quran verses."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faiss

# Add parent project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.audio import load_audio

INDICES_DIR = Path(__file__).resolve().parent / "indices"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_search_model(model_key: str = "hubert"):
    """Load model for embedding extraction during search."""
    from transformers import HubertModel, Wav2Vec2Model, AutoFeatureExtractor

    MODEL_CONFIGS = {
        "hubert": ("facebook/hubert-base-ls960", HubertModel),
        "wav2vec2": ("facebook/wav2vec2-large-xlsr-53", Wav2Vec2Model),
    }

    model_name, model_class = MODEL_CONFIGS[model_key]
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    device = get_device()
    model = model.to(device)
    model.eval()
    return model, extractor, device


def load_index(model_key: str = "hubert", reciter: str = "alafasy"):
    """Load FAISS index and metadata from disk."""
    prefix = f"{model_key}_{reciter}"
    index_path = INDICES_DIR / f"{prefix}.index"
    meta_path = INDICES_DIR / f"{prefix}_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Loaded index: {index.ntotal} vectors ({model_key}/{reciter})")
    return index, metadata


def extract_query_embedding(audio_path: str, model, extractor, device) -> np.ndarray:
    """Extract embedding from query audio file."""
    audio = load_audio(audio_path, sr=SAMPLE_RATE)

    # Truncate if too long
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden = outputs.last_hidden_state
        embedding = hidden.mean(dim=1).squeeze(0)

    emb = embedding.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def search_verse(
    audio_path: str,
    index: faiss.Index,
    metadata: list[dict],
    model,
    extractor,
    device,
    top_k: int = 5,
) -> tuple[list[dict], float, float]:
    """Search for matching verses given audio input.

    Returns:
        results: list of top-k matches with surah, ayah, name, score
        embed_time: time to extract embedding (seconds)
        search_time: time for FAISS search (seconds)
    """
    # Extract embedding
    t0 = time.time()
    embedding = extract_query_embedding(audio_path, model, extractor, device)
    embed_time = time.time() - t0

    # Search
    t1 = time.time()
    query = embedding.reshape(1, -1)
    distances, indices = index.search(query, top_k)
    search_time = time.time() - t1

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            "surah": meta["surah"],
            "ayah": meta["ayah"],
            "surah_name_en": meta["surah_name_en"],
            "score": float(dist),
            "filename": meta.get("filename", ""),
        })

    return results, embed_time, search_time


def main():
    """Interactive search mode."""
    model_key = "hubert"
    reciter = "alafasy"
    top_k = 5

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_key = sys.argv[idx + 1]

    if "--reciter" in sys.argv:
        idx = sys.argv.index("--reciter")
        reciter = sys.argv[idx + 1]

    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k")
        top_k = int(sys.argv[idx + 1])

    # Get audio path from args
    audio_files = [a for a in sys.argv[1:] if not a.startswith("--") and
                   (a.endswith(".mp3") or a.endswith(".m4a") or a.endswith(".wav"))]

    if not audio_files:
        print("Usage: python search.py <audio_file> [--model hubert|wav2vec2] [--top-k 5]")
        sys.exit(1)

    # Load model and index
    print(f"Loading model: {model_key}...")
    model, extractor, device = load_search_model(model_key)
    index, metadata = load_index(model_key, reciter)

    for audio_path in audio_files:
        print(f"\nSearching: {audio_path}")
        print("-" * 50)

        results, embed_time, search_time = search_verse(
            audio_path, index, metadata, model, extractor, device, top_k
        )

        for i, r in enumerate(results, 1):
            print(f"  #{i}: {r['surah_name_en']} {r['surah']}:{r['ayah']} "
                  f"(score: {r['score']:.4f})")

        print(f"  Embedding: {embed_time*1000:.0f}ms | Search: {search_time*1000:.2f}ms | "
              f"Total: {(embed_time+search_time)*1000:.0f}ms")


if __name__ == "__main__":
    main()
