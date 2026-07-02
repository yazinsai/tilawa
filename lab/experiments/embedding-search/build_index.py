#!/usr/bin/env python3
"""Extract audio embeddings and build FAISS indices for verse search."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from tqdm import tqdm

# Add parent project to path for audio loading
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.audio import load_audio

QURAN_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "quran.json"
CORPUS_DIR = Path(__file__).resolve().parent / "audio_corpus"
INDICES_DIR = Path(__file__).resolve().parent / "indices"

# Model configurations
MODELS = {
    "hubert": {
        "model_name": "facebook/hubert-base-ls960",
        "model_class": "HuBertModel",
        "extractor_class": "AutoFeatureExtractor",
        "dim": 768,
    },
    "wav2vec2": {
        "model_name": "facebook/wav2vec2-large-xlsr-53",
        "model_class": "Wav2Vec2Model",
        "extractor_class": "AutoFeatureExtractor",
        "dim": 1024,
    },
}

BATCH_SIZE = 8  # Smaller batches to reduce padding overhead
SAMPLE_RATE = 16000
# Maximum audio length in seconds (truncate longer audio to save memory)
MAX_AUDIO_SECONDS = 15  # Most verses are under 15s; truncating saves significant compute
# Save checkpoints every N items to allow resuming
CHECKPOINT_INTERVAL = 500


def get_device():
    """Get best available device.

    Note: MPS is intentionally excluded for index building because it causes
    hangs with HuBERT/wav2vec2 inference. CPU is fast enough (~200ms/verse)
    and processes 6K+ verses in ~20 minutes.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_key: str):
    """Load a model and its feature extractor."""
    from transformers import HubertModel, Wav2Vec2Model, AutoFeatureExtractor

    config = MODELS[model_key]
    print(f"Loading model: {config['model_name']}...")

    extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])

    if model_key == "hubert":
        model = HubertModel.from_pretrained(config["model_name"])
    elif model_key == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained(config["model_name"])
    else:
        raise ValueError(f"Unknown model: {model_key}")

    device = get_device()
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}, dim={config['dim']}")
    return model, extractor, device, config["dim"]


def extract_embedding(audio: np.ndarray, model, extractor, device) -> np.ndarray:
    """Extract a single embedding from audio. Returns L2-normalized embedding."""
    # Truncate very long audio
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # Mean-pool the last hidden state across time
        hidden = outputs.last_hidden_state  # (1, T, D)
        embedding = hidden.mean(dim=1).squeeze(0)  # (D,)

    emb = embedding.cpu().numpy().astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def extract_embeddings_batch(audio_list: list[np.ndarray], model, extractor, device) -> np.ndarray:
    """Extract embeddings for a batch of audio. Returns L2-normalized embeddings."""
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    truncated = [a[:max_samples] if len(a) > max_samples else a for a in audio_list]

    inputs = extractor(
        truncated,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)
    else:
        attention_mask = None

    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, D)

        if attention_mask is not None:
            # Mask padding before mean pooling
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            hidden = hidden * mask
            embeddings = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        else:
            embeddings = hidden.mean(dim=1)  # (B, D)

    embs = embeddings.cpu().numpy().astype(np.float32)
    # L2 normalize each embedding
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs = embs / norms
    return embs


def build_index_for_model(model_key: str, reciter: str = "alafasy"):
    """Build a FAISS index for a given model and reciter."""
    # Load verse list
    with open(QURAN_JSON) as f:
        verses = json.load(f)

    # Load model
    model, extractor, device, dim = load_model(model_key)

    # Find available audio files
    audio_dir = CORPUS_DIR / reciter
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    available_files = set(f.stem for f in audio_dir.glob("*.mp3"))
    print(f"Found {len(available_files)} audio files for {reciter}")

    # Filter verses to those with available audio
    verse_list = []
    for v in verses:
        filename = f"{v['surah']:03d}{v['ayah']:03d}"
        if filename in available_files:
            verse_list.append({
                "surah": v["surah"],
                "ayah": v["ayah"],
                "surah_name_en": v["surah_name_en"],
                "filename": filename,
                "audio_path": str(audio_dir / f"{filename}.mp3"),
            })

    print(f"Processing {len(verse_list)} verses with audio")

    # Check for checkpoint to resume from
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_key}_{reciter}"
    ckpt_emb_path = INDICES_DIR / f"{prefix}_checkpoint_emb.npy"
    ckpt_meta_path = INDICES_DIR / f"{prefix}_checkpoint_meta.json"
    start_idx = 0
    all_embeddings = []
    metadata = []

    if ckpt_emb_path.exists() and ckpt_meta_path.exists():
        print("Resuming from checkpoint...")
        prev_embs = np.load(str(ckpt_emb_path))
        with open(ckpt_meta_path) as f:
            prev_meta = json.load(f)
        all_embeddings.append(prev_embs)
        metadata.extend(prev_meta)
        start_idx = len(prev_meta)
        print(f"  Loaded {start_idx} embeddings from checkpoint")

    failed = 0

    # Process one-by-one to avoid padding overhead (much faster for variable-length audio)
    for i in tqdm(range(start_idx, len(verse_list)), desc=f"{model_key}/{reciter}",
                  initial=start_idx, total=len(verse_list)):
        v = verse_list[i]
        try:
            audio = load_audio(v["audio_path"], sr=SAMPLE_RATE)
            if len(audio) < SAMPLE_RATE * 0.1:  # Skip very short audio (<0.1s)
                failed += 1
                continue
            emb = extract_embedding(audio, model, extractor, device)
            all_embeddings.append(emb.reshape(1, -1))
            metadata.append({
                "surah": v["surah"],
                "ayah": v["ayah"],
                "surah_name_en": v["surah_name_en"],
                "filename": v["filename"],
            })
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"\n  Failed {v['filename']}: {e}")

        # Save checkpoint periodically
        if (i + 1) % CHECKPOINT_INTERVAL == 0 and len(all_embeddings) > 0:
            ckpt_embs = np.vstack(all_embeddings)
            np.save(str(ckpt_emb_path), ckpt_embs)
            with open(ckpt_meta_path, "w") as f:
                json.dump(metadata, f)
            print(f"\n  Checkpoint saved: {len(metadata)} embeddings")

    if not all_embeddings:
        print("No embeddings extracted!")
        return

    # Stack all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Metadata entries: {len(metadata)}")
    print(f"Failed: {failed}")

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim on normalized vectors
    index.add(embeddings)
    print(f"Index size: {index.ntotal} vectors")

    # Save everything
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_key}_{reciter}"

    index_path = INDICES_DIR / f"{prefix}.index"
    meta_path = INDICES_DIR / f"{prefix}_meta.json"
    emb_path = INDICES_DIR / f"{prefix}_embeddings.npy"

    faiss.write_index(index, str(index_path))
    print(f"Saved index: {index_path}")

    # Save metadata (already in portable format without audio_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata: {meta_path}")

    np.save(str(emb_path), embeddings)
    print(f"Saved embeddings: {emb_path}")

    # Clean up checkpoint files
    ckpt_emb_path = INDICES_DIR / f"{prefix}_checkpoint_emb.npy"
    ckpt_meta_path = INDICES_DIR / f"{prefix}_checkpoint_meta.json"
    ckpt_emb_path.unlink(missing_ok=True)
    ckpt_meta_path.unlink(missing_ok=True)

    return index, metadata, embeddings


def main():
    # Parse args
    model_keys = list(MODELS.keys())
    reciter = "alafasy"

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_key = sys.argv[idx + 1]
            if model_key in MODELS:
                model_keys = [model_key]
            else:
                print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
                sys.exit(1)

    if "--reciter" in sys.argv:
        idx = sys.argv.index("--reciter")
        if idx + 1 < len(sys.argv):
            reciter = sys.argv[idx + 1]

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Building index: {model_key} / {reciter}")
        print(f"{'='*60}")
        t0 = time.time()
        build_index_for_model(model_key, reciter)
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
