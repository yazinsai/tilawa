"""
Evaluation script for QuranCLAP contrastive model.

Evaluates verse identification by:
1. Pre-computing text embeddings for all 6,236 Quran verses
2. For each test audio, computing audio embedding
3. Finding nearest verse text embedding via cosine similarity
4. Reporting top-1, top-3, top-5 accuracy

Also evaluates zero-shot (no training): raw HuBERT + AraBERT embeddings
without projection heads, to establish baseline performance.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model import QuranCLAP, QuranCLAPZeroShot, get_tokenizer
from dataset import load_quran_verses, TARGET_SR

from shared.audio import load_audio
from shared.normalizer import normalize_arabic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"

# Test cases with known audio files
TEST_CASES = [
    {
        "file": str(PROJECT_ROOT / "data" / "test_audio" / "nisa_51.m4a"),
        "expected_surah": 4,
        "expected_ayah": 51,
        "description": "Surah An-Nisa verse 51 (user recording)",
    },
    {
        "file": str(PROJECT_ROOT / "data" / "test_audio" / "ikhlas_2_3.m4a"),
        "expected_surah": 112,
        "expected_ayah": 2,
        "description": "Surah Al-Ikhlas verses 2-3 (user recording, multi-verse)",
    },
    {
        "file": str(PROJECT_ROOT / "data" / "reference_audio" / "001001.mp3"),
        "expected_surah": 1,
        "expected_ayah": 1,
        "description": "Al-Fatiha v1 (Alafasy reference)",
    },
    {
        "file": str(PROJECT_ROOT / "data" / "reference_audio" / "002255.mp3"),
        "expected_surah": 2,
        "expected_ayah": 255,
        "description": "Ayat al-Kursi (Alafasy reference)",
    },
    {
        "file": str(PROJECT_ROOT / "data" / "reference_audio" / "112001.mp3"),
        "expected_surah": 112,
        "expected_ayah": 1,
        "description": "Al-Ikhlas v1 (Alafasy reference)",
    },
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def precompute_text_embeddings(
    model,
    tokenizer,
    verses: dict,
    device: torch.device,
    batch_size: int = 32,
    max_text_len: int = 128,
    use_method: str = "encode_text",
):
    """
    Pre-compute text embeddings for all verses.

    Returns:
        embeddings: [N, embed_dim] tensor
        verse_keys: list of (surah, ayah) tuples in same order
    """
    model.eval()

    verse_keys = sorted(verses.keys())
    texts = [verses[k]["text_uthmani"] for k in verse_keys]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        encode_fn = getattr(model, use_method)
        embeds = encode_fn(input_ids, attention_mask)
        all_embeddings.append(embeds.cpu())

        if (i // batch_size) % 20 == 0:
            print(f"  Text embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0), verse_keys


@torch.no_grad()
def encode_audio_file(
    model,
    audio_path: str,
    device: torch.device,
    max_seconds: float = 30.0,
    use_method: str = "encode_audio",
):
    """Encode a single audio file to an embedding vector."""
    model.eval()

    audio_np = load_audio(audio_path, sr=TARGET_SR)

    # Truncate if too long
    max_samples = int(max_seconds * TARGET_SR)
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]

    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)

    encode_fn = getattr(model, use_method)
    embedding = encode_fn(audio_tensor)
    return embedding.cpu().squeeze(0)


def find_nearest_verses(
    audio_embedding: torch.Tensor,
    text_embeddings: torch.Tensor,
    verse_keys: list,
    top_k: int = 10,
):
    """
    Find the nearest verse text embeddings to an audio embedding.

    Returns list of (surah, ayah, similarity_score) tuples.
    """
    # Cosine similarity (embeddings are already L2-normalized)
    similarities = audio_embedding @ text_embeddings.T

    top_indices = similarities.argsort(descending=True)[:top_k]
    results = []
    for idx in top_indices:
        surah, ayah = verse_keys[idx]
        score = similarities[idx].item()
        results.append((surah, ayah, score))

    return results


def evaluate_test_cases(
    model,
    tokenizer,
    device: torch.device,
    model_name: str = "model",
    use_method_audio: str = "encode_audio",
    use_method_text: str = "encode_text",
):
    """
    Run evaluation on test cases and compute accuracy.

    Returns dict with results.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    # Load verses
    print("Loading Quran verses...")
    verses = load_quran_verses()
    print(f"  {len(verses)} verses loaded")

    # Pre-compute text embeddings
    print("Pre-computing text embeddings for all verses...")
    t0 = time.time()
    text_embeddings, verse_keys = precompute_text_embeddings(
        model, tokenizer, verses, device,
        use_method=use_method_text,
    )
    text_time = time.time() - t0
    print(f"  Text embeddings shape: {text_embeddings.shape}")
    print(f"  Time: {text_time:.1f}s")

    # Evaluate test cases
    print(f"\nEvaluating {len(TEST_CASES)} test cases...")
    results = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    for tc in TEST_CASES:
        if not os.path.exists(tc["file"]):
            print(f"\n  SKIP: {tc['description']} (file not found: {tc['file']})")
            continue

        print(f"\n  Test: {tc['description']}")
        print(f"  Expected: Surah {tc['expected_surah']}:{tc['expected_ayah']}")

        t0 = time.time()
        audio_emb = encode_audio_file(
            model, tc["file"], device,
            use_method=use_method_audio,
        )
        audio_time = time.time() - t0

        nearest = find_nearest_verses(audio_emb, text_embeddings, verse_keys, top_k=10)

        expected = (tc["expected_surah"], tc["expected_ayah"])
        top1_match = nearest[0][:2] == expected
        top3_match = any(n[:2] == expected for n in nearest[:3])
        top5_match = any(n[:2] == expected for n in nearest[:5])

        if top1_match:
            top1_correct += 1
        if top3_match:
            top3_correct += 1
        if top5_match:
            top5_correct += 1

        result = {
            "description": tc["description"],
            "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
            "top1": f"{nearest[0][0]}:{nearest[0][1]} (score={nearest[0][2]:.4f})",
            "top1_correct": top1_match,
            "top3_correct": top3_match,
            "top5_correct": top5_match,
            "audio_encode_time_s": audio_time,
        }
        results.append(result)

        # Print top-5 results
        status = "CORRECT" if top1_match else "WRONG"
        print(f"  Result: {status}")
        print(f"  Audio encode time: {audio_time:.2f}s")
        print(f"  Top-5 predictions:")
        for i, (s, a, score) in enumerate(nearest[:5]):
            verse_text = verses[(s, a)]["text_uthmani"][:50]
            marker = " <-- EXPECTED" if (s, a) == expected else ""
            print(f"    {i+1}. Surah {s}:{a} (score={score:.4f}) {verse_text}...{marker}")

    total = len(results)
    if total == 0:
        print("\nNo test cases could be evaluated.")
        return {"error": "no test cases"}

    summary = {
        "model": model_name,
        "total_test_cases": total,
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "top5_accuracy": top5_correct / total,
        "text_embedding_time_s": text_time,
        "text_embedding_shape": list(text_embeddings.shape),
        "results": results,
    }

    print(f"\n{'='*70}")
    print(f"Summary for {model_name}:")
    print(f"  Top-1 Accuracy: {summary['top1_accuracy']:.1%} ({top1_correct}/{total})")
    print(f"  Top-3 Accuracy: {summary['top3_accuracy']:.1%} ({top3_correct}/{total})")
    print(f"  Top-5 Accuracy: {summary['top5_accuracy']:.1%} ({top5_correct}/{total})")
    print(f"  Text embedding time: {text_time:.1f}s for {len(verses)} verses")
    print(f"{'='*70}")

    return summary


def evaluate_zero_shot(device: torch.device):
    """
    Evaluate zero-shot: raw HuBERT + AraBERT embeddings, no projection heads.
    """
    print("\nLoading zero-shot model (raw HuBERT + AraBERT)...")
    model = QuranCLAPZeroShot()
    model = model.to(device)
    tokenizer = get_tokenizer()

    return evaluate_test_cases(
        model, tokenizer, device,
        model_name="Zero-Shot (HuBERT + AraBERT, no projection)",
        use_method_audio="encode_audio",
        use_method_text="encode_text",
    )


def evaluate_trained(checkpoint_path: str, device: torch.device, embed_dim: int = 256):
    """
    Evaluate a trained QuranCLAP model from a checkpoint.
    """
    print(f"\nLoading trained model from: {checkpoint_path}")
    model = QuranCLAP(embed_dim=embed_dim, freeze_audio=True, freeze_text=True)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    tokenizer = get_tokenizer()

    return evaluate_test_cases(
        model, tokenizer, device,
        model_name=f"Trained ({Path(checkpoint_path).stem})",
    )


def evaluate_reference_audio_retrieval(
    model,
    tokenizer,
    device: torch.device,
    model_name: str = "model",
    use_method_audio: str = "encode_audio",
    use_method_text: str = "encode_text",
):
    """
    Evaluate using all reference audio files (23 files from EveryAyah).
    This gives us a larger test set.
    """
    print(f"\n{'='*70}")
    print(f"Reference Audio Retrieval Evaluation: {model_name}")
    print(f"{'='*70}")

    ref_dir = PROJECT_ROOT / "data" / "reference_audio"
    audio_files = sorted(ref_dir.glob("*.mp3"))

    if not audio_files:
        print("No reference audio files found.")
        return {}

    # Load verses and precompute text embeddings
    verses = load_quran_verses()
    text_embeddings, verse_keys = precompute_text_embeddings(
        model, tokenizer, verses, device,
        use_method=use_method_text,
    )

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0

    for audio_file in audio_files:
        name = audio_file.stem
        if len(name) != 6 or not name.isdigit():
            continue

        surah = int(name[:3])
        ayah = int(name[3:])
        expected = (surah, ayah)

        if expected not in verses:
            continue

        audio_emb = encode_audio_file(
            model, str(audio_file), device,
            use_method=use_method_audio,
        )

        nearest = find_nearest_verses(audio_emb, text_embeddings, verse_keys, top_k=10)

        top1 = nearest[0][:2] == expected
        top3 = any(n[:2] == expected for n in nearest[:3])
        top5 = any(n[:2] == expected for n in nearest[:5])
        top10 = any(n[:2] == expected for n in nearest[:10])

        if top1: top1_correct += 1
        if top3: top3_correct += 1
        if top5: top5_correct += 1
        if top10: top10_correct += 1
        total += 1

        status = "OK" if top1 else f"MISS (got {nearest[0][0]}:{nearest[0][1]})"
        print(f"  {surah}:{ayah} -> {status}")

    print(f"\nReference Audio Results ({total} files):")
    print(f"  Top-1:  {top1_correct}/{total} = {top1_correct/max(total,1):.1%}")
    print(f"  Top-3:  {top3_correct}/{total} = {top3_correct/max(total,1):.1%}")
    print(f"  Top-5:  {top5_correct}/{total} = {top5_correct/max(total,1):.1%}")
    print(f"  Top-10: {top10_correct}/{total} = {top10_correct/max(total,1):.1%}")

    return {
        "total": total,
        "top1": top1_correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
        "top5": top5_correct / max(total, 1),
        "top10": top10_correct / max(total, 1),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate QuranCLAP model")
    parser.add_argument("--mode", default="zero_shot",
                        choices=["zero_shot", "trained", "both", "all_ref"],
                        help="Evaluation mode")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained checkpoint (for 'trained' mode)")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--device", default=None, help="Force device")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    all_results = {}

    if args.mode in ("zero_shot", "both"):
        results = evaluate_zero_shot(device)
        all_results["zero_shot"] = results

        # Also run reference audio retrieval
        print("\n--- Also running reference audio retrieval for zero-shot ---")
        model = QuranCLAPZeroShot().to(device)
        tokenizer = get_tokenizer()
        ref_results = evaluate_reference_audio_retrieval(
            model, tokenizer, device,
            model_name="Zero-Shot",
        )
        all_results["zero_shot_ref"] = ref_results

    if args.mode in ("trained", "both"):
        checkpoint = args.checkpoint
        if not checkpoint:
            # Try to find the best checkpoint
            for name in ["best_model.pt", "phase1_final.pt", "final_model.pt"]:
                path = CHECKPOINTS_DIR / name
                if path.exists():
                    checkpoint = str(path)
                    break

        if checkpoint and os.path.exists(checkpoint):
            results = evaluate_trained(checkpoint, device, args.embed_dim)
            all_results["trained"] = results

            # Also run reference audio retrieval
            print("\n--- Also running reference audio retrieval for trained model ---")
            model = QuranCLAP(embed_dim=args.embed_dim, freeze_audio=True, freeze_text=True)
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)
            tokenizer = get_tokenizer()
            ref_results = evaluate_reference_audio_retrieval(
                model, tokenizer, device,
                model_name="Trained",
            )
            all_results["trained_ref"] = ref_results
        else:
            print(f"\nNo trained checkpoint found. Skipping trained evaluation.")

    if args.mode == "all_ref":
        # Just run reference audio evaluation for zero-shot
        model = QuranCLAPZeroShot().to(device)
        tokenizer = get_tokenizer()
        ref_results = evaluate_reference_audio_retrieval(
            model, tokenizer, device,
            model_name="Zero-Shot",
        )
        all_results["zero_shot_ref"] = ref_results

    # Save results
    results_path = EXPERIMENT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
