"""
Benchmark multiple Quran audio recognition approaches against real recordings.

Approaches tested:
1. mlx-whisper with standard whisper models (base, small, medium, large-v3)
2. mlx-whisper with tarteel-ai fine-tuned model
3. HuggingFace transformers + PEFT with KheemP's LoRA adapter
4. Whisper large-v3-turbo (good speed/accuracy tradeoff)
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# ── Test cases ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "file": "nisa_51.m4a",
        "expected_surah": 4,
        "expected_ayah": 51,
        "description": "An-Nisa:51",
        "expected_text": "الم تر الي الذين اوتوا نصيبا من الكتاب يومنون بالجبت والطاغوت ويقولون للذين كفروا هولاء اهدي من الذين امنوا سبيلا",
    },
    {
        "file": "ikhlas_2_3.m4a",
        "expected_surah": 112,
        "expected_ayah": 2,  # starts at ayah 2
        "description": "Al-Ikhlas:2-3 (multi-ayah)",
        "expected_text": "الله الصمد لم يلد ولم يولد",
    },
]

AUDIO_DIR = Path(__file__).parent.parent / "data" / "test_audio"
db = QuranDB()


def test_mlx_whisper(model_id: str, label: str):
    """Test an mlx-whisper model."""
    import mlx_whisper

    results = []
    for tc in TEST_CASES:
        audio_path = AUDIO_DIR / tc["file"]
        audio = load_audio(str(audio_path))

        start = time.time()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_id,
            language="ar",
        )
        elapsed = time.time() - start

        text = result.get("text", "").strip()
        text_clean = normalize_arabic(text)

        # Match against quran DB
        match = db.match_verse(text)

        results.append({
            "test": tc["description"],
            "transcription_raw": text,
            "transcription_clean": text_clean,
            "match": f"{match['surah_name_en']} ({match['surah']}):{match['ayah']}" if match else "NO MATCH",
            "match_score": f"{match['score']:.3f}" if match else "N/A",
            "correct": match is not None and match["surah"] == tc["expected_surah"] and match["ayah"] == tc["expected_ayah"],
            "time_s": f"{elapsed:.2f}",
        })

    return label, results


def test_hf_lora(base_model: str, lora_adapter: str, label: str):
    """Test a HuggingFace model with LoRA adapter."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    print(f"  Loading base model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)

    print(f"  Loading LoRA adapter: {lora_adapter}")
    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()

    # Use MPS if available (Apple Silicon)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    results = []
    for tc in TEST_CASES:
        audio_path = AUDIO_DIR / tc["file"]
        audio = load_audio(str(audio_path))

        start = time.time()
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start

        text_clean = normalize_arabic(text)
        match = db.match_verse(text)

        results.append({
            "test": tc["description"],
            "transcription_raw": text,
            "transcription_clean": text_clean,
            "match": f"{match['surah_name_en']} ({match['surah']}):{match['ayah']}" if match else "NO MATCH",
            "match_score": f"{match['score']:.3f}" if match else "N/A",
            "correct": match is not None and match["surah"] == tc["expected_surah"] and match["ayah"] == tc["expected_ayah"],
            "time_s": f"{elapsed:.2f}",
        })

    return label, results


def test_hf_model(model_id: str, label: str):
    """Test a HuggingFace model directly (no LoRA)."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"  Loading model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    results = []
    for tc in TEST_CASES:
        audio_path = AUDIO_DIR / tc["file"]
        audio = load_audio(str(audio_path))

        start = time.time()
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start

        text_clean = normalize_arabic(text)
        match = db.match_verse(text)

        results.append({
            "test": tc["description"],
            "transcription_raw": text,
            "transcription_clean": text_clean,
            "match": f"{match['surah_name_en']} ({match['surah']}):{match['ayah']}" if match else "NO MATCH",
            "match_score": f"{match['score']:.3f}" if match else "N/A",
            "correct": match is not None and match["surah"] == tc["expected_surah"] and match["ayah"] == tc["expected_ayah"],
            "time_s": f"{elapsed:.2f}",
        })

    return label, results


def print_results(label, results):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    for r in results:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n  [{status}] {r['test']}")
        print(f"  Raw:   {r['transcription_raw']}")
        print(f"  Clean: {r['transcription_clean']}")
        print(f"  Match: {r['match']} (score: {r['match_score']})")
        print(f"  Time:  {r['time_s']}s")


def main():
    approaches = []

    # ── Approach 1: mlx-whisper with standard models ──
    mlx_models = [
        ("mlx-community/whisper-base-mlx", "MLX Whisper Base"),
        ("mlx-community/whisper-small-mlx", "MLX Whisper Small"),
        ("mlx-community/whisper-medium-mlx", "MLX Whisper Medium"),
        ("mlx-community/whisper-large-v3-mlx", "MLX Whisper Large-v3"),
        ("mlx-community/whisper-large-v3-turbo", "MLX Whisper Large-v3-Turbo"),
    ]

    for model_id, label in mlx_models:
        print(f"\nTesting: {label} ({model_id})")
        try:
            result = test_mlx_whisper(model_id, label)
            approaches.append(result)
            print_results(*result)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Approach 2: Tarteel's fine-tuned model (baseline) ──
    print(f"\nTesting: Tarteel Whisper Base (HF transformers)")
    try:
        result = test_hf_model("tarteel-ai/whisper-base-ar-quran", "Tarteel Whisper Base (HF)")
        approaches.append(result)
        print_results(*result)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Approach 3: KheemP's LoRA adapter ──
    print(f"\nTesting: KheemP LoRA (whisper-base + quran-lora)")
    try:
        result = test_hf_lora(
            "openai/whisper-base",
            "KheemP/whisper-base-quran-lora",
            "KheemP LoRA (Whisper Base + Quran LoRA)"
        )
        approaches.append(result)
        print_results(*result)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Summary ──
    print(f"\n\n{'#'*80}")
    print(f"  SUMMARY")
    print(f"{'#'*80}")
    for label, results in approaches:
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        print(f"  {label}: {correct}/{total} correct")
    print()

    # Save full results
    out = Path(__file__).parent.parent / "data" / "benchmark_results.json"
    serializable = []
    for label, results in approaches:
        serializable.append({"approach": label, "results": results})
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {out}")


if __name__ == "__main__":
    main()
