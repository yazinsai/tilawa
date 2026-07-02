"""
Evaluate our LoRA adapter against user test recordings.
Compares: base whisper-base vs whisper-base + our LoRA adapter.
"""
import sys
import time
import warnings
import re

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from Levenshtein import ratio
from pathlib import Path

ADAPTER_DIR = Path(__file__).parent.parent / "data" / "lora-adapter"
TEST_DIR = Path(__file__).parent.parent / "data" / "test_audio"

db = QuranDB()

TEST_CASES = [
    {
        "file": "nisa_51.m4a",
        "expected_surah": 3,
        "expected_ayah": 23,
        "description": "Aal-i-Imraan:23",
    },
    {
        "file": "ikhlas_2_3.m4a",
        "expected_surah": 112,
        "expected_ayah": 2,
        "description": "Al-Ikhlas:2-3 (multi-ayah)",
    },
]


def clean_text(text):
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def match_with_pairs(text_clean):
    """Match against single verses AND concatenated pairs."""
    # Single verse match
    single_best = None
    single_best_score = 0.0
    for v in db.verses:
        score = ratio(text_clean, v["text_clean"])
        if score > single_best_score:
            single_best_score = score
            single_best = {**v, "score": score, "type": "single"}

    # Pair match
    pair_best = None
    pair_best_score = 0.0
    for surah_num in range(1, 115):
        surah_verses = db.get_surah(surah_num)
        for i in range(len(surah_verses) - 1):
            v1 = surah_verses[i]
            v2 = surah_verses[i + 1]
            concat = v1["text_clean"] + " " + v2["text_clean"]
            score = ratio(text_clean, concat)
            if score > pair_best_score:
                pair_best_score = score
                pair_best = {
                    **v1,
                    "score": score,
                    "type": f"pair {v1['ayah']}-{v2['ayah']}",
                }

    # Return best overall
    if pair_best and pair_best_score > single_best_score:
        return pair_best
    return single_best


def test_model(model, processor, label, device):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for tc in TEST_CASES:
        audio = load_audio(str(TEST_DIR / tc["file"]))
        start = time.time()

        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, language="ar", task="transcribe", max_new_tokens=256
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        text = clean_text(text)
        elapsed = time.time() - start

        text_clean = normalize_arabic(text)
        match = match_with_pairs(text_clean)

        correct = (
            match
            and match["surah"] == tc["expected_surah"]
            and match["ayah"] == tc["expected_ayah"]
        )
        status = "PASS" if correct else "FAIL"

        print(f"\n  [{status}] {tc['description']}")
        print(f"  Raw:   {text}")
        print(f"  Clean: {text_clean}")
        if match:
            print(
                f"  Match: {match['surah_name_en']} ({match['surah']}):{match['ayah']} "
                f"(score: {match['score']:.3f}, {match['type']})"
            )
        else:
            print("  Match: NONE")
        print(f"  Time:  {elapsed:.1f}s")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Test 1: Base whisper-base (no LoRA) ──
    print("\nLoading base model...")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base_model.eval()
    base_model = base_model.to(device)
    test_model(base_model, processor, "Whisper Base (no LoRA)", device)
    del base_model
    torch.mps.empty_cache() if device == "mps" else None

    # ── Test 2: whisper-base + our LoRA ──
    if not ADAPTER_DIR.exists():
        print(f"\nLoRA adapter not found at {ADAPTER_DIR}")
        print("Run train_lora.py first!")
        return

    print("\nLoading base model + LoRA adapter...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    model = model.to(device)
    test_model(model, processor, "Whisper Base + Our LoRA", device)


if __name__ == "__main__":
    main()
