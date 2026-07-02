"""
New Models Benchmark for Offline Quran Verse Identification.

Tests newer/alternative ASR models NOT already in the parent benchmark:
  Priority 1:
    - openai/whisper-large-v3-turbo (HuggingFace transformers, not mlx)
    - tarteel-ai/whisper-base-ar-quran via mlx-whisper
    - distil-whisper/distil-large-v3 (distilled, faster)
  Priority 2:
    - UsefulSensors/moonshine-tiny-ar (Moonshine Arabic)
    - facebook/seamless-m4t-v2-large (Meta multilingual)
    - facebook/mms-1b-all (Meta 1107-language ASR)
  Priority 3:
    - Nuwaisir/Quran_speech_recognizer (wav2vec2 Quran fine-tuned)
    - HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Reuse parent project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# ── Test cases ──────────────────────────────────────────────────────────────

TEST_AUDIO_DIR = PROJECT_ROOT / "data" / "test_audio"
REF_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_audio"

TEST_CASES = [
    # User recordings
    {
        "file": str(TEST_AUDIO_DIR / "nisa_51.m4a"),
        "expected_surah": 4,
        "expected_ayah": 51,
        "description": "An-Nisa:51 (user recording)",
    },
    {
        "file": str(TEST_AUDIO_DIR / "ikhlas_2_3.m4a"),
        "expected_surah": 112,
        "expected_ayah": 2,
        "description": "Al-Ikhlas:2-3 (user, multi-ayah)",
    },
    # Reference audio (Alafasy)
    {
        "file": str(REF_AUDIO_DIR / "001001.mp3"),
        "expected_surah": 1,
        "expected_ayah": 1,
        "description": "Al-Fatiha:1 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "001002.mp3"),
        "expected_surah": 1,
        "expected_ayah": 2,
        "description": "Al-Fatiha:2 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "002255.mp3"),
        "expected_surah": 2,
        "expected_ayah": 255,
        "description": "Ayat al-Kursi (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "112001.mp3"),
        "expected_surah": 112,
        "expected_ayah": 1,
        "description": "Al-Ikhlas:1 (ref)",
    },
    {
        "file": str(REF_AUDIO_DIR / "036001.mp3"),
        "expected_surah": 36,
        "expected_ayah": 1,
        "description": "Ya-Sin:1 (ref)",
    },
]

db = QuranDB()

# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate_transcription(text: str, tc: dict, elapsed: float) -> dict:
    """Evaluate a transcription against expected surah:ayah."""
    text_clean = normalize_arabic(text)
    match = db.match_verse(text)

    correct = (
        match is not None
        and match["surah"] == tc["expected_surah"]
        and match["ayah"] == tc["expected_ayah"]
    )

    return {
        "test": tc["description"],
        "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
        "transcription_raw": text,
        "transcription_clean": text_clean,
        "match": (
            f"{match['surah_name_en']} ({match['surah']}):{match['ayah']}"
            if match
            else "NO MATCH"
        ),
        "match_score": round(match["score"], 4) if match else 0.0,
        "correct": correct,
        "time_s": round(elapsed, 3),
    }


def print_result(r: dict):
    status = "CORRECT" if r["correct"] else "WRONG"
    print(f"  [{status}] {r['test']} (expected {r['expected']})")
    print(f"    Raw:   {r['transcription_raw'][:120]}")
    print(f"    Clean: {r['transcription_clean'][:120]}")
    print(f"    Match: {r['match']} (score: {r['match_score']})")
    print(f"    Time:  {r['time_s']}s")


def print_model_header(label: str):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")


# ── Model test functions ─────────────────────────────────────────────────────

def test_hf_whisper(model_id: str, label: str) -> dict:
    """Test a HuggingFace Whisper model (encoder-decoder) via transformers."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print_model_header(label)
    print(f"  Loading model: {model_id}")

    load_start = time.time()
    processor = WhisperProcessor.from_pretrained(model_id)
    # Force float32 to avoid MPS half-precision issues
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    # Estimate model size
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)  # float32

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        # Use generation config instead of deprecated forced_decoder_ids
        gen_kwargs = {
            "max_new_tokens": 256,
            "language": "ar",
            "task": "transcribe",
        }

        with torch.no_grad():
            predicted_ids = model.generate(input_features, **gen_kwargs)

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "whisper-hf",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


def test_mlx_whisper_tarteel(label: str) -> dict:
    """Test tarteel-ai/whisper-base-ar-quran via HF transformers (mlx-whisper has compatibility issues)."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model_id = "tarteel-ai/whisper-base-ar-quran"
    print_model_header(label)
    print(f"  Loading model: {model_id} via HF transformers")

    load_start = time.time()
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        # Tarteel has an outdated generation config; manually set decoder prompt ids
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ar", task="transcribe"
        )
        model.config.forced_decoder_ids = forced_decoder_ids

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, max_new_tokens=256
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "whisper-hf-tarteel",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


def test_moonshine_ar(label: str) -> dict:
    """Test UsefulSensors/moonshine-tiny-ar via transformers."""
    import torch
    from transformers import MoonshineForConditionalGeneration, AutoProcessor

    model_id = "UsefulSensors/moonshine-tiny-ar"
    print_model_header(label)
    print(f"  Loading model: {model_id}")

    load_start = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32  # MPS doesn't support float16 well for all ops
    processor = AutoProcessor.from_pretrained(model_id)
    model = MoonshineForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device).to(torch_dtype)
    model.eval()
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        sr = processor.feature_extractor.sampling_rate
        inputs = processor(audio, return_tensors="pt", sampling_rate=sr)
        inputs = inputs.to(device)

        # Hallucination prevention: limit max tokens based on audio length
        token_limit_factor = 13 / sr
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            seq_lens = inputs.attention_mask.sum(dim=-1)
            max_length = int((seq_lens * token_limit_factor).max().item())
            max_length = max(max_length, 50)  # minimum 50 tokens
        else:
            max_length = 256

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=max_length)

        text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "moonshine",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


def test_seamless_m4t(label: str) -> dict:
    """Test facebook/seamless-m4t-v2-large for Arabic ASR."""
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    model_id = "facebook/seamless-m4t-v2-large"
    print_model_header(label)
    print(f"  Loading model: {model_id}")

    load_start = time.time()
    # Try loading with sp_model_kwargs to handle tokenizer issues
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"  Processor load failed: {e}")
        print(f"  Trying with legacy sentencepiece...")
        processor = AutoProcessor.from_pretrained(
            model_id, sp_model_kwargs={"enable_sampling": False}
        )

    # Use the dedicated S2T model (smaller memory footprint)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # SeamlessM4T is large; keep on CPU if MPS causes issues
    try:
        model = model.to(device)
    except Exception:
        device = "cpu"
        model = model.to(device)
    model.eval()
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        audio_inputs = processor(
            audio=audio, return_tensors="pt", sampling_rate=16000
        )
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            # S2T submodel doesn't need generate_speech=False
            output_tokens = model.generate(
                **audio_inputs, tgt_lang="arb"
            )

        text = processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        ).strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "seamless-m4t",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


def test_mms(label: str) -> dict:
    """Test facebook/mms-1b-all with Arabic adapter."""
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    model_id = "facebook/mms-1b-all"
    print_model_header(label)
    print(f"  Loading model: {model_id}")

    load_start = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # Load Arabic adapter
    processor.tokenizer.set_target_lang("ara")
    model.load_adapter("ara")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_ids = torch.argmax(outputs.logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0].strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "wav2vec2-mms",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


def test_wav2vec2_quran(model_id: str, label: str) -> dict:
    """Test a wav2vec2-based Quran ASR model."""
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

    print_model_header(label)
    print(f"  Loading model: {model_id}")

    load_start = time.time()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s on {device}")

    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)

    results = []
    for tc in TEST_CASES:
        audio = load_audio(tc["file"])

        start = time.time()
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0].strip()
        elapsed = time.time() - start

        r = evaluate_transcription(text, tc, elapsed)
        results.append(r)
        print_result(r)

    return {
        "model": label,
        "model_id": model_id,
        "type": "wav2vec2-quran",
        "param_count": param_count,
        "size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 1),
        "results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    # Verify test files exist
    for tc in TEST_CASES:
        if not Path(tc["file"]).exists():
            print(f"WARNING: Test file not found: {tc['file']}")

    models_to_test = [
        # Priority 1
        (
            "test_hf_whisper",
            {"model_id": "openai/whisper-large-v3-turbo", "label": "Whisper Large-v3-Turbo (HF)"},
        ),
        (
            "test_mlx_whisper_tarteel",
            {"label": "Tarteel Whisper Base (mlx-whisper)"},
        ),
        (
            "test_hf_whisper",
            {"model_id": "distil-whisper/distil-large-v3", "label": "Distil-Whisper Large-v3"},
        ),
        # Priority 2
        (
            "test_moonshine_ar",
            {"label": "Moonshine Tiny Arabic"},
        ),
        (
            "test_seamless_m4t",
            {"label": "SeamlessM4T-v2 Large"},
        ),
        (
            "test_mms",
            {"label": "MMS-1B-All (Arabic)"},
        ),
        # Priority 3
        (
            "test_wav2vec2_quran",
            {"model_id": "Nuwaisir/Quran_speech_recognizer", "label": "Nuwaisir Quran Recognizer"},
        ),
        (
            "test_wav2vec2_quran",
            {"model_id": "HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr", "label": "HamzaSidhu Quran ASR"},
        ),
    ]

    for func_name, kwargs in models_to_test:
        func = globals()[func_name]
        try:
            result = func(**kwargs)
            all_results.append(result)
        except Exception as e:
            label = kwargs.get("label", func_name)
            print(f"\n{'='*80}")
            print(f"  FAILED: {label}")
            print(f"{'='*80}")
            print(f"  Error: {e}")
            traceback.print_exc()
            all_results.append({
                "model": label,
                "model_id": kwargs.get("model_id", "N/A"),
                "type": "FAILED",
                "error": str(e),
                "results": [],
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'#'*80}")
    print(f"  SUMMARY")
    print(f"{'#'*80}")

    for entry in all_results:
        if entry.get("type") == "FAILED":
            print(f"  {entry['model']}: FAILED ({entry['error'][:80]})")
            continue

        results = entry["results"]
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        avg_time = sum(r["time_s"] for r in results) / max(total, 1)
        avg_score = sum(r["match_score"] for r in results) / max(total, 1)
        print(
            f"  {entry['model']}: {correct}/{total} correct, "
            f"avg {avg_time:.2f}s, avg score {avg_score:.3f}, "
            f"~{entry.get('size_mb', '?')}MB"
        )

    print()

    # Save results
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
