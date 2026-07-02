"""
Test script: feed a full audio recording through the same pipeline
as the WebSocket server and check if Surah Al-Asr (103:1-3) is
correctly identified.

Usage:
    python tests/test_al_asr.py [path_to_audio]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB, partial_ratio

# ---------------------------------------------------------------------------
# Constants (mirror server.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 2.0
TRIGGER_SAMPLES = int(SAMPLE_RATE * TRIGGER_SECONDS)
MAX_WINDOW_SECONDS = 10.0
MAX_WINDOW_SAMPLES = int(SAMPLE_RATE * MAX_WINDOW_SECONDS)
SILENCE_RMS_THRESHOLD = 0.005
VERSE_MATCH_THRESHOLD = 0.45
FIRST_MATCH_THRESHOLD = 0.75
RAW_TRANSCRIPT_THRESHOLD = 0.25

EXPECTED_VERSES_MAP = {
    103: [(103, 1), (103, 2), (103, 3)],
    108: [(108, 1), (108, 2), (108, 3)],
    105: [(105, 1), (105, 2), (105, 3), (105, 4), (105, 5)],
}
EXPECTED_VERSES = EXPECTED_VERSES_MAP.get(103, [])


def load_audio(path: str) -> np.ndarray:
    """Load audio file, convert to 16kHz mono float32."""
    p = Path(path)
    # Convert m4a to wav via ffmpeg if needed
    if p.suffix.lower() in (".m4a", ".mp4", ".aac", ".ogg", ".webm"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", tmp.name],
            capture_output=True,
        )
        audio, sr = sf.read(tmp.name, dtype="float32")
        Path(tmp.name).unlink(missing_ok=True)
    else:
        audio, sr = sf.read(str(p), dtype="float32")

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32)


def load_model():
    """Load FastConformer (same as server.py)."""
    import os
    import types

    # kaldialign fallback
    try:
        import kaldialign  # noqa: F401
    except Exception:
        def align(ref, hyp, eps="<eps>"):
            return list(zip(ref, hyp))
        mod = types.ModuleType("kaldialign")
        mod.align = align
        sys.modules["kaldialign"] = mod

    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils import logging as nemo_logging
    nemo_logging.set_verbosity(nemo_logging.ERROR)

    NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    LOCAL_MODEL_DIR = PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    print(f"Loading model from {source} on {device}...")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=source, map_location=device
    )
    model.eval()
    try:
        model.change_decoding_strategy(decoder_type="ctc")
    except Exception:
        pass
    print("Model loaded.")
    return model


def transcribe(model, audio: np.ndarray) -> str:
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    sf.write(str(tmp_path), audio, SAMPLE_RATE)
    try:
        try:
            outputs = model.transcribe(
                audio=[str(tmp_path)], batch_size=1,
                return_hypotheses=True, verbose=False,
            )
        except TypeError:
            outputs = model.transcribe(
                paths2audio_files=[str(tmp_path)], batch_size=1,
                return_hypotheses=True,
            )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list) and outputs:
            t = outputs[0]
        else:
            t = outputs
        if isinstance(t, str):
            return normalize_arabic(t)
        if hasattr(t, "text"):
            return normalize_arabic(t.text)
        return normalize_arabic(str(t))
    finally:
        tmp_path.unlink(missing_ok=True)


def is_silence(audio: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms < SILENCE_RMS_THRESHOLD


def simulate_server(audio: np.ndarray, model, db: QuranDB):
    """Simulate the server's rolling window exactly as the WebSocket handler does."""
    print(f"\n{'='*70}")
    print(f"Audio length: {len(audio)/SAMPLE_RATE:.1f}s")
    print(f"{'='*70}\n")

    full_audio = np.empty(0, dtype=np.float32)
    new_audio_count = 0
    last_emitted_ref = None
    last_emitted_text = ""
    emitted = []

    # Feed audio in chunks (simulating browser sending ~100ms chunks)
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms
    for i in range(0, len(audio), chunk_size):
        samples = audio[i:i + chunk_size]
        full_audio = np.concatenate([full_audio, samples])
        new_audio_count += len(samples)

        if len(full_audio) > MAX_WINDOW_SAMPLES:
            full_audio = full_audio[-MAX_WINDOW_SAMPLES:]

        if new_audio_count < TRIGGER_SAMPLES:
            continue
        new_audio_count = 0

        tail = full_audio[-TRIGGER_SAMPLES:]
        if is_silence(tail):
            print(f"  [silence at {i/SAMPLE_RATE:.1f}s]")
            continue

        text = transcribe(model, full_audio.copy())
        if not text or len(text.strip()) < 5:
            continue

        audio_len = len(full_audio) / SAMPLE_RATE
        print(f"Transcribed ({audio_len:.1f}s): {text[:100]}")

        # Skip if transcription is mostly residual from the last emitted verse
        if last_emitted_text:
            residual = partial_ratio(text, last_emitted_text)
            if residual > 0.70:
                print(f"  (residual overlap {residual:.2f} with last emitted, skipping)")
                continue

        match = db.match_verse(
            text,
            threshold=RAW_TRANSCRIPT_THRESHOLD,
            max_span=4,
            hint=last_emitted_ref,
            return_top_k=5,
        )

        hint_str = f"{last_emitted_ref[0]}:{last_emitted_ref[1]}" if last_emitted_ref else "none"

        if match:
            end = match.get("ayah_end", "")
            end_str = f"-{end}" if end else ""
            print(f"  MATCH  {match['surah']}:{match['ayah']}{end_str}  "
                  f"score={match['score']:.3f} (raw={match.get('raw_score', match['score']):.3f} "
                  f"+bonus={match.get('bonus', 0.0):.3f})  hint={hint_str}")
            for j, r in enumerate(match.get("runners_up", [])[:3], 1):
                print(f"    #{j}  {r['surah']}:{r['ayah']}  score={r['score']:.3f}  "
                      f"{r['text_clean'][:50]}")
        else:
            print(f"  NO MATCH  hint={hint_str}")

        effective_threshold = FIRST_MATCH_THRESHOLD if last_emitted_ref is None else VERSE_MATCH_THRESHOLD
        if match and match["score"] >= effective_threshold:
            ref = (match["surah"], match["ayah"])
            if ref == last_emitted_ref:
                print("  (dedup — skipped)")
                continue
            ayah_end = match.get("ayah_end")
            if ayah_end:
                # Multi-verse span: emit all verses, advance hint to end
                for a in range(ref[1], ayah_end + 1):
                    emitted.append((ref[0], a))
                    print(f"  >>> EMIT {ref[0]}:{a}")
                effective_ref = (ref[0], ayah_end)
            else:
                emitted.append(ref)
                print(f"  >>> EMIT {ref[0]}:{ref[1]}")
                effective_ref = ref
            last_emitted_ref = effective_ref
            last_emitted_text = normalize_arabic(
                match.get("text_clean", "") or ""
            )
            full_audio = tail.copy()
        else:
            score = round(match["score"], 2) if match else 0.0
            print(f"  (below threshold {effective_threshold}, score={score})")

    return emitted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=str(PROJECT_ROOT / "data" / "test-al-asr.m4a"))
    parser.add_argument("--surah", type=int, default=0, help="Expected surah number")
    args = parser.parse_args()

    audio_path = args.audio
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s, {len(audio)} samples")

    db = QuranDB()
    print(f"QuranDB: {db.total_verses} verses")

    model = load_model()

    emitted = simulate_server(audio, model, db)

    # Determine expected verses
    surah_num = args.surah
    if not surah_num and emitted:
        # Infer from most common surah in emitted
        from collections import Counter
        surah_num = Counter(s for s, _ in emitted).most_common(1)[0][0]
    expected = EXPECTED_VERSES_MAP.get(surah_num, [])
    if not expected and surah_num:
        expected = [(surah_num, v["ayah"]) for v in db.get_surah(surah_num)]

    print(f"\n{'='*70}")
    print(f"RESULTS (Surah {surah_num})")
    print(f"{'='*70}")
    print(f"Emitted: {emitted}")
    print(f"Expected: {expected}")

    matched_expected = [v for v in emitted if v in expected]
    missed = [v for v in expected if v not in emitted]
    wrong = [v for v in emitted if v not in expected]

    print(f"\nCorrect: {len(matched_expected)}/{len(expected)}")
    if missed:
        print(f"Missed: {missed}")
    if wrong:
        print(f"Wrong emits: {wrong}")

    if expected and set(expected).issubset(set(emitted)):
        print("\n✅ ALL VERSES DETECTED")
    else:
        print("\n❌ SOME VERSES MISSED")


if __name__ == "__main__":
    main()
