"""
Improved matching approaches:
1. ASR (Whisper Large-v3) + multi-ayah concatenation matching
2. MFCC+DTW with concatenated reference audio for multi-ayah detection
"""
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from itertools import combinations

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from Levenshtein import ratio

import librosa

REF_DIR = Path(__file__).parent.parent / "data" / "reference_audio"
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
        "expected_ayah": 2,  # starts at ayah 2
        "description": "Al-Ikhlas:2-3 (multi-ayah)",
    },
]


# ── Approach A: ASR with multi-ayah matching ──

def test_asr_multi_ayah():
    """Use Whisper transcription, but match against single AND concatenated verse pairs."""
    import mlx_whisper

    print(f"\n{'='*70}")
    print(f"  ASR (Whisper Large-v3-Turbo) + Multi-Ayah Matching")
    print(f"{'='*70}")

    model_id = 'mlx-community/whisper-large-v3-turbo'

    for tc in TEST_CASES:
        audio = load_audio(str(TEST_DIR / tc["file"]))
        start = time.time()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=model_id, language='ar')
        text = result.get('text', '').strip()
        text_clean = normalize_arabic(text)
        elapsed_asr = time.time() - start

        # Match against single verses
        start = time.time()
        single_scores = []
        for v in db.verses:
            score = ratio(text_clean, v["text_clean"])
            single_scores.append((score, v["surah"], v["ayah"], v["text_clean"], "single"))
        single_scores.sort(reverse=True)

        # Match against concatenated consecutive verse PAIRS
        pair_scores = []
        for surah_num in range(1, 115):
            surah_verses = db.get_surah(surah_num)
            for i in range(len(surah_verses) - 1):
                v1 = surah_verses[i]
                v2 = surah_verses[i + 1]
                concat_text = v1["text_clean"] + " " + v2["text_clean"]
                score = ratio(text_clean, concat_text)
                pair_scores.append((
                    score,
                    v1["surah"], v1["ayah"],
                    f"{v1['text_clean']} | {v2['text_clean']}",
                    f"pair {v1['ayah']}-{v2['ayah']}"
                ))
        pair_scores.sort(reverse=True)

        # Match against TRIPLES
        triple_scores = []
        for surah_num in range(1, 115):
            surah_verses = db.get_surah(surah_num)
            for i in range(len(surah_verses) - 2):
                v1 = surah_verses[i]
                v2 = surah_verses[i + 1]
                v3 = surah_verses[i + 2]
                concat_text = v1["text_clean"] + " " + v2["text_clean"] + " " + v3["text_clean"]
                score = ratio(text_clean, concat_text)
                triple_scores.append((
                    score,
                    v1["surah"], v1["ayah"],
                    f"...",
                    f"triple {v1['ayah']}-{v3['ayah']}"
                ))
        triple_scores.sort(reverse=True)

        elapsed_match = time.time() - start

        # Combine all and find best
        all_scores = single_scores[:5] + pair_scores[:5] + triple_scores[:5]
        all_scores.sort(reverse=True)

        correct_single = single_scores[0][1] == tc["expected_surah"] and single_scores[0][2] == tc["expected_ayah"]
        correct_any = any(
            s[1] == tc["expected_surah"] and s[2] == tc["expected_ayah"]
            for s in all_scores[:3]
        )

        status = "PASS" if correct_any else "FAIL"
        print(f"\n  [{status}] {tc['description']}")
        print(f"  Transcription: {text}")
        print(f"  Normalized:    {text_clean}")
        print(f"  ASR time: {elapsed_asr:.1f}s | Match time: {elapsed_match:.1f}s")
        print(f"\n  Top 5 single-verse matches:")
        for i, (score, s, a, txt, kind) in enumerate(single_scores[:5]):
            marker = " <--" if s == tc["expected_surah"] and a == tc["expected_ayah"] else ""
            print(f"    {i+1}. {s}:{a} ({score:.3f}) {kind}{marker}")
        print(f"\n  Top 5 verse-pair matches:")
        for i, (score, s, a, txt, kind) in enumerate(pair_scores[:5]):
            marker = " <--" if s == tc["expected_surah"] and a == tc["expected_ayah"] else ""
            print(f"    {i+1}. {s}:{a} ({score:.3f}) {kind}{marker}")
        print(f"\n  Top 3 verse-triple matches:")
        for i, (score, s, a, txt, kind) in enumerate(triple_scores[:3]):
            marker = " <--" if s == tc["expected_surah"] and a == tc["expected_ayah"] else ""
            print(f"    {i+1}. {s}:{a} ({score:.3f}) {kind}{marker}")


# ── Approach B: MFCC+DTW with concatenated reference audio ──

def test_dtw_multi_ayah():
    """MFCC+DTW matching with concatenated reference audio for multi-ayah."""
    print(f"\n{'='*70}")
    print(f"  MFCC+DTW with Multi-Ayah Reference Concatenation")
    print(f"{'='*70}")

    # Load reference audio
    ref_files = {}
    for fname in sorted(Path(REF_DIR).glob("*.mp3")):
        surah = int(fname.stem[:3])
        ayah = int(fname.stem[3:])
        ref_files[(surah, ayah)] = load_audio(str(fname))

    # Build single + pair references
    references = {}
    for (s, a), audio in ref_files.items():
        references[(s, a, "single")] = audio
        # Check for next ayah
        if (s, a + 1) in ref_files:
            concat = np.concatenate([audio, ref_files[(s, a + 1)]])
            references[(s, a, f"pair {a}-{a+1}")] = concat

    print(f"  {len(references)} reference entries (singles + pairs)")

    for tc in TEST_CASES:
        test_audio = load_audio(str(TEST_DIR / tc["file"]))
        test_mfcc = librosa.feature.mfcc(y=test_audio, sr=16000, n_mfcc=13)

        start = time.time()
        scores = []
        for (s, a, kind), ref_aud in references.items():
            ref_mfcc = librosa.feature.mfcc(y=ref_aud, sr=16000, n_mfcc=13)
            D, wp = librosa.sequence.dtw(test_mfcc, ref_mfcc, metric='cosine')
            dtw_cost = D[-1, -1] / len(wp)
            sim = 1.0 / (1.0 + dtw_cost)
            scores.append((sim, s, a, kind))

        scores.sort(reverse=True)
        elapsed = time.time() - start

        correct = any(
            s == tc["expected_surah"] and a == tc["expected_ayah"]
            for _, s, a, _ in scores[:3]
        )

        status = "PASS" if correct else "FAIL"
        print(f"\n  [{status}] {tc['description']}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Top 10 matches:")
        for i, (sim, s, a, kind) in enumerate(scores[:10]):
            marker = " <--" if s == tc["expected_surah"] and a == tc["expected_ayah"] else ""
            print(f"    {i+1}. {s}:{a} ({sim:.4f}) [{kind}]{marker}")


if __name__ == "__main__":
    test_asr_multi_ayah()
    test_dtw_multi_ayah()
