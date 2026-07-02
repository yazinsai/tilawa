"""
Test acoustic matching: compare user recordings against EveryAyah reference audio.

Approach 1: MFCC + Cosine Similarity on mean vectors
Approach 2: Chroma features + Cosine Similarity
Approach 3: MFCC + DTW (Dynamic Time Warping)
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.audio import load_audio

import librosa
from scipy.spatial.distance import cosine as cosine_dist

REF_DIR = Path(__file__).parent.parent / "data" / "reference_audio"
TEST_DIR = Path(__file__).parent.parent / "data" / "test_audio"

REFERENCE_VERSES = {
    "004049.mp3": "An-Nisa:49",
    "004050.mp3": "An-Nisa:50",
    "004051.mp3": "An-Nisa:51",
    "004052.mp3": "An-Nisa:52",
    "004053.mp3": "An-Nisa:53",
    "112001.mp3": "Al-Ikhlas:1",
    "112002.mp3": "Al-Ikhlas:2",
    "112003.mp3": "Al-Ikhlas:3",
    "112004.mp3": "Al-Ikhlas:4",
    "003022.mp3": "Aal-i-Imraan:22",
    "003023.mp3": "Aal-i-Imraan:23",
    "003024.mp3": "Aal-i-Imraan:24",
    "001001.mp3": "Al-Fatiha:1",
    "001002.mp3": "Al-Fatiha:2",
    "001003.mp3": "Al-Fatiha:3",
    "001004.mp3": "Al-Fatiha:4",
    "001005.mp3": "Al-Fatiha:5",
    "001006.mp3": "Al-Fatiha:6",
    "001007.mp3": "Al-Fatiha:7",
    "002255.mp3": "Al-Baqara:255",
    "036001.mp3": "Ya-Sin:1",
    "036002.mp3": "Ya-Sin:2",
    "036003.mp3": "Ya-Sin:3",
}

TEST_CASES = [
    {
        "file": "nisa_51.m4a",
        "expected": "Aal-i-Imraan:23",
        "expected_file": "003023.mp3",
        "description": "Aal-i-Imraan:23 (user recording)",
    },
    {
        "file": "ikhlas_2_3.m4a",
        "expected": "Al-Ikhlas:2",
        "expected_file": "112002.mp3",
        "description": "Al-Ikhlas:2-3 (user recording, multi-ayah)",
    },
]


def extract_mfcc(audio, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])


def extract_chroma(audio, sr=16000):
    return librosa.feature.chroma_stft(y=audio, sr=sr)


def mfcc_mean_similarity(audio1, audio2, sr=16000):
    mfcc1 = extract_mfcc(audio1, sr).mean(axis=1)
    mfcc2 = extract_mfcc(audio2, sr).mean(axis=1)
    return 1.0 - cosine_dist(mfcc1, mfcc2)


def chroma_similarity(audio1, audio2, sr=16000):
    c1 = extract_chroma(audio1, sr).mean(axis=1)
    c2 = extract_chroma(audio2, sr).mean(axis=1)
    return 1.0 - cosine_dist(c1, c2)


def mfcc_dtw_similarity(audio1, audio2, sr=16000):
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13).T
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13).T
    D, wp = librosa.sequence.dtw(mfcc1.T, mfcc2.T, metric='cosine')
    dtw_cost = D[-1, -1] / len(wp)
    return 1.0 / (1.0 + dtw_cost)


def test_approach(name, similarity_fn, ref_audio):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    for tc in TEST_CASES:
        test_path = TEST_DIR / tc["file"]
        test_audio = load_audio(str(test_path))

        start = time.time()
        scores = []
        for fname, (ref_aud, label) in ref_audio.items():
            sim = similarity_fn(test_audio, ref_aud)
            scores.append((sim, fname, label))

        scores.sort(reverse=True)
        elapsed = time.time() - start

        correct_in_top1 = scores[0][1] == tc["expected_file"]
        correct_rank = next(
            (i+1 for i, (_, f, _) in enumerate(scores) if f == tc["expected_file"]),
            -1
        )

        status = "PASS" if correct_in_top1 else "FAIL"
        print(f"\n  [{status}] {tc['description']}")
        print(f"  Expected: {tc['expected']} ({tc['expected_file']})")
        print(f"  Correct answer rank: #{correct_rank}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Top 5 matches:")
        for i, (sim, fname, label) in enumerate(scores[:5]):
            marker = " <-- CORRECT" if fname == tc["expected_file"] else ""
            print(f"    {i+1}. {label} ({fname}) - similarity: {sim:.4f}{marker}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    print("Testing acoustic matching approaches against user recordings\n")
    print(f"Reference audio: {len(REFERENCE_VERSES)} verses from EveryAyah (Alafasy)")
    print(f"Test cases: {len(TEST_CASES)}")

    # Pre-load all reference audio once
    ref_audio = {}
    for fname, label in REFERENCE_VERSES.items():
        fpath = REF_DIR / fname
        if fpath.exists():
            ref_audio[fname] = (load_audio(str(fpath)), label)
    print(f"Loaded {len(ref_audio)} reference files\n")

    test_approach("MFCC Mean Vector + Cosine Similarity", mfcc_mean_similarity, ref_audio)
    test_approach("Chroma Features + Cosine Similarity", chroma_similarity, ref_audio)
    test_approach("MFCC + DTW (Dynamic Time Warping)", mfcc_dtw_similarity, ref_audio)
