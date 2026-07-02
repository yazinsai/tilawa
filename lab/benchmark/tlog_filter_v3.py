"""
Replace RetaSy samples in test_corpus_v3 with cleaned TLOG samples.

TLOG (tarteel-ai/tlog) is a large crowd-sourced recitation dataset where the
"clean" split still contains many noisy / misrecited entries. We apply two
filters:

1. Model-quality filter: run each candidate through the shipped FastConformer
   phoneme ONNX (greedy CTC), decode to a phoneme string, and compare against
   the ground-truth phoneme string for the (surah, ayah) referenced by the
   filename. Keep only samples where Levenshtein ratio >= MIN_RATIO (0.85).

2. Audio-quality filter: reject samples shorter than 0.5s or longer than 30s,
   or with audio bytes below some threshold.

Output: overwrite test_corpus_v3/manifest.json replacing all retasy_v3_* rows
with up to N good TLOG rows (category balance preserved).

Usage:
    .venv/bin/python benchmark/tlog_filter_v3.py --target-short 60 --target-medium 50 --target-long 20
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = Path(__file__).parent / "test_corpus_v3"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PHONEMES_PATH = PROJECT_ROOT / "web" / "frontend" / "public" / "quran_phonemes.json"
ONNX_PATH = PROJECT_ROOT / "web" / "frontend" / "public" / "fastconformer_phoneme_q8.onnx"

PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)

FILENAME_RE = re.compile(r"^(\d+)_(\d+)_")  # {surah}_{ayah}_{id}.flac

MIN_RATIO = 0.85
MIN_SEC = 0.5
MAX_SEC = 30.0
SAMPLE_RATE = 16000


def categorize(wc: int) -> str:
    if wc <= 5:
        return "short"
    elif wc <= 15:
        return "medium"
    return "long"


def decode_audio_bytes(raw: bytes) -> np.ndarray | None:
    """Decode flac/wav/mp3 bytes to 16kHz mono float32 via ffmpeg."""
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "flac", "-i", "pipe:0",
                "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
            ],
            input=raw, capture_output=True, timeout=20,
        )
        if r.returncode != 0 or not r.stdout:
            # fallback: let ffmpeg autodetect
            r = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", "pipe:0",
                    "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
                ],
                input=raw, capture_output=True, timeout=20,
            )
            if r.returncode != 0 or not r.stdout:
                return None
        return np.frombuffer(r.stdout, dtype=np.float32).copy()
    except Exception:
        return None


def load_ref_phonemes():
    with open(QURAN_PHONEMES_PATH) as f:
        data = json.load(f)
    by_ref: dict[tuple[int, int], dict] = {}
    for v in data:
        by_ref[(v["surah"], v["ayah"])] = {
            "phonemes": v["phonemes"],                   # space-separated tokens
            "phonemes_joined": v["phonemes_joined"],     # compact form
            "word_count": len(v["phonemes_joined"].split()),
        }
    return by_ref


class PhonemeOnnx:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.sess.get_inputs()]

    def transcribe(self, audio: np.ndarray) -> list[int]:
        """Return greedy CTC token-IDs (after blank collapsing)."""
        import librosa
        # Match fastconformer-phoneme-lm preprocessing
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=512, hop_length=160, win_length=400,
            n_mels=80, fmax=8000, htk=True, norm="slaney",
        )
        mel = np.log(mel + 1e-5)
        mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)
        features = mel.astype(np.float32)[np.newaxis]
        length = np.array([mel.shape[1]], dtype=np.int64)
        out = self.sess.run(None, {self.input_names[0]: features, self.input_names[1]: length})
        logprobs = out[0][0]  # [T, V]
        argmax = np.argmax(logprobs, axis=-1)
        # Collapse repeats, drop blanks
        tokens: list[int] = []
        prev = -1
        for t in argmax:
            t = int(t)
            if t == BLANK_ID:
                prev = t
                continue
            if t != prev:
                tokens.append(t)
            prev = t
        return tokens


def ids_to_phoneme_str(ids: list[int]) -> str:
    return " ".join(PHONEME_VOCAB[i] for i in ids if 0 <= i < len(PHONEME_VOCAB))


def levenshtein_ratio(a: str, b: str) -> float:
    from Levenshtein import ratio
    return ratio(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-short", type=int, default=50)
    ap.add_argument("--target-medium", type=int, default=50)
    ap.add_argument("--target-long", type=int, default=20)
    ap.add_argument("--min-ratio", type=float, default=MIN_RATIO)
    ap.add_argument("--max-scan", type=int, default=3000)
    ap.add_argument("--max-per-ayah", type=int, default=2,
                    help="Limit how many samples per (surah,ayah) to promote reciter diversity.")
    args = ap.parse_args()

    targets = {"short": args.target_short, "medium": args.target_medium, "long": args.target_long}
    total_target = sum(targets.values())

    print("Loading reference phonemes...")
    ref_by_ayah = load_ref_phonemes()
    print(f"  {len(ref_by_ayah)} verses indexed")

    print(f"Loading ONNX ({ONNX_PATH.name})...")
    model = PhonemeOnnx(ONNX_PATH)

    print("Streaming TLOG-clean...")
    from datasets import Audio, load_dataset
    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Need HUGGING_FACE_TOKEN in env")
    ds = load_dataset("tarteel-ai/tlog", split="clean", streaming=True, token=token)
    ds = ds.cast_column("audio", Audio(decode=False))

    accepted: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    per_ayah_count: Counter[tuple[int, int]] = Counter()
    scanned = rejected_audio = rejected_ratio = 0

    # Drop any previously-added tlog_* rows + files so this is a clean rebuild.
    with open(MANIFEST_PATH) as f:
        _m = json.load(f)
    _m["samples"] = [s for s in _m["samples"] if not s["id"].startswith("tlog_")]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(_m, f, indent=2, ensure_ascii=False)
    for p in CORPUS_DIR.glob("tlog_*.wav"):
        p.unlink()

    saved_ids: set[str] = set()
    for i, ex in enumerate(ds):
        scanned = i + 1
        if scanned > args.max_scan:
            break
        total = sum(len(v) for v in accepted.values())
        if total >= total_target:
            break
        all_cats_full = all(len(accepted[c]) >= targets[c] for c in targets)
        if all_cats_full:
            break

        a = ex.get("audio") or {}
        path = a.get("path") or ""
        raw = a.get("bytes")
        m = FILENAME_RE.match(path)
        if not m or not raw:
            continue
        surah, ayah = int(m.group(1)), int(m.group(2))

        ref = ref_by_ayah.get((surah, ayah))
        if ref is None:
            continue
        wc = ref["word_count"]
        cat = categorize(wc)
        if len(accepted[cat]) >= targets[cat]:
            continue
        if per_ayah_count[(surah, ayah)] >= args.max_per_ayah:
            continue

        # audio filter
        audio = decode_audio_bytes(raw)
        if audio is None:
            rejected_audio += 1
            continue
        dur = len(audio) / SAMPLE_RATE
        if dur < MIN_SEC or dur > MAX_SEC:
            rejected_audio += 1
            continue

        # quality filter
        hyp_ids = model.transcribe(audio)
        hyp_str = ids_to_phoneme_str(hyp_ids)
        ratio = levenshtein_ratio(hyp_str, ref["phonemes"])
        if ratio < args.min_ratio:
            rejected_ratio += 1
            if scanned % 50 == 0:
                print(f"  [{scanned}] reject ratio={ratio:.2f} ({cat} {surah}:{ayah}) kept={dict((k, len(v)) for k, v in accepted.items())}")
            continue

        # Save WAV
        idx = len(accepted[cat])
        sid = f"tlog_{cat[:1]}{idx:03d}_{surah:03d}_{ayah:03d}"
        if sid in saved_ids:
            continue
        fn = f"{sid}.wav"
        fp = CORPUS_DIR / fn
        import soundfile as sf
        sf.write(str(fp), audio, SAMPLE_RATE)
        accepted[cat].append({
            "id": sid,
            "file": fn,
            "surah": surah,
            "ayah": ayah,
            "ayah_end": None,
            "category": cat,
            "source": "tlog",
            "ratio": round(float(ratio), 3),
            "duration_s": round(float(dur), 2),
            "expected_verses": [{"surah": surah, "ayah": ayah}],
        })
        per_ayah_count[(surah, ayah)] += 1
        saved_ids.add(sid)
        if (sum(len(v) for v in accepted.values())) % 10 == 0:
            print(f"  [{scanned}] accepted so far: " + ", ".join(f"{c}={len(accepted[c])}/{targets[c]}" for c in targets))

    print(f"\nScan done: scanned={scanned}, accepted={sum(len(v) for v in accepted.values())}, "
          f"rejected_audio={rejected_audio}, rejected_ratio={rejected_ratio}")

    # Merge into manifest: drop retasy_v3_* rows, add tlog rows
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    before = len(manifest["samples"])
    manifest["samples"] = [s for s in manifest["samples"] if not s["id"].startswith("retasy_v3_")]
    # also delete the files
    for p in CORPUS_DIR.glob("retasy_v3_*.wav"):
        p.unlink()
    for cat in targets:
        manifest["samples"].extend(accepted[cat])
    after = len(manifest["samples"])

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cats = Counter(s["category"] for s in manifest["samples"])
    srcs = Counter(s["source"] for s in manifest["samples"])
    print(f"\nManifest updated: {before} -> {after} samples")
    print(f"  by category: {dict(cats)}")
    print(f"  by source:   {dict(srcs)}")


if __name__ == "__main__":
    main()
