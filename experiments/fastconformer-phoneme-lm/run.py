"""FastConformer phoneme CTC + beam search with phoneme word vocabulary.

Uses pyctcdecode beam search constrained to valid Quran phoneme-words
(~18K unique words). This forces CTC output to be valid Quran phoneme
sequences instead of arbitrary phoneme token combinations.

Only exposes predict() — no transcribe(). The benchmark runner will
use predict() directly for batch evaluation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from Levenshtein import ratio
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ONNX_MODEL_PATH = Path(
    os.getenv(
        "PHONEME_LM_ONNX_PATH",
        str(PROJECT_ROOT / "web" / "frontend" / "public" / "fastconformer_phoneme_q8.onnx"),
    )
)

QURAN_PHONEMES_PATH = PROJECT_ROOT / "web" / "frontend" / "public" / "quran_phonemes.json"
UNIGRAMS_PATH = PROJECT_ROOT / "data" / "kenlm_phoneme" / "quran_phoneme_unigrams.txt"
KENLM_MODEL_PATH = os.getenv("PHONEME_KENLM_MODEL", "").strip()
NGRAMS_PATH = PROJECT_ROOT / "data" / "kenlm_phoneme" / "phoneme_ngrams.json"
USE_CUSTOM_LM = os.getenv("PHONEME_LM_USE_CUSTOM", "1").strip() == "1"

BEAM_WIDTH = int(os.getenv("PHONEME_LM_BEAM_WIDTH", "24"))
BEAM_PRUNE_LOGP = float(os.getenv("PHONEME_LM_BEAM_PRUNE", "-10.0"))
TOKEN_MIN_LOGP = float(os.getenv("PHONEME_LM_TOKEN_MIN", "-5.0"))
ALPHA = float(os.getenv("PHONEME_LM_ALPHA", "0.7"))
BETA = float(os.getenv("PHONEME_LM_BETA", "1.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("PHONEME_LM_CONFIDENCE", "0.45"))

# Must match the vocabulary used during training
PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)  # 69

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_onnx_session = None
_beam_decoder = None
_verses = None  # list of {surah, ayah, phonemes_joined}


# ---------------------------------------------------------------------------
# Custom phoneme n-gram LM for pyctcdecode
# ---------------------------------------------------------------------------

from pyctcdecode.language_model import AbstractLanguageModel


class PhonemeNgramLM(AbstractLanguageModel):
    """Phoneme-word n-gram LM for pyctcdecode beam search."""

    def __init__(self, ngrams_path: Path, order: int = 3, smoothing: float = 0.1, weight: float = 0.3):
        data = json.loads(ngrams_path.read_text(encoding="utf-8"))

        self._unigrams = data["unigrams"]
        self._bigrams = data["bigrams"]
        self._trigrams = data.get("trigrams", {})
        self._total = data["total_words"]
        self._vocab_size = data["vocab_size"]
        self._order = order
        self._smoothing = smoothing
        self._weight = weight

        # Build word prefix set for partial token validation
        self._word_set = set(self._unigrams.keys())
        self._word_prefixes: set[str] = set()
        for w in self._word_set:
            for i in range(1, len(w) + 1):
                self._word_prefixes.add(w[:i])

        print(f"PhonemeNgramLM: {self._vocab_size} words, "
              f"{len(self._bigrams)} bigram contexts, "
              f"{len(self._trigrams)} trigram contexts")

    @property
    def order(self) -> int:
        return self._order

    def _ngram_logprob(self, context: tuple[str, ...], word: str) -> float:
        """Smoothed n-gram log probability."""
        import math

        # Trigram
        if len(context) >= 2:
            key = f"{context[-2]} {context[-1]}"
            tri = self._trigrams.get(key, {})
            total = sum(tri.values()) if tri else 0
            if total > 0:
                count = tri.get(word, 0)
                prob = (count + self._smoothing) / (total + self._smoothing * self._vocab_size)
                return math.log(prob)

        # Bigram fallback
        if len(context) >= 1:
            bi = self._bigrams.get(context[-1], {})
            total = sum(bi.values()) if bi else 0
            if total > 0:
                count = bi.get(word, 0)
                prob = (count + self._smoothing) / (total + self._smoothing * self._vocab_size)
                return math.log(prob)

        # Unigram fallback
        count = self._unigrams.get(word, 0)
        prob = (count + self._smoothing) / (self._total + self._smoothing * self._vocab_size)
        return math.log(prob)

    def score(self, prev_state: object, word: str, is_last_word: bool = False) -> tuple[float, object]:
        """Score a complete word given previous state. Returns (log_prob, new_state)."""
        context = prev_state if prev_state else ()
        logp = self._ngram_logprob(context, word) * self._weight
        new_state = (context + (word,))[-(self._order - 1):]
        return logp, new_state

    def score_partial_token(self, partial_token: str) -> float:
        """Score a partial word (character-level prefix). Return 0 if valid, penalty if not."""
        if not partial_token:
            return 0.0
        if partial_token in self._word_prefixes:
            return 0.0
        return -10.0  # heavy penalty for impossible word prefixes

    def get_start_state(self):
        return ()

    def reset_params(self, **params):
        pass

    def save_to_dir(self, filepath: str):
        pass


def _read_unigrams(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _ensure_loaded():
    global _onnx_session, _beam_decoder, _verses

    if _onnx_session is not None:
        return

    import onnxruntime as ort

    # Load ONNX model
    print(f"Loading ONNX model: {ONNX_MODEL_PATH}")
    _onnx_session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    # Load verse data
    _verses = json.loads(QURAN_PHONEMES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(_verses)} verses")

    # Build pyctcdecode labels
    # Index 68 ("|") maps to " " (space) for word segmentation
    # Index 69 (blank) maps to "" (empty, pyctcdecode convention)
    labels = []
    for i, token in enumerate(PHONEME_VOCAB):
        if token == "|":
            labels.append(" ")  # word boundary = space
        else:
            labels.append(token)
    labels.append("")  # blank at end

    # Build beam decoder
    from pyctcdecode import build_ctcdecoder

    unigrams = _read_unigrams(UNIGRAMS_PATH)
    print(f"Unigrams: {len(unigrams)} phoneme-words")

    if KENLM_MODEL_PATH:
        print(f"Using KenLM model: {KENLM_MODEL_PATH}")
        _beam_decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=KENLM_MODEL_PATH,
            unigrams=unigrams,
            alpha=ALPHA,
            beta=BETA,
        )
    elif USE_CUSTOM_LM and NGRAMS_PATH.exists():
        # Use custom phoneme n-gram LM via direct constructor
        from pyctcdecode.decoder import Alphabet, BeamSearchDecoderCTC
        from pyctcdecode.language_model import AbstractLanguageModel

        # Make PhonemeNgramLM inherit from AbstractLanguageModel
        lm = PhonemeNgramLM(NGRAMS_PATH, order=3, smoothing=0.1, weight=0.3)

        alphabet = Alphabet.build_alphabet(labels)
        _beam_decoder = BeamSearchDecoderCTC(alphabet, language_model=lm)
        print("Using custom phoneme n-gram LM")
    else:
        # Unigram-only mode
        _beam_decoder = build_ctcdecoder(
            labels=labels,
            unigrams=unigrams,
        )
        print("Using unigram-only beam search (no KenLM)")


def _compute_logprobs(audio: np.ndarray) -> np.ndarray:
    """Run ONNX inference and return logprobs [T, vocab_size]."""
    # NeMo-compatible mel spectrogram
    import librosa

    # Preprocessing matching NeMo defaults
    dither = 1e-5
    audio = audio + dither * np.random.randn(len(audio))
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # preemphasis

    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=512, hop_length=160, win_length=400,
        n_mels=80, fmax=8000, htk=True, norm="slaney",
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)

    features = mel.astype(np.float32)[np.newaxis]  # [1, 80, T]
    length = np.array([mel.shape[1]], dtype=np.int64)

    input_names = [inp.name for inp in _onnx_session.get_inputs()]
    results = _onnx_session.run(None, {
        input_names[0]: features,
        input_names[1]: length,
    })

    logprobs = results[0]  # [1, T, vocab_size]
    return logprobs[0]  # [T, vocab_size]


PHONEME_TOKEN_MAP = {t: i for i, t in enumerate(PHONEME_VOCAB)}
WORD_BOUNDARY_ID = PHONEME_TOKEN_MAP["|"]

TOP_K_LEVENSHTEIN = int(os.getenv("PHONEME_LM_TOP_K", "10"))
TOP_SURAHS = int(os.getenv("PHONEME_LM_TOP_SURAHS", "8"))
MAX_SPAN = int(os.getenv("PHONEME_LM_MAX_SPAN", "6"))


def _tokenize_phonemes(phoneme_str: str) -> list[int]:
    """Convert space-separated phoneme string to token IDs."""
    ids = []
    for token in phoneme_str.strip().split():
        tid = PHONEME_TOKEN_MAP.get(token)
        if tid is not None:
            ids.append(tid)
    return ids


def _match_phoneme_text(phoneme_text: str, top_k: int = 10) -> list[dict]:
    """Match decoded phoneme-word text against all Quran verses. Returns top-K."""
    if not phoneme_text.strip():
        return []

    scored = []
    for verse in _verses:
        ref = verse.get("phonemes_joined", "")
        if not ref:
            continue
        score = ratio(phoneme_text, ref)
        scored.append({
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "ayah_end": None,
            "score": round(score, 4),
            "phonemes": verse.get("phonemes", ""),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _collect_candidate_surahs(top_matches: list[dict]) -> list[int]:
    """Get unique surahs from top Levenshtein matches."""
    seen = set()
    ordered = []
    for m in top_matches:
        s = m["surah"]
        if s not in seen:
            seen.add(s)
            ordered.append(s)
        if len(ordered) >= TOP_SURAHS:
            break
    return ordered


def _build_phoneme_candidates(surahs: list[int]) -> list[dict]:
    """Build single + multi-verse phoneme candidates for CTC scoring."""
    # Group verses by surah
    by_surah: dict[int, list[dict]] = {}
    for v in _verses:
        s = v["surah"]
        if s in surahs:
            by_surah.setdefault(s, []).append(v)

    candidates = []
    for surah_num in surahs:
        verses = by_surah.get(surah_num, [])
        # Sort by ayah
        verses.sort(key=lambda v: v["ayah"])

        # Single verses
        for v in verses:
            candidates.append({
                "surah": surah_num,
                "ayah": v["ayah"],
                "ayah_end": None,
                "phonemes": v.get("phonemes", ""),
            })

        # Multi-verse spans
        for i in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                j = i + span_len
                if j > len(verses):
                    break
                # Only contiguous ayahs
                if verses[i + span_len - 1]["ayah"] - verses[i]["ayah"] != span_len - 1:
                    continue
                chunk = verses[i:j]
                phonemes = " | ".join(v.get("phonemes", "") for v in chunk)
                candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "phonemes": phonemes,
                })

    return candidates


def _score_candidates_ctc(logprobs_np: np.ndarray, candidates: list[dict]) -> list[tuple[dict, float]]:
    """Score phoneme candidates against logprobs using CTC forward algorithm."""
    import torch.nn.functional as F

    logits = torch.tensor(logprobs_np, dtype=torch.float32).unsqueeze(0)  # [1, T, V]
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [T, 1, V]
    T = log_probs.size(0)

    encoded = []
    target_lengths = []
    feasible_indices = []

    for i, c in enumerate(candidates):
        ids = _tokenize_phonemes(c["phonemes"])
        if len(ids) == 0:
            continue
        if len(ids) * 2 + 1 > T:
            continue
        encoded.append(ids)
        target_lengths.append(len(ids))
        feasible_indices.append(i)

    if not encoded:
        return [(c, float("inf")) for c in candidates]

    N = len(encoded)
    log_probs_batch = log_probs.expand(T, N, -1).contiguous()
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)
    all_targets = torch.tensor([idx for seq in encoded for idx in seq], dtype=torch.long)

    losses = F.ctc_loss(
        log_probs_batch, all_targets, input_lengths, target_lengths_t,
        blank=BLANK_ID, reduction="none", zero_infinity=False,
    )
    losses = torch.where(torch.isfinite(losses), losses, torch.tensor(1e9))
    scores = (losses / target_lengths_t.float()).tolist()

    score_map = {feasible_indices[j]: scores[j] for j in range(N)}
    results = [(candidates[i], score_map.get(i, float("inf"))) for i in range(len(candidates))]
    results.sort(key=lambda x: x[1])
    return results


def _greedy_decode_phonemes(logprobs: np.ndarray) -> str:
    """Greedy CTC decode → phoneme-word text."""
    ids = logprobs.argmax(axis=1)
    prev = -1
    tokens = []
    for idx in ids:
        if idx != prev and idx != BLANK_ID:
            if idx < len(PHONEME_VOCAB):
                tokens.append(PHONEME_VOCAB[idx])
        prev = idx

    words = []
    cur: list[str] = []
    for t in tokens:
        if t == "|":
            if cur:
                words.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append("".join(cur))
    return " ".join(words)


USE_BEAM = os.getenv("PHONEME_LM_USE_BEAM", "0").strip() == "1"


def predict(audio_path: str) -> dict:
    """Predict surah/ayah from audio using greedy decode + CTC rescoring."""
    _ensure_loaded()
    import math

    audio = load_audio(audio_path)
    logprobs = _compute_logprobs(audio)

    # Step 1: Decode (greedy by default, beam optional)
    if USE_BEAM and _beam_decoder is not None:
        phoneme_text = _beam_decoder.decode(
            logprobs, beam_width=BEAM_WIDTH,
            beam_prune_logp=BEAM_PRUNE_LOGP, token_min_logp=TOKEN_MIN_LOGP,
        ).strip()
    else:
        phoneme_text = _greedy_decode_phonemes(logprobs)

    # Step 2: Levenshtein match against all verses
    top_matches = _match_phoneme_text(phoneme_text, top_k=TOP_K_LEVENSHTEIN)

    if not top_matches:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": phoneme_text}

    text_best = top_matches[0]

    return {
        "surah": text_best["surah"],
        "ayah": text_best["ayah"],
        "ayah_end": text_best.get("ayah_end"),
        "score": text_best["score"],
        "transcript": phoneme_text,
    }


def model_size() -> int:
    """Return model file size in bytes."""
    if ONNX_MODEL_PATH.exists():
        return ONNX_MODEL_PATH.stat().st_size
    return 131 * 1024 * 1024
