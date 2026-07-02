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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from Levenshtein import ratio
from collections import defaultdict

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

    # Load verse data and pre-compute derived fields
    _verses = json.loads(QURAN_PHONEMES_PATH.read_text(encoding="utf-8"))
    for v in _verses:
        joined = v.get("phonemes_joined", "")
        v["_phonemes_joined_ns"] = joined.replace(" ", "")

        # Strip bismillah prefix for ayah 1 (except surahs 1 and 9)
        no_bsm = None
        if v["ayah"] == 1 and v["surah"] != 1 and v["surah"] != 9 and joined.startswith(_BSM_PHONEMES_JOINED):
            no_bsm = joined[len(_BSM_PHONEMES_JOINED):].strip() or None
        v["_phonemes_joined_no_bsm"] = no_bsm
        v["_phonemes_joined_no_bsm_ns"] = no_bsm.replace(" ", "") if no_bsm else None

        # Index by surah
        _by_surah.setdefault(v["surah"], []).append(v)

    # Sort each surah's verses by ayah
    for verses in _by_surah.values():
        verses.sort(key=lambda x: x["ayah"])

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

    # Preprocessing matching NeMo defaults (no dither — inference should be deterministic)
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
TOP_SURAHS = int(os.getenv("PHONEME_LM_TOP_SURAHS", "20"))
MAX_SPAN = int(os.getenv("PHONEME_LM_MAX_SPAN", "4"))

_BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi"

# Pre-computed verse data (populated in _ensure_loaded)
_by_surah: dict[int, list[dict]] = {}


# ---------------------------------------------------------------------------
# Matching helpers (ported from TypeScript levenshtein.ts / quran-db.ts)
# ---------------------------------------------------------------------------

def semi_global_distance(query: str, ref: str) -> int:
    """Min edit distance aligning entire query against any substring of ref.
    Free gaps at start/end of ref."""
    if not query:
        return 0
    if not ref:
        return len(query)
    m, n = len(query), len(ref)
    prev = list(range(m + 1))
    best = prev[m]
    for j in range(1, n + 1):
        curr = [0] * (m + 1)  # free to start anywhere in ref
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == ref[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        best = min(best, curr[m])  # free to end anywhere in ref
        prev = curr
    return best


def fragment_score(query: str, ref: str) -> float:
    """How well does query match as a fragment of ref? 0.0-1.0."""
    if not query:
        return 1.0
    return max(0.0, 1.0 - semi_global_distance(query, ref) / len(query))


def _short_query_boost(no_space_text: str, verse: dict, use_no_bsm: bool = False) -> float:
    """Boost score for short queries by matching against prefix/first word."""
    if use_no_bsm:
        candidate = verse.get("_phonemes_joined_no_bsm_ns", "") or verse.get("_phonemes_joined_ns", "")
    else:
        candidate = verse.get("_phonemes_joined_ns", "")
    if not candidate:
        return 0.0

    prefix_window = min(len(candidate), len(no_space_text) + 6)
    prefix = ratio(no_space_text, candidate[:prefix_window])

    if use_no_bsm:
        joined = verse.get("_phonemes_joined_no_bsm", "") or ""
    else:
        joined = verse.get("phonemes_joined", "")
    first_word = joined.split(" ")[0] if joined else ""
    first_word_score = ratio(no_space_text, first_word) if first_word else 0.0

    return max(prefix, first_word_score)


def _match_phoneme_text(phoneme_text: str, top_k: int = 10) -> list[dict]:
    """Multi-pass matching of decoded phoneme text against all Quran verses.

    Ported from TypeScript QuranDB.retrieveCandidates / matchVerse:
      Pass 1:   ratio() for global ranking + short-query boost
      Pass 1.5: fragmentScore boost for partial matches (query >= 8 chars no-space)
      Pass 2:   multi-verse span matching on top surahs
    """
    if not phoneme_text.strip():
        return []

    no_space_text = phoneme_text.replace(" ", "")

    # --- Pass 1: ratio() scoring with short-query boost ---
    # Each entry: [verse, raw_score, boosted_score]
    scored: list[list] = []
    for verse in _verses:
        ref = verse.get("phonemes_joined", "")
        if not ref:
            continue
        raw = ratio(phoneme_text, ref)

        if len(no_space_text) <= 10:
            raw = max(raw, _short_query_boost(no_space_text, verse))

        no_bsm = verse.get("_phonemes_joined_no_bsm")
        if no_bsm:
            raw = max(raw, ratio(phoneme_text, no_bsm))
            if len(no_space_text) <= 10:
                raw = max(raw, _short_query_boost(no_space_text, verse, use_no_bsm=True))

        scored.append([verse, raw, raw])

    scored.sort(key=lambda x: x[2], reverse=True)

    # Collect top surahs for Pass 2 (ratio-only, before fragment boost)
    pass2_surahs: list[int] = []
    for entry in scored:
        s = entry[0]["surah"]
        if s not in pass2_surahs:
            pass2_surahs.append(s)
        if len(pass2_surahs) >= TOP_SURAHS:
            break

    # --- Pass 1.5: fragment score boost ---
    if len(no_space_text) >= 8:
        resorted = False
        for i, (verse, raw, _) in enumerate(scored):
            ref_ns = verse.get("_phonemes_joined_ns", "")
            if not ref_ns:
                continue
            # Only boost when query is shorter than ~80% of verse
            if len(no_space_text) >= len(ref_ns) * 0.8:
                continue

            frag = fragment_score(no_space_text, ref_ns)
            no_bsm_ns = verse.get("_phonemes_joined_no_bsm_ns")
            if no_bsm_ns:
                frag = max(frag, fragment_score(no_space_text, no_bsm_ns))

            if frag > raw:
                boosted = raw + (frag - raw) * 0.7
                scored[i] = [verse, boosted, boosted]
                resorted = True

        if resorted:
            scored.sort(key=lambda x: x[2], reverse=True)

    # --- Pass 2: multi-verse span matching on top surahs ---
    span_results: list[dict] = []
    for surah_num in pass2_surahs:
        verses = _by_surah.get(surah_num, [])
        for i in range(len(verses)):
            for span in range(2, MAX_SPAN + 1):
                if i + span > len(verses):
                    break
                chunk = verses[i:i + span]
                # Use no-bsm for first verse
                first_phonemes = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
                span_phonemes = first_phonemes + " " + " ".join(
                    v.get("phonemes_joined", "") for v in chunk[1:]
                )
                raw = ratio(phoneme_text, span_phonemes)
                span_results.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "score": round(raw, 4),
                    "phonemes": " | ".join(v.get("phonemes", "") for v in chunk),
                })

    # --- Combine singles + spans, return top-K ---
    singles = []
    for verse, raw, boosted in scored[:max(top_k, 32)]:
        singles.append({
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "ayah_end": None,
            "score": round(boosted, 4),
            "phonemes": verse.get("phonemes", ""),
        })

    combined = singles + span_results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


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
    """Predict surah/ayah from audio using greedy decode + multi-pass matching."""
    _ensure_loaded()

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
