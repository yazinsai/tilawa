"""Quran-aligner phoneme CTC experiment (base + large models).

Uses wav2vec2 CTC models to decode entire audio into IPA phonemes via
CTC greedy decode, then matches against pre-phonemized Quran reference
using Levenshtein ratio + n-gram anchor voting.

Models:
  - base:  hetchyy/r15_95m  (~95M params, 388MB)
  - large: hetchyy/r7       (~1B params, 1200MB)

Pipeline:
  audio -> full CTC decode -> phoneme string
       -> Pass 1: Levenshtein vs all 6236 verses (top 5 surahs)
       -> Pass 1b: N-gram voting for additional candidate surahs
       -> Pass 2: multi-verse spans across all candidate surahs
       -> best match wins
"""

import os
import sys
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor
from Levenshtein import ratio as lev_ratio

from shared.audio import load_audio

# ── Constants ────────────────────────────────────────────────────────────────

MODELS = {
    "base": {"id": "hetchyy/r15_95m", "size_mb": 388},
    "large": {"id": "hetchyy/r7", "size_mb": 1200},
    "base-int8": {"id": "hetchyy/r15_95m_onnx_int8", "size_mb": 116, "onnx": True,
                  "onnx_file": "model.onnx"},
    "large-int8": {"id": "hetchyy/r7_onnx_int8", "size_mb": 970, "onnx": True,
                   "onnx_file": "model_quantized.onnx"},
}

PHONEME_CACHE_PATH = PROJECT_ROOT / "data" / "phoneme_cache.pkl"
NGRAM_INDEX_PATH = PROJECT_ROOT / "data" / "phoneme_ngram_index_5.pkl"
ENV_PATH = PROJECT_ROOT / ".env"

TOP_SURAHS = 5       # surahs from Levenshtein pass 1
NGRAM_TOP_SURAHS = 5 # additional surahs from n-gram voting
MAX_SPAN = 7         # max consecutive verses in a span
MATCH_THRESHOLD = 0.3 # minimum Levenshtein ratio to accept

# ── Stub dataclasses for unpickling phoneme_cache.pkl ────────────────────────
# The pickle was built with quran_aligner_app's src.alignment.phoneme_matcher
# classes. We define identical stubs here so we don't depend on that package.


@dataclass
class RefWord:
    text: str
    phonemes: List[str]
    surah: int
    ayah: int
    word_num: int

    @property
    def location(self) -> str:
        return f"{self.surah}:{self.ayah}:{self.word_num}"


@dataclass
class ChapterReference:
    surah: int
    words: List[RefWord]
    avg_phones_per_word: float
    flat_phonemes: List[str]
    flat_phone_to_word: List[int]
    word_phone_offsets: List[int]

    @property
    def num_words(self) -> int:
        return len(self.words)


@dataclass
class PhonemeNgramIndex:
    """Pre-computed n-gram index for the entire Quran."""
    ngram_positions: Dict[Tuple[str, ...], List[Tuple[int, int]]]
    ngram_counts: Dict[Tuple[str, ...], int]
    ngram_size: int
    total_ngrams: int


class _StubUnpickler(pickle.Unpickler):
    """Remap quran_aligner_app pickle classes to our local stubs."""

    _CLASS_MAP = {
        ("src.alignment.phoneme_matcher", "RefWord"): RefWord,
        ("src.alignment.phoneme_matcher", "ChapterReference"): ChapterReference,
        ("src.alignment.ngram_index", "PhonemeNgramIndex"): PhonemeNgramIndex,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in self._CLASS_MAP:
            return self._CLASS_MAP[key]
        return super().find_class(module, name)


# ── Lazy globals ─────────────────────────────────────────────────────────────

_loaded_models = {}   # model_name -> {"model": ..., "processor": ..., "device": ...}
_verse_phonemes = None   # list of {"surah", "ayah", "phoneme_str"}
_surah_verses = None     # dict[int, list[dict]]
_ngram_index = None      # PhonemeNgramIndex (loaded from pickle)


def _load_hf_token():
    """Load HF_TOKEN from .env file or environment."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = token
                return token
    return None


def _ids_to_phoneme_list(ids, tokenizer, pad_id):
    """CTC collapse: remove blanks, collapse consecutive duplicates, strip |."""
    toks = tokenizer.convert_ids_to_tokens(ids)
    if not toks:
        return []
    pad_tok = tokenizer.convert_ids_to_tokens([pad_id])[0] if pad_id is not None else "[PAD]"
    collapsed = []
    prev = None
    for t in toks:
        if t == pad_tok or t == "|":
            prev = t
            continue
        if t == prev:
            continue
        collapsed.append(t)
        prev = t
    return collapsed


def _build_verse_phoneme_db():
    """Load phoneme_cache.pkl and build per-verse phoneme strings."""
    with open(PHONEME_CACHE_PATH, "rb") as f:
        chapters = _StubUnpickler(f).load()

    verse_phonemes = []
    surah_verses = {}

    for surah_num in sorted(chapters):
        ch = chapters[surah_num]
        # Group words by ayah
        ayah_words = {}
        for word in ch.words:
            ayah_words.setdefault(word.ayah, []).append(word)

        surah_list = []
        for ayah_num in sorted(ayah_words):
            words = ayah_words[ayah_num]
            # Join all phonemes for this verse (space-separated)
            phoneme_str = " ".join(p for w in words for p in w.phonemes)
            entry = {
                "surah": surah_num,
                "ayah": ayah_num,
                "phoneme_str": phoneme_str,
            }
            verse_phonemes.append(entry)
            surah_list.append(entry)
        surah_verses[surah_num] = surah_list

    return verse_phonemes, surah_verses


def _ensure_reference_loaded():
    """Load phoneme reference DB and n-gram index (shared across all models)."""
    global _verse_phonemes, _surah_verses, _ngram_index
    if _verse_phonemes is not None:
        return
    print(f"Loading phoneme reference from {PHONEME_CACHE_PATH}...")
    _verse_phonemes, _surah_verses = _build_verse_phoneme_db()
    print(f"Loaded {len(_verse_phonemes)} verses across {len(_surah_verses)} surahs")

    if NGRAM_INDEX_PATH.exists():
        print(f"Loading n-gram index from {NGRAM_INDEX_PATH}...")
        with open(NGRAM_INDEX_PATH, "rb") as f:
            _ngram_index = _StubUnpickler(f).load()
        print(f"Loaded {len(_ngram_index.ngram_positions)} unique "
              f"{_ngram_index.ngram_size}-grams")
    else:
        print(f"Warning: n-gram index not found at {NGRAM_INDEX_PATH}")


def _ngram_vote(phonemes: list[str]) -> list[Tuple[int, int, int, float]]:
    """Run n-gram voting on ASR phonemes to find candidate (surah, ayah) locations.

    Returns list of (surah, run_start_ayah, run_end_ayah, run_weight) sorted
    by weight descending.
    """
    if _ngram_index is None or not phonemes:
        return []

    n = _ngram_index.ngram_size
    if len(phonemes) < n:
        return []

    # Extract n-grams from ASR phoneme list
    asr_ngrams = [tuple(phonemes[i:i + n]) for i in range(len(phonemes) - n + 1)]

    # Accumulate rarity-weighted votes per (surah, ayah)
    votes: Dict[Tuple[int, int], float] = defaultdict(float)
    for ng in asr_ngrams:
        if ng not in _ngram_index.ngram_positions:
            continue
        weight = 1.0 / _ngram_index.ngram_counts[ng]
        for surah, ayah in _ngram_index.ngram_positions[ng]:
            votes[(surah, ayah)] += weight

    if not votes:
        return []

    # Aggregate by surah
    surah_totals: Dict[int, float] = defaultdict(float)
    surah_ayah_weights: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (s, a), w in votes.items():
        surah_totals[s] += w
        surah_ayah_weights[s][a] = w

    # For each top surah, find best contiguous ayah run
    ranked_surahs = sorted(surah_totals.items(), key=lambda kv: kv[1], reverse=True)
    results = []
    for surah, _ in ranked_surahs[:NGRAM_TOP_SURAHS * 2]:  # check more, keep top N
        ayah_weights = surah_ayah_weights[surah]
        sorted_ayahs = sorted(ayah_weights.keys())

        # Build runs of consecutive ayahs
        runs = []
        rs, re, rw = sorted_ayahs[0], sorted_ayahs[0], ayah_weights[sorted_ayahs[0]]
        for i in range(1, len(sorted_ayahs)):
            a = sorted_ayahs[i]
            if a == re + 1:
                re = a
                rw += ayah_weights[a]
            else:
                runs.append((rs, re, rw))
                rs, re, rw = a, a, ayah_weights[a]
        runs.append((rs, re, rw))

        best_run = max(runs, key=lambda r: r[2])
        results.append((surah, best_run[0], best_run[1], best_run[2]))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def _ensure_model_loaded(model_name: str = "base"):
    """Lazy-load a specific model (cached in _loaded_models)."""
    global _loaded_models
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    _ensure_reference_loaded()

    model_info = MODELS[model_name]
    model_id = model_info["id"]
    token = _load_hf_token()

    if model_info.get("onnx"):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        onnx_file = model_info.get("onnx_file", "model_quantized.onnx")
        local_dir = snapshot_download(model_id, token=token)
        onnx_path = os.path.join(local_dir, onnx_file)
        print(f"Loading ONNX model ({model_name}) from {model_id}...")
        processor = AutoProcessor.from_pretrained(local_dir)
        sess = ort.InferenceSession(onnx_path)
        print(f"ONNX model loaded ({model_name})")
        entry = {"onnx_session": sess, "processor": processor, "device": "cpu"}
        _loaded_models[model_name] = entry
        return entry

    print(f"Loading {model_id} ({model_name})...")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForCTC.from_pretrained(model_id, token=token)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    print(f"Model {model_name} loaded on {device}")

    entry = {"model": model, "processor": processor, "device": device}
    _loaded_models[model_name] = entry
    return entry


# ── Public API ───────────────────────────────────────────────────────────────

def list_models() -> list[str]:
    return [k for k in MODELS if k.endswith("-int8")]


def _decode_phonemes(audio_path: str, model_name: str = "base") -> list[str]:
    """Load audio, run full CTC decode, return phoneme list."""
    m = _ensure_model_loaded(model_name)
    audio = load_audio(audio_path)

    if "onnx_session" in m:
        inputs = m["processor"](audio, sampling_rate=16000, return_tensors="np", padding=True)
        logits = m["onnx_session"].run(
            ["logits"], {"input_values": inputs["input_values"]}
        )[0]
        pred_ids = np.argmax(logits, axis=-1)[0].tolist()
    else:
        inputs = m["processor"](audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(m["device"]) for k, v in inputs.items()}
        with torch.no_grad():
            logits = m["model"](**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    pad_id = m["processor"].tokenizer.pad_token_id or 0
    return _ids_to_phoneme_list(pred_ids, m["processor"].tokenizer, pad_id)



def predict(audio_path: str, model_name: str = "base", debug: bool = False) -> dict:
    _ensure_model_loaded(model_name)

    # ASR phase
    phonemes = _decode_phonemes(audio_path, model_name)
    asr_str = " ".join(phonemes)

    if not asr_str.strip():
        return {
            "surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": "",
        }

    # Pass 1: score against all 6236 verses
    scored = []
    for v in _verse_phonemes:
        score = lev_ratio(asr_str, v["phoneme_str"])
        scored.append((v, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    best_v, best_single_score = scored[0]
    best = {
        "surah": best_v["surah"],
        "ayah": best_v["ayah"],
        "ayah_end": None,
        "score": best_single_score,
    }
    best_score = best_single_score

    # Pass 1b: N-gram voting for additional candidate surahs
    ngram_results = _ngram_vote(phonemes)
    ngram_surahs = [s for s, _, _, _ in ngram_results[:NGRAM_TOP_SURAHS]]

    # Pass 2: multi-verse spans around top surahs (Levenshtein + n-gram union)
    seen_surahs = set()
    lev_top_surahs = []
    for v, _ in scored[:20]:
        s = v["surah"]
        if s not in seen_surahs:
            seen_surahs.add(s)
            lev_top_surahs.append(s)
            if len(lev_top_surahs) >= TOP_SURAHS:
                break

    # Merge: Levenshtein surahs + n-gram surahs (deduplicated)
    span_surahs = list(lev_top_surahs)
    for s in ngram_surahs:
        if s not in seen_surahs:
            seen_surahs.add(s)
            span_surahs.append(s)

    all_spans = []
    for surah_num in span_surahs:
        if surah_num not in _surah_verses:
            continue
        verses = _surah_verses[surah_num]
        for i in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                if i + span_len > len(verses):
                    break
                chunk = verses[i:i + span_len]
                combined = " ".join(c["phoneme_str"] for c in chunk)
                score = lev_ratio(asr_str, combined)
                all_spans.append((surah_num, chunk[0]["ayah"], chunk[-1]["ayah"], score))
                if score > best_score:
                    best_score = score
                    best = {
                        "surah": surah_num,
                        "ayah": chunk[0]["ayah"],
                        "ayah_end": chunk[-1]["ayah"],
                        "score": score,
                    }

    if best_score < MATCH_THRESHOLD:
        result = {
            "surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": asr_str,
        }
    else:
        result = {
            "surah": best["surah"],
            "ayah": best["ayah"],
            "ayah_end": best.get("ayah_end"),
            "score": round(best_score, 4),
            "transcript": asr_str,
        }

    if debug:
        all_spans.sort(key=lambda x: x[3], reverse=True)
        result["debug"] = {
            "asr_phonemes": asr_str,
            "top_5_verses": [
                {"surah": v["surah"], "ayah": v["ayah"], "score": round(sc, 4)}
                for v, sc in scored[:5]
            ],
            "top_3_spans": [
                {"surah": s, "ayah_start": a1, "ayah_end": a2, "score": round(sc, 4)}
                for s, a1, a2, sc in all_spans[:3]
            ],
            "ngram_top_5": [
                {"surah": s, "ayah_start": rs, "ayah_end": re, "weight": round(w, 3)}
                for s, rs, re, w in ngram_results[:5]
            ],
            "span_surahs_searched": span_surahs,
            "best_single_score": round(best_single_score, 4),
            "best_span_score": round(all_spans[0][3], 4) if all_spans else 0.0,
            "num_phonemes_asr": len(phonemes),
            "num_phonemes_best_ref": len(best_v["phoneme_str"].split()) if best_v else 0,
        }

    return result


def model_size(model_name: str = "base") -> int:
    return MODELS[model_name]["size_mb"] * 1024 * 1024
