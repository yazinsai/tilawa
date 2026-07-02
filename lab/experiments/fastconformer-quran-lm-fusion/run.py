"""FastConformer + in-beam Quran LM fusion + constrained ayah decoding.

Second-pass experiment:
  1) Use pyctcdecode beam search with a custom Quran-trained language model
     plugged into decoder beam expansion (true shallow fusion).
  2) Apply tighter prefix constraints during expansion via LM penalties on
     invalid word prefixes / partial tokens.
  3) Restrict final prediction to known ayah text (single + multi-ayah spans)
     through CTC forced alignment on candidate surahs.
"""

import math
import os
import sys
import types
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from Levenshtein import ratio
from pyctcdecode.decoder import Alphabet, BeamSearchDecoderCTC
from pyctcdecode.language_model import AbstractLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

score_candidates = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Decoder search
N_BEST = int(os.getenv("FASTCONFORMER_QURAN_LM_NBEST", "12"))
DECODE_BEAM_WIDTH = int(os.getenv("FASTCONFORMER_QURAN_BEAM_WIDTH", "24"))
BEAM_PRUNE_LOGP = float(os.getenv("FASTCONFORMER_QURAN_BEAM_PRUNE_LOGP", "-10.0"))
TOKEN_MIN_LOGP = float(os.getenv("FASTCONFORMER_QURAN_TOKEN_MIN_LOGP", "-5.0"))

CONFIDENCE_THRESHOLD = float(os.getenv("FASTCONFORMER_QURAN_LM_CONFIDENCE", "0.72"))

# Custom LM knobs
LM_ORDER = int(os.getenv("FASTCONFORMER_QURAN_WORD_LM_ORDER", "3"))
LM_SMOOTHING = float(os.getenv("FASTCONFORMER_QURAN_WORD_LM_SMOOTHING", "0.2"))
LM_FUSION_WEIGHT = float(os.getenv("FASTCONFORMER_QURAN_WORD_LM_WEIGHT", "0.2"))

LM_PREFIX_MAX_SPAN = int(os.getenv("FASTCONFORMER_QURAN_LM_PREFIX_MAX_SPAN", "4"))
LM_PREFIX_WORD_LIMIT = int(os.getenv("FASTCONFORMER_QURAN_LM_PREFIX_WORD_LIMIT", "8"))
LM_INVALID_PREFIX_PENALTY = float(os.getenv("FASTCONFORMER_QURAN_LM_INVALID_PREFIX_PENALTY", "0.0"))
LM_PARTIAL_INVALID_PENALTY = float(os.getenv("FASTCONFORMER_QURAN_LM_PARTIAL_INVALID_PENALTY", "0.0"))
LM_PARTIAL_VALID_BONUS = float(os.getenv("FASTCONFORMER_QURAN_LM_PARTIAL_VALID_BONUS", "0.0"))
LM_EOS_MATCH_BONUS = float(os.getenv("FASTCONFORMER_QURAN_LM_EOS_MATCH_BONUS", "0.0"))
LM_EOS_PREFIX_PENALTY = float(os.getenv("FASTCONFORMER_QURAN_LM_EOS_PREFIX_PENALTY", "0.0"))
LM_EOS_MISS_PENALTY = float(os.getenv("FASTCONFORMER_QURAN_LM_EOS_MISS_PENALTY", "0.0"))

# True KenLM path (used if available)
KENLM_MODEL_PATH = os.getenv("FASTCONFORMER_QURAN_KENLM_MODEL", "").strip()
KENLM_UNIGRAMS_PATH = os.getenv(
    "FASTCONFORMER_QURAN_KENLM_UNIGRAMS",
    str(PROJECT_ROOT / "data" / "kenlm" / "quran_unigrams.txt"),
).strip()
KENLM_ALPHA = float(os.getenv("FASTCONFORMER_QURAN_KENLM_ALPHA", "0.7"))
KENLM_BETA = float(os.getenv("FASTCONFORMER_QURAN_KENLM_BETA", "1.0"))

# Closed-vocab CTC stage
TOP_HYPOTHESES_FOR_SURAHS = int(os.getenv("FASTCONFORMER_QURAN_TOP_HYPS", "6"))
TOP_SURAHS_BRUTEFORCE = int(os.getenv("FASTCONFORMER_QURAN_TOP_SURAHS", "8"))
SEARCH_TOP_K = int(os.getenv("FASTCONFORMER_QURAN_SEARCH_TOPK", "8"))
MAX_SPAN = int(os.getenv("FASTCONFORMER_QURAN_MAX_SPAN", "6"))
SPAN_THRESHOLD = float(os.getenv("FASTCONFORMER_QURAN_SPAN_THRESHOLD", "0.25"))

# FastConformer source
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
MODEL_SIZE_BYTES = 115 * 1024 * 1024


# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_fastconformer_model = None
_beam_decoder = None
_decoder_lm = None
_db = None
_device = "cpu"


class QuranWordFusionLM(AbstractLanguageModel):
    """Word-level Quran LM + prefix constraints for pyctcdecode shallow fusion."""

    def __init__(
        self,
        db: QuranDB,
        order: int,
        smoothing: float,
        lm_weight: float,
        prefix_max_span: int,
        prefix_word_limit: int,
        invalid_prefix_penalty: float,
        partial_invalid_penalty: float,
        partial_valid_bonus: float,
        eos_match_bonus: float,
        eos_prefix_penalty: float,
        eos_miss_penalty: float,
    ):
        self._db = db
        self._order = max(2, order)
        self._smoothing = max(1e-6, smoothing)
        self._lm_weight = lm_weight

        self._prefix_max_span = max(1, prefix_max_span)
        self._prefix_word_limit = max(1, prefix_word_limit)
        self._invalid_prefix_penalty = invalid_prefix_penalty
        self._partial_invalid_penalty = partial_invalid_penalty
        self._partial_valid_bonus = partial_valid_bonus
        self._eos_match_bonus = eos_match_bonus
        self._eos_prefix_penalty = eos_prefix_penalty
        self._eos_miss_penalty = eos_miss_penalty

        # N-gram tables
        self._counts = defaultdict(Counter)
        self._totals = Counter()
        self._vocab = {"<eos>"}

        # Prefix constraints and final-span hits
        self._valid_prefixes = set()
        self._valid_full_sequences = set()

        # Partial-token constraints
        self._word_prefixes = set()

        self._build_resources()
        self._vocab_size = max(1, len(self._vocab))

    @property
    def order(self) -> int:
        return self._order

    @staticmethod
    def _split_words(text: str) -> list[str]:
        return [w for w in normalize_arabic(text).strip().split() if w]

    def _add_ngram_training_text(self, words: list[str]) -> None:
        if not words:
            return
        self._vocab.update(words)
        bos = ["<bos>"] * (self._order - 1)
        seq = bos + words + ["<eos>"]
        for i in range(self._order - 1, len(seq)):
            ctx = tuple(seq[i - self._order + 1:i])
            nxt = seq[i]
            self._counts[ctx][nxt] += 1
            self._totals[ctx] += 1

    def _build_resources(self) -> None:
        # Train LM on single-ayah text.
        for v in self._db.verses:
            words = self._split_words(v["text_clean"])
            self._add_ngram_training_text(words)

            for w in words:
                for i in range(1, len(w) + 1):
                    self._word_prefixes.add(w[:i])

        # Build prefix constraints from multi-ayah spans to support continuation.
        for surah in range(1, 115):
            verses = self._db.get_surah(surah)
            if not verses:
                continue
            verse_words = [self._split_words(v["text_clean"]) for v in verses]

            for start in range(len(verses)):
                acc = []
                for span in range(1, self._prefix_max_span + 1):
                    idx = start + span - 1
                    if idx >= len(verses):
                        break
                    acc.extend(verse_words[idx])
                    if not acc:
                        continue

                    seq = tuple(acc)
                    self._valid_full_sequences.add(seq)

                    for n in range(1, min(len(seq), self._prefix_word_limit) + 1):
                        self._valid_prefixes.add(seq[:n])

    def _ngram_log_prob(self, prev_words: tuple[str, ...], word: str) -> float:
        ctx = prev_words[-(self._order - 1):] if self._order > 1 else ()
        counts = self._counts.get(ctx)
        count = counts.get(word, 0) if counts else 0
        denom = self._totals.get(ctx, 0) + self._smoothing * self._vocab_size
        prob = (count + self._smoothing) / denom
        return math.log(prob)

    def get_start_state(self):
        return tuple()

    def score_partial_token(self, partial_token: str) -> float:
        token = normalize_arabic(partial_token).strip()
        if not token:
            return 0.0
        if token in self._word_prefixes:
            return self._partial_valid_bonus
        return self._partial_invalid_penalty

    def _eos_adjustment(self, words: tuple[str, ...]) -> float:
        if not words:
            return 0.0
        if words in self._valid_full_sequences:
            return self._eos_match_bonus
        if len(words) <= self._prefix_word_limit and words in self._valid_prefixes:
            return self._eos_prefix_penalty
        return self._eos_miss_penalty

    def score(self, prev_state, word: str, is_last_word: bool = False):
        prev_words = tuple(prev_state) if prev_state is not None else tuple()

        norm_word = normalize_arabic(word).strip()
        if not norm_word:
            score = self._eos_adjustment(prev_words) if is_last_word else 0.0
            return score, prev_words

        words = [w for w in norm_word.split() if w]
        if not words:
            score = self._eos_adjustment(prev_words) if is_last_word else 0.0
            return score, prev_words

        running = prev_words
        delta = 0.0

        for w in words:
            candidate = running + (w,)
            if len(candidate) <= self._prefix_word_limit and candidate not in self._valid_prefixes:
                delta += self._invalid_prefix_penalty

            delta += self._lm_weight * self._ngram_log_prob(running, w)
            running = candidate

        if is_last_word:
            delta += self._eos_adjustment(running)

        return delta, running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _install_kaldialign_fallback() -> None:
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _tokenize_for_ctc(text: str) -> list[int]:
    t = normalize_arabic(text).strip()
    if not t:
        return []

    tok = _fastconformer_model.tokenizer

    methods = []
    for name in ("text_to_ids", "encode_as_ids", "encode"):
        fn = getattr(tok, name, None)
        if callable(fn):
            methods.append(fn)

    inner = getattr(tok, "tokenizer", None)
    if inner is not None:
        for name in ("encode_as_ids", "encode"):
            fn = getattr(inner, name, None)
            if callable(fn):
                methods.append(fn)

    for fn in methods:
        try:
            ids = fn(t)
            if isinstance(ids, dict):
                ids = ids.get("input_ids") or ids.get("ids")
            if isinstance(ids, tuple):
                ids = list(ids)
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if not isinstance(ids, list):
                continue
            out = []
            for i in ids:
                out.append(int(i))
            if out:
                return out
        except Exception:
            continue

    return []


def _read_unigrams(path: str) -> list[str] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                w = normalize_arabic(line.strip())
                if w:
                    out.append(w)
        return out if out else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def _ensure_loaded():
    global _fastconformer_model, _beam_decoder, _decoder_lm, _db, _device, score_candidates

    if _fastconformer_model is not None:
        return

    if score_candidates is None:
        scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
        spec = importlib.util.spec_from_file_location("ctc_scorer", str(scorer_path))
        scorer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer_mod)
        score_candidates = scorer_mod.score_candidates

    _install_kaldialign_fallback()

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    except Exception as exc:
        raise ImportError(
            "NeMo ASR dependencies are required. Install with: pip install 'nemo_toolkit[asr]'"
        ) from exc

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    print(f"Loading FastConformer from {source} on {_device}...")

    try:
        _fastconformer_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=source,
            map_location=_device,
        )
    except Exception:
        if LOCAL_MODEL_DIR.exists():
            nemo_files = sorted(LOCAL_MODEL_DIR.glob("*.nemo"))
            if not nemo_files:
                raise
            _fastconformer_model = EncDecHybridRNNTCTCBPEModel.restore_from(
                str(nemo_files[0]),
                map_location=_device,
            )
        else:
            raise

    _fastconformer_model.eval()
    try:
        _fastconformer_model.change_decoding_strategy(decoder_type="ctc")
    except Exception:
        pass

    _db = QuranDB()

    vocab = _fastconformer_model.tokenizer.vocab
    labels = list(vocab) + [""]

    # Try true KenLM shallow fusion first, if provided.
    _beam_decoder = None
    if KENLM_MODEL_PATH:
        try:
            from pyctcdecode import build_ctcdecoder

            unigrams = _read_unigrams(KENLM_UNIGRAMS_PATH)
            _beam_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=KENLM_MODEL_PATH,
                unigrams=unigrams,
                alpha=KENLM_ALPHA,
                beta=KENLM_BETA,
            )
            print(
                "Built pyctcdecode KenLM decoder "
                f"(vocab={len(vocab)}, beam_width={DECODE_BEAM_WIDTH}, lm={KENLM_MODEL_PATH})"
            )
        except Exception as e:
            print(f"KenLM decoder init failed ({e}), falling back to custom LM")
            _beam_decoder = None

    # Fallback: custom in-process LM + prefix constraints.
    if _beam_decoder is None:
        try:
            alphabet = Alphabet.build_alphabet(labels)
            _decoder_lm = QuranWordFusionLM(
                db=_db,
                order=LM_ORDER,
                smoothing=LM_SMOOTHING,
                lm_weight=LM_FUSION_WEIGHT,
                prefix_max_span=LM_PREFIX_MAX_SPAN,
                prefix_word_limit=LM_PREFIX_WORD_LIMIT,
                invalid_prefix_penalty=LM_INVALID_PREFIX_PENALTY,
                partial_invalid_penalty=LM_PARTIAL_INVALID_PENALTY,
                partial_valid_bonus=LM_PARTIAL_VALID_BONUS,
                eos_match_bonus=LM_EOS_MATCH_BONUS,
                eos_prefix_penalty=LM_EOS_PREFIX_PENALTY,
                eos_miss_penalty=LM_EOS_MISS_PENALTY,
            )
            _beam_decoder = BeamSearchDecoderCTC(alphabet, language_model=_decoder_lm)
            print(
                "Built pyctcdecode custom-LM decoder "
                f"(vocab={len(vocab)}, beam_width={DECODE_BEAM_WIDTH}, order={LM_ORDER})"
            )
        except Exception as e:
            print(f"pyctcdecode custom LM setup failed ({e}), fallback to greedy")
            _beam_decoder = None


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _fastconformer_logits(audio_path: str) -> torch.Tensor:
    _ensure_loaded()

    audio = load_audio(audio_path)
    audio_signal = torch.tensor(audio, dtype=torch.float32, device=_device).unsqueeze(0)
    audio_length = torch.tensor([len(audio)], dtype=torch.long, device=_device)

    with torch.no_grad():
        processed_signal, processed_signal_length = _fastconformer_model.preprocessor(
            input_signal=audio_signal, length=audio_length
        )
        encoded, _encoded_len = _fastconformer_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        logits = _fastconformer_model.ctc_decoder(encoder_output=encoded)

    return logits


def _greedy_decode(logits: torch.Tensor) -> dict:
    greedy_ids = torch.argmax(logits, dim=-1)[0]
    blank_id = logits.shape[-1] - 1

    prev = -1
    tokens = []
    for idx in greedy_ids.tolist():
        if idx != prev and idx != blank_id:
            tokens.append(idx)
        prev = idx

    vocab = _fastconformer_model.tokenizer.vocab
    text = "".join(vocab[t] for t in tokens if t < len(vocab))
    text = normalize_arabic(text.replace("▁", " ").strip())

    return {
        "text": text,
        "acoustic": 0.0,
        "combined": 0.0,
    }


def _decode_nbest(logits: torch.Tensor) -> list[dict]:
    if _beam_decoder is None:
        return [_greedy_decode(logits)]

    lp = F.log_softmax(logits[0], dim=-1).detach().cpu().numpy()

    try:
        beams = _beam_decoder.decode_beams(
            lp,
            beam_width=DECODE_BEAM_WIDTH,
            beam_prune_logp=BEAM_PRUNE_LOGP,
            token_min_logp=TOKEN_MIN_LOGP,
            prune_history=False,
        )
    except Exception:
        return [_greedy_decode(logits)]

    by_text = {}
    for beam in beams:
        if not beam:
            continue
        text = normalize_arabic(str(beam[0]).strip())
        if not text:
            continue
        acoustic = float(beam[3]) if len(beam) > 3 else 0.0
        combined = float(beam[4]) if len(beam) > 4 else acoustic
        prev = by_text.get(text)
        if prev is None or combined > prev["combined"]:
            by_text[text] = {
                "text": text,
                "acoustic": acoustic,
                "combined": combined,
            }

    if not by_text:
        return [_greedy_decode(logits)]

    hyps = sorted(by_text.values(), key=lambda x: x["combined"], reverse=True)
    return hyps[:N_BEST]


# ---------------------------------------------------------------------------
# Candidate generation and scoring
# ---------------------------------------------------------------------------


def _collect_candidate_surahs(hypotheses: list[dict]) -> list[int]:
    ordered = []
    seen = set()

    def add_surah(surah: int) -> None:
        if surah in seen:
            return
        seen.add(surah)
        ordered.append(surah)

    for hyp in hypotheses[:TOP_HYPOTHESES_FOR_SURAHS]:
        text = hyp["text"]

        match = _db.match_verse(text, threshold=SPAN_THRESHOLD, max_span=MAX_SPAN)
        if match:
            add_surah(match["surah"])

        for r in _db.search(text, top_k=SEARCH_TOP_K):
            add_surah(r["surah"])

        if len(ordered) >= TOP_SURAHS_BRUTEFORCE:
            break

    return ordered[:TOP_SURAHS_BRUTEFORCE]


def _build_candidates(surahs: list[int]) -> list[dict]:
    candidates = []
    for surah_num in surahs:
        verses = _db.get_surah(surah_num)

        for v in verses:
            candidates.append(
                {
                    "surah": surah_num,
                    "ayah": v["ayah"],
                    "ayah_end": None,
                    "text_clean": v["text_clean"],
                    "text_uthmani": v["text_uthmani"],
                    "text": v["text_uthmani"],
                }
            )

        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                candidates.append(
                    {
                        "surah": surah_num,
                        "ayah": chunk[0]["ayah"],
                        "ayah_end": chunk[-1]["ayah"],
                        "text_clean": " ".join(v["text_clean"] for v in chunk),
                        "text_uthmani": " ".join(v["text_uthmani"] for v in chunk),
                        "text": " ".join(v["text_uthmani"] for v in chunk),
                    }
                )

    return candidates


def _best_match_from_hypotheses(hypotheses: list[dict]) -> dict | None:
    best = None
    best_score = 0.0
    for hyp in hypotheses:
        m = _db.match_verse(hyp["text"], threshold=SPAN_THRESHOLD, max_span=MAX_SPAN)
        if m and m["score"] > best_score:
            best_score = m["score"]
            best = m
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(audio_path: str) -> str:
    logits = _fastconformer_logits(audio_path)
    hyps = _decode_nbest(logits)
    return hyps[0]["text"] if hyps else ""


def predict(audio_path: str) -> dict:
    _ensure_loaded()

    logits = _fastconformer_logits(audio_path)
    hypotheses = _decode_nbest(logits)

    if not hypotheses:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }

    transcript = hypotheses[0]["text"]
    best_match = _best_match_from_hypotheses(hypotheses)

    if best_match and best_match["score"] >= CONFIDENCE_THRESHOLD:
        return {
            "surah": best_match["surah"],
            "ayah": best_match["ayah"],
            "ayah_end": best_match.get("ayah_end"),
            "score": best_match["score"],
            "transcript": transcript,
        }

    candidate_surahs = _collect_candidate_surahs(hypotheses)
    if not candidate_surahs:
        if best_match:
            return {
                "surah": best_match["surah"],
                "ayah": best_match["ayah"],
                "ayah_end": best_match.get("ayah_end"),
                "score": best_match["score"],
                "transcript": transcript,
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    candidates = _build_candidates(candidate_surahs)
    print(f"  [Constrained decode] {len(candidate_surahs)} surahs, {len(candidates)} candidates")

    blank_id = logits.shape[-1] - 1
    scored = score_candidates(
        logits.detach().cpu(),
        candidates,
        _tokenize_for_ctc,
        blank_id=blank_id,
    )

    if not scored:
        if best_match:
            return {
                "surah": best_match["surah"],
                "ayah": best_match["ayah"],
                "ayah_end": best_match.get("ayah_end"),
                "score": best_match["score"],
                "transcript": transcript,
            }
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    best_candidate, best_loss = scored[0]
    ctc_conf = math.exp(-best_loss) if math.isfinite(best_loss) else 0.0

    if best_match and best_match["score"] >= max(CONFIDENCE_THRESHOLD, ctc_conf):
        return {
            "surah": best_match["surah"],
            "ayah": best_match["ayah"],
            "ayah_end": best_match.get("ayah_end"),
            "score": best_match["score"],
            "transcript": transcript,
        }

    return {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(ctc_conf, 4),
        "transcript": transcript,
    }


def model_size() -> int:
    _ensure_loaded()
    if LOCAL_MODEL_DIR.exists():
        return _dir_size_bytes(LOCAL_MODEL_DIR)
    return MODEL_SIZE_BYTES
