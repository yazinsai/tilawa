"""c2c-direct — CTC re-rank of FastConformer's own log-probs.

audio -> ctc log-probs -> greedy transcript
        -> three candidate retrieval strategies (trigram+Lev, Lev search, spaceless scan)
        -> multi-ayah spans around top single-verse hits
        -> confidence-gated: text match >= threshold skips the slow step
        -> otherwise: batched F.ctc_loss against all candidates using
           the SAME [T, 1025] log-prob matrix; pick best by -ctc_norm_loss

The key insight: re-scoring FastConformer's own log-probs corrects
*decoding* errors (BPE merge/split, argmax on tied tokens) — not
acoustic errors. A separate Stage 2 model (our old fastconformer-ctc-rescore)
fails the same way on the same hard cases. Same model for ASR + re-score
gives the re-rank real signal to find a better interpretation.

The same exported CTC head is used for both the greedy transcript (text
match) and the candidate re-rank, so the failure modes are aligned and
the re-rank can recover from bad argmax decisions.

On the upstream v1 (53 samples, multi-verse spans) the reference
implementation hits 99.2% recall / 0.3 s latency / 115 MB declared.
This port is intentionally small: no new deps beyond what `nvidia-fastconformer`
already requires.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from Levenshtein import ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
MODEL_SIZE_BYTES = (
    115 * 1024 * 1024
)  # legacy: matches upstream claim; replaced by real file size

# Tunables (all overridable via env)
TOP_TEXT = int(os.getenv("CTC_DIRECT_TOP_TEXT", "100"))
TOP_SPAN_REFS = int(os.getenv("CTC_DIRECT_TOP_SPAN_REFS", "80"))
MAX_SPAN = int(os.getenv("CTC_DIRECT_MAX_SPAN", "6"))
CTC_BATCH = int(os.getenv("CTC_DIRECT_BATCH", "16"))
FALLBACK_THRESHOLD = float(os.getenv("CTC_DIRECT_THRESHOLD", "0.80"))
TEXT_WEIGHT = float(os.getenv("CTC_DIRECT_TEXT_WEIGHT", "0.0"))
# Penalty per extra verse in a multi-ayah span. Default 0.5 (was 0.05) to
# avoid the false-positive span in retasy_019 (3:1+3:2 instead of 3:2).
# With 0.5, the 2-verse span needs ctc_norm_loss to be 0.5 better than
# the single verse -- in practice this is rare, so we prefer the single
# verse. Real multi-verse recitations (e.g. multi_002_285_286) still
# pick the 2-verse span because the alignment advantage is bigger.
SPAN_PENALTY = float(os.getenv("CTC_DIRECT_SPAN_PENALTY", "0.5"))

VOCAB_SIZE = 1025
BLANK_ID = VOCAB_SIZE - 1

_model = None
_db = None
_token_ids_cache: dict[str, list[int]] = {}


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
                bt[i][j] = "S" if best == sub else "I" if best == ins else "D"
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


def _ensure_loaded():
    global _model, _db
    if _model is not None:
        return

    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils import logging as nemo_logging

    nemo_logging.set_verbosity(nemo_logging.ERROR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    print(f"[c2c-direct] loading FastConformer from {source} on {device}...")
    try:
        _model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=source,
            map_location=device,
        )
    except Exception:
        if LOCAL_MODEL_DIR.exists():
            nemo_files = sorted(LOCAL_MODEL_DIR.glob("*.nemo"))
            if not nemo_files:
                raise
            _model = EncDecHybridRNNTCTCBPEModel.restore_from(
                str(nemo_files[0]),
                map_location=device,
            )
        else:
            raise
    _model.eval()

    _db = QuranDB()


def _ctc_logprobs(audio_path: str) -> np.ndarray:
    """Return [T, 1025] CTC log-probs (variable T)."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    device = next(_model.parameters()).device
    with torch.no_grad():
        sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        length = torch.tensor([sig.shape[1]], dtype=torch.int64).to(device)
        processed, plen = _model.preprocessor(input_signal=sig, length=length)
        encoded, elen = _model.encoder(audio_signal=processed, length=plen)
        log_probs = _model.ctc_decoder(encoder_output=encoded)
        lp = log_probs[0].detach().cpu().numpy().astype(np.float32)
        t = int(elen[0]) if elen is not None else lp.shape[0]
        return lp[:t]


def _greedy_decode(log_probs: np.ndarray) -> str:
    """Argmax + CTC collapse (drop repeated tokens + blanks), then let the
    SentencePiece tokenizer do the BPE→text merge. This is the same path
    NeMo's transcribe() takes internally and is the only reliable way to
    decode the BPE vocabulary used by stt_ar_fastconformer_hybrid_large.
    """
    ids = log_probs.argmax(-1)
    deduped: list[int] = []
    prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK_ID:
            deduped.append(i)
        prev = i
    if not deduped:
        return ""
    # ids_to_text uses SentencePiece internal decoder to merge BPE pieces.
    return normalize_arabic(_model.tokenizer.ids_to_text(deduped).strip())


def _span_len(c: dict) -> int:
    return (c.get("ayah_end") or c["ayah"]) - c["ayah"] + 1


def _candidate_text(c: dict) -> str:
    return c.get("ctc_text") or c.get("text_clean") or ""


def _token_ids(text: str) -> list[int]:
    cached = _token_ids_cache.get(text)
    if cached is not None:
        return cached
    token_ids = _model.tokenizer.text_to_ids(text)
    _token_ids_cache[text] = token_ids
    return token_ids


def _add_candidate(out: list[dict], seen: set, cand: dict) -> None:
    c = dict(cand)
    c["ayah_end"] = c.get("ayah_end") or c["ayah"]
    if not c.get("ctc_text"):
        c["ctc_text"] = c.get("text_clean") or ""
    key = (c["surah"], c["ayah"], c["ayah_end"])
    if key not in seen and c["ctc_text"].strip():
        seen.add(key)
        out.append(c)


def _make_span(surah: int, start: int, end: int) -> dict | None:
    chunk = [_db.get_verse(surah, ayah) for ayah in range(start, end + 1)]
    if any(v is None for v in chunk):
        return None
    first = chunk[0].get("text_clean_no_bsm") or chunk[0]["text_clean"]
    text = " ".join([first] + [v["text_clean"] for v in chunk[1:]])
    return {
        "surah": surah,
        "ayah": start,
        "ayah_end": end,
        "text_clean": text,
        "ctc_text": text,
        "score": 0.0,
    }


def _build_candidates(transcript: str) -> tuple[list[dict], dict | None]:
    """Three retrieval strategies + multi-ayah spans."""
    out: list[dict] = []
    seen: set = set()
    single_refs: list[tuple[int, int]] = []

    # Pass 1: trigram-indexed match (returns single verse OR span).
    base = _db.match_verse(
        transcript,
        threshold=0.0,
        max_span=MAX_SPAN,
        return_top_k=TOP_TEXT,
        use_trigram_index=True,
    )
    if base:
        _add_candidate(out, seen, base)
        single_refs.append((base["surah"], base["ayah"]))
        for runner_up in base.get("runners_up", []):
            verse = _db.get_verse(runner_up["surah"], runner_up["ayah"])
            if verse:
                c = dict(verse)
                c["score"] = runner_up.get("score", 0.0)
                _add_candidate(out, seen, c)
                single_refs.append((c["surah"], c["ayah"]))

    # Pass 2: full-search top-k (catches what trigram index missed).
    for verse in _db.search(transcript, top_k=TOP_TEXT):
        _add_candidate(out, seen, verse)
        single_refs.append((verse["surah"], verse["ayah"]))

    # Pass 3: full Lev scan, taking max of spaced + spaceless match. The
    # spaceless comparison handles BPE splits that strip the inter-word
    # space marker from the model's output.
    norm = transcript
    spaceless = norm.replace(" ", "")
    scored = []
    for verse in _db.verses:
        s = max(
            ratio(norm, verse["text_clean"]),
            ratio(spaceless, verse["text_clean"].replace(" ", "")),
        )
        scored.append((s, verse))
    for s, verse in sorted(scored, key=lambda x: x[0], reverse=True)[:TOP_TEXT]:
        c = dict(verse)
        c["score"] = s
        _add_candidate(out, seen, c)
        single_refs.append((c["surah"], c["ayah"]))

    # Multi-ayah span candidates around the top single hits.
    for surah, ayah in single_refs[:TOP_SPAN_REFS]:
        verses = _db.get_surah(surah)
        max_ayah = len(verses)
        for start in range(max(1, ayah - MAX_SPAN + 1), min(ayah, max_ayah) + 1):
            for end in range(
                max(ayah, start + 1), min(max_ayah, start + MAX_SPAN - 1) + 1
            ):
                span = _make_span(surah, start, end)
                if span:
                    _add_candidate(out, seen, span)

    return out, base


def _ctc_rerank(log_probs_np: np.ndarray, candidates: list[dict]) -> list[dict]:
    """Batched F.ctc_loss against FastConformer's own [T, 1025] log-probs.

    Skips candidates whose token sequence is too long to align within T
    (CTC requires target_len * 2 + 1 <= T, otherwise zero_infinity kicks in
    and the loss becomes meaningless).
    """
    if not candidates:
        return []

    t, vocab = log_probs_np.shape
    log_probs = torch.from_numpy(log_probs_np).unsqueeze(1)  # [T, 1, V]

    encoded: list[list[int]] = []
    feasible_indices: list[int] = []
    lengths: list[int] = []
    for i, cand in enumerate(candidates):
        token_ids = _token_ids(_candidate_text(cand))
        if token_ids and len(token_ids) * 2 + 1 <= t:
            encoded.append(token_ids)
            feasible_indices.append(i)
            lengths.append(len(token_ids))

    for cand in candidates:
        cand["ctc_loss"] = float("inf")
        cand["ctc_norm_loss"] = float("inf")
        cand["ctc_len"] = 0
        cand["final_score"] = -float("inf")

    for start in range(0, len(encoded), CTC_BATCH):
        batch_tokens = encoded[start : start + CTC_BATCH]
        batch_indices = feasible_indices[start : start + CTC_BATCH]
        batch_lengths = lengths[start : start + CTC_BATCH]
        n = len(batch_tokens)
        targets = torch.tensor(
            [tok for seq in batch_tokens for tok in seq],
            dtype=torch.long,
        )
        target_lengths = torch.tensor(batch_lengths, dtype=torch.long)
        input_lengths = torch.full((n,), t, dtype=torch.long)
        losses = F.ctc_loss(
            log_probs.expand(t, n, vocab).contiguous(),
            targets,
            input_lengths,
            target_lengths,
            blank=BLANK_ID,
            reduction="none",
            zero_infinity=True,
        )
        norm_losses = losses / target_lengths.float()
        for idx, raw_loss, norm_loss, length in zip(
            batch_indices,
            losses.tolist(),
            norm_losses.tolist(),
            batch_lengths,
        ):
            cand = candidates[idx]
            cand["ctc_loss"] = float(raw_loss)
            cand["ctc_norm_loss"] = float(norm_loss)
            cand["ctc_len"] = int(length)
            text_score = float(cand.get("score") or 0.0)
            penalty = SPAN_PENALTY * (_span_len(cand) - 1)
            cand["final_score"] = -float(norm_loss) + TEXT_WEIGHT * text_score - penalty

    ranked = [c for c in candidates if math.isfinite(c["ctc_norm_loss"])]
    ranked.sort(key=lambda c: c["final_score"], reverse=True)
    return ranked


def _empty(transcript: str = "") -> dict:
    return {
        "surah": 0,
        "ayah": 0,
        "ayah_end": None,
        "score": 0.0,
        "transcript": transcript,
        "candidates": [],
    }


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    log_probs = _ctc_logprobs(audio_path)
    transcript = _greedy_decode(log_probs)
    if not transcript.strip():
        return _empty("")

    candidates, base = _build_candidates(transcript)
    if not candidates and not base:
        return _empty(transcript)

    # Confidence gate: high-confidence text match -> skip the slow CTC step.
    use_ctc = base is None or float(base.get("score", 0.0)) < FALLBACK_THRESHOLD
    ranked = _ctc_rerank(log_probs, candidates) if use_ctc else []

    if use_ctc and ranked:
        best = ranked[0]
        source = "ctc"
        score = (
            math.exp(-best["ctc_norm_loss"])
            if math.isfinite(best["ctc_norm_loss"])
            else 0.0
        )
    elif base:
        best = base
        source = "text"
        score = float(base.get("score", 0.0))
    else:
        return _empty(transcript)

    out_candidates = []
    for cand in ranked[:5] if ranked else [best]:
        out_candidates.append(
            {
                "surah": cand["surah"],
                "ayah": cand["ayah"],
                "ayah_end": cand.get("ayah_end") or cand["ayah"],
                "score": round(
                    float(cand.get("final_score", cand.get("score", 0.0))), 4
                ),
            }
        )

    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": best.get("ayah_end") or best["ayah"],
        "score": round(score, 4),
        "transcript": transcript,
        "source": source,
        "candidates": out_candidates,
    }


def transcribe(audio_path: str) -> str:
    """Stage-1 only — required by the streaming benchmark path."""
    _ensure_loaded()
    return _greedy_decode(_ctc_logprobs(audio_path))


def model_size() -> int:
    """Real on-disk size of the cached .nemo (whichever path it lives at).

    Falls back to the legacy 115 MB constant only when the model hasn't
    been resolved from HF or a local dir.
    """
    if LOCAL_MODEL_DIR.exists():
        return sum(f.stat().st_size for f in LOCAL_MODEL_DIR.rglob("*") if f.is_file())
    # Walk the HF hub cache (the .nemo is always cached there after first run)
    hf_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--nvidia--stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    )
    if hf_root.exists():
        blobs = hf_root / "blobs"
        if blobs.exists():
            total = sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
            if total > 0:
                return total
    return MODEL_SIZE_BYTES
