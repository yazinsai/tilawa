"""NVIDIA FastConformer Arabic experiment.

This experiment uses NeMo's hybrid RNNT+CTC Arabic model:
  nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0

Pipeline:
  audio -> FastConformer transcript -> QuranDB span-aware match

Note: Requires `nemo_toolkit[asr]`.
"""

import os
import sys
import tempfile
from pathlib import Path
import types

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import torch

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
MODEL_SIZE_BYTES = 115 * 1024 * 1024
DECODER_TYPE = os.getenv("NVIDIA_FASTCONFORMER_DECODER", "ctc")
SPAN_THRESHOLD = float(os.getenv("NVIDIA_FASTCONFORMER_SPAN_THRESHOLD", "0.25"))
MAX_SPAN = int(os.getenv("NVIDIA_FASTCONFORMER_MAX_SPAN", "4"))

_model = None
_db = None


def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback when package is absent.

    NeMo imports `kaldialign` in context-biasing utilities, even for inference
    flows that do not use those codepaths. This fallback unblocks model import.
    """
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


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _ensure_loaded():
    global _model, _db
    if _model is not None:
        return

    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from nemo.utils import logging as nemo_logging
    except Exception as exc:
        raise ImportError(
            "NeMo ASR dependencies are required for nvidia-fastconformer. "
            "Install with: pip install 'nemo_toolkit[asr]'"
        ) from exc

    nemo_logging.set_verbosity(nemo_logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    print(f"Loading NVIDIA FastConformer from {source} on {device}...")

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

    # Prefer CTC decode path for text stability on short utterances.
    try:
        _model.change_decoding_strategy(decoder_type=DECODER_TYPE)
    except Exception:
        pass

    _db = QuranDB()


def transcribe(audio_path: str) -> str:
    _ensure_loaded()

    audio = load_audio(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(tmp_path, audio, 16000)
        try:
            outputs = _model.transcribe(
                audio=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
                verbose=False,
            )
        except TypeError:
            outputs = _model.transcribe(
                paths2audio_files=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
            )

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list) and outputs:
            transcript = _extract_text(outputs[0])
        else:
            transcript = _extract_text(outputs)

        return normalize_arabic(transcript)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def predict(audio_path: str) -> dict:
    transcript = transcribe(audio_path)

    if not transcript.strip():
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }

    best = _db.match_verse(transcript, threshold=SPAN_THRESHOLD, max_span=MAX_SPAN)
    if not best:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": best.get("ayah_end"),
        "score": best["score"],
        "transcript": transcript,
    }


def model_size() -> int:
    if LOCAL_MODEL_DIR.exists():
        return sum(f.stat().st_size for f in LOCAL_MODEL_DIR.rglob("*") if f.is_file())
    return MODEL_SIZE_BYTES
