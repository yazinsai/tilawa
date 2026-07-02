"""FastConformer Phoneme CTC experiment.

This experiment uses a phoneme-level CTC model for mispronunciation detection
and verse identification. The phoneme model outputs IPA-like phoneme sequences
which are compared against reference phoneme transcriptions.

Pipeline:
  audio -> Phoneme CTC model -> phoneme string
  audio -> Text model (nvidia-fastconformer) -> verse identification
  predicted phonemes + reference phonemes -> alignment -> error detection

Requires:
  - nemo_toolkit[asr]
  - shared.phoneme_aligner (for alignment)
  - data/quran_phonemes.json (reference phoneme data)
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import torch

from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_DIR = Path(
    os.getenv(
        "FASTCONFORMER_PHONEME_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "fastconformer-phoneme-v1"),
    )
)
MODEL_SIZE_BYTES = 115 * 1024 * 1024  # fallback estimate

# Must match the vocabulary used during training (order matters for ID mapping)
PHONEME_VOCAB = [
    # Vowels (12)
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    # Consonants (28)
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    # Geminated (28)
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    # Word boundary (1)
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)  # CTC blank at last index

_phoneme_model = None
_db = None
_phoneme_refs = None
_text_module = None


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
    """Load the phoneme CTC model and QuranDB."""
    global _phoneme_model, _db
    if _phoneme_model is not None:
        return

    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(MODEL_DIR)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Phoneme model directory not found: {model_dir}. "
            "Set FASTCONFORMER_PHONEME_MODEL_DIR or place model in data/fastconformer-phoneme-v1/"
        )

    nemo_files = sorted(model_dir.glob("*.nemo"))
    if not nemo_files:
        raise FileNotFoundError(
            f"No .nemo files found in {model_dir}. "
            "Expected a phoneme CTC model file (e.g. model.nemo)."
        )

    try:
        from nemo.utils import logging as nemo_logging
    except Exception:
        pass
    else:
        nemo_logging.set_verbosity(nemo_logging.ERROR)

    print(f"Loading phoneme CTC model from {nemo_files[0]} on {device}...")

    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    import torch.nn as nn
    import tarfile
    import tempfile as tmp_mod

    # NeMo's restore_from doesn't support strict=False. We manually:
    # 1. Load the default model from HuggingFace (original 1025-class)
    # 2. Replace CTC decoder head to 70 classes
    # 3. Load our fine-tuned weights from the .nemo checkpoint

    # Extract weights from .nemo archive
    with tmp_mod.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_files[0], "r:") as tar:
            tar.extractall(tmpdir)
        ckpt_path = Path(tmpdir) / "model_weights.ckpt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Determine CTC decoder size from checkpoint
    ctc_w = ckpt["ctc_decoder.decoder_layers.0.weight"]
    num_classes = ctc_w.shape[0]  # 70
    in_features = ctc_w.shape[1]  # 512

    # Load base model
    _phoneme_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        map_location=device,
    )

    # Replace CTC decoder to match our phoneme vocab
    _phoneme_model.ctc_decoder.decoder_layers[0] = nn.Conv1d(
        in_features, num_classes, kernel_size=1
    )

    # Load fine-tuned weights (now shapes match)
    _phoneme_model.load_state_dict(ckpt, strict=True)
    print(f"  Loaded fine-tuned weights, CTC decoder: {num_classes} classes")

    _phoneme_model.eval()
    _phoneme_model.to(device)

    # Switch to CTC decoding
    try:
        _phoneme_model.change_decoding_strategy(decoder_type="ctc")
    except Exception:
        pass

    _db = QuranDB()


def _ensure_text_model():
    """Lazily load the text (nvidia-fastconformer) model for verse identification."""
    global _text_module
    if _text_module is not None:
        return

    text_run_path = PROJECT_ROOT / "experiments" / "nvidia-fastconformer" / "run.py"
    if not text_run_path.exists():
        raise FileNotFoundError(
            f"Text model experiment not found: {text_run_path}"
        )

    import importlib.util
    spec = importlib.util.spec_from_file_location("nvidia_fastconformer_run", str(text_run_path))
    _text_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_text_module)


def _load_phoneme_refs():
    """Load reference phoneme sequences from data/quran_phonemes.json."""
    global _phoneme_refs
    if _phoneme_refs is not None:
        return

    ref_path = PROJECT_ROOT / "data" / "quran_phonemes.json"
    if not ref_path.exists():
        print(f"Warning: Reference phonemes not found at {ref_path}")
        _phoneme_refs = {}
        return

    with open(ref_path) as f:
        data = json.load(f)

    _phoneme_refs = {}
    for entry in data:
        _phoneme_refs[(entry["surah"], entry["ayah"])] = entry["phonemes"]


def transcribe_phonemes(audio_path: str) -> str:
    """Transcribe audio to a space-separated phoneme string.

    Uses manual CTC greedy decoding since the model's BPE tokenizer
    doesn't know about our phoneme vocabulary.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Space-separated phoneme string.
    """
    _ensure_loaded()

    audio = load_audio(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(tmp_path, audio, 16000)

        # Run encoder + CTC decoder manually
        import numpy as np
        audio_signal = torch.tensor(np.expand_dims(audio, 0), dtype=torch.float32)
        audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)

        device = next(_phoneme_model.parameters()).device
        audio_signal = audio_signal.to(device)
        audio_len = audio_len.to(device)

        with torch.no_grad():
            encoded, encoded_len = _phoneme_model.forward(
                input_signal=audio_signal, input_signal_length=audio_len
            )
            log_probs = _phoneme_model.ctc_decoder(encoder_output=encoded)

        # CTC greedy decode: argmax -> collapse -> remove blanks
        ids = log_probs[0].argmax(dim=-1).cpu().tolist()
        length = encoded_len[0].item()
        ids = ids[:length]

        # Collapse consecutive duplicates
        collapsed = []
        prev = None
        for idx in ids:
            if idx != prev:
                collapsed.append(idx)
            prev = idx

        # Map to phoneme tokens, skip blanks
        tokens = []
        for idx in collapsed:
            if idx == BLANK_ID:
                continue
            if 0 <= idx < len(PHONEME_VOCAB):
                tokens.append(PHONEME_VOCAB[idx])

        return " ".join(tokens)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def detect_mispronunciations(audio_path: str, surah: int, ayah: int) -> dict:
    """Detect mispronunciations by comparing predicted vs reference phonemes.

    Args:
        audio_path: Path to the audio file.
        surah: Surah number (1-114).
        ayah: Ayah number.

    Returns:
        Dict with predicted_phonemes, reference_phonemes, errors list, and PER.
    """
    _ensure_loaded()
    _load_phoneme_refs()

    predicted = transcribe_phonemes(audio_path)

    ref_key = (surah, ayah)
    if ref_key not in _phoneme_refs:
        return {
            "predicted_phonemes": predicted,
            "reference_phonemes": "",
            "errors": [],
            "per": 0.0,
            "error": f"No reference phonemes for surah {surah}, ayah {ayah}",
        }

    reference = _phoneme_refs[ref_key]

    from shared.phoneme_aligner import align_phoneme_strings
    alignment = align_phoneme_strings(predicted, reference)

    return {
        "predicted_phonemes": predicted,
        "reference_phonemes": reference,
        "errors": alignment["errors"],
        "per": alignment["per"],
    }


def transcribe(audio_path: str) -> str:
    """Transcribe audio to Arabic text for verse identification.

    Delegates to the nvidia-fastconformer text model. The benchmark runner
    feeds this through the streaming pipeline (run_on_full_transcript) which
    uses QuranDB span matching — handles multi-verse recordings properly.
    """
    _ensure_text_model()
    return _text_module.transcribe(audio_path)


def model_size() -> int:
    """Return the model file size in bytes."""
    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        return sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return MODEL_SIZE_BYTES
