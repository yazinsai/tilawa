"""Export FastConformer phoneme CTC model to ONNX and quantize for web deployment.

Defaults to the restored local v4-tlog checkpoint, but callers can override the
checkpoint path explicitly to export other runs without silently changing the
shipped browser artifact.

Output:
  - web/frontend/public/fastconformer_phoneme_q8.onnx
  - web/frontend/public/phoneme_vocab.json
  - web/frontend/public/export_metadata.json
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must match training vocabulary exactly
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
NUM_CLASSES = len(PHONEME_VOCAB)  # 69
BLANK_ID = NUM_CLASSES  # 69 = blank; total outputs = 70

DEFAULT_NEMO_CHECKPOINT = (
    PROJECT_ROOT / "data" / "fastconformer-phoneme-v4-tlog" / "model" / "model.nemo"
)
WEB_PUBLIC = PROJECT_ROOT / "web" / "frontend" / "public"
METADATA_PATH = WEB_PUBLIC / "export_metadata.json"


def resolve_checkpoint_path(checkpoint_override: str | None) -> Path:
    if checkpoint_override:
        return Path(checkpoint_override).expanduser().resolve()

    env_override = os.environ.get("OFFLINE_TARTEEL_NEMO_CHECKPOINT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    return DEFAULT_NEMO_CHECKPOINT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_metadata(checkpoint_path: Path) -> None:
    onnx_path = WEB_PUBLIC / "fastconformer_phoneme.onnx"
    q8_path = WEB_PUBLIC / "fastconformer_phoneme_q8.onnx"
    vocab_path = WEB_PUBLIC / "phoneme_vocab.json"

    missing = [path for path in (onnx_path, q8_path, vocab_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot write metadata; missing files: {missing}")

    with vocab_path.open() as f:
        vocab = json.load(f)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "default_checkpoint_path": str(DEFAULT_NEMO_CHECKPOINT),
        "onnx_path": str(onnx_path),
        "q8_path": str(q8_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "q8_size_bytes": q8_path.stat().st_size,
        "onnx_sha256": _sha256(onnx_path),
        "q8_sha256": _sha256(q8_path),
        "vocab_path": str(vocab_path),
        "vocab_sha256": _sha256(vocab_path),
        "vocab_tokens": len(vocab),
        "variants": [
            {"name": "fp32", "path": str(onnx_path), "size_bytes": onnx_path.stat().st_size},
            {"name": "q8", "path": str(q8_path), "size_bytes": q8_path.stat().st_size},
        ],
        "output_name": checkpoint_path.parent.parent.name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved: {METADATA_PATH}")


def export_onnx(checkpoint_path: Path):
    import nemo.collections.asr as nemo_asr
    import torch

    model_id = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    onnx_path = WEB_PUBLIC / "fastconformer_phoneme.onnx"
    vocab_path = WEB_PUBLIC / "phoneme_vocab.json"

    print(f"Loading base model: {model_id}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    # Replace CTC head (ctc_decoder, not decoder which is RNNT)
    vocab_size = NUM_CLASSES + 1  # 69 phonemes + blank = 70
    old_decoder = model.ctc_decoder
    if hasattr(old_decoder, "decoder_layers") and len(old_decoder.decoder_layers) > 0:
        last_layer = old_decoder.decoder_layers[-1]
        in_features = getattr(last_layer, "in_channels", getattr(last_layer, "in_features", 512))
        old_decoder.decoder_layers[-1] = torch.nn.Conv1d(in_features, vocab_size, kernel_size=1)
        print(f"Replaced CTC head: Conv1d({in_features}, {vocab_size})")
    else:
        in_features = model.cfg.encoder.get("d_model", 512)
        model.ctc_decoder = torch.nn.Linear(in_features, vocab_size)
        print(f"Replaced entire ctc_decoder: Linear({in_features}, {vocab_size})")

    # Load fine-tuned weights
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading fine-tuned weights from: {checkpoint_path}")
    import tempfile, tarfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(checkpoint_path, "r:*") as tar:
            tar.extractall(tmpdir)
        weights_path = Path(tmpdir) / "model_weights.ckpt"
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Export CTC-only ONNX
    model.set_export_config({"decoder_type": "ctc"})
    print(f"Exporting to: {onnx_path}")
    model.export(str(onnx_path))
    print(f"ONNX export complete: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Save phoneme vocabulary
    vocab = {}
    for i, token in enumerate(PHONEME_VOCAB):
        vocab[str(i)] = token
    vocab[str(BLANK_ID)] = "<blank>"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved: {vocab_path} ({len(vocab)} tokens)")


def validate_onnx():
    """Validate ONNX phoneme model produces correct output."""
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    onnx_path = WEB_PUBLIC / "fastconformer_phoneme_q8.onnx"
    if not onnx_path.exists():
        onnx_path = WEB_PUBLIC / "fastconformer_phoneme.onnx"
    vocab_path = WEB_PUBLIC / "phoneme_vocab.json"

    with open(vocab_path) as f:
        vocab = json.load(f)
    blank_id = max(int(k) for k in vocab.keys())

    # Find test audio
    test_files = sorted((PROJECT_ROOT / "benchmark" / "test_corpus").glob("*.wav"))
    if not test_files:
        print("No test audio files found. Skipping validation.")
        return
    test_audio = test_files[0]
    print(f"Validating with: {test_audio.name}")

    # Load and preprocess audio
    audio_data, sr = sf.read(str(test_audio), dtype="float32")
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    # Run ONNX inference
    import torch
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    )
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    audio_length = torch.tensor([audio_tensor.shape[1]])
    processed, proc_len = model.preprocessor(
        input_signal=audio_tensor, length=audio_length
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    ort_inputs = {
        input_names[0]: processed.detach().numpy(),
        input_names[1]: proc_len.detach().numpy(),
    }
    logprobs = sess.run(None, ort_inputs)[0]

    # CTC greedy decode
    predicted_ids = np.argmax(logprobs[0], axis=-1)
    tokens = []
    prev = -1
    for idx in predicted_ids:
        if idx != prev and idx != blank_id:
            tokens.append(vocab.get(str(idx), "?"))
        prev = idx
    phoneme_str = " ".join(tokens)
    print(f"Phoneme output: {phoneme_str[:200]}")

    # Join into words
    words = []
    current_word = []
    for tok in tokens:
        if tok == "|":
            if current_word:
                words.append("".join(current_word))
            current_word = []
        else:
            current_word.append(tok)
    if current_word:
        words.append("".join(current_word))
    print(f"Joined words:   {' '.join(words)}")


def quantize():
    """Quantize the ONNX model to uint8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    input_path = WEB_PUBLIC / "fastconformer_phoneme.onnx"
    output_path = WEB_PUBLIC / "fastconformer_phoneme_q8.onnx"

    if not input_path.exists():
        print(f"ONNX model not found: {input_path}")
        return

    print(f"Quantizing: {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)")
    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QUInt8)
    print(f"Quantized:  {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FastConformer phoneme ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the .nemo checkpoint to export. Defaults to the restored "
            "v4-tlog checkpoint, or OFFLINE_TARTEEL_NEMO_CHECKPOINT if set."
        ),
    )
    parser.add_argument("--validate", action="store_true", help="Validate an existing ONNX export.")
    parser.add_argument("--quantize", action="store_true", help="Quantize an existing fp32 ONNX export.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation after export.")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if args.validate:
        validate_onnx()
    elif args.quantize:
        quantize()
        write_export_metadata(checkpoint_path)
    else:
        export_onnx(checkpoint_path)
        quantize()
        write_export_metadata(checkpoint_path)
        if not args.skip_validate:
            validate_onnx()
