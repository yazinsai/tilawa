"""Export FastConformer phoneme CTC model to ONNX on Modal, quantize, and download.

Usage:
  # Export fp32 + q8 + metadata and download to web/frontend/public/
  modal run scripts/export_phoneme_onnx_modal.py

  # Export only (no download)
  modal run scripts/export_phoneme_onnx_modal.py --no-download

  # Download previously exported model
  modal run scripts/export_phoneme_onnx_modal.py --download-only
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import modal

app = modal.App("fastconformer-phoneme-export")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "onnxruntime>=1.16",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
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


def _install_kaldialign_fallback() -> None:
    """Install kaldialign fallback for NeMo import compatibility."""
    import sys
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref, hyp = list(ref), list(hyp)
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i][j - 1] + 1, dp[i - 1][j] + 1)
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/vol": vol},
)
def export_and_quantize(output_name: str = "fastconformer-phoneme-v2"):
    """Export the fine-tuned phoneme model to ONNX and quantize to uint8."""
    import os
    import tarfile
    import tempfile

    import torch

    _install_kaldialign_fallback()
    import nemo.collections.asr as nemo_asr
    from onnxruntime.quantization import QuantType, quantize_dynamic

    vol.reload()

    checkpoint_path = f"/vol/{output_name}/model/model.nemo"
    out_dir = Path(f"/vol/{output_name}/export")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "fastconformer_phoneme.onnx"
    q8_path = out_dir / "fastconformer_phoneme_q8.onnx"
    vocab_path = out_dir / "phoneme_vocab.json"
    metadata_path = out_dir / "export_metadata.json"
    if not os.path.exists(checkpoint_path):
        # Try listing what's available on the entire volume
        print("Checkpoint not at expected path, searching volume...")
        vol_root = Path("/vol")
        print(f"Volume root contents: {list(vol_root.iterdir()) if vol_root.exists() else 'NOT FOUND'}")
        for p in sorted(vol_root.rglob("*.nemo")):
            print(f"  Found: {p}")
            checkpoint_path = str(p)
        # Also check for .ckpt files
        if not os.path.exists(checkpoint_path):
            for p in sorted(vol_root.rglob("*.ckpt")):
                print(f"  Found ckpt: {p}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No .nemo checkpoint found on volume")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load base model
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)

    # Move model to CPU for export
    model = model.cpu()

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

    # Load fine-tuned weights from .nemo checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # .nemo files can be tar.gz or just tar
        try:
            with tarfile.open(checkpoint_path, "r:gz") as tar:
                tar.extractall(tmpdir)
        except tarfile.ReadError:
            with tarfile.open(checkpoint_path, "r:") as tar:
                tar.extractall(tmpdir)

        weights_path = Path(tmpdir) / "model_weights.ckpt"
        if not weights_path.exists():
            # Search for the weights file
            for p in Path(tmpdir).rglob("*.ckpt"):
                print(f"  Found weights: {p}")
                weights_path = p
                break
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys:    {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()

    # Export ONNX (CTC-only)
    model.set_export_config({"decoder_type": "ctc"})
    print(f"Exporting ONNX to: {onnx_path}")
    model.export(str(onnx_path))
    print(f"ONNX export: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Quantize to uint8
    print(f"Quantizing to uint8...")
    quantize_dynamic(str(onnx_path), str(q8_path), weight_type=QuantType.QUInt8)
    print(f"Quantized:   {q8_path.stat().st_size / 1e6:.1f} MB")

    # Save phoneme vocabulary
    vocab = {}
    for i, token in enumerate(PHONEME_VOCAB):
        vocab[str(i)] = token
    vocab[str(BLANK_ID)] = "<blank>"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary: {vocab_path} ({len(vocab)} tokens)")

    metadata = {
        "checkpoint_path": checkpoint_path,
        "onnx_path": str(onnx_path),
        "q8_path": str(q8_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "q8_size_bytes": q8_path.stat().st_size,
        "vocab_path": str(vocab_path),
        "vocab_tokens": len(vocab),
        "variants": [
            {"name": "fp32", "path": str(onnx_path), "size_bytes": onnx_path.stat().st_size},
            {"name": "q8", "path": str(q8_path), "size_bytes": q8_path.stat().st_size},
        ],
    }
    metadata["output_name"] = output_name
    # Embed training metadata if available
    training_meta_path = Path(f"/vol/{output_name}/model/training_metadata.json")
    if training_meta_path.exists():
        try:
            training_meta = json.loads(training_meta_path.read_text(encoding="utf-8"))
            metadata["training_metadata"] = training_meta
        except Exception:
            pass

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata:   {metadata_path}")

    # Commit volume
    vol.commit()
    print("Done! Files saved to Modal volume.")

    return {
        "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "q8_size_mb": round(q8_path.stat().st_size / 1e6, 1),
        "vocab_tokens": len(vocab),
        "metadata_path": str(metadata_path),
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/vol": vol},
    timeout=600,
)
def read_file(path: str) -> bytes:
    """Read a file from the volume and return its contents."""
    vol.reload()
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    output_name: str = "fastconformer-phoneme-v2",
    no_download: bool = False,
    download_only: bool = False,
):
    local_dir = Path(__file__).resolve().parent.parent / "web" / "frontend" / "public"
    local_dir.mkdir(parents=True, exist_ok=True)

    if not download_only:
        print("=== Exporting and quantizing on Modal ===")
        result = export_and_quantize.remote(output_name=output_name)
        print(f"\nExport results: {result}")

    # Download files
    if not no_download:
        print("\n=== Downloading files ===")
        export_dir = f"/vol/{output_name}/export"
        files = [
            (f"{export_dir}/fastconformer_phoneme.onnx", "fastconformer_phoneme.onnx"),
            (f"{export_dir}/fastconformer_phoneme_q8.onnx", "fastconformer_phoneme_q8.onnx"),
            (f"{export_dir}/phoneme_vocab.json", "phoneme_vocab.json"),
            (f"{export_dir}/export_metadata.json", "export_metadata.json"),
        ]
        for remote_path, local_name in files:
            print(f"Downloading {local_name}...")
            data = read_file.remote(remote_path)
            local_path = local_dir / local_name
            local_path.write_bytes(data)
            print(f"  Saved: {local_path} ({len(data) / 1e6:.1f} MB)")

        print("\nAll files downloaded to web/frontend/public/")
