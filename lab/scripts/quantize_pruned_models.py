"""Quantize pruned CTC models to int8 for deployment.

Supports:
  - PyTorch dynamic int8 (qnnpack backend, works on macOS ARM)
  - ONNX export + ORT dynamic int8

Usage:
    .venv/bin/python scripts/quantize_pruned_models.py \
        --model-path data/rabah-ctc-pruned-8l-evenly_spaced-finetuned \
        --format both
"""

import argparse
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def quantize_pytorch_int8(model_path: Path, output_path: Path) -> int:
    """Apply PyTorch dynamic int8 quantization with qnnpack backend."""
    print(f"Loading model from {model_path}...")
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
    model.eval()

    # Force qnnpack backend (fixes "No engine was found" on macOS ARM)
    torch.backends.quantized.engine = "qnnpack"
    print(f"Quantization backend: {torch.backends.quantized.engine}")

    print("Applying dynamic int8 quantization...")
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), str(output_path / "pytorch_model_int8.bin"))

    model.config.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    size = _dir_size_bytes(output_path)
    print(f"PyTorch int8 saved to {output_path} ({size / 1e6:.1f} MB)")
    return size


def quantize_onnx_int8(model_path: Path, output_path: Path) -> int:
    """Export to ONNX and apply ORT dynamic int8 quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ONNX quantization requires: pip install onnx onnxruntime")
        return 0

    print(f"Loading model from {model_path}...")
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    onnx_fp32_path = output_path / "model_fp32.onnx"
    onnx_int8_path = output_path / "model_int8.onnx"

    print("Exporting to ONNX (fp32)...")
    dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_fp32_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=14,
    )

    print("Quantizing ONNX to int8...")
    quantize_dynamic(
        str(onnx_fp32_path),
        str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )

    # Remove fp32 ONNX to save space
    onnx_fp32_path.unlink()

    processor.save_pretrained(str(output_path))

    size = _dir_size_bytes(output_path)
    print(f"ONNX int8 saved to {output_path} ({size / 1e6:.1f} MB)")
    return size


def main():
    parser = argparse.ArgumentParser(description="Quantize pruned CTC models to int8")
    parser.add_argument(
        "--model-path", required=True, help="Path to fine-tuned pruned model"
    )
    parser.add_argument(
        "--format",
        choices=["pytorch", "onnx", "both"],
        default="pytorch",
        help="Quantization format (default: pytorch)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to sibling of model-path)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    base_output = Path(args.output_dir) if args.output_dir else model_path.parent

    if args.format in ("pytorch", "both"):
        pytorch_out = base_output / f"{model_path.name}-pytorch-int8"
        quantize_pytorch_int8(model_path, pytorch_out)

    if args.format in ("onnx", "both"):
        onnx_out = base_output / f"{model_path.name}-onnx-int8"
        quantize_onnx_int8(model_path, onnx_out)


if __name__ == "__main__":
    main()
