# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx==1.19.0", "onnxruntime==1.22.1", "onnxconverter-common==1.16.0", "numpy<2.3"]
# ///
"""Quantised variants of a streaming student export (ticket 10).

    uv run --python cpython-3.12-macos-aarch64-none tools/quantize_onnx.py .scratch/models/spliced-clean int8
    ... fp16 | int8-qdq (static, calibrated on help clips' fbank windows; needs --calib <dir of .wav>)

Writes `<src>-<variant>/{model.onnx,manifest.json}` next to the source dir. int8 is icefall's recipe: dynamic,
MatMul weights only (ConvInteger is missing from onnxruntime-web). fp16 keeps fp32 inputs and outputs, so the SDK's
state tensors and fbank don't change.
"""

import hashlib
import json
import sys
from pathlib import Path


def int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)


# Ops compared against fp32 constants in the Zipformer graph; converting them breaks type binding at load.
BLOCK = ["Equal", "Where"]


def fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    m = onnx.load(str(src))
    m = convert_float_to_float16(m, keep_io_types=True, op_block_list=DEFAULT_OP_BLOCK_LIST + BLOCK)
    onnx.save(m, str(dst))


def main() -> None:
    src, variant = Path(sys.argv[1]).resolve(), sys.argv[2]
    out = src.parent / f"{src.name}-{variant}"
    out.mkdir(exist_ok=True)
    {"int8": int8, "fp16": fp16}[variant](src / "model.onnx", out / "model.onnx")
    data = (out / "model.onnx").read_bytes()
    m = json.loads((src / "manifest.json").read_text())
    m.update(format=f"onnx-{variant}", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), quantizedFrom=m["sha256"])
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"{out}: {len(data) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
