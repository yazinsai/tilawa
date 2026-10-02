"""Expose the frozen encoder frames as an ONNX output. No weights change.

The CTC matmul reads ``/Transpose_226_output_0``. Adding that tensor to the
graph outputs leaves ``log_probs`` on the same path (the previous probe
checked bit-identity). The int8 graph quantizes a copy of it for the CTC
matmul; the exposed tensor is the fp32 encoder output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ENC = "/Transpose_226_output_0"


def expose_encoder(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    produced = {out for node in model.graph.node for out in node.output}
    if ENC not in produced:
        raise SystemExit(f"{src} has no {ENC}")
    if not any(out.name == ENC for out in model.graph.output):
        model.graph.output.append(helper.make_tensor_value_info(ENC, TensorProto.FLOAT, [1, 12, 512]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    # Reload and confirm both outputs exist. Do not print the whole graph.
    check = onnx.load(str(dst))
    names = [out.name for out in check.graph.output]
    if "log_probs" not in names or ENC not in names:
        raise SystemExit(f"bad outputs: {names[:4]}")
    print(f"encoder output added  bytes={dst.stat().st_size}")


if __name__ == "__main__":
    expose_encoder(Path(sys.argv[1]), Path(sys.argv[2]))
