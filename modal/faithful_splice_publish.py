"""Quantize the faithful-splice student and publish it as an isolated lab listener."""

import hashlib
import json
from pathlib import Path
from shutil import copyfile

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("onnx==1.19.0", "onnxruntime==1.22.1")
app = modal.App("tilawa-faithful-splice-publish", image=image)
student = modal.Volume.from_name("tilawa-student")
lab = modal.Volume.from_name("tilawa-lab")
LISTENER = "faithful-splice-int8"


@app.function(cpu=8, memory=32768, volumes={"/student": student, "/lab": lab}, timeout=3600)
def publish(run: str) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path("/student/runs") / run / "onnx"
    out = Path("/lab/data/models") / LISTENER
    out.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(src / "model.onnx"),
        str(out / "model.onnx"),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    data = (out / "model.onnx").read_bytes()
    manifest = json.loads((src / "manifest.json").read_text())
    manifest.update(
        id=LISTENER,
        format="onnx-int8",
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        quantizedFrom=manifest["sha256"],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    copyfile(src / "parity.json", out / "parity.json")
    lab.commit()
    return manifest


@app.local_entrypoint()
def main(run: str = "faithful-splice-v1"):
    print(publish.remote(run))
