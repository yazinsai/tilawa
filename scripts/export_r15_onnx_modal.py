"""Export and quantize Ahmed's r15 phoneme CTC model on Modal.

The upstream int8 repo (`hetchyy/r15_95m_onnx_int8`) currently 404s, while the
fp32 PyTorch model (`hetchyy/r15_95m`) is accessible with an HF token. This
script creates our own ONNX artifacts and stores them on the Modal volume
`w2v-phonemes-r15`.

Usage:
  # One-time, if no Modal HF secret exists yet:
  # modal secret create hf-token HF_TOKEN=...

  # Export fp32 ONNX + dynamic int8 ONNX to the Modal volume
  modal run --detach scripts/export_r15_onnx_modal.py --no-download

  # Download previously exported artifacts into data/r15-onnx/
  modal run scripts/export_r15_onnx_modal.py --download-only --output-dir data/r15-onnx
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("w2v-phonemes-r15-export")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.5",
        "transformers>=4.40",
        "huggingface_hub>=0.23",
        "onnx>=1.16",
        "onnxruntime>=1.18",
        "soundfile>=0.12",
    )
)

vol = modal.Volume.from_name("w2v-phonemes-r15", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

MODEL_ID = "hetchyy/r15_95m"
EXPORT_DIR = Path("/vol/r15_95m_onnx")


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def export_and_quantize():
    import os
    import time

    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForCTC, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; create Modal secret `hf-token` with HF_TOKEN=...")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = EXPORT_DIR / "model_fp32.onnx"
    int8_path = EXPORT_DIR / "model_int8.onnx"
    metadata_path = EXPORT_DIR / "export_metadata.json"

    started = time.time()
    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCTC.from_pretrained(MODEL_ID, token=token)
    model.eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_values):
            return self.wrapped(input_values=input_values).logits

    export_model = LogitsOnly(model).eval()

    # Keep this long enough to include multiple Wav2Vec2 conv/attention frames,
    # but small enough that export is quick. Dynamic axes handle longer audio.
    dummy = torch.randn(1, 16000 * 10)

    print(f"Exporting fp32 ONNX to {fp32_path}...")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy,),
            str(fp32_path),
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "time"},
            },
            opset_version=17,
        )

    print(f"Quantizing dynamic int8 ONNX to {int8_path}...")
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    print("Saving processor/tokenizer files...")
    processor.save_pretrained(str(EXPORT_DIR))

    files = sorted(
        {
            path.name: path.stat().st_size
            for path in EXPORT_DIR.iterdir()
            if path.is_file()
        }.items()
    )
    metadata = {
        "model_id": MODEL_ID,
        "artifacts_dir": str(EXPORT_DIR),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "fp32_size_bytes": fp32_path.stat().st_size,
        "int8_size_bytes": int8_path.stat().st_size,
        "files": [{"name": name, "size_bytes": size} for name, size in files],
        "export_seconds": round(time.time() - started, 2),
        "torch_version": torch.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    vol.commit()
    print(json.dumps(metadata, indent=2))
    return metadata


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/vol": vol},
    timeout=600,
)
def read_file(path: str) -> bytes:
    vol.reload()
    return Path(path).read_bytes()


@app.local_entrypoint()
def main(
    no_download: bool = False,
    download_only: bool = False,
    output_dir: str = "data/r15-onnx",
):
    if not download_only:
        result = export_and_quantize.remote()
        print(f"Export result: {json.dumps(result, indent=2)}")

    if no_download:
        return

    local_dir = Path(output_dir)
    if not local_dir.is_absolute():
        local_dir = Path(__file__).resolve().parent.parent / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(read_file.remote(str(EXPORT_DIR / "export_metadata.json")))
    for item in metadata["files"]:
        name = item["name"]
        remote_path = str(EXPORT_DIR / name)
        print(f"Downloading {remote_path}...")
        (local_dir / name).write_bytes(read_file.remote(remote_path))

    print(f"Downloaded artifacts to {local_dir}")
