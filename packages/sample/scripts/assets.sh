#!/usr/bin/env bash
# Links the exported model and onnxruntime-web's wasm into public/ (gitignored).
# MODEL_DIR defaults to the main checkout's `.scratch/models/spliced-clean` (`modal run modal/student_export.py`).
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(git rev-parse --path-format=absolute --git-common-dir)/..
model=${MODEL_DIR:-$root/.scratch/models/spliced-clean}
mkdir -p public/models/spliced-clean public/ort
[ -f "$model/model.onnx" ] || { echo "No model at $model; run modal/student_export.py or set MODEL_DIR" >&2; exit 1; }
ln -sfn "$model" public/models/spliced-clean/1-fp32
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.* public/ort/
