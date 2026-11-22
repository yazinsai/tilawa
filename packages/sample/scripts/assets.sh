#!/usr/bin/env bash
# Links every registry tier's exported model, onnxruntime-web's wasm and the device page's clips into public/ (gitignored).
# Tier `<id>@1-<variant>` comes from the main checkout's `.scratch/models/<id>-<variant>` (`<id>` for fp32;
# `modal run modal/student_export.py`, then `tools/quantize_onnx.py`). MODELS_DIR overrides `.scratch/models`.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(git rev-parse --path-format=absolute --git-common-dir)/..
models=${MODELS_DIR:-$root/.scratch/models}
mkdir -p public/ort
node -e '
const r = require("../sdk/src/models-v1.json");
for (const t of r.tiers) console.log(t.base, t.id + (t.version.endsWith("-fp32") ? "" : "-" + t.version.split("-").slice(1).join("-")));
' | while read -r base dir; do
  if [ -f "$models/$dir/model.onnx" ]; then
    mkdir -p "public/$(dirname "$base")"
    ln -sfn "$models/$dir" "public/${base%/}"
  else
    echo "No model at $models/$dir for $base (skipped)" >&2
  fi
done
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.* public/ort/
node scripts/clips.mjs "$root"
