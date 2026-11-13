#!/usr/bin/env bash
# Fixed training pipeline; practice target inspection is a separate command.
set -euo pipefail
span_work=/scratch/listener-acoustic-span-contrast/fit
span_model=$(python3 modal/student_acoustic_span_bundle.py resolve \
  --card lab-experiment.json --models "$LAB_DATA/models")
mkdir -p "$span_work"
python3 modal/student_acoustic_span_export.py --model "$span_model" --out "$span_work"
# LAB_DATA/help-train excludes release-held-out speakers. Also remove any speaker
# associated with a non-training row before extracting or matching any recording.
python3 - "$span_work/excluded-speakers.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps([]))
PY
pnpm --filter @tilawa/core build
pnpm --filter @tilawa/sdk build
pnpm --filter @tilawa/sdk exec tsx src/bench/acoustic-span-cli.ts \
  --model "$span_model" --extractor "$span_work" \
  --labels "$LAB_DATA/help-train/labels.jsonl" --excluded-speakers "$span_work/excluded-speakers.json" \
  --out "$span_work/training.jsonl"
python3 modal/test_student_acoustic_span.py
python3 modal/test_student_acoustic_span_bundle.py
python3 modal/student_acoustic_span.py --input "$span_work/training.jsonl" \
  --metadata "$span_work/acoustic-span-features-metadata.json" \
  --excluded-speakers "$span_work/excluded-speakers.json" --out "$span_work"
cp "$span_work/acoustic-span-training-report.json" "$LAB_OUT/"
bundle=$(mktemp -d /scratch/listener-acoustic-span-contrast/bundle.XXXXXX)
python3 modal/student_acoustic_span_bundle.py assemble \
  --model "$span_model" --fitted "$span_work" --out "$bundle" \
  --report "$LAB_OUT/bundle-inheritance.json"
pnpm --filter @tilawa/sdk exec tsx src/bench/acoustic-span-cli.ts \
  --model "$bundle" --parity "$span_work" --out "$LAB_OUT/export-parity.json"
