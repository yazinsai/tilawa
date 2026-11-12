#!/usr/bin/env bash
# Fixed training pipeline; practice target inspection is a separate command.
set -euo pipefail
duration_work=/scratch/listener-duration-matched-hard-negatives/fit
mkdir -p "$duration_work"
python3 modal/student_duration_negatives_export.py --model "$LAB_DATA/models/spliced-clean-int8" --out "$duration_work"
# LAB_DATA/help-train excludes release-held-out speakers. Also remove any speaker
# associated with a non-training row before extracting or matching any recording.
python3 - "$duration_work/excluded-speakers.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps([]))
PY
pnpm --filter @tilawa/core build
pnpm --filter @tilawa/sdk build
pnpm --filter @tilawa/sdk exec tsx src/bench/duration-negatives-cli.ts \
  --model "$LAB_DATA/models/spliced-clean-int8" --extractor "$duration_work" \
  --labels "$LAB_DATA/help-train/labels.jsonl" --excluded-speakers "$duration_work/excluded-speakers.json" \
  --out "$duration_work/training.jsonl"
python3 modal/test_student_duration_negatives.py
python3 modal/student_duration_negatives.py --input "$duration_work/training.jsonl" \
  --metadata "$duration_work/duration-negatives-features-metadata.json" \
  --excluded-speakers "$duration_work/excluded-speakers.json" --out "$duration_work"
cp "$duration_work/duration-negatives-training-report.json" "$LAB_OUT/"
bundle=/scratch/listener-duration-matched-hard-negatives/bundle
mkdir -p "$bundle"
# Preserve the primary model, manifest and existing insertion assets exactly.
cp -a "$LAB_DATA/models/spliced-clean-int8/." "$bundle/"
cp "$duration_work"/duration-negatives{.json,-assets.json,-features.onnx,-matched.json,-random.json} "$bundle/"
pnpm --filter @tilawa/sdk exec tsx src/bench/duration-negatives-cli.ts \
  --model "$bundle" --parity "$duration_work" --out "$LAB_OUT/export-parity.json"
