#!/usr/bin/env bash
# Training-only preparation and frozen calibration. Target inspection is a separate command.
set -euo pipefail
completeness_work=/scratch/listener-pronunciation-completeness
mkdir -p "$completeness_work/fit"
python3 modal/student_completeness_export.py --model "$LAB_DATA/models/spliced-clean-int8" --out "$completeness_work/fit"
# The exported help-train set already excludes release-held-out speakers. The
# preparer additionally excludes every speaker appearing in any non-train split.
python3 - "$completeness_work/excluded-speakers.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text('[]')
PY
pnpm --filter @tilawa/core build
pnpm --filter @tilawa/sdk build
pnpm --filter @tilawa/sdk exec tsx src/bench/completeness-cli.ts \
  --model "$LAB_DATA/models/spliced-clean-int8" --extractor "$completeness_work/fit" \
  --labels "$LAB_DATA/help-train/labels.jsonl" --excluded-speakers "$completeness_work/excluded-speakers.json" \
  --out "$completeness_work/fit/training.jsonl"
python3 modal/test_student_pronunciation_completeness.py
python3 modal/student_pronunciation_completeness.py --input "$completeness_work/fit/training.jsonl" \
  --metadata "$completeness_work/fit/completeness-features-metadata.json" \
  --excluded-speakers "$completeness_work/fit/excluded-speakers.json" --out "$completeness_work/fit"
cp "$completeness_work/fit/completeness-training-report.json" "$LAB_OUT/"
