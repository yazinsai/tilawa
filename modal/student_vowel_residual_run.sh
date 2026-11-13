#!/usr/bin/env bash
# Fixed training selection; any practice inspection is a separate diagnostic run.
set -euo pipefail
residual_work=/scratch/listener-vowel-conditioned-residual/fit
residual_model=$(python3 modal/student_vowel_residual_bundle.py resolve --card lab-experiment.json --models "$LAB_DATA/models")
mkdir -p "$residual_work"
python3 modal/student_vowel_residual_export.py --model "$residual_model" --out "$residual_work"
pnpm --filter @tilawa/core build
pnpm --filter @tilawa/sdk build
pnpm --filter @tilawa/sdk exec tsx src/bench/vowel-residual-cli.ts --model "$residual_model" \
  --extractor "$residual_work" --labels "$LAB_DATA/help-train/labels.jsonl" --require-auxiliaries --out "$residual_work/training.jsonl"
python3 modal/test_student_vowel_residual.py
python3 modal/student_vowel_residual.py --input "$residual_work/training.jsonl" \
  --metadata "$residual_work/vowel-residual-features-metadata.json" \
  --extraction-report "$residual_work/vowel-residual-extraction-report.json" \
  --excluded-speakers "$residual_work/vowel-residual-excluded-speakers.json" --out "$residual_work"
cp "$residual_work/vowel-residual-training-report.json" "$LAB_OUT/"
residual_bundle=$(mktemp -d /scratch/listener-vowel-conditioned-residual/bundle.XXXXXX)
python3 modal/student_vowel_residual_bundle.py assemble --model "$residual_model" --fitted "$residual_work" \
  --out "$residual_bundle" --report "$LAB_OUT/bundle-inheritance.json"
pnpm --filter @tilawa/sdk exec tsx src/bench/vowel-residual-cli.ts --model "$residual_bundle" \
  --parity "$residual_work" --out "$LAB_OUT/export-parity.json"
# Publish the path only after bundle integrity and exported runtime parity pass.
python3 - "$residual_bundle" "$LAB_OUT/bundle.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({'path':str(Path(sys.argv[1]).relative_to('/scratch/listener-vowel-conditioned-residual'))}))
PY
