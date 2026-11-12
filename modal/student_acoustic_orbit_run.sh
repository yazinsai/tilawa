#!/usr/bin/env bash
# Bounded training recipe, invoked only through the authorized supervisor.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo 'usage: bash modal/student_acoustic_orbit_run.sh PRIMARY LABELS EXCLUDED_SPEAKERS OUTPUT' >&2
  exit 2
fi
orbit_primary="$1"
orbit_labels="$2"
orbit_excluded="$3"
orbit_output="$4"
mkdir -p "$orbit_output"
# Preserve the complete primary bundle and every already declared side asset.
python3 - "$orbit_primary" "$orbit_output" <<'PY'
import shutil,sys
from pathlib import Path
source,out=map(Path,sys.argv[1:])
if source.resolve()==out.resolve():raise ValueError('Training destination must be separate from primary')
for path in source.iterdir():
    dest=out/path.name
    if path.is_dir():shutil.copytree(path,dest,dirs_exist_ok=True)
    else:shutil.copy2(path,dest)
PY
pnpm --filter @tilawa/core... --filter @tilawa/sdk... build
python3 modal/student_acoustic_orbit_export.py --model "$orbit_primary" --out "$orbit_output"
pnpm --filter @tilawa/sdk exec tsx src/bench/acoustic-orbit-cli.ts \
  --model "$orbit_primary" --labels "$orbit_labels" --excluded-speakers "$orbit_excluded" --out "$orbit_output/transcripts.json"
python3 modal/student_acoustic_orbit.py --prepare --model "$orbit_primary" --labels "$orbit_labels" \
  --reference "$orbit_output/transcripts.json" --metadata "$orbit_output/acoustic-orbit-metadata.json" \
  --excluded-speakers "$orbit_excluded" --input "$orbit_output/training.npz" --out "$orbit_output"
python3 modal/student_acoustic_orbit.py --metadata "$orbit_output/acoustic-orbit-metadata.json" \
  --excluded-speakers "$orbit_excluded" --input "$orbit_output/training.npz" --out "$orbit_output"
python3 -m unittest discover -s modal -p test_student_acoustic_orbit.py -v
pnpm --filter @tilawa/sdk exec tsx src/bench/acoustic-orbit-cli.ts \
  --model "$orbit_output" --verify --out "$orbit_output/verified.json"
