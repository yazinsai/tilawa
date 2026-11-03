#!/bin/bash
# SDK streaming decode of one target session with the wide int8 student, then true-position replay.
set -euo pipefail
ids=(lab-v1.5-p31 lab-v1.5-p50 lab-v1.5-p55 lab-v1.5-p56 lab-v1.5-p79)
shard="${LAB_SHARD:-0}"
if [ "$shard" -ge "${#ids[@]}" ]; then
  echo "idle shard $shard"
  exit 0
fi
id="${ids[$shard]}"
pnpm --filter @tilawa/core build
d=$(mktemp -d)
mkdir -p "$d/wavs"
ln -s "/lab/data/lab-v1.5/sessions/${id}.phone.wav" "$d/wavs/${id}.wav"
echo "decode $id" 
pnpm --filter @tilawa/sdk decode --model /scratch/listener-wide-ear-refit-calibration/wide-clean-int8 --wavs "$d/wavs" --out "$d/out" --push 20
mkdir -p /scratch/listener-wide-ear-refit-calibration/sdk-evidence
cp "$d/out/${id}.evidence.json" /scratch/listener-wide-ear-refit-calibration/sdk-evidence/
cp "$d/out/${id}.evidence.json" "$LAB_OUT/"
echo "replay $id bytes $(wc -c < "$d/out/${id}.evidence.json")"
args=()
case "$id" in
  lab-v1.5-p31) ats="55.47 288.67" ;;
  lab-v1.5-p50) ats="146.47" ;;
  lab-v1.5-p55) ats="269.44" ;;
  lab-v1.5-p56) ats="106.76" ;;
  lab-v1.5-p79) ats="136.10" ;;
esac
for at in $ats; do args+=(--session "$id" --at "$at"); done
pnpm --filter @tilawa/core exec tsx ../../tools/replay-targets.ts --data /lab/data/lab-v1.5/sessions --evidence "$d/out" "${args[@]}"
