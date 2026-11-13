#!/usr/bin/env bash
set -euo pipefail
pnpm --filter @tilawa/core... --filter @tilawa/sdk... build
python3 -m pip install --quiet numpy
pnpm --filter @tilawa/core exec vitest run test/location-margin.test.ts
pnpm --filter @tilawa/core exec tsx /app/packages/core/train/location-parity.ts "$LAB_OUT/location-parity.json"
LOCATION_PARITY="$LAB_OUT/location-parity.json" python3 packages/core/train/test_location_margin.py
