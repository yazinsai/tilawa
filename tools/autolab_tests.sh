#!/usr/bin/env bash
# Autolab validation: build, then this shard of the core and sdk tests (LAB_SHARD/LAB_SHARDS from modal/lab_run.py).
# test/checker.test.ts runs past Vitest's fixed 60 s worker RPC deadline on a 2-CPU Modal container, which fails the run
# with 'Timeout calling "onTaskUpdate"' after every test passed; that one error, with nothing failed, counts as a pass.
set -o pipefail
pnpm --filter @tilawa/core... --filter @tilawa/sdk... build || exit 1
shard="--shard=$((LAB_SHARD + 1))/$LAB_SHARDS"
for pkg in @tilawa/core @tilawa/sdk; do
  out=$(pnpm --filter "$pkg" exec vitest run --passWithNoTests "$shard" 2>&1)
  code=$?
  echo "$out"
  if [ $code -ne 0 ]; then
    if echo "$out" | grep -q 'Timeout calling "onTaskUpdate"' && echo "$out" | grep -qE 'Tests +[0-9]+ passed' \
      && ! echo "$out" | grep -qE '(Tests|Test Files) +.*failed'; then
      echo "autolab: only Vitest's RPC deadline failed in $pkg; every test passed"
    else
      exit $code
    fi
  fi
done
