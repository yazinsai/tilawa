# Preserve canonical near-repeats during tracker acquisition

Status: implemented; practice result pending. Fold or discard belongs to the user.

Test set: lab-v1.4.v3, practice only. Base: main 5716d49; its engine source is identical to the stored practice baseline b544715 (14 wrong calls, 11 missed and 3 false warnings over 88.136 minutes).

## Prediction, before scoring

Three fewer tracker-attributed and overall phone wrong calls. Preserve clean recitation and studio; inspect false jumps, on-place share and lock delay. No listener, checker, scorer or test-set changes.

## Cause and change

The p01 and p18 startup traces first lock to 25:57 at 5.00 and 8.80 seconds. Their winner leads by 1.1 weighted edits. The tracker has collapsed part of the canonical phrase مِن أجر إن أجري as a near-repeat. The shortened evidence now favours 25:57 over the five shared passages in surah 26. This causes the p01 warnings at 25:57:8 and :14 and the p18 warning at :8.

Before accepting a proposed repeat collapse, compare the best global reference fit of the full and shortened evidence. Preserve the full evidence whenever its cost is no greater. This uses the existing alignment costs; it adds no duration, confidence or case-specific threshold.

The initial prediction proposed startup persistence using the existing jump span. That prototype failed cadence and chunk-equivalence tests and was removed before any practice score. The refined hypothesis and original prediction are both preserved in lab-experiment.json and the ledger. The final guard addresses the corrupted evidence directly.

## Validation

Real first-22-second student phone prefixes from p01 and p18 reproduce wrong locks on main. Two regression tests fail on main and pass with this guard, checking both tracker locks and checker warnings. The unique single-push control passes. Existing state, cadence, reacquisition, phrase, lead-in, local-tracking and jump suites, plus the TypeScript build, pass on Modal.

Red run: startup-red-f40f66c-20261030T174745Z. Passing run: startup-canonical-unit-f40f66c-dirty-20261030T175154Z. The earlier persistence prototype's rejected test run is startup-green-f40f66c-dirty-20261030T174846Z.

Practice-only baseline traces: /runs/wrong-calls-probe-5716d49-20261030T174304Z/{0,1}/probe.json on tilawa-lab. No exam scoring or inspection.
