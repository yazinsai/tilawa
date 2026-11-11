# Tracker reference acceptance volume

Status: needs-human

## Prediction

Tracker −1, overall 0: recovering the near continuation could remove the target false
skipped-passage warning while exposing the previously documented nearby listener
warning. This is an exploratory experiment, with no standalone fold promised.

## Mechanism

For a reference pronunciation `r` and observed-token count `n`, estimate
`Z(r,n) = E_background[exp(-C(X,r))]`. Rank local reference alternatives with
`C(heard,r) + log Z(r,n)`, keeping movement penalties separate. A pronunciation
that matches many unrelated phone strings should lose its generic acoustic
advantage. The target restart currently beats its available near continuation by
0.3915 plain edits; the training-frozen log-volume difference must exceed that
advantage before the premise can survive.

The prepared estimator uses only none-labelled help-train recordings from training
speakers, decoded by the shipped streaming student. It fits marginal initial-phone
frequencies and a first-order phone transition distribution, with fixed half-count
smoothing. Neither labels from practice nor passage or reader identity enter the
runtime parameters. Five speaker-grouped partitions within training provide
predictive checks. No model selection uses practice.

The sampler fixes its seed and 4,096 shared samples before any target measurement.
Each sample has a deterministic stream, so longer strings extend the same samples
used for shorter lengths. Each sampled string contributes its minimum acoustic
edit likelihood once; alternative alignments do not add counting weight. The cache
uses only pronunciation and observed length. The acoustic cost uses the existing
directed substitution, insertion and deletion prices with fixed entry and an open
terminal suffix, matching anchored local fitting rather than charging an unfinished
word's remaining phones.

## Work prepared

- `packages/core/src/acceptance-volume.ts`: training-only phone-background fitting,
  deterministic shared sampling, minimum acoustic cost, stable log-volume
  estimation and a pronunciation/length cache.
- `packages/core/test/acceptance-volume.test.ts`: acoustic-price and endpoint
  equivalence, once-per-sample estimation, broad-reference penalties, deterministic
  shared prefixes, and held-out-speaker/scripted-edit exclusion.
- `.scratch/acceptance-volume/train.ts`: clean training-speaker decoding, source
  audio hashes, frozen background output and grouped training-only checks.
- `.scratch/acceptance-volume/trace.ts`: disposable instrumentation of the existing
  local search snapshots and unchanged product replay; it does not modify the
  engine or scorer source.
- `.scratch/acceptance-volume/compare.ts`: a diagnostic comparison and sample-block
  uncertainty output. Its range of lengths is a diagnostic, not a substitute for
  recovering the target's actual common comparison interval.

## Validation and limits

All four supervised Modal requests returned
`NotFoundError: No such file or directory`, with no output directory or command
log. The first requested read-only inspection of the authorized training export;
the second requested build, focused tests, fitting and target replay; the third
requested the prescribed six-shard build/core/SDK suite; the fourth requested the
corrected estimator, focused tests, fitting, replay and volume comparison. Their
result records report `usd: 0.0`. The error gives no precise failure stage, so
even execution of these commands cannot be confirmed.

Source inspection verified that local acoustic costs use the same directed fuzzy
prices and leave unconsumed terminal reference phones uncharged. Tests have been
written but have not been observed to pass. No training decode or fit has been
verified, no target volume difference has been received, and no target warning
replay has been verified. Practice scoring was invoked
zero times. No numeric result or target-win count is claimed.

Tracker ranking is intentionally not wired to untrained parameters. Runtime
transitions, commitment margins and checker behavior remain at baseline. This is
an infrastructure block, not a killed premise or a completed experiment. Restore
the supervisor's execution environment, rerun fitting and the snapshot audit,
recover the actual common observation interval, and compare the frozen volume
difference with 0.3915 before implementing and replaying the rank change. If the
premise or target-win rule kills the experiment, remove its engine code and tests
and record the measured result here.

The read-only snapshot tool takes session, listener, near-word reference,
restart-word reference and time in seconds as arguments. The comparison tool takes
the returned snapshot file and time. For this incident those diagnostic arguments
are `lab-v1.5-p46 exp-listener-teacher-insert-marginal-gate-ad6eb8c0 54:36:5 54:37:9
121.84`. They are not engine parameters. Fitting reads only `LAB_DATA/help-train`
and the shipped student bundle and writes its frozen result to `LAB_OUT` and the
authorized persistent experiment scratch directory.
