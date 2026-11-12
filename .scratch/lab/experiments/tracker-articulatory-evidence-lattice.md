# Joint articulatory observation lattice

Status: needs-human — authorized Modal validation infrastructure failed. Prototype implemented; no fitted model is active.

## Prediction

Card `tracker-articulatory-evidence-lattice` predicts tracker -1 and overall 0: recovering the false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84` may expose the previously documented nearby listener false warning. No additional target wins are assumed. The experiment card is unchanged.

## Mechanism

The local tracker consumes hard emitted phones. `trackerInput` explicitly removes frame posteriors before both product and practice replay. The prototype retains a compact nonblank posterior distribution on each emitted token while continuing to remove the full rows from acquisition input. It averages unconditioned rows over the token's original inclusive frame interval before conditioning on nonblank, reconstructing omitted top-k mass uniformly. Original token identities, confidence and timestamps are unchanged, and listener evidence is not mutated.

`articulatory-channel.ts` projects phone probabilities onto categorical, complete place/manner/voicing bundles. It sums probabilities of phones sharing each complete bundle; it never multiplies independently plausible feature marginals. Thus uncertainty between a voiced bilabial stop and an unvoiced alveolar stop cannot manufacture posterior mass for a voiced alveolar stop. Bundles intentionally omit distinctions outside the card's three features, including emphasis and vowel duration; the existing phone identity and duration charges remain in place.

The observation model is a normalized emission matrix conditioned on reference bundles, with a separate insertion row. A pair-channel forward/backward calculation marginalizes reference-to-observation placement and allocates fractional emission counts, including uncertain observed category. Phone-pair priors derived from existing acoustic substitution costs are pooled into complete bundle pairs. The estimator also fits a phone-identity control on exactly the same recordings, posterior distributions, placement graph and regularization choices.

At runtime the conditional-versus-pooled observation likelihood contributes in the existing three-nats-per-edit units to every consuming local transition. Ordinary continuation and restart diagonals receive the identical contribution for the same reference phone and observation. The reference-only deletion closure, movement edges and prices, duration charges, commitment margins, acquisition, events and checker handling are preserved. A pooled emission model contributes zero. Missing evidence or an inventory mismatch falls back to the existing local recurrence.

The SDK's training-only `articulatory-fit.ts` reads only `LAB_DATA/help-train` and the supplied deployed model. It selects scored, none-labelled, event-free, repeat-free, non-excluded train recordings and excludes every speaker having any non-train label. Deterministic speaker groups select smoothing mass separately for joint and identity models; the models' differing categorical alphabets mean their raw training log likelihoods are not compared directly. It saves fitted parameters and a report with model, eligible-label and speaker-partition hashes. Calibration uses the existing edit conversion, with no practice-derived parameter selection.

## Validation and limits

Added unit tests cover unsupported feature combinations, posterior residual mass and absolute frame coordinates, fractional alignment estimation, normalized emissions, parameter sharing versus the identity control, neutral and missing evidence, listener-input immutability, duration-charge preservation, symmetric consuming-transition contributions and rewind behavior. The component-replay unit assertion accommodates local acoustic metadata while still checking unchanged heard sounds and full posterior rows at the checker.

Local Node syntax checks passed for the new estimator, model code, tests and modified tracker/local search. `git diff --check` passed. These checks establish syntax and patch cleanliness, not TypeScript type correctness or unit-test success.

All three supervisor Modal requests failed before command execution with `NotFoundError: No such file or directory`, each reporting zero spend:

- Request 1: training-data layout inspection.
- Request 2: six-shard `bash tools/autolab_tests.sh`, including core and SDK build/tests.
- Request 3: core/SDK build followed by `pnpm --filter @tilawa/sdk exec tsx src/articulatory-fit.ts`.

The authorized runner was used for every request. No direct Modal invocation, alternate service, external filesystem search or storage modification was attempted. Repeating these startup failures provides no acoustic or test result. `.autolab/outcome.json` records infrastructure recovery as needed, with no invented measurements.

There is deliberately no guessed or hand-populated parameter table: `ARTICULATORY_MODEL` remains undefined until authorized training finishes. Consequently, this prototype currently leaves deployed tracking behavior unchanged. No training data, practice evidence, target comparison, target win or practice score has been obtained. The recorded 0.3915-edit disadvantage comes from the card's previous audits, not a new measurement. The premise kill criterion is not established, and infrastructure failure must not be recorded as zero target wins.

## Remaining validation

1. Run the six-shard build/core/SDK tests on the final source.
2. Run the training estimator, freeze its joint model in `articulatory-model.ts`, and retain the fit report in this write-up.
3. Compare joint and identical-data identity controls on the target's identical local observation interval, then replay the full target through tracker and checker. Keep dataset identifiers in diagnostic commands and notes only.
4. If the trained joint model leaves the near continuation behind or the built change wins fewer than three targets, revert engine and test changes, record the measured premise kill, and commit the write-up/outcome. If three or more targets are won, run the authorized practice score before deciding. No practice-score allowance has been consumed.
