# Confidence-bearing local edit channel

Status: needs-human. The supplied supervisor runner failed before execution on all four requests.
The prototype and training estimator are implemented; training-frozen calibration, target comparison,
default activation and required Modal build/tests remain unfinished. `lab-experiment.json` is unchanged.

## Prediction and premise

Tracker −1, overall 0 on lab-v1.5u practice. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The documented near continuation remains available but loses to the rhyme restart by 0.3915 edits
over an identical heard interval. That measurement is supplied in the brief and has not been
reproduced here. Correcting the binding may expose the documented nearby listener false warning.

Code inspection verifies one premise: `tracker.ts` formerly called `local.step(ph.p)`, discarding
the token's `conf` when pricing local insertions, substitutions and restart entries. It now calls
`local.step(ph.p, ph.conf)`. This establishes missing reliability information, not that the target's
two paths have different reliability evidence or that restoring it reverses their ordering.

## Mechanism implemented

`packages/core/src/confidence-channel.ts` estimates a normalized joint distribution of heard
identity and confidence bin given reference identity. Identity emissions have a reference-phone
row; confidence emissions share parameters by reference manner class, heard manner class and
identity relation (same, existing soft confusion, other, or insertion). Sparse confidence counts
pool toward the heard class's distribution. Reference-independent background distributions are
conditioned on heard identity.

Token alignment is latent. EM uses forward/backward expected diagonal and insertion counts in
a normalized pair edit channel, with fixed match/deletion/insertion probabilities .8/.1/.1 and
terminal stop probability .9. Complete clean training references are aligned to complete heard
recordings. Confidence stays attached to each observation through every alternative alignment.
Estimation runs at most 30 iterations; old emission tables are released through a weak cache.
This training alignment model is separate from production movement and deletion pricing.

Production local matching retains its existing identity costs and adds only:

`−(10 / 3) log[P(confidence bin | reference/heard classes, identity relation) / P(confidence bin | heard identity)]`.

Ten units remain one edit and three nats remain one edit, using the tracker's existing conversion.
The same contribution is applied to continuation diagonals and restart diagonals; insertion
observations use the fitted insertion confidence channel. Reference-only deletion charges,
duration charges, movement arcs, search horizon, commit margins and downstream handling retain
their existing definitions. Missing or invalid confidence and unknown inventory phones contribute
zero. Marginalizing confidence contributes zero relative evidence because both distributions
normalize. A selected one-bin model disables all confidence contributions.

The channel is supplied through the optional `TrackerOptions.confidenceChannel`. No synthetic
or unfitted default model is installed. Ordinary product and benchmark calls still have the
original scoring until a training-frozen model is returned and deliberately installed as the
default. Consequently, scoring this unfinished branch cannot test the card's hypothesis.

## Training isolation and selection

`packages/sdk/src/confidence-channel-fit.ts` opens only help-train labels/audio, the reference
asset and the supplied acoustic model. It keeps `kind: none`, `split: train`, scored,
non-excluded recordings with no labelled mistakes or repeats. Any speaker with a non-training
label is excluded entirely. It requires at least two eligible training speakers and never opens
practice evidence or release-held-out audio.

The estimator decodes with the shipped SDK listener and caches phonemes under the acoustic
model hash. Up to five speaker-grouped training folds choose equal-width confidence bin counts
1, 2, 4 or 8 and pooling masses 16, 64 or 256. Selection uses held-partition predictive log
density, including bin-width normalization so different bin counts have a common scale.
It then refits on all eligible training recordings and writes `confidence-channel.json` and
`fit-report.json` to `LAB_OUT`. The report records model identity/hash, selected-label hash,
recording/speaker counts and every fold result. None of these results has been returned yet.

## Validation and limits

Eight focused test bodies passed locally under Node 22 TypeScript stripping with a temporary
assertion adapter for Vitest. They check normalized rows, confidence-dependent pricing,
phonetic sharing for unseen pairs, missing/marginalized confidence, insertion and confidence
endpoints, ambiguous edits and unequal lengths, comparable density scales, local confidence
plumbing and rewind, and unchanged local behavior for uninformative confidence.
This is numerical and behavioral prototype validation; it is **not** a TypeScript build or
a required Vitest suite result. The temporary adapter is removed from the final branch.

Supervisor requests:

- Run 1: inspect training availability and storage; failed before execution with
  `NotFoundError: No such file or directory`.
- Run 2: six-shard core/SDK build and tests; the same pre-execution failure.
- Run 3: repeat six-shard build/tests after local integration; the same failure.
- Run 4: core/SDK build followed by training-only calibration; the same failure.

No build logs, training decodes, selected hyperparameters or fitted model were returned.
No target replay or practice-score request was made (0 of 3 practice scores used). Target wins
and overall delta are unknown. Infrastructure failure does not establish a premise kill;
the outcome is `needs-human` with no measured count.

Resume through the supplied runner after its missing resource is restored:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && CONFIDENCE_CACHE=/scratch/tracker-confidence-bearing-edit-channel/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/confidence-channel-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Freeze the returned model without practice-driven selection, install it as the tracker default,
and inspect both local candidates over the identical observed interval. Then replay the target
through the unchanged checker. If the contribution does not reverse the ordering or the measured
target wins are fewer than three, apply the user's kill rule: revert engine code/tests, retain
the measured write-up and commit a killed outcome. Otherwise obtain a practice score before
making a gate decision. No binding recovery or warning improvement is claimed here.
