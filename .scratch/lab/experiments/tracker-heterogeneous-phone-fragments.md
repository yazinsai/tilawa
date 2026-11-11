# Heterogeneous phone fragments

Status: needs-human. The optional fragmentation channel, training estimator and unit tests are
implemented. All three requests through the supplied supervisor runner failed before execution
with `NotFoundError: No such file or directory`. No trained model, target replay, TypeScript
build or full Vitest result was returned. The experiment card is unchanged.

## Prediction and premise

Tracker −1, overall 0 on lab-v1.5u practice. The target is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The card reports that the near continuation remains available but loses to the restart by
0.3915 edits. This is supplied audit evidence, not a measurement reproduced here.

The premise is that one reference articulation can produce several different decoded phone
tokens. Independent token edits would then treat the recognition fragments as separate spoken
sounds. A normalized one-reference-to-many-observations event could change the near path's
relative cost. The predicted overall change includes the previously documented nearby listener
warning that correcting the binding may expose; this experiment does not alter checker handling.

The premise kill requires a training-frozen multi-token explanation on the target and a reversal
of the near-continuation versus restart preference. Neither could be evaluated without remote
training and evidence. An infrastructure failure is not evidence that the premise is false.
No target win or kill is claimed.

## Implementation

`packages/core/src/fragmentation.ts` defines a normalized recognition event for each articulation:
an ordinary single-token/deletion branch, or an ordered heterogeneous sequence of two or three
observed tokens. Identical-token sequences do not qualify. Sequence excess probabilities are
stored sparsely; a residual distribution has support over every heterogeneous inventory sequence,
with half its mass at each length. This avoids selecting a vocabulary using held-partition data.
Every row's ordinary, sequence and residual probabilities sum to one.

Complete-recording forward/backward marginalizes insertion, deletion, ordinary recognition and
fragmentation boundaries. Hierarchical fitting shares counts from exact phone identity through
fixed manner classes to pooled sequences. It runs at most 30 iterations. The ordinary emission
distribution normalizes exponential weights from the existing identity/deletion costs at three
nats per edit. Insertion has fixed probability 0.1 with uniform phone outcomes, and the final
state partitions insertion and stopping. The pooled prior gives ordinary recognition 0.95,
fragmentation 0.05, and has two pseudocounts. These definitions are fixed before practice access.

`local.ts` composes bounded observation-span arcs with the existing local matcher. An event
consumes one reference symbol and two or three observations exactly once. It retains the reference
offset as its owner and leaves subsequent pronunciation-expanded symbols to the existing duration
charges. Only a run's first identity symbol can fragment; a word boundary starts another owner.
The ordinary branch receives its probability-allocation charge. Fragment costs are likelihood
ratios against ordinary exact recognition, expressed in the existing three-nats-per-edit units.
No extra route receives free probability mass.

The matcher retains at most three immutable observation rows, including their cumulative
normalization totals. Continuation and word-start restart arcs therefore consume the same heard
interval even when the chosen path revises an earlier token's alignment. Restart arcs use the
original jump price, distance charge and surah-entry charge. A later window recenter cannot add a
destination unavailable at the event's onset. Save/restore retains the observation history and
latest fragment ownership. Search windows, ordinary edit prices, movement edges, commitment
margins and checker code retain their existing definitions.

`TrackerOptions.fragmentation` supplies the frozen model to local matching. A model type is
exported by core. There is **no fitted default model**: ordinary product and practice invocations
still use the original channel until a returned training model is installed. Scoring the current
default would not test this experiment. Default activation is unfinished rather than fabricated
from a synthetic example or the target.

## Training isolation

`packages/sdk/src/fragmentation-fit.ts` reads only help-train labels/audio, the shipped reference
and the deployed model at `models/spliced-clean-int8`. It keeps scored, non-excluded none-labelled
training takes with no events or repeats. Every speaker with any non-training annotation is
excluded entirely, including mixed-split speakers. Decodes cache under the model hash.

Up to five deterministic speaker-grouped partitions compare no fragmentation with pooling
masses 8, 32 and 128 by predictive log probability on the training-held partitions. Ties prefer
the disabled channel, then stronger pooling. Selection freezes before refitting on all eligible
training speakers. Outputs in `LAB_OUT` are `fragmentation-model.json` and `fit-report.json`, with
model/label hashes, recording/speaker counts and all partition results. No training outputs have
been returned, and no selection was made using practice evidence.

## Validation and limits

Nine core and two SDK focused test bodies passed under Node 22 TypeScript stripping with a
temporary assertion adapter. This checks numerical and runtime behavior, not a TypeScript build
or a full Vitest run. The temporary files were removed. The tests cover:

- Complete-string normalization on a two-phone inventory through observed length eight, within
  0.000005 of total probability one; exact normalization of recognition-event rows.
- Two- and three-token ordered events, one reference owner and one recognition edit.
- Save/restore, window recentering and cumulative common-interval costs.
- The same event on continuation and restart, with movement still charged.
- Exact ordinary-path equivalence when fragmentation is disabled.
- Remaining duration-symbol deletion charges, and exclusion of interior duration symbols.
- A genuine whole-word omission retaining a charged movement explanation.
- Latent fitting, normalized phonetic-class backoff and unknown-phone rejection.
- Exclusion of mistakes, repeats, held-out/mixed-split speakers, and deterministic speaker grouping.

`git diff --check` passes. No scorer, lab tooling, evidence, test data or manifest was changed.

Supervisor requests, all failed before execution with the same NotFoundError:

1. One-shard scratch/data availability and the supplied target trace command (run 1).
2. Core/SDK build followed by training-only calibration, 60-minute limit (run 2).
3. Required core/SDK build and sharded test suite, six shards (run 3).

Practice scores used: **0 of 3**. Target wins, component/overall deltas, training selection and
gate verdict remain unknown. There is no evidence to report that the target contains the proposed
fragmentation, or that a training-frozen channel would reverse its preference.

## Resume after infrastructure recovery

Use only the supplied supervisor runner:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && FRAGMENTATION_CACHE=/scratch/tracker-heterogeneous-phone-fragments/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/fragmentation-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Freeze the returned selection before opening target evidence. Replay both local candidates with
that exact model over their common heard interval, inspect the selected fragment's reference
owner and sequence, and replay the product through the unchanged checker. Install a default only
from that training-frozen result. If no preferred multi-token single-articulation explanation
reverses the comparison, or the built change wins fewer than three targets, follow the requested
kill procedure: revert engine changes/tests, retain the measured write-up and commit a killed
outcome. If three or more targets win, obtain the required practice score before deciding on any
narrowing. Infrastructure failure alone does not authorize a premise kill.
