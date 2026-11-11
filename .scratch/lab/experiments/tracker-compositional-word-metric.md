# Compositional whole-word identity metric

Status: needs-human. The training and comparison prototype is implemented, but the supplied
supervisor runner fails before command execution. No training-frozen model or target result
has been obtained. The metric is not enabled by default.

## Prediction

Tracker −1 wrong call and overall 0 on lab-v1.5u practice. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting localization can expose the documented nearby listener false warning; checker
handling remains unchanged. `lab-experiment.json` is unchanged.

The supplied earlier audits put the retained near continuation 0.3915 edits behind the
rhyme restart over a common observation interval. That number is prior evidence from the
brief, not a measurement reproduced in this checkout. The experiment tests whole-word
identity rather than a missing search state, omitted charge or pronunciation rule.

## Implemented mechanism

`packages/core/src/word-metric.ts` implements two small CPU phone encoders. Each composes
left/centre/right phone contexts using a learned convolution, tanh, mean/max pooling and
a normalized linear projection. Parameters are indexed by phone symbols and hidden
dimensions; no word, reader, session or verse entry is learned. Unseen written words and
observed fragments use the same composition. Boundary padding participates explicitly.

Match cost is a nonnegative training-calibrated multiple of squared embedding distance,
minus the corresponding identical-string distance. This ordinary-reading control removes
the offset between the two encoders and gives an exact observation zero identity cost.
Unknown or empty strings retain the original charge. The runtime owns a copy of the
model and bounds embedding caches by phone strings.

`packages/core/src/local.ts` preserves the original phone DP representatives and attaches
immutable predecessor records. Completed-word prefix summaries share earlier work without
recharging an audited open word. When the original winner changes run origin, the optional
metric scores every finite retained trajectory over the complete heard interval since
that local search was initialized. Each word occurrence uses exactly its consumed written
and observed fragments. Moving back to the same word creates a separate occurrence.
Unconsumed terminal suffixes are not charged. Entirely deleted words retain their original
charges because there is no acoustic word match to replace.

Only word-identity charges are replaced. Every existing movement, jump-distance, surah-entry
and duration charge survives. Candidate observation counts must agree. The selected result
is an existing trajectory; no transition or coverage witness is added. The existing raw
costs still drive the DP alignment representatives and recent misfit accounting. Tracker
confirmation, margins, event delivery, lock-drop handling and checker rules are unchanged.
`TrackerOptions.localWordMetric` provides `audit` and `apply` modes. Audit reports the
learned alternative while preserving original positions; apply selects its retained path.

No placeholder or hand-fitted default model is installed. A fitted artifact must be
obtained and validated before enabling the experiment for the supervisor's practice run.

## Training isolation and frozen selection

`packages/sdk/src/word-metric-decode.ts` uses only help-train and the supplied deployed
`models/spliced-clean-int8` listener. Eligibility requires explicit `split: train`,
`kind: none`, `scored: true`, no exclusion, no scripted events and no labelled repeats.
Any speaker with a non-train row is excluded altogether. Audio paths must stay inside
help-train. Cached decodes are keyed by model and audio hashes. The training document
contains pronunciation strings and speaker folds, with model/reference/label hashes;
it contains no practice observations or passage-location features.

`packages/core/src/word_metric_train.py` samples four latent edit paths per recording from
a transcript-constrained forward sum using the original directed phone prices at the
existing three nats per edit. Every training path consumes the full transcript and heard
sequence. These samples yield alternative word boundaries. Duration indels remain outside
identity examples. Completed words and partial prefixes teach the streaming frontier.

Every observation contrasts against eight phonologically similar Quran pronunciation
strings, selected through a bigram shortlist and directed phone distance. Identical-string
ordinary controls accompany training observations. Both encoders start identically and
then learn separately; training uses contrastive retrieval plus positive-pair proximity.

Dimensions 8/16 and checkpoints 20/40 epochs are compared in up to five deterministic
training-speaker folds. Each fold fits its model and distance-to-edit slope using only
the other speakers. Selection minimizes held-partition retrieval errors, then squared
match-cost error, then model size. The selected architecture/checkpoint is refitted on
the authorized training pool and the nonnegative slope is fitted once. All selection
finishes before opening a practice case. Training also exports a numerical fixture for
Python/TypeScript inference parity checks.

## Validation and limits

Eleven focused TypeScript test bodies passed locally under Node 22 TypeScript stripping
and a small assertion adapter. This is a numerical/synthetic check, not a Vitest suite
result. They cover unseen composition and order, encoder normalization, owned models,
incompatible/invalid parameters, duration/movement/deletion prices, a synthetic identity
ranking reversal, full observation accounting, original audit-mode positions, rewind and
snapshot ownership, open-word extension without duplicate charges, selection of an existing
path, and training-speaker exclusion/grouping.

The complete core TypeScript project passed a no-emit check with TypeScript 5.9.3 and
the repository's Node type version. The SDK training eligibility module also passed
its standalone strict type check. The new SDK decoder and replay helper also passed a
source-based type check with an ambient declaration for the existing ONNX binary import;
this does not validate ONNX loading or replace the SDK build. Python source and Python
test syntax compile. Local Python numerical execution is unavailable because the installed
NumPy binary has the wrong architecture; no local training was attempted.
Four additional Python tests cover latent-boundary reproducibility, separate duration
costs, batch-padding invariance and identical-string controls; execution requires the
Modal NumPy/PyTorch environment. No full build or suite result is available yet.

The prototype reranks the existing minimum-phone-cost alignment retained at each cell;
it does not search every latent alignment under the learned metric. Training uses finite
posterior boundary samples and a bounded negative shortlist. First and last word matches
are consumed fragments, not reconstructed complete words. Runtime predecessor memory and
comparison cost have not been measured on real sessions. Ordinary-reading controls are
training/synthetic controls, not a measured clean-session safety result.

The supplied supervisor requests were:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Inspect authorized data/schema, storage and PyTorch | `NotFoundError: No such file or directory`, before execution |
| 2 | Core/SDK build, training-only decode and frozen calibration | Same pre-execution failure; no model or fit report |
| 3 | Core/SDK builds/tests on six shards, plus Python numerical tests | Same pre-execution failure; no build or test logs |

No practice-score requests were made (0 of 3). There is no measured target-win count,
target ranking reversal or overall delta. An infrastructure failure does not establish
that the hypothesis is false, so this is needs-human rather than killed.

## Remaining validation

Restore the supplied supervisor execution environment, then run:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && WORD_METRIC_CACHE=/scratch/tracker-compositional-word-metric/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/word-metric-decode.ts && python3 packages/core/src/word_metric_train.py "$LAB_OUT/word-metric-training.json" "$LAB_OUT"'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh && if [ "$LAB_SHARD" = 0 ]; then python3 -m unittest discover -s packages/core/test -p test_word_metric_train.py; fi'
```

Copy the returned model and parity fixture into this worktree's scratch directory.
The read-only `.scratch/lab/word-metric-target-audit.ts` takes all case selection through
arguments. Its `audit` mode records original and learned costs, fixed charges and shared
observation counts, and its `apply` mode runs the unchanged checker on the selected paths:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- \
  'pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/word-metric-target-audit.ts --set lab-v1.5u --session lab-v1.5-p46 --at 121.84 --model /app/.scratch/lab/word-metric-model.json --parity /app/.scratch/lab/word-metric-parity.json --mode audit'
```

Establish whether the frozen metric ranks the retained near continuation above the restart
on the identical interval, then use `apply` to determine actual target wins and exposed
warnings. A failed premise or fewer than three measured target wins requires reverting the
engine changes and recording a killed outcome. If enough targets are won, install the
training-frozen model as the experiment default and request a practice score before any
gate-dependent narrowing. Do not refit selection or conversion from practice outcomes.
