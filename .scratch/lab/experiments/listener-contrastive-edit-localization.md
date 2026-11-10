# Contextual edit localization

Status: source implemented; model fitting and full validation blocked by the
supervisor compute service. The acoustic premise is untested. The experiment
card is unchanged, and no fold or discard was performed.

## Prediction and mechanism

The prediction is two false warnings removed and one caught mistake lost:
listener −1, overall −1. The mechanism is recording-level competition between
edit locations and classes, learned from frozen encoder features and help-train
training speakers rather than independent accused-word crops.

`modal/student_edit_localization.py` exposes the pre-CTC projection input in a
separate copy of the supplied ONNX graph. It pools encoder means, deviations,
phone probabilities and text histograms under canonical CTC forward-backward
occupancies. Acoustic boundaries remain latent; the primary graph is untouched.
The contextual head predicts a normalized recording-level none state versus
location/class alternatives (lexical change, omission, pronunciation/other, and
repeat). Scripts provide weak location/class supervision with a latent
compliance mixture, rather than performed phone transcripts. None-labelled
recordings and repeats are explicit controls.

The independent head receives the identical raw features and compliance model,
but predicts each word independently. Speaker hashing creates disjoint fitting,
selection and preservation-calibration partitions inside split=train. Input with
an eval or unspecified split is rejected before audio access. Each partition
must contain substitutions, none recordings and repeats; missing coverage fails
rather than silently relaxing these requirements. Regularization and checkpoint
selection use only the selection speakers. Scripted substitutions from the
separate calibration speakers set each head's minimum preservation score; ties
remain preserved. Practice inputs never select a weight or operating point.

Recording competition is distilled into contextual per-word sigmoid scores.
Inference has no cross-word normalization or one-mistake limit. Export includes
the distilled head and independent diagnostic control, verifies their dynamic
word counts against ONNX Runtime, checks primary CTC equality on a training
recording, and copies the complete supplied baseline directory into a new
bundle. Only its generated manifest receives the new auxiliary declarations.
The supplied primary file, its SHA and its acoustic geometry are preserved.

The SDK discovers actual non-graph wrong-word confirmations and overlapping
ayah alignment windows without changing the engine. It merges overlapping
views of one occurrence, keeps disjoint re-reads separate, and abstains unless
both ends of the ayah have been observed. A separate encoder stream supplies
frozen features; both heads use identical pooled rows. Certificates identify
the exact written word, expected and observed strings, and acoustic frame
interval. The checker may only make a matching confirmed wrong word uncertain.
Graph confirmations, insertions, primary tokens/posteriors and tracker events
retain their existing paths. Conflicting calibration certificates abstain.

The Node loader requires both declared files and their hashes, rejecting a
missing or mismatched asset. No declaration means the ordinary listener path.
`packages/sdk/src/bench/edit-localization-cli.ts` provides label-free diagnostics
on caller-supplied audio and evidence, including both frozen decisions and a
primary-evidence equality check; it does not score or choose thresholds.

## Validation completed

Python syntax compilation passes. Seven local mathematical tests pass under
`arch -x86_64 python3`; one PyTorch-dependent multiple-change test is skipped
because PyTorch is unavailable locally. Tests cover exhaustive CTC path
marginals, impossible alignments, word pooling, repeat targets, stable speaker
partitions, split rejection, and preservation ties.

Changed TypeScript sources pass Node's TypeScript syntax transformation.
Bounded local source smoke checks using Node's transformer pass for read-only
checker discovery, an exact certificate changing confirmation to uncertainty
with unchanged text/timestamp, separate zero-initialized feature caches, shared
feature pooling, and end-to-end synthetic SDK certificate enrichment. The last
check uses stub scores and uniform frozen features: it establishes wiring,
not acoustic discrimination.

Core/SDK unit tests were added for certificate matching and conflicts, actual
checker consumption, exhaustive CTC occupancies, serialization/rechunking,
overlapping versus repeated contexts, separate feature states, complete SDK
candidate-to-certificate transport, and missing/hash-mismatched declared assets.
The complete Vitest/build suite has not run successfully.

## Compute block and limits

The previous attempt's requests 1–3 failed before providing container logs.
Fresh requests in this attempt had the same result:

| Run | Requested work | Result |
| --- | --- | --- |
| 4 | Working directory, scratch capacity, help-train listing and ORT version | `NotFoundError: No such file or directory` |
| 5 | Supplied six-shard `bash tools/autolab_tests.sh` | Same error |
| 6 | Six-shard test script plus Python localization tests on implemented sources | Same error |

All three fresh result files report zero spend. No container log, audio, data
listing or trained model returned. The failed path is unknown; storage exhaustion
has not been established. No other compute route or outside-worktree dataset was
used.

No help-train recording was processed, no head was fitted, and no preservation
score was frozen. Zero registered targets have been tested, and no practice
score was invoked; all three allowed practice scores remain unused. The card's
contrastive diagnostic (two rejections retained by the independent control) and
the task's three-target continuation requirement are both unevaluated. This is
an infrastructure block, not a premise kill or a measured negative result.

There is no deployable model bundle and no `.autolab/model.json`; consequently
running the main listener would not evaluate this experiment. The optional
engine implementation remains on the branch for the retry, unlike the previous
write-up-only snapshot. No scorer, lab tooling, supplied manifest/evidence, test
data or experiment-card changes were made.

Additional limitations remain even after compute is restored. Script compliance
is uncertain, canonical occupancies can misplace frames around real edits, and
long or repeated passages may not match single-ayah training. The current SDK
path enriches recorded evidence at end-of-stream and sends live clients an
explicit final uncertainty update; it cannot prevent an earlier live alert.
This final-recheck timing is not a verified production streaming veto.

Resume through the authorized supervisor: inspect actual training partitions,
fit and verify the complete bundle with `modal/student_edit_localization.py`,
then run full core/SDK checks and both frozen heads on actual engine-requested
target windows. Verify primary decoding, tracker events and inherited teacher
insertion behavior before declaring the bundle. Apply the stated continuation
rule and practice gate only after real target results exist.
