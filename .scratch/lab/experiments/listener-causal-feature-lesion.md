# Causal mean lesions of deployed acoustic features

Status: provisional implementation; Modal execution is blocked. No
training-selected mask, acoustic target result, or practice score is available.
The experiment card is unchanged. This is not a measured kill or a ready model
bundle.

## Prediction and mechanism

The card predicts listener −1 and overall −1: a sparse set of harmful encoder
channels may supply false substitution evidence. Replace those channels with
their training mean in a separate recheck, retaining the shipped projection
weights. Only a strict reversal from an original rival preference to a lesioned
complete-written-pronunciation preference permits a veto.

`modal/student_feature_lesion_export.py` exposes the deployed final head input
and logits in a copy of the primary ONNX graph. For the dynamically quantized
head, the input is the dequantized activation codes actually consumed by the
projection, and the effective weight includes its scales and zero points. The
export requires byte-identical primary outputs and caches over eight stateful
windows and checks head reconstruction. Those checks are implemented but have
not executed on the shipped graph here.

`modal/student_feature_lesion_cache.ts` streams complete recordings using the SDK
frontend and clock, caches features and full logits/posteriors, and discovers
complete comparisons through a read-only checker hook. Mask fitting accepts only
eligible help-train training speakers; any nontraining take excludes its whole
speaker. No practice input can select a mask.

`modal/student_feature_lesion.py` partitions speakers into fitting, selection and
untouched audit groups. Fitting none-labelled recordings determine the channel
means. Every proposed lesion recomputes full softmax probabilities and complete
CTC forward sums. Sparse channel selection targets false wrong-word comparisons
on none takes, with zero sign losses on fitting preservation controls. Selection
chooses among that path using the second speaker group and its preservation
controls. Controls include complete correctly recognized training words and
already-confirmed accusations at confirmed scripted substitution locations;
the planted script is never imposed as a performed phoneme transcript. Both
control categories and false comparisons in fitting and selection are required.
The mask is capped at one channel per 32 feature dimensions. Nineteen seeded,
equally sized random masks are frozen and reported for comparison. No threshold
or model weight is selected on practice.

The SDK recheck applies `(training_mean - deployed_feature) * shipped_weight`
to the original logits for the selected channels. Both hypotheses use the same
complete CTC topology and acoustic interval. Exact word, sequences, frame bounds
and four finite scores travel in the ordinary evidence metadata. Checker public
verdicts may change only for string-path wrong-word confirmations; internal
confirmation and ownership state remain intact. Graph confirmations and
insertion decisions are exempt. The live session retracts only an identical
occurrence removed by the corresponding certified replay.

The Node loader binds the mask to the primary SHA and alphabet, checks declared
side-asset sizes and hashes, and rejects an incomplete bundle. For the shipped
inventory it also requires the existing insertion teacher. The bundle utility
copies the caller's full primary directory and leaves its model and manifest
unchanged. None of these assets has been produced here, so no `.autolab/model.json`
has been declared; decoding the main model would not measure this experiment.

## Validation and limits

Added synthetic unit checks cover complete CTC path enumeration (including
repeated labels), exact mean replacement with unchanged weights, strict finite
reversals, sparse-mask and model binding, incomplete asset rejection, native
streaming input/cache parity including final padding, exact occurrence matching,
and preservation of graph confirmations. The existing insertion-loader mock
now explicitly represents an absent lesion descriptor. Python syntax compilation
and `git diff --check` pass. These are syntax and source checks, not executed unit
tests or acoustic validation.

Supervisor runs 1 and 2 were environment probes; run 3 requested the supplied
six-shard `tools/autolab_tests.sh` harness. All three failed before command
execution with `NotFoundError: No such file or directory`, with $0 recorded
compute. A repair request was sent to the user. Run 4 requested the synthetic
Python tests, shipped-graph export, complete help-train cache and training-only
selection and failed before execution with the same error and $0 compute.
Consequently neither required test suite, the export parity checks, nor training
selection has executed. Retrying unchanged commands cannot resolve that blocker.

No held-out-speaker recordings or exam data were accessed. No scorer, lab
tooling, supplied manifest, evidence, test data or experiment-card edits were
made. No practice score was requested. Targets won and the random-mask median
are unknown. There is no evidence on which to invoke the premise kill criterion,
and no basis for claiming a gain or fold eligibility.

The provisional session integration buffers audio and performs its review at
`end()`, matching the existing auxiliary-session pattern. It therefore does not
provide an immediate streaming veto, and its memory and latency costs remain
unmeasured. Training preservation cannot establish practice safety; genuine
substitutions can still depend on the removed channels.

## Remaining work

Restore supervisor Modal execution, then run the added tests and the supplied
six-shard harness on the final source snapshot. Complete the training cache and
freeze the selected mask/report before opening the registered target audio.
Diagnose those targets through the generic cache/diagnose commands and compare
target reversals with the nineteen random-mask controls. The task requires a
rollback and measured kill if fewer than three targets are won, or if the causal
premise is absent. If at least three are won and the random-mask control passes,
finish and verify the complete bundle (including the existing teacher), declare
it in ignored `.autolab/model.json`, and run the required practice score before
deciding. Until then this branch must not be treated as an evaluated listener.
