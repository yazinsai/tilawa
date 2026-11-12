# Averaging one phone-head optimization trajectory

Status: implementation prepared; training and acoustic validation blocked by
the supervisor execution service. The experiment card is unchanged. No trained
candidate bundle is declared.

## Prediction and mechanism

The preregistered prediction is listener −1 and overall −1 wrong calls on
lab-v1.5u, with no predicted losses. The proposed cause is checkpoint-specific
consonant boundary placement: a terminal head can prefer a rival pronunciation
while nearby iterates share a decision geometry that favors the written word.
This cause has not been established on the registered incidents.

`modal/student_trajectory_head.py` freezes the shipped encoder and refits its
linear CTC phone head on cached features. After training-only convergence, it
continues one fixed-learning-rate optimization trajectory and accumulates an
equal parameter mean after every update in a fixed tail. Inference uses one
averaged head, with no output voting or extra inference pass.

The terminal iterate is retained as a comparison. An ordinary continuation
branches from the same converged parameters and momentum state, sees identical
minibatches and performs the same number of optimizer updates, with cosine
learning-rate annealing instead of the fixed-rate averaging tail. Its checkpoint
is retained separately; it is not incorporated into the parameter average.

## Data, objective and training selection

The loader requires authorized original clean-match parquet recordings under
`/student/s1`, with their performed reference transcripts and speaker-cluster
metadata. It excludes an entire original speaker cluster if any member is
outside the training split or on the existing release exclusion list. Unknown
clusters are not admitted. The existing augmentation banks are required under
`/student/aug`; their availability in the supervisor container is unverified.

Help-train canonical supervision admits only training speakers' none-labelled
takes with no mistake, repeat or unsupported region. Any speaker with an
excluded or non-training take is excluded in full. Mistake/repeat takes supply
only transcript-free preservation of the shipped output, never their planted
scripts as performed transcripts. Complete recordings retain ordinary speech
and pauses, and every cached frame participates in preservation. Exact waveform
duplication across different speaker identities is rejected before caching.

Fit and validation partitions reserve entire speakers separately within each
supervised source. Speakers having only preservation takes enter the fit bank
and are never claimed as independent validation speakers. Each supervised take
minimizes its complete CTC negative log likelihood divided by frame count,
plus twice the mean framewise KL from the shipped distribution. Preservation
takes minimize that KL alone. Each update averages eight supervised losses and
four preservation losses separately and adds the two means.

The recipe was fixed before acoustic measurements: seed 831, at most 512
original recordings, at most 30 seconds per recording, three cached views
(ordinary plus two using the existing shipped augmentation), SGD learning rate
0.001 and momentum 0.9. The warm-up checks the deterministic fit-bank objective
after every epoch. Four consecutive relative changes of at most 0.002 constitute
convergence; failure to converge within 48 epochs declares no candidate.

The converged trajectory continues for eight epochs. Float64 accumulators form
equal per-update parameter averages over the last eight, four and two epochs.
Separate training-validation speakers select the tail with the lowest
speaker-averaged CTC-plus-preservation loss; ties favor the longer tail. Neither
practice recordings nor their labels, warnings or scores participate in the
refit, convergence check, schedule choice or checkpoint selection. Cached
features, initial head, feature graph, frozen plan and selected checkpoints have
checksum provenance.

## Export and validation

`modal/student_head_refit.py` captures the features actually consumed by the
shipped phone head, including rounded/dequantized activations in the integer
graph. Capture checks that the original outputs and cache states remain exactly
unchanged. Streaming caching trims final padding to the SDK's output count.
The exporter replaces only classifier parameters and their weight-quantization
constants, retaining the existing quantization geometry. It checks output
geometry, finite probabilities and exact cache-state parity.

Each of the averaged, terminal and ordinary bundles must include all inherited
SDK insertion-teacher assets with their declared sizes and checksums. Export
refuses missing or corrupt assets. The large immutable teacher weights are
copied once and hardlinked between comparison bundles on the same volume. The
existing SDK directory loader can load these bundles without a code change.
Vocabulary, frontend, frame clock, encoder graph, tracker, checker and
confirmation requirements remain unchanged; supplied manifests are not edited.

Source compilation passes. Added synthetic unit fixtures cover exact equal
parameter averaging, training-only convergence, speaker exclusions, refusing
scripted canonical supervision, transcript-free preservation, validation-tail
selection, actual matched optimizer inputs and update counts, float/int8 feature
capture, per-channel weight quantization, stream-padding trimming, cache parity
and complete auxiliary bundle checks. All 11 fixtures passed locally in 106.19
seconds in an isolated workspace environment (NumPy 2.5.4, torch 2.15.0, ONNX
1.23.2 and ONNX Runtime 1.31.1). Only small synthetic inputs were used; no
recording training or acoustic validation ran on the Mac.

Supervisor run 1 requested a read-only data/runtime inventory; run 2 requested
the supplied six-shard core/SDK test harness. Run 3 retried with the training
image, requesting the Python fixtures, a prerequisite inventory and sequential
prepare/fit/export. Run 4 retried the core/SDK harness with the explicit node
image. All four returned `NotFoundError: No such file or directory`, recorded
zero compute expenditure and produced no output directory. Core/SDK tests
could not execute. The original recording/augmentation mounts and inherited
teacher bundle could not be inventoried. No acoustic job or core/SDK test
started.

## Limits and continuation

No training convergence, selected tail, terminal-versus-averaged pronunciation
comparison or recovered target warning has been measured. No practice score has
run; all three permitted scores remain available. There is no
`.autolab/model.json`, because no complete verified candidate bundle exists.
Missing compute is not evidence against the acoustic premise, so no killed
outcome is asserted without an acoustic result.

In the authorized supervisor container, set
`PYTHONPATH=/app/tools:/app/modal` and run the CLI's `prepare`, `fit`, then
`export`, each with
`--out /scratch/listener-trajectory-averaged-readout/refit-v1`.
`prepare` and `export` also require
`--model /lab/data/models/spliced-clean-int8`. Verify all three complete bundles
through the ordinary SDK loader before declaring the averaged model.

After training selection is frozen, obtain identical full written-versus-rival
windows from the actual baseline checker requests for all four registered
incidents. Compare the terminal, averaged and ordinary exported models, then
replay the target sessions through the unchanged engine. Convenient crops or
shortened phone comparisons do not test this premise. Under the user's stopping
rule, fewer than three recovered targets requires reverting the engine changes
and recording a premise kill; three or more requires an authorized practice
score before assessing collateral losses. The card's one-warning forecast does
not satisfy that continuation threshold by itself.

Until those checks run, target gain, genuine-mistake preservation, clean-session
safety, studio preservation and aggregate improvement remain unknown. No fold
has been made.
