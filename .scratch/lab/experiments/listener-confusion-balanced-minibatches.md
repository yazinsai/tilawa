# Confusion-balanced recording minibatches

Status: implementation prepared; training and acoustic validation blocked by
the supervisor execution service. The experiment card is unchanged. No trained
candidate is declared.

## Prediction and mechanism

The card predicts listener −1 and overall −1 wrong calls, with no predicted
losses. The proposed cause is that updates containing only one side of a rare
consonant contrast move the multiclass decision boundary without contrasting
evidence. Grouping recordings containing both alternatives into one update may
stabilize that boundary. Whether this cause applies to the four registered
warnings remains unmeasured.

`modal/student_balanced_head.py` freezes the shipped encoder and refits its
existing linear phone head. The objective remains complete-transcript,
multiclass CTC, with a shipped-output preservation term. There is no pairwise
classifier, extra inference pass, warning veto or change to tracker/checker
confirmation. The existing SDK directory loader supports the resulting head.

`modal/student_confusion_batches.py` computes exact transcript-conditioned CTC
forward/backward phone occupancies, including blank and legal repeated-label
placement. Each recording's descriptor is the equal mean of its cached views'
frame-averaged occupancies. Confusion mining compares these soft transcript
occupancies with the shipped decoder's frame winners, using only fit speakers.
The geometric mean of two directed confusion rates ranks contrasts supported
in both directions. Contributions are normalized by recording duration.

The balanced scheduler normalizes each phone's occupancy mass by its fit-bank
mean. Seeded random recording anchors are followed by greedy additions that
first provide both alternatives and then minimize normalized mass imbalance.
This changes grouping, without resampling, cropping, dropping or duplicating
recordings. Matched controls shuffle recordings or sort them only by recording
length, with randomized ties and update order. Final partial batch sizes stay
identical. Exact schedules are saved and recording multiplicity and update
counts are audited after every cached view.

## Training data and selection

Original authorized clean-match recordings require training-speaker cluster
metadata and complete performed reference transcripts under `/student/s1`.
An entire original cluster is excluded if any member has a non-training split
or appears in the existing release exclusion list; unknown clusters are
refused. Help-train admits training speakers' none-labelled takes only, with
no events, repeats or unsupported regions. Any help speaker having an excluded
or non-training take is excluded in full. Scripted mistakes are never taught as
canonical readings. Waveform duplicates with conflicting speakers or partitions
are refused.

Separate whole-speaker partitions reserve one fifth of speakers within each
source, with a minimum of one reserved speaker and at least three admitted
speakers per source. Both confusion mining and batch descriptors use fit
speakers exclusively. Reserved training speakers select the grouping procedure
and checkpoint by speaker-averaged ordinary CTC-plus-preservation loss.

The recipe was fixed before acoustic measurements: seed 831, at most 512
original recordings, maximum recording duration 30 seconds, three cached views,
effective batch size eight, SGD learning rate 0.001 and momentum 0.9, and twelve
epochs. The unchanged view and two views using existing shipped augmentation
functions are shared exactly between arms. Every view minimizes complete CTC
negative log likelihood per frame plus twice the frame-averaged KL from the
shipped posterior. All real speech and pause frames participate. Recording
gradients accumulate before one optimizer step; even a final partial batch
uses the fixed divisor eight so its recordings are not implicitly upweighted.

Four arms train: shuffled, length-only, and balanced using the first four or
eight mined contrasts. Reserved speakers choose between the two balanced
procedures and their checkpoints. Both control heads are saved at the same
selected epoch and recording exposure. All arms finish the same twelve-epoch
schedule before a final selection file is written. Preparation and fitting
share a 23-hour elapsed-compute budget. Practice data, warnings and scores do
not enter preparation, mining, fitting or selection.

## Feature and export checks

`modal/student_head_refit.py` replays each complete recording through the
deployed streaming ONNX graph, retaining chunk/cache semantics and trimming
the final padded output rows to the SDK count. Quantized heads use their actual
rounded and dequantized activation inputs. Feature capture checks exact
primary-output and encoder-cache parity against the shipped graph.

The exporter replaces only phone-head constants and their weight quantization
parameters, retaining the existing quantization geometry, all encoder nodes
and constants, inventory and timing. It verifies finite output geometry and
exact encoder-cache parity. All inherited SDK insertion-teacher assets must be
present with their declared sizes and hashes in each bundle; missing assets
block export. Generated manifests contain the new model hashes; supplied
manifests are unchanged.

Frozen plan, head, feature graph, caches, grouping, schedules and selected
checkpoints have checksum provenance. Reading a plan rejects missing or
duplicated views, changed speaker assignments, excluded speakers and modified
caches. Export additionally requires complete training and equal selected
control exposure. A partially started fit cannot be silently appended to by a
second invocation.

## Validation and limits

Source compilation passes. The eight synthetic occupancy/scheduler fixtures
pass, including exhaustive CTC path agreement, repeated labels, impossible
targets, soft descriptors, bidirectional mining, improved contrast balance,
deterministic schedules and exact recording/partial-batch preservation.
All 21 synthetic fixtures passed locally in 44.902 seconds in an isolated
workspace environment (NumPy 2.5.4, torch 2.15.0, ONNX 1.23.2 and ONNX Runtime
1.31.1). They additionally exercise a complete miniature four-arm refit,
training-selected checkpoints with equal control exposure, refusal of altered
selection audits, whole-speaker exclusions, missing/duplicated cache views,
simultaneous recording gradients with momentum and fixed partial-batch
weighting, float/int8 feature capture, per-channel quantization, streaming
padding trimming, and complete auxiliary bundle export. Only small synthetic
inputs ran locally; no acoustic recording training ran on the Mac.

Supervisor requests 1–3 returned `NotFoundError: No such file or directory`
before launch, with zero compute expenditure. These requested a runtime/data
inventory, the supplied six-shard core/SDK test harness, and a training-image
prerequisite inventory. The direct prepare/fit/export request (run 4), explicit
node-image test retry (run 5), and final training-image request with the passing
fixtures and sequential prepare/fit/export (run 6) failed identically before
launch. All six recorded zero compute expenditure and produced no output
directory. Core/SDK tests could not execute. No acoustic preparation, fit,
export or target-session replay has executed. The available original training
data, augmentation banks and inherited teacher assets could not be inventoried.

No practice score has run; all three permitted scores remain available. There
is no `.autolab/model.json`, because no complete verified bundle exists. No
target gain, clean-session safety, studio preservation or aggregate gain is
claimed. An execution-service failure does not establish the acoustic premise
as false, so no killed outcome is asserted.

In the authorized supervisor training image, set
`PYTHONPATH=/app/modal:/app/tools`. Run this CLI's `prepare`, `fit`, then
`export` with
`--out /scratch/listener-confusion-balanced-minibatches/refit-v1`; preparation
and export also require `--model /lab/data/models/spliced-clean-int8`. Stop on
any failed command and retain the audits. A completed export creates balanced,
shuffled and length bundles.

Once training selection is frozen, decode identical complete target-session
audio with all three matched heads and replay the unchanged engine. Compare
actual registered warnings against both controls; convenient crops, local
spelling proxies and training loss do not establish warning recovery. Under
the user's continuation rule, fewer than three recovered targets requires
rollback of engine changes and a measured killed outcome. Three or more
requires an authorized practice score before evaluating collateral losses.
The card's one-warning forecast alone does not meet that continuation rule.
No fold has been made.
