# Transcription edit-risk phone-head refit

Status: implementation prepared; real-data validation blocked by supervisor
execution. No replacement listener is declared, and the experiment is unmeasured.
The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1: two recovered missed additions
with one possible collateral wrong call. The proposed defect is that a better
transcript-conditioned CTC likelihood need not produce fewer errors in the
unrestricted collapsed phone string.

`modal/student_edit_risk.py` builds a bounded acoustic-only CTC prefix lattice
from the current head and shipped posteriors, keeping blank and nonblank prefix
states separate. Identical collapsed transcriptions are merged. Every candidate
is rescored with its complete CTC forward sum, including all alignments and
blank separation between repeated labels. Candidate generation receives no
transcript, lexicon, reader identity or practice incident. The current-head
lattice is refreshed before each recording update.

For candidates H, exact probabilities P(h), retained mass M, recorded transcript
r and N=max(1,|r|), the conditional risk is
`sum_h [P(h)/M * Levenshtein(h,r)] / N`. This gives graded credit for recovering
a missing phone and penalizes invented or misordered phones. Log-space scoring
keeps normalization stable even when absolute retained mass underflows.

The risk arm adds that conditional risk and a coverage cost `−log(M)/T` to
ordinary transcript CTC per valid frame and shipped-posterior preservation KL
per valid frame. The coverage cost discourages moving probability outside the
lattice to improve its conditional risk. All objective weights are one, fixed
before any practice inspection. The ordinary-CTC control retains precisely the
same transcript and preservation objectives.

The approximation is explicit. The omitted probability is OTHER=1−M, not an
invented blank hypothesis or a probability floor. Validation reports M, OTHER,
conditional risk and full-expectation bounds:
`M * conditional <= full risk <= M * conditional + OTHER * max(T,|r|)/N`.
The omitted transcriptions' actual edit costs are unknown, so this is not exact
unbounded expected edit-distance training. A large OTHER mass can make the
bounds uninformative; coverage regularization does not guarantee a good lattice.

## Matched training and export

`modal/student_edit_risk_train.py` enrolls authorized, verified performed
transcripts with audio SHA-256 identities, plus help-train training-speaker
none takes with empty mistake/repeat/unsupported labels. Scripted mistakes
are never treated as canonical performed transcripts. Nontraining speakers
discovered in either corpus are excluded across both before enrollment;
duplicate audio and paths leaving the enrolled corpus are rejected.

Whole-speaker hashing creates fit and selection partitions entirely within
training speakers. Both corpora must occur in each partition. Preparation
caches native streaming encoder features for an original recording and two
deterministic shipped-augmentation views, retaining full utterances and real
pauses. Features, audio, model, auxiliary assets and augmentation banks have
recorded hashes. The frozen feature graph must reproduce every original
posterior and state output exactly before caching.

Both existing affine heads start from the same dequantized shipped parameters.
They receive identical recording order, views, weights, SGD recipe and update
count: seed 1729, twelve epochs, learning rate 0.02, gradient clipping at five.
Shared checkpoint selection uses paired speaker-balanced greedy phone error,
then the common CTC/preservation objective, including epoch zero. A separate
training-only limits step chooses among registered 8/16/32 prefix bounds by
geometric retained shipped mass relative to the largest bound; its 95% rule
does not claim 95% absolute probability coverage. Preparation, bound selection,
refit and export account for a cumulative one-day elapsed/CPU budget.

`modal/student_edit_risk_export.py` exposes the native preprojection features
and replaces only the affine phone head, preserving the quantized encoder,
ordered inventory, 40 ms clock, filterbank, streaming state and window geometry.
It copies and checksum-verifies the unchanged insertion teacher and all its
external weight files. Both selected bundles must pass optimized-runtime
posterior parity on training-speaker selection recordings before a final
training report is written. The scaffolding reuses
`exp/listener-collapse-sequence-distill`; tracker, checker, decoder and insertion
verification code have no changes.

## Validation and limits

Synthetic tests cover exhaustive transcription probabilities and gradients,
blank-separated repeats, unique reference-free candidates, omitted mass and
full-risk bounds, finite differences, rare-lattice stability, frozen evidence,
matched updates, refreshed proposals, cross-corpus exclusions, authorization,
audio identity, bounds independent of transcripts, native float/int8 projection
extraction, complete teacher assets and paired fit/export streaming parity.

The supplied supervisor rejected run 1 (data/model probe), run 2 (six-shard
`bash tools/autolab_tests.sh`) and run 3 (training-image data probe and numerical
tests) before command execution with `NotFoundError: No such file or directory`.
Run 4 (a minimal `pwd` command in the training image) failed identically.
Each result records zero expenditure and no command output. No remote jobs
remain pending. This is an execution-access failure, not a training or acoustic
result. The runtime issue has been raised with the user.

All 22 local synthetic tests pass, with no skips, in 9.929 seconds:

```
.scratch/edit-risk-venv/bin/python -m unittest discover -s modal -p 'test_student_edit_risk*.py' -v
```

The isolated local environment uses Python 3.13.7, Torch 2.8.0, NumPy 2.5.4,
ONNX 1.19.0 and ONNX Runtime 1.22.1. Only synthetic numerical/unit work ran
locally; no recitation was decoded or trained on the Mac. Python syntax
compilation and the staged diff whitespace check also pass. The synthetic
fit/export test exercises both bundle paths; its metrics are not acoustic
evidence about the targets.

No real recording has been enrolled or cached; neither real head has been
trained. No complete real model bundle exists, so `.autolab/model.json` is
absent. Core/SDK tests, actual data schemas, shipped-model extraction/parity,
target consonant recovery and engine warning changes remain unvalidated.
No practice scores have been requested; listener/overall changes and target
wins are unknown. Infrastructure failure does not satisfy a premise kill.

After execution is restored, run the numerical tests and core/SDK suite;
enroll eligible training assets, prepare caches, record training-only bounds,
fit and verify both bundles. Compare full-session primary tokens against the
matched ordinary-CTC control: the card requires at least two target spans to
recover previously missing planted consonants in order. The later builder rule
also requires at least three target wins before practice scoring. Neither has
been measured. Declare only a verified complete risk bundle, then score if
eligible; do not score the main model as a substitute. All three practice-score
allowances remain available.
