# Per-example early-learning retention

Status: implementation prepared; acoustic premise and candidate validation are
blocked by supervisor execution failures. The experiment card is unchanged.
No trained model bundle is declared.

## Prediction and premise

The preregistered prediction is two false warnings removed and one caught
mistake lost: listener −1 and overall −1 wrong calls on lab-v1.5u. The proposed
failure is late memorization of recording-specific acoustic peculiarities after
transferable phone distinctions have been learned. This cause is unverified.

The premise requires at least two registered targets to prefer the complete
written pronunciation at a training-selected early control checkpoint, then
prefer the complete accused pronunciation later in the same control trajectory.
The targets are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

No target identifiers or verse references enter the training or export code.
Unavailable execution is not evidence that the premise failed.

## Training mechanism and control

`modal/student_early_learning.py` initializes one fresh linear phone head over
the frozen shipped encoder. A temporal bank stores detached framewise phone
predictions separately for each recording/view. It records predictions during
transcript-only warm-up. After warm-up, a retention arm adds
`lambda * mean(log(1 - dot(history, current_probabilities)))` to CTC. History is
an exponential moving average with decay 0.9, read before the current update;
current predictions are detached before updating the bank. It changes learning,
without averaging classifier parameters or consulting the reference at inference.

Each retention arm branches from its control's exact warm-up parameters,
momentum state and prediction history. The control continues with identical
examples, permutations, learning rate and preservation batches, without the
retention penalty. All schedules start from the same freshly initialized head.
A zero-strength synthetic arm checks exact parameter and validation equality
with its control after every matched epoch.

Supervised clean takes use complete-recording transcript CTC, normalized by
frame count. They do not receive a post-warm-up KL anchor toward the shipped
head: introducing that anchor could directly manufacture the alleged reversal
toward the shipped head's mistaken rival. Mistake, repeat and other non-clean
takes instead provide only transcript-free KL preservation of shipped outputs
after warm-up. Their scripts never become performed transcripts. Full recordings
include speech and pauses; no loss is applied to extra end-padding rows.

The frozen recipe uses seed 831, up to 512 original clean-match recordings,
at most 30 seconds per recording, three cached views, SGD learning rate 0.001
and momentum 0.9, batches of eight supervised examples and four preservation
examples, and preservation weight 2. Native audio is one view; two views draw
from the existing shipped RIR, noise, gain and SpecAugment recipe. Acoustic
augmentation, initialization and example order are fixed before target feedback.

Training-only validation selects among warm-ups of 4, 8 and 12 epochs,
retention strengths 0.1, 0.3 and 1, and checkpoints through epoch 24. The
validation objective is CTC on clean takes and KL preservation on non-clean
takes, averaged equally across speakers. Ties use fixed strength, warm-up and
epoch ordering. No practice recording, warning, score or target-window preference
participates in selection. The selected schedule's warm-up endpoint is its early
control; its final control is the late premise comparator. The control at the
selected retention epoch is also retained for equal-update comparison.

## Data and provenance

`modal/student_head_refit.py` requires original cached performed transcripts and
speaker metadata under `/student/s1`, plus the existing augmentation banks under
`/student/aug`. Their availability through this supervisor is unverified. Missing
or ambiguous provenance fails preparation rather than admitting another source.
Original speakers are excluded in full when any member is outside the training
split or on the existing release exclusion list; unknown speaker clusters are
not admitted. Help-train similarly excludes every speaker with a non-training
or excluded take. Canonical supervision admits only none-labelled takes without
events, repeats or unsupported regions.

Whole training speakers are split separately by supervised source, reserving
approximately 20% for validation. All views and preservation takes of a
supervised speaker follow its partition. Speakers with preservation takes only
enter training, without being claimed as validation speakers. Identical
waveforms with conflicting speaker identities are rejected. Initial head,
feature graph, cached features, plan and selected checkpoints have checksum
provenance checked before fitting, comparison and export.

## Feature capture, export and diagnosis

The shared refit helper captures the actual frozen classifier inputs, including
rounded/dequantized activations consumed by the int8 graph. It checks original
output and cache parity. Complete-recording feature capture carries streaming
states, verifies per-window row counts, and trims final padding to the SDK's
`outputsFor` count. Training retains the shipped phoneme inventory, blank class,
40 ms clock and frontend.

Export changes only phone-head parameters and their weight-quantization
constants. It retains per-tensor or per-channel quantization geometry, checks
finite output geometry and exact encoder-cache parity, and verifies every
inherited auxiliary SDK asset before writing any candidate. The insertion
teacher graph and external weights are required; comparison bundles share one
physical copy through hardlinks. The existing SDK directory loader accepts this
bundle contract. Tracker, checker and confirmation requirements are unchanged.
Supplied or tracked manifests are not edited; each exported bundle receives its
own generated manifest.

The separate `compare` command reads externally captured full-stream feature
windows from the actual baseline accusation and complete written/accused phone
sequences. It requires the frozen feature-graph hash, checks selected-head and
plan hashes, and reports full CTC written-minus-accused advantages at early,
late and retained checkpoints. It cannot alter selection. These float-head
acoustic preferences are not engine-warning wins; exported models still need
ordinary engine replay, including the existing auxiliary evidence routes.

## Validation and remaining work

Python source compilation and staged whitespace checks pass. All 13 synthetic
tests passed in 31.178 seconds in an isolated workspace environment using
NumPy 2.5.4, Torch 2.15.0, ONNX 1.23.2 and ONNX Runtime 1.31.1. Only tiny
synthetic inputs were used; no recording training or acoustic evaluation ran
on the Mac. The fixtures cover speaker separation, script-free preservation, detached
independent histories, retention gradient direction, repeated-articulation CTC,
zero-strength paired trajectories, target-probe separation, feature-cache
tampering, float/int8 capture, head-only export, per-channel quantization,
stream-padding trimming, cache state and complete auxiliary assets.

Supervisor run 1 requested a read-only data inventory; run 2 requested the
provided six-shard core/SDK harness. Both returned
`NotFoundError: No such file or directory`, without command output. Run 3
retried only a working-directory/environment print; run 4 retried the focused
Python fixtures. Both returned the identical error. Core/SDK tests could not
execute. No original-cache, augmentation-bank or teacher inventory has succeeded.
No recording training has run.

No early-to-late target reversals, warning recoveries or aggregate changes have
been measured. No practice score has run; all three permitted attempts remain.
There is no `.autolab/model.json` or killed outcome: no complete trained bundle
exists, and neither acoustic stopping criterion has been measured. Genuine
mistake preservation, clean-session safety and studio behavior remain unknown.

When supervisor execution is available, run `prepare`, then `fit`, with
`PYTHONPATH=/app/modal:/app/tools`,
`--out /scratch/listener-early-learning-retention/refit-v1`, and for preparation
`--model /lab/data/models/spliced-clean-int8`. The environment needs NumPy,
Torch/Torchaudio, ONNX/ONNX Runtime, PyArrow, SoundFile and SciPy, as well as the
authorized source mounts above. Keep the same frozen plan for subsequent stages.

After training selection is frozen, capture full-stream features for the four
baseline accusations and run `compare --windows <diagnostic.json>`. Each request
contains `key`, `features`, `feature_graph_sha256`, `from_frame`, `to_frame`,
and complete `written`/`accused` ID arrays. Fewer than two early-to-late reversals
kills the premise. Otherwise export with the same `--out` and `--model`, verify
the complete bundles through the ordinary SDK loader, and replay the target
sessions with the retention model. Fewer than three actual target wins requires
reverting engine changes and recording a killed outcome under the user's
stopping rule. Three or more requires an authorized practice score before
assessing the fold gate. The forecast alone does not satisfy that threshold.
