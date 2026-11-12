# Listener peak selection confidence

Status: needs-human. Implementation is committed, but calibration training and
audio validation are blocked by supervisor execution failures. No fitted bundle
exists; this experiment is not ready for scoring or folding. The card is unchanged.

## Prediction and mechanism

Predict two false warnings removed and one caught mistake lost: listener −1 and
overall −1. The three registered targets are the phone wrong-word warnings at
26:106:1 in p11, 55:48:1 in p36, and 92:13:1 in realuser-v1-sob-049.
The task's continuation rule requires three target wins before practice scoring;
the card additionally asks whether selection adjustment removes confirmation
from at least two targets relative to the peak-only control.

The SDK reports a maximum rounded class probability over each emitted CTC run
fragment. A fragment still open at an ONNX window boundary is emitted immediately,
and its continuation in later windows is suppressed. The new calibration uses
exactly this initial emitted fragment; completing runs later would change the
delivery behavior and the timestamps used by the engine.

`modal/student_peak_collect.ts` runs the unchanged Node listener on prepared
training recordings and caches its full-vocabulary outputs and emitted fragments.
`modal/student_peak_confidence.py` prepares only none-labelled training-speaker
recordings. It rejects excluded, unscored, unsupported, mistake-labelled and repeat
recordings, and rejects a speaker appearing in both training and non-training
partitions. Listener identity and training-label hashes travel with the cache;
mixed exports are rejected. The existing help importer establishes the expected
labels schema, but the actual supervisor export has not been inspected.

Correctness supervision sums over connected/pausal transcript forms and all CTC
placements. For each emitted fragment it computes the conditional probability
that a matching transcript phone occupies at least one of its frames. This is a
soft transcript-conditioned label obtained from the same acoustic model, rather
than independent listening ground truth. Consecutive pronunciation length repeats
are collapsed within each reference word, preserving repeats between words.

The fit models logit-transformed frame probabilities with separate logit-normal
distributions for correct and incorrect occurrences. Its maximum density includes
the CDF raised to the effective number of opportunities minus one. Effective
opportunities are `1 + (length − 1) × (1 − correlation)/(1 + correlation)`;
lag-one dependence is shrunk to the training estimate. Bayes' rule converts the
two maximum densities and the training correctness prior into occurrence
confidence. This effective-sample approximation is a modelling assumption whose
adequacy still requires validation. It does not establish that CTC peaks behave
like unconstrained independent samples.

Four speaker-grouped training folds select shrinkage and regularization using
speaker-averaged soft-label cross entropy. The peak-only control uses the same
training procedure with one opportunity regardless of run length. These choices
are fixed independently of practice incidents. No calibration parameters have
yet been fitted.

The Node directory and cached-tier loaders accept an adjacent
`run-confidence.json`, verifying its model hash, inventory and schema. The shared
listener changes only the existing token `conf` field; identities, frame bounds,
posterior rows and acoustic logits follow the original code. The public SDK also
exports the calibration type and functions. Registry downloads and browser/native
loaders do not install a new calibration asset. Checker rules are unchanged.
Because the tracker also consumes token confidence, unchanged identities alone
do not guarantee unchanged position events; that remains a full-replay check.

## Validation and limits

Local synthetic checks passed for selection discount at a fixed peak,
dependence shrinkage, the length-independent peak control, finite extreme
probabilities, and incompatible-asset rejection. A synthetic backend exercised
the real listener implementation and frontend: the calibration retained token
identities, frame bounds and posterior chunks; token output was invariant across
push sizes; and a run crossing a window boundary was calibrated from its initial
eight-frame fragment and emitted once. These checks used synthetic calibration
parameters, not fitted training parameters or practice audio.

Added SDK unit coverage includes these behaviors and an actual tiny ONNX fixture
comparison. Added Python unit coverage compares transcript probability and
fragment occupancy with exhaustive CTC paths, including repeated phones,
alternative transcripts and an absent class. Python syntax compilation and
`git diff --check` passed. The ONNX/Vitest suite and numerical Python unit tests
have not run. Local NumPy cannot import because its installed extension has an
incompatible architecture; fitting remains assigned to Modal as required.

All four authorized supervisor requests failed with
`NotFoundError: No such file or directory`, before any command output:

1. Training-data and model-directory inspection, one shard.
2. The prescribed `bash tools/autolab_tests.sh`, six shards.
3. A second read-only data inspection, one shard.
4. Minimal `pwd`, one shard, five-minute timeout.

The failure therefore prevents training-data access, calibration fitting, SDK
tests, target/control replay and bundle verification. It does not establish a
failure of the acoustic premise. Target wins and losses are unknown. No practice
scores were requested; all three remain available. There is no
`.autolab/model.json` and no fitted `run-confidence.json`. Running the primary
model without the calibration would not evaluate this change.

Resume by restoring supervisor execution, inspecting the help-train export,
running the numerical and SDK/core tests, collecting training fragments, and
freezing both fits through the speaker partitions. Then replay the three targets
with the unchanged tracker/checker in baseline, peak-control and selection arms.
If fewer than three targets are won, roll back the engine implementation and
record a killed result under the task's rule. Otherwise finish and verify a bundle
containing the unchanged primary model, its existing auxiliary assets, and the
frozen calibration before an authorized practice score. Neither continuation nor
kill is justified by the currently unmeasured targets.
