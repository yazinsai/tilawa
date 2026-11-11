# Recording-level substitution training support

Status: implementation draft; real training, target validation and the complete
model bundle are blocked by the authorized supervisor execution path. The
experiment card is unchanged. This branch is not ready for practice scoring.

## Prediction and mechanism

Predict listener −1 and overall −1: an existing false substitution may depend
on one isolated training recording rather than a supported acoustic distinction.
The card's premise kill is zero qualifying reversals among four complete target
comparisons. The task additionally requires at least three actual target warnings
removed before continuation to practice scoring. Neither criterion has been
measured on recorded audio.

The CPU trainer fits a compact diagonal-Gaussian pronunciation scorer on fixed
native acoustic features: eight coordinates, each the mean of ten consecutive
log-mel bins from the SDK frontend. There are no encoder updates, recognition
labels, target-specific forms or word-identity parameters. Emission means and
variances shrink toward phone-manner pools, excluding the phone's own data.

Forward/backward EM integrates every ordered phone boundary within each complete
none-labelled recording. Every pronunciation label must consume an observation;
no target label is optional. The scoring duration prior is normalized conditional
on complete pronunciation, and the same topology and prior apply to both
hypotheses. Repeated pronunciation labels remain separate required states.

Source WAV hashes identify recordings. Preparation rejects duplicate recordings
and excludes every speaker with a non-training take, together with an explicit
exclusion list. Only help-train training speakers' none-labelled takes fit the
emissions. Each jackknife fit removes every frame of exactly one recording and
refits normalization, pooled priors and all EM parameters from scratch. It does
not delete individual frames or reuse fractional occupancies from the full fit.

Three prior strengths (2, 8 and 32) are selected by canonical likelihood on
three deterministic speaker-disjoint folds, giving each speaker equal weight.
The selected strength is then fixed. Substitution-labelled recordings supply
weak preservation controls: every already supported string-path accusation
in those recordings is preserved conservatively, without assuming the script
identifies performed words or exact acoustic boundaries. For each held-out
speaker, the control comparisons use emissions and recording-deletion fits
trained without any of that speaker's audio.

For a comparison, define D as complete written log likelihood minus complete
accused log likelihood. Recording influence is (D after deletion − D full)
divided by the number of acoustic observations. Select the recording with the
largest absolute influence, with deterministic ties. A larger influence in the
opposite direction therefore prevents a smaller positive reversal from vetoing.
The preservation cutoff is the maximum positive influence among held-out
substitution controls whose full fit prefers the rival and deletion prefers the
written sequence, or zero when no such control reverses. Strictly exceeding that
cutoff preserves every available calibration reversal. Empty or incomplete
control calibration fails before asset export. This is a training-only operating
point, not a guarantee about unfamiliar speakers or real substitutions.

The SDK compares the exact complete sequences and endpoints used by an existing
supported non-graph wrong-word confirmation. A veto requires the full scorer
to prefer the accusation, the most influential deletion to prefer the written
pronunciation, and that influence to exceed the frozen cutoff. Unknown phones,
insufficient observations, non-finite scores and unsupported fits abstain.

Only the public confirmed wrong-word verdict becomes uncertain. The checker's
internal confirmation and ownership ledger remain baseline so insertion-tail
merges and later candidate generation retain their decisions. Certificates match
the word, both complete sequences and both endpoints exactly. They travel as
metadata on primary tokens, survive JSON serialization and native evidence-cell
rechunking, and can be consumed without waveform access. Graph confirmations and
all other warning classes are exempt.

Audio sessions compute the recheck before publishing live confirmations. After
existing gap and teacher-insertion enrichment, recorded evidence receives its
own certificates for its final primary windows. Primary decoding, confidence,
posterior rows, tracker input and events are unchanged. The Node directory and
tier loaders verify the optional training-support descriptor, fixed asset name,
byte count, SHA-256, inventory, observation clock and fit shapes. A declared
missing or corrupt file fails and releases the model backend. Legacy bundles
without the descriptor retain their ordinary behavior.

## Validation and limits

The three authorized supervisor requests all failed before container execution:

1. Run 1: read-only help-train layout, labels and scratch-capacity inspection.
2. Run 2: the supplied six-shard `bash tools/autolab_tests.sh` harness.
3. Run 3: the same harness after the draft implementation and initial checks.

Each returned `NotFoundError: No such file or directory`. No alternative Modal
route was attempted. No real training label, recording, target waveform or
provided model was accessed. The missing execution path needs restoration by
the supervisor; storage, training coverage and acoustic premise remain unknown.

Local checks used repository fixtures and small synthetic arrays only:

- Six core tests pass: actual complete-comparison discovery; strict finite
  reversal and cutoff checks; exact frame/pronunciation matching; baseline issue
  creation with only the public verdict changed; unchanged insertion ownership
  and decisions; and exempt parallel-graph confirmations.
- Seven SDK scorer tests pass: exhaustive complete boundary likelihood;
  normalized timing mass; a real synthetic recording-deletion reversal;
  per-observation influence; maximum-absolute selection, including an opposite
  influence; candidate-order invariance and required repeated labels; missing
  coverage abstention; frontend rechunking; and corrupted-model rejection.
- Six Node loader tests pass: the three existing teacher tests and new verified,
  missing and corrupt support-asset cases. These mock ONNX backend creation.
- Three Python tests pass: exhaustive boundary sums and forward/backward
  occupancies; whole-recording removal matching a fresh independent refit; and
  source, split, speaker-exclusion and duplicate-recording enforcement.
- A synthetic six-recording, three-speaker training run completed grouped model
  selection, preservation calibration, every deletion refit and JSON export.
  Python and SDK complete scores, selected recording and influence agree within
  1e-8 nats on that export. Its cutoff and model are toy results, not candidate
  assets or evidence about the four targets.
- TypeScript core/SDK compilation and `git diff --check` pass. Local compilation
  uses an ignored ambient declaration for the unavailable React Native ONNX
  peer. It does not validate that platform or substitute for the required Modal
  build/test suites. No full practice or release suite ran on the Mac.

The independent acoustic view is deliberately small and loses within-bin detail.
Diagonal emissions and the stationary timing model are limited approximations;
recording influence measures sensitivity of this new scorer, not causation in
the shipped student's original training. Sparse phone coverage can force
abstention, and weak script controls can include mislabeled or unperformed edits.
The acoustic fit may fail to distinguish any target, and unseen genuine swaps
may depend on an isolated training example too. Runtime cost and memory for the
full deletion bank have not been measured on real recordings.

## Remaining work

Restore the authorized supervisor path, audit help-train training speakers and
excluded-speaker provenance, then prepare recordings with the SDK training CLI
and run `modal/student_training_support.py` on Modal. Freeze its real selection
report and cutoff before reading the target comparisons.

Use the CLI's arbitrary audio/evidence probe to discover complete candidates
from the unchanged free engine. Incident keys select results for reporting only;
they cannot choose acoustic boundaries or pronunciations. Check all four actual
warnings, full/deletion preferences, influential recordings and frozen cutoffs.
If no comparison qualifies, or fewer than three actual warnings are removed,
roll back engine changes, record the premise result and write the requested
killed outcome. Otherwise finish and verify a bundle containing the unchanged
primary ONNX and manifest, all existing teacher assets required by that model,
and the fitted support JSON and descriptor. Verify ordinary SDK loading and
primary/tracker/insertion preservation before declaring `.autolab/model.json`.
Then request practice scoring, with at most three runs and no practice tuning.

No real fitted support asset or complete bundle exists yet. No model selection
file or killed outcome was written: an infrastructure failure does not establish
that the acoustic premise is false. Targets tested: zero; practice scores
requested: zero. All three practice-score allowances remain. No gain, loss,
clean-session result, studio result or fold is claimed. No scorer, lab tooling,
test data, provided manifest/evidence or card was edited.
