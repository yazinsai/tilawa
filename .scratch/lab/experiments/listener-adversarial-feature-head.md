# listener-adversarial-feature-head

Status: needs-info

Prediction: remove two listener false wrong-word warnings and lose one caught
substitution: listener −1, overall −1. This is below the fold threshold and is
an exploration, not a claimed release gain. The experiment card is unchanged.

## Implementation

The ordinary and adversarial heads are identical 32→48→phoneme tanh networks.
They receive the same frozen pre-CTC encoder features, projected with the same
fixed random matrix. Complete pronunciation scores marginalize CTC boundaries,
normalizing the timing partition conditional on complete emission. The difference
of candidate scores is antisymmetric; training randomizes candidate order and
its target together. Neither candidate receives a “written text” flag.

Training anchors are exclusively none-labelled help-train training speakers.
The preparation utility rejects contradictory none labels, excludes every
speaker appearing outside the training split, and additionally accepts an
explicit release-speaker exclusion list. It aligns whole canonical none takes
to infer word intervals; no mistake script supplies a performed transcript.
Attested competitors are taken from canonical words with recorded positive
examples. The nearest two different pronunciations supply contrasts when both
complete candidates fit the acoustic interval. Each cross-validation training
fold further requires the rival to have a positive example in that fold.

Both heads start from identical weights and use identical examples, candidate
orders, optimizer settings and epoch counts. Only the adversarial head performs
three projected gradient-ascent steps on each example before its update. The
attack increases incorrect-pronunciation preference, with coefficient RMS
bounded by its radius. A shared rank-eight PCA basis, estimated from within-phone
residuals of training recordings, constrains feature changes to observed
acoustic-variation directions. Features, normalization and that basis are fitted
separately within each training fold.

The candidate radii are 0.1, 0.3 and 0.6. Three speaker-grouped training-only
partitions select by speaker-averaged natural contrast accuracy, requiring all
covered ordinary-head rival preferences on weak substitution preservation
controls to survive. These controls are actual engine-requested windows at
labelled substitution locations; scripts never assign phonemes to their frames.
The training utility refuses to export if no radius meets its training-only
preservation constraint. Scales and checkpoint selection never read practice.

The SDK replays the primary streaming geometry with a separate graph and cache
instance that exposes pre-head encoder features. The exporter only adds an output
to a separate graph copy and verifies original outputs and caches on eight
streaming windows. The primary model and its manifest are not changed.

The checker exposes the complete written and already-accused pronunciations and
exact frame interval for an existing supported string-path confirmation. The
SDK supplies a certificate only for a strict finite reversal: adversarial prefers
written and ordinary prefers rival. Exact pronunciation, word occurrence and
frame interval must match. Graph confirmations are exempt. Internal confirmed
issue ownership remains unchanged; the veto changes the public verdict only,
so it does not feed different ownership into insertion decisions. Audio sessions
publish a downgrade at completion and retain the certificate in recorded
evidence for replay through the existing scored listener path.

The optional bundle descriptor requires both the trained head JSON and feature
extractor, with sizes and SHA-256 hashes. The head also binds to the exact primary
hash, inventory and dimensions. Missing or corrupt declared files fail loading;
no placeholder artifact or missing-path model declaration has been created.

## Validation and present limit

Local `python3 -m py_compile` passed for the exporter, trainer and mechanism
tests. `git diff --check` passed. Unit tests were added for candidate-order
symmetry, normalized complete timing, bounded attack direction, unknown and
nonfinite observations, exact certificate matching, graph preservation,
primary token/posterior preservation, audio-session downgrades and missing or
corrupt declared bundles. **These tests have not run successfully.**

Four supervisor requests all failed with
`NotFoundError: No such file or directory`, before returning shell output:

1. Storage/help-train/model census.
2. Minimal `pwd`, directory listing and storage probe.
3. Required six-shard `bash tools/autolab_tests.sh`.
4. Python dependency and adversarial mechanism-test invocation.

The returned result files report zero compute spend. The two probes establish
that this failure occurs before model training or SDK execution; it is not a
measured failure of the hypothesis. No alternative Modal route or external
training data was sought.

Consequently there is **no fitted bundle, no training validation result, no target
preference measurement, and no practice score**. Practice scoring was invoked
zero times. Target recovery is unknown; it is not zero measured recovery. The
implementation is retained as an unvalidated draft for resuming after the
supervisor runtime/storage is restored. It must not be scored as a completed
experiment using the unchanged primary model.

The card's two-flip criterion and the task's stricter continuation instruction
have not been evaluated. Once measured, fewer than three actual target warnings
removed requires a premise kill and engine reversion; three or more requires a
supervisor practice score before a decision. No constants may be changed from
those training-frozen choices in response to target results.

## Resume

On the authorized supervisor only:

1. Run the required six-shard SDK/core checks and the Python mechanism tests.
2. Copy the complete shipped primary bundle, including its existing auxiliary
   assets, into `/scratch/listener-adversarial-feature-head/bundle`. Keep primary
   files byte-identical. Run `modal/student_adversarial_export.py` with `--model`
   pointing to the shipped bundle and `--out` pointing to that scratch bundle.
3. Run `packages/sdk/src/bench/adversarial-feature-cli.ts` with `--model`,
   `--extractor`, `--labels /lab/data/help-train/labels.jsonl`, an explicit
   `--excluded-speakers` training exclusion file, and `--out` for training rows.
   Review coverage and provenance before fitting.
4. Run `modal/student_adversarial_feature.py --input <rows> --metadata
   <bundle>/adversarial-feature-metadata.json --excluded-speakers <file> --out
   <bundle>`. Inspect speaker partitions, control coverage and frozen radius.
5. Verify the complete bundle loader and primary output/event/insertion parity
   on training audio. Inspect the four registered targets with the SDK CLI's
   `--audio` and `--evidence` mode, using actual engine-requested intervals and
   complete pronunciations. Labels must not choose comparisons or boundaries.
6. Only after successful bundle verification, declare the relative scratch
   bundle path in untracked `.autolab/model.json`. If at least three targets
   are won, invoke supervisor `practice-score .`; otherwise revert the engine
   changes, record the measured premise kill and commit the result.

Risks remain: coarse frozen features can already have lost the distinguishing
cues; word interval inference may be poor; finite help-train lexical coverage
may be inadequate; a feature attack can suppress a genuine pronunciation
contrast. Training-speaker preservation is a control, not proof of generalization.
The whole-session auxiliary feature pass and end-of-session downgrade also have
latency and memory costs. None of these limits has been measured here.
