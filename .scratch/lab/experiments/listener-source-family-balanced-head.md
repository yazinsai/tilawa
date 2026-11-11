# Source-family-balanced phone-head refit

Status: implementation committed; experiment unvalidated because the supervisor's
Modal requests failed before container execution. No trained candidate bundle or
practice result exists. This is an infrastructure block, not a premise kill.
The experiment card is unchanged.

## Prediction and mechanism

Predict one removed listener false warning and no losses: listener −1, overall
−1. The forecast is deliberately below the fold bar. The proposed cause is
pseudoreplication: multiple copies of one performed recording can give its
consonant realizations more influence than the apparent recording count suggests.

`modal/student_source_families.py` identifies exact decoded waveform copies and
conservatively verifies re-encodings. It permits codec delay and positive gain,
requires at least 99% waveform coverage, and requires correlation of at least
0.995 globally and in every supported 250 ms block. A full-recording energy
envelope screens candidates; it cannot prove identity. Matching text, reader or
verse never proves membership. Complete-link clustering prevents a sequence of
weak matches from joining different takes. Silence alone cannot establish a
shared performance. A separate pairwise audit rejects any verified alias across
training and validation, including pairs that complete-link clustering separates.

`modal/student_source_head.py` admits original clean-match training recordings
with speaker-cluster provenance and help-train's scored, none-labelled training
takes without labelled events, repeats or unsupported mistakes. It removes
held-out/gold original clusters and excluded help speakers. Recordings are capped
at 30 seconds. A deterministic subset of at most 4,096 original recordings is
selected before acoustic grouping, without enriching it for duplicates. A fifth
of eligible training speakers per source is reserved before fitting. These are
training-only validation partitions, not release or practice speakers.

The utility captures the shipped ONNX encoder's features with its streaming
caches. For an integer head, it captures the rounded activations actually used
by that head. It verifies that feature capture leaves all original outputs and
cache states identical, and checks reconstructed head log probabilities against
the original model. Final input padding is trimmed to the SDK's true output
length. Each recording has one ordinary view and two cached views using the
existing student augmentation functions. Clean transcripts collapse held-phone
runs as the shipped spliced CTC recipe does; ordinary speech and in-recording
pauses also receive KL anchoring to the shipped outputs.

Both arms start from the same shipped linear classifier and use the same cached
data, deterministic minibatches, SGD settings and update counts. The control
preserves uniform recording mass, distributed over its cached views. The
balanced arm gives each source family equal total mass across all copies and
views while retaining their relative weights. Global importance weights are
never renormalized within a minibatch. CTC plus the same KL anchor is used in
both arms; the encoder is frozen.

The balanced checkpoint is selected by reserved-training-speaker loss, averaging
families within speakers. Its original-weight control uses that same epoch and
update count. Full-population initial gradients are recorded separately.
`gradient_effect()` measures the infinitesimal change in the complete
written-versus-rival CTC margin under each gradient. A balancing-only correction
means a positive effect from balancing and a nonpositive effect from the
control. This diagnostic cannot select a checkpoint.

Export replaces only the phone-head constants, retaining encoder nodes, other
tensors, vocabulary, streaming geometry and manifest contract. Both arms are
exported. The exporter verifies cache parity and requires every currently
declared SDK auxiliary asset, including the existing insertion teacher and its
external weights, with the registered sizes and checksums. Missing assets abort
export rather than silently removing that listener behavior. Tracker, checker
and SDK inference rules are unchanged.

## Validation

All 16 local lightweight tests passed in an isolated environment under ignored
`.autolab/`, using generated waveforms and small synthetic ONNX graphs:

- Copies and delay/gain re-encodings group; different performances, local speech
  changes, silence and nontransitive matches do not falsely join.
- The partition audit catches a verified cross-partition pair even when
  complete-link clustering separates it.
- Family weights include copies and all augmentations, retain original relative
  proportions, and equal the control when no balancing difference exists.
- Float and integer ONNX head extraction reproduce the original outputs; head
  replacement preserves nodes and streaming state.
- Stream caching trims final padding, and clean targets collapse held-phone runs.
- Export rejects missing or corrupt auxiliary files, copies a complete synthetic
  bundle and preserves the manifest's inference contract in both arms.
- A tiny synthetic CTC fit demonstrates a corrective gradient present only in
  balancing while selecting the same update count in both arms. This is a unit
  fixture, not evidence about any target incident.

Python compilation and whitespace checks passed. The final test log is retained
locally as `.autolab/source-head-unit-tests.log`.

Supervisor Modal requests 1–5 all returned `NotFoundError: No such file or
directory` before container execution, with no command logs or measured compute.
They included training-data inventory, the required six-shard core/SDK harness,
and a training-image request for the new unit tests followed by prepare, fit and
export. The failed harness does not establish that core/SDK tests pass.

## Limits and outstanding work

No real training inventory, source-family counts, acoustic grouping calibration,
encoder-feature cache, original/balanced refit, training-speaker checkpoint
selection, target-gradient comparison or target-warning recovery was measured.
The strict copy policy has synthetic checks but no measured recall or false-join
rate on the authorized corpus. Real input compatibility, training runtime and
the complete installed teacher bundle also remain unverified.

The card requires at least two target comparisons with a balancing-only
corrective gradient. The task additionally requires at least three recovered
targets before practice scoring. Neither condition has been evaluated, so no
kill verdict is justified. Target wins are unknown, not zero. No practice scores
were used (all three remain available), and no aggregate gain, loss, clean-session
safety or studio result is claimed.

Once Modal execution is restored, first verify the authorized input inventory
and calibrate copy verification within training speakers. Then run prepare, fit
and export under the training image, inspect frozen selection and bundle reports,
and measure the registered complete-pronunciation gradients and target recoveries
without changing grouping or selection from those diagnostics. A qualifying
candidate still requires SDK validation and the authorized practice score.

No `.autolab/model.json` was written: there is no complete real model bundle to
select. No killed outcome was written because infrastructure failure does not
test the premise. The retained code is a refit implementation for continuation;
it does not yet change what the scored engine hears. No scorer, lab tooling,
supplied data, evidence or manifest was changed. No fold was performed.
