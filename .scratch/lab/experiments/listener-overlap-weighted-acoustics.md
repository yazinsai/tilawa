# listener-overlap-weighted-acoustics

Status: blocked by remote execution; premise unmeasured

## Prediction and premise

Predict listener −1 and overall −1: one fewer phone wrong-word false warning,
with no predicted loss. Broad real-phone refits previously changed target
readings but also damaged other readings. This experiment tests whether those
refits learned from acoustic realizations that have little overlap between the
original and learner populations.

The registered comparison is a training-frozen overlap-weighted head against a
uniform-mixture head on identical engine-requested windows, comparing the
complete written and accused pronunciations. The task additionally requires
three recovered targets before practice scoring. Neither a failed remote job
nor an unknown target result is evidence against the acoustic premise.

## Implementation

`modal/student_overlap_head.py` implements prepare, fit and export. Shared
streaming capture/export helpers and their fixtures are adapted from
`exp/listener-transfer-weighted-head`; no transfer-gradient weighting is used.

Preparation requires the primary model and every inherited SDK insertion-teacher
asset, with the declared byte sizes and SHA256 checksums. It admits original
clean-match training recordings with speaker-cluster provenance and help-train
training speakers' scored none-labelled takes without mistakes or repeats.
Speakers occurring in a non-training partition or the explicit exclusions are
removed in full. Whole speakers are reserved within each source; exact waveform
copies attributed to different speakers abort preparation.

The encoder and streaming graph are frozen. For the integer classifier, cached
features are the dequantized rounded activations actually consumed by the head.
Capture checks all existing outputs and encoder caches for parity. Valid output
length follows the SDK, trimming final padding. Each recording receives an
ordinary view and two views using the existing shipped augmentation functions.

An exact CTC forward/backward computation derives soft frame-phone occupancies
from the authorized transcript and frozen baseline outputs, including blank
occupancy for pauses. No greedy word boundaries provide supervision. A separate
regularized logistic domain classifier for each phone supplies feature-by-phone
interactions. Its loss integrates these soft occupancies, gives each speaker
equal aggregate sampling mass, and balances original and learner priors within
each phone so transcript frequencies alone cannot distinguish domains.

The fixed recipe samples at most 512 original recordings, limits recordings to
30 seconds, samples at most 64 frames per cached view for domain fitting, and
uses three whole-speaker cross-fitting folds. Domain L2 candidates are 0.01,
0.1 and 1, fitted for 200 steps; reserved training speakers select regularization.
Each training speaker's weights come from a discriminator fitted without that
speaker. Phones without shared domain supervision receive neutral odds.

Original frame-phone supervision receives learner probability e(x, phone);
learner supervision receives its complement 1−e(x, phone). Global training-only
normalization restores the original mass of every phone, including blank. The
uniform control receives the same class-balance normalization with unit overlap
weights. The implementation records and checks both arms' actual phone mass.
Weights are not renormalized per minibatch.

Both heads begin at the shipped effective linear parameters and use identical
cached examples, minibatches and updates. The head objective is cross-entropy
against the frozen soft transcript occupancies plus KL to the shipped outputs
with coefficient 2. Both receive 12 epochs, batch size 8 and SGD step size 0.0001.
Reserved-training-speaker mean transcript-plus-preservation loss selects the
weighted checkpoint; the uniform control uses that same epoch and update count.
Neither target audio nor practice outcomes enter fitting or selection.

Export changes only classifier constants, retaining the existing integer
quantization geometry, encoder nodes/tensors, inventory, clock and manifest
contract. It checks cache parity, exports both arms and copies the complete
inherited teacher bundle. Frozen feature, discriminator, overlap-weight and
checkpoint hashes are checked before export. SDK inference, tracker, checker
and warning requirements receive no changes.

## Validation

All 14 local synthetic tests passed using isolated dependencies under ignored
`.autolab/`. The final log is `.autolab/overlap-unit-tests.log`.

- Exhaustive enumeration agrees with soft CTC occupancies, including repeated
  labels and intervening blanks; impossible complete transcripts are rejected.
- Symmetric overlap weights suppress domain extremes while retaining original
  phone mass. Neutral domain odds reproduce the uniform arm exactly.
- Domain priors balance sources within phones; missing shared-phone support
  receives neutral odds. Repeats, mistakes and unscored takes cannot supply
  canonical supervision. Whole-speaker folds keep all takes together.
- A tiny synthetic fit exercises discriminator selection, cross-fitting,
  matched head updates and checkpoint selection. Reserved speakers never enter
  discriminator fitting. Altering frozen weight files aborts validation.
- Float/int8 feature capture reproduces classifier outputs. Head replacement
  preserves encoder nodes and caches, including per-channel quantization.
  Streaming output trims final padding. Export rejects missing/corrupt inherited
  assets and copies a complete synthetic bundle with its manifest contract.

Python compilation and staged whitespace checks passed. These fixtures use no
practice or real training recordings and establish no target recovery.

Supervisor requests 1 and 2 (data inventories), 3 (the required six-shard
core/SDK harness), and 5 (CPU training-image tests, preparation, fit and export)
all returned `NotFoundError: No such file or directory`, with zero recorded
compute and no container logs. Request 4 was rejected because GPUs are permitted
only for bet cards; request 5 removed the GPU request. The recipe itself is
unchanged by that correction. The failed test harness does not establish that
core/SDK tests pass.

No practice scores were used; all three remain available. Target wins are
unknown, not zero. No aggregate listener/overall result, preference reversal,
clean-session safety or studio result is claimed.

## Limits

The real training run and export did not execute. Input compatibility, acoustic
overlap, checkpoint selection, exact-window preference reversals, target wins
and full-set behavior remain unmeasured. Real speaker counts and installed
teacher assets could not be inventoried. Runtime within the CPU budget is also
unverified. Frozen supervision and output anchoring do not guarantee preservation
of genuine mistakes.

Once supervisor execution is restored, run the following under `modal-run
--image train --timeout 60`, with `PYTHONPATH=/app/modal:/app/tools`:

```sh
python3 -m unittest discover -s modal -p 'test_student_*head*.py' -v
python3 modal/student_overlap_head.py prepare --model "$LAB_DATA/models/spliced-clean-int8" --out /scratch/listener-overlap-weighted-acoustics/overlap-v1
python3 modal/student_overlap_head.py fit --out /scratch/listener-overlap-weighted-acoustics/overlap-v1
python3 modal/student_overlap_head.py export --model "$LAB_DATA/models/spliced-clean-int8" --out /scratch/listener-overlap-weighted-acoustics/overlap-v1
```

Then inspect the frozen selection and complete export reports, compare written
and already-accused complete pronunciations on actual engine-requested target
windows, and measure target-warning recoveries without adjusting training from
those diagnostics. Fewer than three wins requires the instructed premise kill
and reversion; three or more requires the authorized practice score. A qualifying
bundle would be selected as `overlap-v1/weighted-bundle` in `.autolab/model.json`.
Core/SDK tests still need to run through the prescribed six-shard harness.

No model selection or killed outcome was written. There is no complete real
bundle to declare, and infrastructure failure does not justify a kill verdict.
The retained implementation is for continuation; it does not yet change what
the scored engine hears. No scorer, lab tooling, supplied data, evidence,
manifest, experiment card, tracker or checker was changed. No fold was performed.
