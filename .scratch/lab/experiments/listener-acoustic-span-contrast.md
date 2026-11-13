# listener-acoustic-span-contrast

Status: needs-info

## Prediction and premise

The card predicts listener -1 and overall -1: a false rival may combine plausible
individual acoustic fragments without a coherent complete pronunciation. The
experiment contrasts complete recorded words against compositions of the same
constituent phones from different occurrences. The user's continuation rule
requires at least three actual target wins before practice scoring. None of the
four registered targets has been evaluated, so neither that rule nor the card's
zero-reversal kill criterion has been measured.

## Implementation

The feature extraction, exact-window certificate transport and inheritance
checks reuse `exp/listener-duration-matched-hard-negatives`. The new training
examples and temporal readout replace that experiment's different-word negatives
and independent-frame head; no previous trained head is reused.

A separate instrumented graph exposes the inherited listener's frozen pre-CTC
features. Export requires bit-for-bit primary-output/cache parity over eight
successive windows. The fixed 32-dimensional projection limits cached feature
size. The untouched card resolves the inherited listener directory; there is no
fallback to a different primary model. Bundle assembly validates every declared
new asset before copying, preserves every inherited file including teacher and
external ONNX data, and checks the resulting complete file tree. Python/SDK head
parity on training-only fixtures is required before declaring a bundle.

Preparation uses only help-train training speakers and excludes any speaker
with a non-training row, in addition to the supplied release-speaker exclusion.
None-labelled takes must have no planted events, repeats or unsupported changes.
Complete scripted substitutions can enter as coherent positives only with an
explicit, unambiguous attested spoken spelling and greater full-transcript CTC
mass than the canonical transcript. They are aligned to the spoken candidate,
not corrected to canonical text. This acoustic check is a weak provenance guard,
not an independent listening judgment. Substitution positives are also retained
when rare phones prevent fragment matching.

Complete supported transcripts supply CTC peak anchors. Midpoints in uncertain
blank intervals split constituent phones. Each negative phone has identical
identity to the target and comes from a distinct occurrence in the exact same
recording; a composition requires at least two donor occurrences. Original
fragment durations must be within a factor of 1.5. Both negative fragments and
same-occurrence controls pass through the same intermediate-length interpolation
and join positions, yielding identical complete phone sequences and final
frame counts. Three boundary-jitter views are marginalized during fitting.
Unmatched words cannot become fragment negatives. Availability of enough such
matches remains unverified.

Both arms have the same positives, ordinary absent-pronunciation negatives,
initial weights, architecture, optimizer and update count. Additional negative
slots use fragment compositions in the contrast arm and ordinary absent
pronunciations in the control, which never sees composed negatives. The readout
combines each frozen frame, its adjacent-frame difference and the queried phone
embedding in a nonlinear hidden layer. Unlike independent emissions, this can
use transitions between acoustic fragments. Scoring marginalizes all ordered,
nonempty phone segments, consuming every candidate phone and every frame, with a
normalized timing prior and equal-prior outer-boundary alternatives.

A fixed speaker hash assigns fitting, checkpoint-selection and independent audit
partitions. Normalization uses fitting speakers only; matching stays inside each
partition. Each arm selects its checkpoint by selection-partition BCE. A fixed
300-word fragment cap per partition, 100-frame word cap, 24 epochs and enforced
one-day cached-feature training ceiling bound compute. Training reports record
pair provenance, speaker partitions, preservation counts, input/code hashes,
checkpoint histories and audit results. Practice targets select none of these.

At an existing confirmed string-path wrong word, the SDK compares complete
written and already-accused pronunciations on the exact original native-frame
window. A certificate requires a strictly positive contrast margin and strictly
negative control margin. It binds both strings, location and bounds; invalid
scores, ties or changed bounds preserve the warning. Graph confirmations retain
their existing path. Internal issue ownership is preserved so insertion merging
keeps baseline behavior. The SDK transports certificates in ordinary evidence
and applies corresponding final uncertainty revisions at session completion.

## Validation and blocker

Checks completed locally: Python compilation, shell syntax, six bundle unit
tests, TypeScript syntax checks and `git diff --check`. Training transcript smoke
checks exercise spoken substitutions, exclusion of repeats, full CTC path mass
and repeated-phone consumption. Syntax and smoke checks do not replace the
required core/SDK suites. Four numerical Python tests could not run locally:
NumPy's installed extension has the wrong architecture and torch is absent.
Training remains assigned to Modal.

Added unit tests cover same-phone/same-take fragment matching, identical duration
and join geometry, finite temporal-head gradients, boundary marginalization,
complete pronunciation consumption, strict preference reversal, certificate
serialization/identity, graph and insertion preservation, end-of-session
revisions, spoken-transcript construction, inherited teacher coexistence and
missing/corrupt/wrong-primary bundle rejection.

Four authorized supervisor requests returned
`NotFoundError: No such file or directory`, with zero compute recorded and no
container logs or LAB_OUT:

1. Read-only data/help-train/model inventory.
2. Required six-shard core/SDK tests.
3. Complete extraction, training, assembly and parity pipeline (60 minutes).
4. Retry of the required six-shard tests.

The supervisor resource must be restored before dependent work can continue.
This is an infrastructure blocker, not a measured acoustic kill. No real
training recording was inspected, no head was fitted, and no complete model
bundle exists. No `.autolab/model.json` or killed outcome was declared; running
the inherited model alone would not evaluate this implementation. The untracked
`.autolab/blocker.json` records the failed requests. All three practice-score
allowances remain, and the card, scorer, lab tooling, data, evidence and supplied
manifests are unchanged.

After restoration, rerun the final code's core/SDK suites and the fixed training
pipeline, verify inherited-output and exported-head parity, inspect training-only
matching/audit results, and evaluate the four actual engine warning windows. If
fewer than three targets are won, roll back engine changes and record a measured
killed result. If at least three are won, declare the verified scratch bundle and
practice-score before deciding. No fold is authorized or performed.

Unknowns include transfer to phone-degraded practice recordings, gains on the
four targets, preservation of coherent mistakes, and full-set tradeoffs.
Completion-time revisions do not establish timely suppression of live warnings;
full audio retention and a second encoder pass are not device-budget results.
The committed code is an unvalidated implementation checkpoint, not a completed
or score-ready experiment.
