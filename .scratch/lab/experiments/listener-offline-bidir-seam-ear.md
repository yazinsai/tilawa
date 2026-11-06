# Full bidirectional student seam ear

Status: implementing; acoustic premise not yet measured. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1 wrong calls on lab-v1.5u, with two
of six named additions recovered and about one loss. The task's continuation
rule requires three retained target detections before practice scoring.

The primary student uses streaming chunk masks and chunkwise convolutions.
Its fixed-window ONNX export cannot become a full-lookahead model just by
buffering more audio. This experiment trains a separate copy with
`Zipformer2(causal=False)`, unrestricted attention and ordinary symmetric
convolutions, then exports a variable-length offline ONNX graph without
streaming cache inputs.

Training uses gold-excluded clean-match real recitation, random 2–8 second
sample crops, and starts after natural teacher-blank pauses. Canonical phones
are Viterbi-aligned to the retained teacher posterior rows. Only complete
canonical phone spans inside each crop enter CTC. KD maps each crop frame to
the nearest source teacher frame. The shipped RIR, noise, gain and
SpecAugment calls and settings are unchanged. No practice labels, splice
positives or teacher relabels enter training.

The intended secondary inference span runs from each detector-labelled pause
at an ayah or waqf-marked inner seam to 300 ms after the primary tracker's
step onto the second written word. Its phones are judged only by the folded
soft-cost lexical surplus rule and existing neighbour, prefix, back-run and
non-Quran exclusions. Primary phonemes, posterior rows, tracker events and
support decisions must remain unchanged.

## Validation and limits

Modal run 7 passed the core and SDK builds, all 756 core tests and 30 SDK
tests, including the five new offline decoder/span tests; one optional SDK
integration test was skipped. The supplied harness accepted its documented
Vitest worker RPC timeout after every affected test passed. No test deadline
or enforcement was changed. A subsequent diagnostic invocation in shard 0
failed because ignored analysis files were omitted from the source snapshot;
that failure did not occur in either test suite.

Pending: checkpoint compatibility, crop label tests, distant-right-context
test, full-context training, dynamic-length export parity, and six-target
greedy consonant census. Fewer than three targets with two planted consonants
in order kills the acoustic premise without practice scoring. If the model
passes that census, complete SDK integration and test the actual warnings
before practice scoring.
