# listener-future-marginal-distill

Status: needs-info

Prediction: listener -1, overall -1, with one recovered added word and no
predicted losses. This is the card's prediction, not a measured result.

The proposed mechanism averages complete teacher probability distributions over
alternative authorized training-recording continuations strictly beyond the
student's deployed receptive field. The student receives its original audio and
canonical transcript. An ordinary-teacher control and a future-marginal arm use
identical frozen native encoder features, examples, initial head, optimization
schedule, and speaker-grouped training selection. The export must preserve the
ordered phone inventory, 40 ms clock, streaming state contract, and existing SDK
insertion teacher, including its external weights.

## Validation so far

The native streaming geometry is a 45-feature-frame input window, advancing by
32 ten-millisecond feature frames and producing eight 40 ms output rows. At
16 kHz with a 25 ms analysis window, output rows 0–7 can observe samples through
exclusive bound 7440, rows 8–15 through 12560, and rows 16–23 through 17680.
This conservatively protects the full deployed input window and cached past.

Eight focused Python tests passed locally under the Python architecture matching
the installed NumPy. They check observable-sample preservation, probability
averaging rather than averaging logits, teacher frame coverage, a runtime audit
that rejects nonlocal student changes, deterministic training-speaker admission
and donor partitioning, and exact streaming replay. Replay coverage exercises
every tail phase for feature lengths 9–129, verifies cache progression and pad
placement, and rejects nonfinite or differently timed native-feature evidence.
These tests use synthetic distributions and a recording runtime; they establish
implementation contracts, not acoustic gains or ONNX-model parity.

Supervisor requests 1 and 2 for data inspection, request 3 for the six-shard
core/SDK suite, and request 4 through the explicit training image all returned
`NotFoundError: No such file or directory` before execution. No authorized audio
or model was inspected remotely, and the required core/SDK suites have not run.
There have been zero practice scores. No trained checkpoint or complete export
bundle exists, and `.autolab/model.json` has not been declared.

## Remaining validity checks

The replacement boundary must preserve the complete pronunciation supervised by
each selected chunk. Protecting only the student sample deadline does not prove
this: a held phone or its word may continue beyond that deadline. The current
helper can cut that continuation. Preparation and diagnostics need a later
protected pronunciation boundary before their results can support the card.

Once execution is restored, inspect authorized help-train metadata, verify
training-only none-labelled speaker admission, and enroll the five named
incidents externally for acoustic diagnosis. Retain full-session student history
and require bit-identical supervised student rows under every teacher-only
intervention. Count an incident toward the premise only when a missing planted
consonant's teacher/student identity disagreement disappears under a valid
future marginal. The correlated teacher diagnostic is not an independent
listening judgment and does not establish a recovered warning.

If at least two incidents meet that premise, run the matched head refits and
select checkpoints using only grouped training partitions. Verify cached
feature and exported-model parity and the complete auxiliary bundle, then
validate recovered warnings at the target keys. The user requires stopping and
reverting engine changes if fewer than three targets recover, and requires a
practice score before deciding if at least three recover. Neither count is
currently known. Infrastructure failure is not evidence that the acoustic
premise failed, so the experiment has not been marked killed.
