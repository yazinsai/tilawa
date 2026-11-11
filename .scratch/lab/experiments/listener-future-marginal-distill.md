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

Twelve focused Python tests passed locally under the Python architecture matching
the installed NumPy; three Torch/ONNX-dependent tests were skipped. They check observable-sample preservation, probability
averaging rather than averaging logits, teacher frame coverage, a runtime audit
that rejects nonlocal student changes, deterministic training-speaker admission
and donor partitioning, native chunk selection, same-frame disagreement
bookkeeping, preservation of held pronunciations beyond the student deadline,
canonical alignment against exhaustive CTC paths (including identical adjacent
labels), and exact streaming replay. Replay coverage exercises
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

The runner now protects the maximum of the deployed input deadline and the end
of every word supervised by the chunk. Training word ends come from the original
teacher's complete canonical CTC alignment, including held and repeated phones,
with a feature-analysis window retained after the final assigned frame.
Diagnostic cases must enroll a stop time covering the complete planted
pronunciation. The helper checks that samples through the resulting boundary
remain identical. Alignment estimates are not ground-truth boundaries: real
training-audio spot checks remain necessary before these targets can support the
card. No such check ran because the supervisor failed before execution.

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

## Prepared implementation and restart commands

`modal/student_future.py` contains the geometry, pronunciation guards, probability
mixture and same-frame diagnostics. `modal/student_future_train.py` enrolls only
scored training-speaker none takes with no events, repeats or unsupported labels,
rejects duplicate audio, and draws four distinct real continuations within each
recipient's fit or selection partition. Preparation selects at most eight
evenly spaced native chunks per recording by geometry, caches the original
student features and full transcript, and audits bit-identical student rows.
The two arms use the same masked KD sites plus full-recording CTC and KL
preservation, including pauses. Only the zero-initialized affine head correction
is trained: twelve SGD epochs at 0.02, with epoch zero eligible for selection
using speaker-balanced CTC plus preservation loss on selection training speakers.
These constants were fixed before acoustic results; no practice tuning occurred.

`modal/student_future_export.py` retains the int8 encoder and state tensors,
grafts the selected head, copies the teacher graph and every external-weight
member, and verifies their checksums. The runner requires full exported-graph
parity on selection-speaker recordings before declaring the bundle complete.
The SDK loads this through its existing primary-model and adjacent-teacher
contract; tracker, checker and insertion verification code are untouched.

Once the service is restored, use its `--image train` mode to run:

```sh
python3 -m unittest discover -s modal -p 'test_student_future*.py'
python3 modal/student_future_train.py diagnose --model /lab/data/models/spliced-clean-int8 --help /lab/data/help-train/labels.jsonl --cases <externally-enrolled-cases.json> --out "$LAB_OUT/premise.json"
python3 modal/student_future_train.py prepare --model /lab/data/models/spliced-clean-int8 --help /lab/data/help-train/labels.jsonl --out /scratch/listener-future-marginal-distill/prepared
python3 modal/student_future_train.py fit --prepared /scratch/listener-future-marginal-distill/prepared --out /scratch/listener-future-marginal-distill/refit
```

Cases are JSON objects with `wav`, `start`, `stop` and planted `phones`, obtained
from the existing labels only for diagnosis. Run preparation/refitting only if
at least two target incidents meet the premise, after pronunciation-boundary
checks. Both arms use original authorized none takes, without synthetic student
transcripts or channel augmentation. The sparse target cache, small four-draw
Monte Carlo approximation, estimated training boundaries and phone-head-only
capacity may limit transfer. None has been measured here.

Finish real-bundle parity, product-path target recovery and the required core/SDK
suite before declaring `.autolab/model.json` for `refit/marginal`. Scoring the
main model in the meantime would not evaluate this experiment. No killed
outcome is written because neither the premise nor target recovery was measured.
