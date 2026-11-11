# Transition-coupled acoustic evidence

Status: blocked by the supervisor's execution service; acoustic premise untested.
No trained candidate is submitted. The experiment card is unchanged.

## Prediction and decision rule

The card predicts listener −1 and overall −1: one false warning removed, with
no predicted losses. The three target phone wrong-word warnings are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

The builder instruction requires three target wins before a practice score;
fewer than three measured wins would kill the experiment. No target comparison
has run, so neither that rule nor the card's acoustic kill criterion has been
evaluated. An execution error is not a measured zero-win result.

## Mechanism attempted

The prototype adds a low-rank consecutive-frame-label potential to frozen
student encoder features, including transitions to and from CTC blank. Its
numerator sums all ordinary CTC paths collapsing to the complete transcript;
its denominator sums all frame-label paths. Repeated identical phones require
an intervening blank, and acceptance requires the final phone or trailing
blank. Thus a partial written pronunciation cannot win by dropping its onset
or ending, and a transition potential cannot improve likelihood just by
adding unnormalized mass.

The matched independent-label control has the same parameter tensors and
adjacent-frame feature access, but contributes only unary label potentials.
Both arms were designed to use recording-balanced frozen-feature PCA,
rank four, and a shared regularizer selected by three training-speaker folds.
Only help-train `kind:none`, `split:train` recordings would be admitted; a
speaker with any non-training recording would be excluded. Neither head
receives verse identity or an expected-text prompt. The authorized training
data could not be inspected, so eligibility, coverage and feasibility remain
unverified.

The proposed SDK side extractor exposes the pre-CTC encoder tensor in a
separate ONNX graph and verifies equality of all primary outputs and streaming
caches before training. The warning hook discovers the complete written and
already-accused pronunciations and exact frame windows requested by existing
string-path support checks. A certificate is permitted only for finite strict
preferences: the independent control favours the rival and the coupled head
favours the written pronunciation on that identical window. Graph
confirmations are exempt. Public wrong-word verdicts are revised without
changing the internal ownership ledger, preserving primary tracker events and
insertion merges. The prototype's audio-side review occurs at session end;
its live warning latency would need further validation before deployment.

## Validation and execution blocker

Local Python syntax compilation and `git diff --check` passed for the
prototype. Unit tests were authored for exhaustive tiny CTC path sums,
forward/backward derivatives, equal parameter counts, strict finite
reversals, exact certificate binding, and graph preservation. These tests
were **not executed successfully**. Local core and SDK test commands cannot
start Vitest because this worktree has no `node_modules`. The local NumPy
installation also cannot import because its binary architecture differs from
the local Python architecture. Heavy computation was not attempted locally.

Only the authorized supervisor runner was used. Its first three requests
failed with `NotFoundError: No such file or directory`, without any returned
container log or output directory:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Probe authorized data, model directories and Python dependencies | Failed before any returned command output |
| 2 | Required six-shard `bash tools/autolab_tests.sh` | Failed before any returned command output |
| 3 | Retry authorized data and dependency discovery | Failed before any returned command output |
| 4 | `python3 modal/test_student_transition.py` | No output after more than seven minutes; pending at cleanup |
| 5 | Minimal `python3 -c "print(1729)"` | No output after more than seven minutes; pending at cleanup |

The failure's underlying cause is unknown. It cannot be attributed to the
training code, missing help-train recordings, or an acoustic mismatch without
a container log. The remaining requests distinguish a basic runner failure
from failures of data access or the prototype's dependencies.

## Limits and disposition

No ONNX parity check, feature extraction, training, speaker-grouped model
selection, SDK bundle-load check, or actual engine-window comparison completed.
Target wins, losses, clean-session changes and studio changes are unmeasured.
**Practice scores used: 0 of 3.** No overall delta is claimed.

The unverified engine, training and test changes were removed after repeated
runner failures and the two unanswered retries. The prototype is retained
only in ignored worktree scratch at `.autolab/dev/transition-prototype/`;
it is not a validated implementation. No `.autolab/model.json` or declared
side bundle has been submitted: scoring the main listener without the trained
side assets would not evaluate this mechanism. No killed outcome is recorded,
because the acoustic premise and target-win threshold were not tested.

To resume, the supervisor must first execute a basic command and the required
tests. Then inspect eligible training-speaker coverage, export and verify the
frozen extractor, fit both matched arms using training-only selection, verify
the complete bundle, and inspect the three actual engine-requested
comparisons. Apply the target-win rule only after those measurements exist.
