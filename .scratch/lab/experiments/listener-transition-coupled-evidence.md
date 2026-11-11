# Transition-coupled acoustic evidence

Status: blocked by the supervisor's execution service; acoustic premise untested.
The second attempt restored and improved the engine implementation, but cannot
submit it with the required trained and verified assets: even a one-line
command fails through the only authorized compute route. No trained candidate
is submitted. The experiment card is unchanged.

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
insertion merges. The second attempt replaced the end-only audio review with
live frozen-feature capture on the identical primary streaming feeds and
synchronous scoring before public confirmation. It also retains exact-window
certificates for evidence replay after existing gap/insertion enrichment.
Synthetic streaming tests passed, but real-model parity and live latency remain
unverified.

## Validation and execution blocker

Local Python syntax compilation and `git diff --check` passed for the
prototype. Three tiny Python tests passed using `arch -x86_64 python3`:
exhaustive CTC path sums with complete acceptance, analytic forward/backward
derivatives against finite differences, and empty-window semantics. The
fourth test, instantiating the matched heads to count their parameters, could
not run because that interpreter has no PyTorch. Native ARM Python cannot
import the installed x86 NumPy extension. No training or heavy computation
was attempted locally.

Node's built-in TypeScript transpiler and a small assert-backed harness ran
four SDK tests and five checker tests successfully. These check exhaustive
path sums, strict finite reversal, identical synthetic streaming feeds and
primary evidence across push sizes, exact certificate binding, baseline
issue creation, public-verdict-only changes, and graph preservation. The
harness used the existing reference asset and authored synthetic fixtures.
These isolated checks do not establish type-checking, real ONNX parity,
actual target behaviour, or a pass of the required suites. Local core and SDK
test commands cannot start Vitest because this worktree has no `node_modules`.

Only the authorized supervisor runner was used. All eight requests have now
finished with `NotFoundError: No such file or directory`, without any returned
container log or output directory. The renewed minimal-command failure shows
that the blocker also occurs without prototype code or data dependencies:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Probe authorized data, model directories and Python dependencies | Failed before any returned command output |
| 2 | Required six-shard `bash tools/autolab_tests.sh` | Failed before any returned command output |
| 3 | Retry authorized data and dependency discovery | Failed before any returned command output |
| 4 | `python3 modal/test_student_transition.py` | Failed before any returned command output |
| 5 | Minimal `python3 -c "print(1729)"` | Failed before any returned command output |
| 6 | Renewed minimal `python3 -c "print(1729)"` | Failed before any returned command output |
| 7 | Python unit tests, then authorized data/model/schema discovery | Failed before any returned command output |
| 8 | Renewed required six-shard `bash tools/autolab_tests.sh` | Failed before any returned command output |

The failure's underlying cause is unknown. It cannot be attributed to the
training code, missing help-train recordings, or an acoustic mismatch without
a container log. Results are retained locally in the ignored
`.autolab/modal-result-1.json` through `.autolab/modal-result-8.json`; each
reports the same error and zero dollars of compute. No request is pending.

## Limits and disposition

No real ONNX parity check, feature extraction, training, speaker-grouped model
selection, SDK bundle-load check, or actual engine-window comparison completed.
Target wins, losses, clean-session changes and studio changes are unmeasured.
**Practice scores used: 0 of 3.** No overall delta is claimed.

The second attempt restored the prior prototype, corrected empty-window
acceptance and added the live streaming hook and its parity test. After all
renewed supervisor requests failed, the unverified engine, training and test
changes were removed from the submission. The latest prototype is retained
in ignored worktree scratch at `.autolab/dev/transition-prototype/`, with an
updated `tracked.patch`, source/test files and `NOTES.md` for resuming. It is
not a validated implementation.

The final tracked change is this explanation of why the implementation cannot
be completed in the available environment. The user's submission requirement
is to finish and verify the complete side-asset bundle; the unavailable runner
prevents that prerequisite. No `.autolab/model.json` or declared side bundle
has been submitted: scoring the main listener without the trained side assets
would not evaluate this mechanism. No killed outcome is recorded, because the
acoustic premise and target-win threshold were not tested.

To resume, the supervisor must first execute a basic command and the required
tests. Then inspect eligible training-speaker coverage, export and verify the
frozen extractor, fit both matched arms using training-only selection, verify
the complete bundle, and inspect the three actual engine-requested
comparisons. Apply the target-win rule only after those measurements exist.
