# Dynamic compression acoustic recheck

Status: blocked on authorized remote execution; the acoustic premise has not been measured. Engine code is unchanged.

## Prediction and mechanism

The card predicts one phone wrong-word false warning removed, listener −1 and
overall −1, with no predicted losses. It proposes history-dependent amplitude
compression as a cause of distorted consonant attacks despite intact samples.

The intended experiment fits an attack/release estimator using known applied
compression on authorized training recordings, with paired identity and
memoryless controls. Training speakers must remain separate across fitting,
model selection and calibration, and release-held-out speakers must be excluded.
Only preceding audio outside an accused word may determine compression
parameters and initial state. A regularized inverse then rechecks the existing
window with the unchanged acoustic model. Certification requires recovery of
the complete written pronunciation under existing spelling rules, loss of rival
support under the existing judgement, and retention of the accusation by both
controls. Primary decoding, tracker events, graph confirmations and insertion
decisions must be preserved.

The card's early premise threshold is two targets exhibiting an explained
history-dependent gain trajectory. The task additionally requires at least
three target recoveries before retaining an implementation, and a practice
score if three or more recover. Neither threshold has been evaluated.

## Work and checks

Read `AGENTS.md`, the unchanged card, the audio-review workflow and the current
SDK/checker evidence paths. Read-only inspection of `tools/lab_sessions.py`
(`degrade`) and `modal/lab_build_sessions.py` shows that the generated phone
channel uses room convolution, band limiting, noise and one global peak scale.
It introduces no attack/release compressor. That does **not** rule out
compression already present in the source recordings, especially the real-user
target, and is not grounds for a premise kill.

A draft bounded utility initially beside the student scripts implements a one-ms RMS
level detector with separate gain attack and release constants, its capped
causal inverse, paired injected controls, and level-invariant crest/envelope
history features. It selects between two regularized tree estimators exclusively
within hashed training-speaker partitions, reserving separate speakers for
control calibration. No practice identifiers or reference text are inputs.

On the implementation retry, the draft was restored temporarily beside the
student scripts. Its iterative inverse was replaced with an exact solution
for each one-ms block, avoiding undercorrection of the high-ratio memoryless
control. Amplification remains capped at 18 dB; invalid parameters, nonfinite
audio and an out-of-bounds initial state are rejected.

Seven small synthetic unit checks passed locally in an isolated environment
under this worktree: preceding energy changes release gain; forward and inverse
state continue causally across windows; the known dynamic inverse recovers an
attack/release signal and beats identity and memoryless inverses; feature values
are stable under global amplitude changes; a high-ratio memoryless inverse is
exact; and the inverse enforces its gain bound and input checks. These establish
the draft's elementary signal mechanics only. They do not establish training
specificity, blind parameter accuracy, written-word recovery or mistake
preservation. The local command was
`.autolab/dynamic-compression-venv/bin/python -m unittest discover -s modal/tests -p test_student_dynamic_compression.py -v`;
all seven passed in 3.516 seconds.

## Execution limitation

Authorized supervisor request 1 attempted a read-only inventory of `LAB_DATA`.
Request 2 retried with only `pwd`. Both returned
`NotFoundError: No such file or directory`, with zero reported compute cost,
no shard output and no usable execution result. The missing resource cannot be
identified from these responses. Request 3 submitted the synthetic checks and
training utility through the same authorized supervisor route and returned the
same error, again without shard output or training results. Remote execution
therefore remains unavailable; no alternative Modal route was attempted.

The retry submitted request 4 for a fresh `LAB_DATA`/help-train inventory. It
returned the identical `NotFoundError: No such file or directory`, again with
zero reported compute cost and no shard output. Request 5 submitted the seven
synthetic unit checks through the supervisor, and request 6 submitted the
required six-shard `bash tools/autolab_tests.sh` build/core/SDK checks. Their
results both returned the same `NotFoundError`, zero reported compute cost and
no shard output. Neither remote test request ran usable checks. Requests 4–6
are recorded in ignored `.autolab/modal-request-{4,5,6}.json` and their matching
`modal-result-{4,5,6}.json` files. All retry requests used the prescribed
`python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run` route; no
practice score was used.

This blocks a complete implementation for concrete reasons: the training
utility cannot read the authorized recordings or verify their speaker/split
schema, fit the estimator, freeze training-speaker selection and calibration,
or export and verify its runtime assets. The target waveforms and traces cannot
be read to identify the actual preceding and disputed intervals or compare the
dynamic model with its controls. Installing a veto without those steps would
substitute an untrained hypothesis for the card's required acoustic evidence.
An optional hook that never loads a fitted estimator would instead leave the
scored listener unchanged and would not implement this experiment.

The untrained utility and its checks were removed from engine sources and saved
under ignored `.autolab/dynamic-compression-draft/` for resumption. The final
committed change is this record only. No scorer, lab tooling, test data,
manifest, supplied evidence or experiment-card edit was made. There is no
`.autolab/model.json` and no misleading `killed` outcome file.

No training recordings or target waveforms have been inspected. No estimator
has been fitted or frozen, no SDK veto has been installed, no asset bundle has
been declared, and no practice score has been requested (zero of three used).
Target recoveries and overall/listener changes are **unmeasured**, not zero.
There is no scientific killed verdict: failed remote execution does not show
that the targets lack compression or that a built recheck wins too few targets.

The next necessary work is to restore authorized supervisor execution, run
training-only selection/calibration, and audit all four targets using preceding
audio outside their actual disputed spans. Proceed to SDK integration and
complete bundle verification only if the premise survives; follow the task's
target-recovery and practice-score rules after integration. Neither a fold
decision nor a substitute local practice evaluation is justified by this record.
