# Listener exemplar density ratio

Status: implementation prepared; training, TypeScript tests and acoustic premise validation blocked by the supervisor's Modal runner. No practice score has run. The experiment card is unchanged.

## Prediction

The card predicts two false warnings removed and one caught substitution lost:
listener −1, overall −1. Its four registered targets are p11 at 26:106:1,
p36 at 55:48:1, p55 at 26:174:8, and sob-049 at 92:13:1.

The task's continuation requirement takes precedence over the card's lower
kill threshold: continue to practice scoring only if the built change wins at
least three targets. No target has been inspected or tested in this worktree
because the authorized compute requests failed before command execution.
Zero tested targets is not a measured zero-gain result or a premise kill.

## Mechanism

`modal/student_exemplar_export.py` identifies the shipped ONNX graph's final
pre-CTC projection input, before dynamic activation quantization, and exposes
it as an output in a separate extractor graph. It retains all model operations
and weights and requires eight consecutive streaming windows to reproduce
all primary outputs and cache tensors exactly. It writes a separate graph;
it does not edit the supplied model or manifest.

The SDK training CLI extracts this representation in the primary window,
lookahead, padding and cache regime on help-train none-labelled training
recordings only. It excludes every speaker appearing in a non-training label,
and accepts a further explicit speaker exclusion list. Release-held-out
speakers are absent from the supervisor's authorized help-train input.

The CPU estimator obtains fractional phone labels with full CTC
forward-backward conditioned on the complete canonical transcript, including
blank placement. It never labels a frame with the model's winning class.
Each recording contributes at most 128 regularly spaced datastore frames
and has total weight one. Whitening uses equal-recording moments and a
16-dimensional PCA projection. Speaker-grouped three-fold validation chooses
covariance shrinkage, neighbourhood size and identity smoothing from a fixed
eight-configuration grid, using complete canonical sequence likelihood and
equal-speaker aggregation. A final fit uses all eligible training speakers;
no practice observation participates in any fit or selection.

For each phone class, retrieval selects one fractionally supported neighbour
per recording, ranked by whitened squared distance minus twice the log soft
label. It retains at most the selected number of recording neighbours.
A common kernel bandwidth comes from the union of those class-balanced
neighbourhoods. Weighted soft-label kernel mass is divided by that class's
weighted datastore frequency. Common identity smoothing handles sparse
neighbourhoods; the primary stream still supplies blank-versus-speech mass.
Thus phone popularity and a recording's number of frames cannot directly
supply repeated unlimited evidence for one label.

The SDK compares the complete written and already-accused pronunciations on
the exact frames requested by the existing posterior confirmation path.
Both use ordinary CTC boundary marginalization and identical timing and blank
treatment. The control uses the original stored primary posterior rows,
including their normal top-k expansion, rather than replacing them with an
extractor decode. A certificate requires a finite strict reversal: the
parametric head prefers the rival and the exemplar likelihood prefers the
written pronunciation. Candidate order and reference identity do not affect
retrieval. Unsupported complete pronunciations abstain.

Certificates carry the word occurrence, both complete pronunciations and
exact window coordinates through ordinary token evidence serialization.
The checker changes only the public final verdict and `mistakes()` result;
its internal confirmation and ownership ledger retain baseline state so
insertion decisions and later candidates remain unchanged. Parallel-graph
confirmations are exempt. The source includes an SDK end-of-audio replay to
publish matching vetoes without replacing primary decoding or tracker events.

An optional `exemplar-assets.json` descriptor declares the side graph and
fitted datastore with hashes. The Node loader rejects incomplete or corrupt
declared bundles. The exporter can build a bundle by copying the original
primary files and fitting the datastore, without changing its manifest.
No `.autolab/model.json` has been declared: there is no trained bundle yet,
and scoring the main model would not evaluate this experiment.

## Validation and limits

Three small Python unit tests passed using the local Python's x86_64 mode:
exhaustive CTC path sums and boundary occupancies, rejection of non-training
or mistake-labelled inputs, and soft-label class-frequency correction.
Python compilation and `git diff --check` passed.

Added TypeScript unit tests cover exhaustive complete-sequence CTC scoring,
candidate-order symmetry, unchanged blank evidence, class-frequency
correction, recording weight validation, unsupported phones, exact-window
certificate transport, unchanged internal issue creation, and parallel-graph
preservation. These tests have not executed successfully because Modal is
unavailable and this worktree has no node_modules.

Supervisor requests 1 and 2 both returned
`NotFoundError: No such file or directory` before executing even the
read-only environment audit. Request 3 submitted the required six-shard
`bash tools/autolab_tests.sh` harness and returned the same pre-execution
`NotFoundError`. Request 4 submitted the Python tests,
frozen extractor export and training-only bundle fit, and also returned the
same pre-execution `NotFoundError`. All four requests failed without a
container log, a fitted datastore or exported side assets. Results are recorded
under the ignored `.autolab` request/result files.

Outstanding required work: restore the runner; pass the TypeScript build and
all core/SDK tests; export and validate the complete bundle; freeze the
training fit; inspect the four actual engine-requested target comparisons;
then apply the continuation requirement. If fewer than three targets are won,
revert engine changes and record a premise kill. If at least three are won,
run an authorized practice score before deciding anything about losses.
All three allowed practice scores remain unused.

This implementation is unvalidated engine code, not a scored improvement.
Whitening can discard useful acoustic directions, finite neighbour coverage
can prefer a wrong label, and CTC-derived soft labels can inherit model bias.
A real substitution may therefore receive a veto. Runtime currently buffers
session audio and runs a separate full-session extractor only when there are
eligible accusations, publishing audio-backed vetoes at session end; it does
not yet meet an immediate streaming-warning latency requirement. The exported
side graph duplicates primary model inference/storage. Its actual size,
runtime, training coverage and population precision remain unmeasured.

No scorer, lab tooling, supplied manifest, test data, evidence or card was
edited. No fold was performed.
