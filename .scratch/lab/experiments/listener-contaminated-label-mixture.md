# Recording-level transcript contamination mixture

Status: implementation prepared; training and acoustic validation blocked by the supervisor execution service. No trained head or deployable candidate bundle exists. The experiment card is unchanged.

## Prediction and mechanism

The prediction is two false warnings removed and one collateral wrong call,
listener −1 and overall −1 on lab-v1.5u. The proposed cause is unscripted
deviations in recordings admitted with supposedly correct transcripts. Such
recordings can teach incorrect phone associations. This cause remains unverified.

`modal/student_contaminated_head.py` refits only the shipped linear phone head.
`modal/student_head_refit.py` captures the encoder features actually consumed by
that head, including rounded/dequantized activations in the shipped integer
graph. Both arms start at the same effective shipped weights. All encoder nodes,
encoder constants, cache states, vocabulary, frontend and frame timing remain
unchanged. No tracker, checker or confirmation rule is changed.

For a supplied transcript y and complete recording x, the mixture minimizes

    −log((1 − π) P_CTC(y | x) + π Q(y)).

Q is an add-one-smoothed phone bigram distribution with explicit termination,
estimated once from fit transcripts. Every transition row sums to one and has
positive termination mass, so Q is normalized over finite transcripts, including
their lengths. It is independent of the recording and never supplies an
alternative transcript. Supplied reference identities are retained throughout
each original recording, including its edges; only held-phone runs are collapsed,
as in the existing CTC target representation. Teacher guesses do not replace
reference edges or internal phones.

The full CTC and background sequence probabilities are mixed before the loss is
divided by recording frame count. Consequently the CTC gradient is multiplied
by the posterior probability that the transcript came from the acoustic source.
An acoustically incompatible pair can receive little corrective supervision.
The same mechanism can downweight difficult, correctly labelled speech; the
predicted collateral wrong call accounts for that substantial risk.

π is estimated with MAP EM and a Beta(2,2) prior, using ordinary, unmasked
recordings from separate calibration speakers before each epoch. Those speakers
receive no training gradients. Separate validation speakers select the mixture
checkpoint with lowest ordinary CTC plus shipped-output preservation loss; the
ordinary-CTC arm is retained at the exact same epoch and update count. Practice
incidents, warnings, scores and speakers enter neither estimation nor selection.

## Data and frozen recipe

The loader requires the authorized original clean-match parquet recordings,
their reference transcripts and speaker-cluster metadata under `/student/s1`.
An original cluster is excluded in full if any member is outside the training
split or in the existing release exclusion list. Unknown speaker clusters are
not admitted. It also admits help-train training speakers' none-labelled takes
with no mistake, repeat or unsupported region. Help speakers with any excluded
or non-training take are removed in full.

Mistake and repeat recordings contribute only transcript-free preservation of
the shipped model's outputs. Their planted scripts are not treated as performed
phone transcripts. Whole recordings retain ordinary speech and pauses. Exact
waveform duplication across speaker identities is rejected before caching.
Fit, calibration and validation partitions keep speakers intact and are drawn
separately within each supervised source. Feature caches, initial head, encoder
graph, background distribution, selection and exports have checksum provenance.

The bounded recipe was fixed before acoustic or practice measurements: seed
831, up to 512 original recordings of at most 30 seconds, three views per
recording (one ordinary and two with the existing shipped augmentation), 12
epochs, eight supervised recordings and four preservation recordings per
update, SGD learning rate 0.0001, and shipped-output KL weight 2. Both arms use
identical recordings, view ordering, batches, update counts and preservation
examples. Only the recording-level transcript loss differs.

Export replaces only phone-head tensors, retains the existing weight
quantization geometry, and checks output geometry and exact cache-state parity.
It requires all inherited SDK insertion-teacher assets to exist with their
declared sizes and checksums, then copies them into both bundles. This prevents
an apparently valid primary model from silently dropping the previously shipped
auxiliary listener. The supplied model registry and manifests are not edited.

## Validation and limits

Source compilation passes. Added unit fixtures cover finite-transcript
normalization, mixture likelihood and gradient responsibility, exact ordinary
CTC at zero contamination, calibration behavior, whole-speaker partitions,
cluster-wide release exclusions, refusing scripted/repeat canonical supervision,
matched tiny refit updates and validation selection, frozen feature capture,
float and integer heads, per-channel quantization, final-padding trimming,
unchanged cache states, and complete auxiliary bundle checks. All 14 fixtures
passed locally in 53.5 seconds using the workspace's isolated environment
(NumPy 2.5.4, torch 2.15.0, ONNX 1.23.2 and ONNX Runtime 1.31.1). Earlier
dependency-import failures were resolved before this successful run. These
small synthetic inputs do not establish performance on any Quran recording.

Supervisor run 1 (read-only inventory), run 2 (the supplied six-shard core/SDK
test harness), and run 3 (the training image with Python fixtures, inventory and
the bounded prepare/fit/export pipeline) each returned
`NotFoundError: No such file or directory`. Their result metadata records zero
compute expenditure and they produced no output directory. Run 4 retried the
supplied core/SDK harness with the explicit node image and returned the same
error. Core/SDK tests could not run. These are execution-service failures, not
measured acoustic failures.

No recording cache, fitted head, contamination estimate, selected checkpoint,
target preference comparison or target-warning recovery has been measured. No
practice score was run; all three permitted practice scores remain available.
There is no `.autolab/model.json`, since declaring a nonexistent or incomplete
bundle would not evaluate this experiment. No killed outcome is asserted:
inaccessible compute does not falsify the contamination premise.

Completion requires the authorized supervisor to run the current source,
verify both complete exports through the ordinary SDK loader, and replay the
registered target sessions. The complete written and already-accused
pronunciations must be compared on identical windows obtained from the actual
baseline checker requests, without selecting more favorable boundaries. At
least two preferences must reverse exclusively under the mixture relative to
the matched ordinary-CTC control. Under the user's implementation stopping
rule, fewer than three recovered target warnings requires a premise kill;
three or more requires a practice score before judging collateral losses.
The card's prediction of two removals does not establish either criterion.

The training CLI has sequential `prepare`, `fit` and `export` commands, each
taking `--out /scratch/listener-contaminated-label-mixture/refit-v1`.
`prepare` and `export` also require `--model` for the verified complete shipped
bundle. In the supervisor container, set
`PYTHONPATH=/app/tools:/app/modal`. Check the authorized original parquet and
augmentation mounts and the inherited teacher assets before running this
pipeline. Run 3 returned no inventory, so their availability is unknown.
Its snapshot also predates the correction that removed teacher-spliced
reference edges; continuation must use the current committed source.

Until those checks run, there is no evidence for target gain, clean-session
safety, studio preservation or aggregate improvement. The committed training
implementation is retained for continuation, rather than representing an
unchanged baseline decode as a result for an untrained head. No fold was made.
