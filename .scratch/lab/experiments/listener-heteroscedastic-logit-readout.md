# Heteroscedastic logit readout

Status: ready-for-human

Implementation draft; acoustic training and target evaluation are blocked by the
authorized supervisor failing before command execution. The experiment card is
unchanged. This is not a measured kill or a model bundle ready for scoring.

## Prediction and mechanism

The card predicts two false warnings removed and one caught substitution lost:
listener -1, overall -1. Its proposed cause is a point-valued phone readout
overstating uncertain sustained consonant distinctions. That acoustic premise
has not been established by this experiment.

`modal/student_logit_uncertainty_export.py` adds the frozen pre-CTC encoder output
to a separate ONNX graph, leaving the supplied primary graph and manifest
unchanged. A seeded 32-dimensional projection feeds a 48-unit tanh readout. The
head predicts phone mean logits and one or two input-dependent loading vectors;
the loading vectors define a low-rank logistic-normal distribution, including
blank. Encoder weights never train.

Latents are independent between blocks of four or eight native 40 ms frames and
shared within each block. Block boundaries use the absolute session frame grid,
including when an accusation begins inside a block. A fixed five-point
Gauss-Hermite rule per latent dimension integrates the whole block's CTC forward
recursion before passing its boundary state distribution to the next block.
This sums every block-latent combination without enumerating their exponential
product. Averaging frame posteriors first would implement a different,
independent-frame uncertainty model. Both written and rival hypotheses use the
same observations, latent prior, block boundaries and standard complete CTC
topology; final-phone or trailing-blank completion is mandatory.

Only help-train training speakers' none-labelled takes supply transcripts. Takes
with scripted errors or repeats cannot become clean anchors. Any speaker named
by a non-training label, or by the explicit exclusion list, is excluded. No
practice recording supplies training features, supervision or model selection.
Additional authorized recording coverage was not available to inspect because
the supervisor did not execute the dataset census.

Both treatment and deterministic control have identical hidden layers, output
dimensions, initialization, data and update counts. All control output channels
participate through deterministic nonlinear residual logits. Four shared
initialization epochs match the frozen parent's frame distributions, followed
by six epochs optimizing the complete canonical transcript's marginalized CTC
likelihood. Both arms regularize their auxiliary channels; covariance fitting
also uses the same weight decay as its control. No hard phone boundaries or
assumed performed substitution transcripts supervise either arm.

Three folds group speakers, spreading none-labelled speakers across folds first.
The training-only grid is rank {1,2}, block extent {4,8}, regularization
{0.001,0.01}. Selection first minimizes the fraction of held-out training-speaker
substitution controls reversed toward the written pronunciation, then compares
speaker-averaged transcript NLL against the equally parameterized deterministic
control. Weak substitution labels only select existing engine-requested
preservation windows. Five-versus-seven-point quadrature error is recorded on
training validation transcripts. The recipe and grid are fixed in code before
any practice target inspection; no selected model or convergence result exists
yet.

The SDK discovers already-supported string-path wrong-word comparisons at the
checker's existing confirmation windows, including contextual and word-local
paths. It compares complete written and already-accused pronunciations. A
certificate requires finite strict disagreement: this readout's mean-only
likelihood prefers the rival, while its integrated likelihood prefers the
written pronunciation. The separately trained deterministic control does not
supply the runtime mean comparison. Ties, unknown phones, incomplete CTC paths
and mismatched frame geometry abstain.

Certificates name the exact occurrence, both pronunciations and frame bounds.
They change only the public wrong-word verdict; the checker's internal
confirmation and ownership ledger stays intact. Primary phones, confidences,
posterior rows and tracker events remain the primary listener's. Graph
confirmations and insertion decisions bypass this veto. The Node loader checks
both side files' lengths and SHA-256 hashes, the primary graph hash, inventory,
head shapes and latent geometry; a missing declared file throws.

## Validation and limits

Local lightweight checks passed for the actual SDK mathematical implementation:
exhaustive symbol-path and block-latent enumeration with a partially intersected
block, zero covariance matching ordinary CTC, Gaussian quadrature mass and unit
variance, and a constructed preference reversal. In that synthetic four-frame
example the mean written likelihood is 0.201270 versus rival 0.230369; integrated
written is 0.331735 versus rival 0.258415. This proves the operation can reverse a
comparison, not that any real incident has this structure.

Python AST parsing passed for the trainer, exporter and six mathematical tests.
Node's TypeScript stripping accepted eleven implementation/test files. This is
syntax checking, not a TypeScript build or a passing test suite.

Added tests cover exhaustive quadrature, repeated-phone blank requirements,
finite transcript-likelihood gradients, input-conditioned covariance, equal
control capacity, exclusion provenance and speaker folds; exact certificate
transport; graph and actual insertion preservation; native feature windows,
cache progression and final-frame truncation; session completion publication;
and rejection of incomplete or corrupt declared bundles. An existing loader
test's filesystem mock now represents the absent optional descriptor correctly.
These suites have not run successfully on the authorized supervisor.

Supervisor requests 1, 2, 3 and 4 returned `NotFoundError: No such file or directory`
before executing commands, each reporting zero compute. They attempted the
dataset census, the required six-shard test harness, and the complete
build/Python-tests/extract/train/export command. Request 4 attempted the required
six-shard harness after implementation and failed before execution as well.
No practice scores have been used. Target reversals and warning removals are
unmeasured, rather than zero. Dataset coverage, trained parameters, selected
configuration, full builds/tests, encoder export parity and normal SDK bundle
loading remain unverified.

The exporter checks eight consecutive cache-carrying windows: bit equality of
the original and extended graphs with optimization disabled, and numerical
agreement plus identical greedy identities against the optimized primary. The
bundle command copies the supplied primary assets unchanged, fits the heads,
writes a hashed side descriptor, and verifies normal SDK loading. None of those
checks has executed here. No `.autolab/model.json` declares absent assets and no
killed outcome claims an unmeasured target result.

This draft buffers session audio and performs its auxiliary feature pass at
`end()`. Memory and latency grow with session duration, and a live warning can
appear before its final downgrade. Immediate streaming silence on these false
warnings, mobile/browser asset loading, device performance and population
safety are not established. Small-data transcript training can teach incorrect
pronunciation preferences and suppress genuine weak substitutions. Five-point
quadrature convergence on actual fitted covariance remains an explicit
measurement to inspect before target evaluation.

## Remaining authorized work

Restore execution of the supplied supervisor and run the six-shard harness plus:

```sh
python3 -m unittest discover -s modal -p test_student_logit_uncertainty.py
python3 modal/student_logit_uncertainty_export.py \
  --model /lab/data/models/spliced-clean-int8 \
  --labels /lab/data/help-train/labels.jsonl \
  --out /scratch/listener-heteroscedastic-logit-readout/bundle
```

After verifying the completed bundle through the ordinary loader, use the SDK
diagnostic CLI on each target's existing audio/evidence to measure exact-window
mean/integrated reversals and actual warning removals, checking unchanged tracker
events and insertion decisions. Do not tune the grid or constants from those
comparisons. The user's later stopping rule requires at least three actual
target wins to continue, superseding the card's two-reversal threshold. Fewer
than three calls for rollback and a measured killed outcome; three or more
requires an official practice score before deciding. Infrastructure failure
establishes neither result.

No scorer, lab tooling, supplied data, evidence, manifest or experiment-card
edits were made. No fold was performed.
