# Preactivation residual score

Status: needs-human. The implementation and synthetic checks are present, but
the supplied supervisor has not executed the data inspection or tests. Training,
bundle verification and target comparisons remain blocked. This is not an
acoustic kill result. The experiment card is unchanged.

## Prediction and mechanism

Predict two false warnings removed and one caught substitution lost: listener
−1, overall −1. The hypothesis is that non-invertible encoder activations discard
weak distinctions that the ordinary post-activation readout cannot recover.
Intervention requires opposing complete-pronunciation preferences: the residual
arm must prefer the written pronunciation while the matched post-only control
prefers the already accused pronunciation.

`modal/student_preactivation_export.py` appends diagnostic outputs to a separate
copy of the frozen primary ONNX graph. It recognizes encoder ReLU, Clip,
HardSwish, GELU, SiLU and expanded Zipformer Swoosh formulas, excluding injective
primitives and the CTC head. Only matching float time-major `[T,1,C]` activation
pairs are eligible. Up to six architecturally spaced sites and seeded projection
matrices are chosen before training. Eight successive streaming windows must
preserve every primary output and recurrent state exactly with graph
optimizations disabled. The supplied weights, computation and manifest are not
edited. Compatibility with the actual supplied graph is still unverified.

The preparation utility accepts only usable help-train `none` recordings on the
`train` split, without labelled mistakes or repeats. It requires an explicit
speaker exclusion file and additionally excludes every speaker with any
nontraining label. The trainer independently rejects nontraining or excluded
records. No practice examples participate in fitting or selection.

For each site, a regularized predictor estimates eight projected preactivation
coordinates from all corresponding post-activation channels, using normalized
linear and fixed random nonlinear features. Regression weights recordings
equally. Its prediction residual is concatenated with eight ordinary post
coordinates. The equal-capacity control uses sixteen ordinary post coordinates,
with the first eight shared. Residuals are calculated on actual paired
observations before native-clock averaging; pooling first could manufacture
apparent information loss. Faster activation grids are averaged after residual
calculation, and slower stages repeat actual observations onto the native grid.
Only integer clock ratios are supported.

Both sixteen-dimensional scorers fit full-Gaussian phone emissions with the
same shrinkage and four EM updates. Parent posteriors initialize transcript
occupancies. Subsequent updates marginalize all CTC phone boundaries under each
arm's likelihood. Timing transitions have normalized outgoing probabilities,
and scores condition on completing the pronunciation, avoiding a path-count
reward. Three speaker-grouped training folds select the activation site and
predictor regularization from a fixed grid. Conditional phone-identity loss,
averaged by speaker, selects the residual arm; the matched control's validation
loss and held-out reconstruction error are also recorded. All choices freeze
before target inspection.

The residual is information unexplained by a finite nonlinear regressor. It is
an estimate of conditional residual information, not a proof that an arbitrary
predictor cannot recover it. A reversal could reflect predictor approximation
or differing projected features rather than information destroyed by the
activation. Training validation and the matched control constrain that claim
but cannot establish the causal hypothesis on their own.

The SDK asks the unchanged checker for already supported non-graph string-path
wrong-word comparisons. It scores the complete written and accused strings,
including their existing neighbours when the primary confirmation uses them,
on identical frame windows and with identical boundary marginalization. Only a
strict finite reversal supplies a certificate. Core matches the word identity,
both pronunciations and exact frame bounds before reporting that confirmation
as uncertain. Internal issue confirmation and ownership remain unchanged so
subsequent candidate generation and insertion merges retain baseline state.
Primary tokens, posterior rows and tracker input remain intact; graph
confirmations do not request this comparison.

The Node loader verifies both declared side-file hashes and the primary hash.
A missing or corrupt declared bundle fails loading. Audio-backed sessions retain
audio for an end-of-session auxiliary review, publish explicit downgrades of
the corresponding live occurrences, and retain certificates in recorded
evidence for replay. This buffered route does not demonstrate timely streaming
warnings, mobile deployment, memory bounds or device latency.

## Validation and limits

Both TypeScript builds pass. The five new core tests and thirteen new SDK tests
pass locally. They cover unchanged candidate discovery, exact frame/string
matching, finite strict preferences, graph exemption, path-count normalization,
repeated articulations, candidate exchange symmetry, missing-phone abstention,
training-label eligibility, paired activation pooling, and corrupt/missing
bundle rejection. The session test uses the real tracker/checker with mocked
auxiliary scores: it verifies an explicit downgrade, certificate persistence,
identical primary position events and unchanged insertion verdicts. It does not
validate trained acoustic discrimination. Across the full SDK run and the
focused rerun after correcting the added clock-ratio test's syntax, all 38
available SDK tests pass; one optional help-data test is skipped. The full run
also exposed a blanket filesystem mock in the existing insertion-model tests
that claimed every read returned a manifest. It now reports absent optional
files as ENOENT and asserts no residual scorer was attached. All three existing
insertion-loader tests pass after that correction.

The final rerun of all six Python numerical/structural tests passed locally
after matched-control validation reporting was added. They check activation formula
identification, exclusion of injective/head sites, complete boundary sums
against exhaustive short paths, posterior occupancy normalization, recoverable
pre-information subtraction, ambiguous ReLU branch retention, and regularized
sparse-phone covariance.

The full local core run completed with 753 tests passed, eight timeout failures
and two Vitest `onTaskUpdate` RPC timeout errors. Seven failures exceeded the
default five-second deadline; the other exceeded an explicit fifteen-second
startup deadline. No assertion failure was reported. All seven default-timeout
cases passed isolated reruns with a sixty-second default deadline, as did the
other onset and excursion tests. The startup file passed two tests and still
timed out on its second acquisition prefix. Replacing only the checker with
its original HEAD version for an isolated comparison produced the same two
passes and one timeout: 17.879 seconds on the original checker versus 17.964
seconds on the experiment checker, against the fifteen-second limit. The
experiment checker was restored after this control. The full core suite is not
certified passing in this environment, and the prescribed remote harness did
not execute.

All four supervisor requests returned `NotFoundError: No such file or directory`
before command execution, with zero compute cost and no shard output. They
requested a read-only model/data audit, the prescribed six-shard test
harness, a minimal `pwd` retry, and the test harness plus Python tests. No
supplied encoder graph, help-train label, practice
audio or target window has been inspected remotely. Target comparisons: zero.
Practice scores: zero of three. Neither target wins nor overall score changes
have been measured, and no fold or discard was performed.

There is no trained bundle and no `.autolab/model.json` declaration. Evaluating
the main model would not evaluate this change. Restore the supplied supervisor,
inspect the actual graph and eligible labels, export and verify the diagnostic
graph, cache eligible pairs, fit/freeze the two arms, then verify the complete
bundle and SDK/Python scoring parity before inspecting target windows. Apply
the task's explicit continuation requirement: fewer than three recovered
targets stops the experiment despite the card's two-reversal criterion. At
least three recoveries require practice-score before a gate decision.

## Reproduction entry points

Export: `python3 modal/student_preactivation_export.py --model <primary-dir>
--out <bundle-dir>`. Preparation: `pnpm --filter @tilawa/sdk exec tsx
src/bench/preactivation-residual-cli.ts --model <primary-dir> --extractor
<bundle-dir> --labels <help-train-labels> --excluded-speakers <exclusions.json>
--out <cached-pairs.jsonl>`. Training: `python3
modal/student_preactivation_residual.py --input <cached-pairs.jsonl> --metadata
<bundle-dir>/preactivation-residual-metadata.json --excluded-speakers
<exclusions.json> --out <bundle-dir>`.

The completed bundle must retain the unmodified primary model, manifest and
all its existing auxiliary assets, and add `preactivation-residual.onnx`,
`preactivation-residual-emissions.json` and
`preactivation-residual-asset.json`. The generic comparison CLI takes `--model
<bundle-dir> --audio <wav> --evidence <evidence.json> --out <comparison.json>`;
engine decisions choose the windows and pronunciations. There is no inference
incident list. Preparation currently expects normalized `speaker`, `split`,
`kind`, `wav` and `ayahs` fields and 16 kHz WAV files; compatibility with the
supplied training data has not been established.

No scorer, lab tooling, supplied manifest, evidence, test data or card was edited.
