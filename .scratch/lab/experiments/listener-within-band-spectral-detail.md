# Within-band spectral residual evidence

Status: all three target accusations vetoed by the training-frozen residual model;
SDK integration and complete-bundle validation in progress. The card is unchanged.

## Prediction

Predict one false wrong-word warning removed, no losses, listener −1 and overall
−1. The card requires at least two of its three complete-sequence comparisons
to prefer the written sequence. The task additionally requires three target
warnings removed before practice scoring; fewer than three means rollback.

## Mechanism

The SDK exposes private copies of the existing power spectrum after precisely
the same DC removal, pre-emphasis, 25 ms Povey window and 512-point FFT used for
the original 80-bin mel features. Original mel frames retain their computation,
values and 10 ms streaming clock. No audio filter or alternate analysis window
is used.

A CPU utility beside the student scripts constructs an orthonormal basis for
the null space of that exact mel matrix, excluding FFT bins outside every mel
filter. Normalizing powers by the sum of the existing mel powers removes gain.
Projecting onto this basis reads only information annihilated by frequency
aggregation. A ridge predictor from an intercept and normalized mel powers
subtracts residual detail predictable from mel observations. Phone-conditioned
Gaussian means with training-estimated shared OAS covariance score what remains.
This is fine-frequency power evidence, independent of the network's features
and of phase.

Only help-train recordings with `split=train`, `kind=none`, `scored=true`, a
speaker ID, and no events, repeats, unsupported labels or exclusions enter the
fit. Frozen primary CTC forced alignment of canonical transcripts supplies
phone supervision. Every 40 ms phone label supervises its four original 10 ms
analysis windows. Five speaker-hash folds refit the conditional predictor,
phone means and covariance on the training fold alone. The mean held-out
recording NLL selects predictor ridge and phone-mean shrinkage before any target
inspection. The final model refits on all eligible training speakers.

The checker exposes only final primary string-path decisions and their exact
complete written and already-accused sequences, common comparison bounds, and
disputed-slot bounds. The SDK sums Gaussian likelihood over legal CTC paths,
marginalizing phone boundaries and normalizing by the number of legal placements.
Repeated equal labels still require a blank. Both hypotheses use identical
original analysis frames and unchanged pronunciation labels. Only a finite,
strict written likelihood advantage permits a veto certificate.

Certificates bind the exact word, complete strings, comparison bounds and slot
bounds through ordinary JSON evidence. A veto changes the public string-path
wrong-word result to uncertain while retaining primary internal ownership.
Parallel-graph confirmations, connected-madd decisions, primary tokens,
posteriors, tracker events and existing insertion decisions keep their paths.
The side scorer is installed beside the unchanged primary ONNX graph. Node
loaders verify its size, checksum, inventory and native sample clock. Browser
and native fetches cache and verify the independently declared side asset. A
live audio session retains private resampled PCM and reads the requested original
FFT windows before publishing the wrong-word decision. Evidence-only sessions
without the bound certificate abstain. Recorded evidence receives the same
certificates after existing gap and insertion enrichment, for ordinary replay.

## Validation and limits

Modal run 3 passed all six shards of the supplied build/test harness: 760
core tests and 29 SDK tests, with one optional SDK integration test skipped.
Both packages built. New tests cover strict finite certificate preference,
exact complete-string and frame binding through JSON, graph preservation,
exhaustive CTC placement sums, equal-label blank separation, unknown-phone
abstention, private-spectrum mutation isolation and primary mel equivalence.
Modal run 2 passed four CPU utility tests and fitted 211 eligible recordings
from 72 training speakers, all at 16 kHz, containing 263,784 original analysis
frames. The five folds held 19, 16, 15, 7 and 15 speakers. Training-only validation
selected predictor ridge 1e-6 and phone-mean shrinkage 1000 from the preregistered
code grid. Held-out mean continuous NLL was −725.4363 nats/frame in the original
normalized-power coordinates; density NLL can be negative in these units.
The retained null space has 164 dimensions and maximum mel leakage
8.6942e-16. The fitted JSON asset is 1,165,426 bytes, SHA-256
`36d97e045ac13fd3afc145c15848c6effa7993f347efabbb896a3d6d2d068a8e`.
Its audit, recording hashes and candidate validation scores are recorded in
`.autolab/modal-out-2/0/training.json`; the persistent fit is under
`/scratch/listener-within-band-spectral-detail/fit/`.
Run 4 rebuilt both packages and passed the eight focused core/SDK tests.
Python and SDK residual observations on the first 40 windows of a native-rate
clean training recording compared exactly (maximum absolute difference zero).
That diagnostic then stopped before target inspection because the supervisor
snapshot excludes `lab-experiment.json`. Run 5 passes the card's declared
listener name explicitly to the temporary diagnostic; the asset and all fitted
parameters remain unchanged. Run 7 repeats the target probe with requested-window
feature extraction in place of full-session extraction; the frontend and all
three sequence scores remain identical. All three target accusations disappear:

| Target | Comparison frames | Disputed slot frames | Written minus accused (nats) |
| --- | --- | --- | ---: |
| lab-v1.5-p11 / 26:106:1 | [2523,2616) | [2593,2601) | +0.601831356 |
| lab-v1.5-p36 / 55:48:1 | [6997,7092) | [7044,7062) | +7.326716259 |
| realuser-v1-sob-049 / 92:13:1 | [864,920) | [895,909) | +0.246842753 |

Every phone in both complete sequences has training coverage. The probes check
that primary evidence and complete tracker events are identical, and that
other warning types are identical on these sessions. Target results are in
`.autolab/modal-out-7/0/targets.json`. The two smaller positive margins remain
weak evidence; no threshold or parameter was adjusted after target inspection.

Practice score 1 is pending. The supervisor's `serve_practice` route replays
the card's existing listener evidence; it does not fresh-decode the bundle
declared in `.autolab/model.json`. A stored-evidence score therefore does not
measure the new acoustic certificates. The supervisor's later fresh phone and
studio decodes must establish the overall effect. The implemented model is an approximation: forced
CTC phone supervision, framewise Gaussian emissions, shared covariance and a
linear conditional predictor cannot establish that every discarded acoustic
cue is recoverable or informative. Speaker-grouped likelihood validation is
not a mistake-recall benchmark.

## Bundle and integration status

Run 8 copied the unchanged primary ONNX graph and manifest plus the frozen
residual asset into the experiment's persistent `bundle/`, then failed while
installing CPU torch in the experiment's `teacher-env/` with `No space left on
device`. The existing insertion teacher must also be present to preserve the
folded insertion behavior. Runs 9 and 10 attempted cleanup only of that partial
install and inspection of authorized model files; both returned `NotFoundError`
without command logs. Run 11, the expanded six-shard test harness, failed at
launch with the same error. The new SDK loading/session tests have not yet run.
The complete model bundle has not been declared for scoring: it still needs
verified teacher graph/data assets and runtime validation. Scratch cleanup has
been requested from the supervisor; this is an infrastructure limitation, not
a premise kill.
