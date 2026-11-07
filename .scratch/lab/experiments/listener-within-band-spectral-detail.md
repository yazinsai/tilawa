# Within-band spectral residual evidence

Status: implementing the training-frozen acoustic premise probe. The experiment
card is unchanged.

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
The premise is tested before declaring an installed side asset or changing the
production session loader.

## Validation and limits

Pending Modal training, unit checks and frozen target comparisons. No practice
score has been requested. The implemented model is an approximation: forced
CTC phone supervision, framewise Gaussian emissions, shared covariance and a
linear conditional predictor cannot establish that every discarded acoustic
cue is recoverable or informative. Speaker-grouped likelihood validation is
not a mistake-recall benchmark.
