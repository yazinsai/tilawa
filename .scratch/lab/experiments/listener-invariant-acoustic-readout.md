# Invariant acoustic phone readout

Status: blocked on the supervisor's Modal environment; implementation committed,
but neither a trained bundle nor a measured result is available.

## Prediction and mechanism

Predict two false substitutions removed and one caught substitution lost:
listener −1, overall −1. The experiment card is unchanged. The task's explicit
continuation rule requires at least three target warnings removed before a
practice score, even though the card specifies a two-reversal premise check.

The readout uses a separate copy of the shipped int8 graph exposing its frozen
pre-CTC encoder features. A fixed random projection retains 32 feature dimensions.
The primary graph, cache outputs, phone inventory and frame clock are unchanged.
Both readouts have a 48-unit tanh layer and one shared final phone classifier.

Only help-train training speakers' none-labelled recordings supply canonical
transcripts. Exact CTC forward/backward produces soft phone occupancies, retaining
blank frames and genuine repeated articulations. Substitution-labelled recordings
provide preservation comparisons only where the unchanged engine actually
requests one; no script is imposed as a performed phone transcript.
Recording-level exclusions and unscorable takes are skipped independently of the
speaker split, so unusable audio and takes with extra unscripted mistakes cannot
supply canonical supervision. None-labelled takes containing labelled mistakes or
repeats are rejected. Supplied labels are never modified.

Recording groups come from eight text-independent acoustic descriptors: level,
dynamic range, four relative spectral-band levels, temporal variation and
within-frame spectral spread. Deterministic k-means fits descriptor normalization
and centroids inside each training fold. Each recording receives equal mass;
soft-label importance weights give the groups identical phone distributions over
their common supported classes. Both arms use these same balanced risks, batches,
initial weights, architecture, optimizer, update count and regularization.

The invariant arm additionally penalizes the squared gradient of each group's
risk with respect to the same final classifier's weight and bias. Second-order
gradients update the nonlinear representation. Gradients are squared separately
before averaging, so opposing group preferences cannot cancel. This is a shared
classifier stationarity objective, rather than average-risk or worst-risk
optimization. The first three epochs are a common warm-up. Two/three groups,
penalty strength and weight decay are selected through three speaker-grouped
training-only folds, prioritizing preservation reversals and then held-out
soft-label NLL. The final grouping and readouts are frozen before target checks.

The SDK compares each complete written pronunciation with the already accused
pronunciation on the actual engine-requested frames. Both hypotheses use the
same CTC boundary marginalization with a normalized timing prior conditioned on
completion. A certificate requires a strict written preference from the invariant
readout and a strict rival preference from its pooled control. Missing or
nonfinite comparisons abstain. Certificates bind the word, both complete
pronunciations and exact frame interval; graph confirmations are exempt.

The checker preserves its internal confirmation and ownership ledger and changes
only the public wrong-word verdict. This retains subsequent candidate generation
and insertion decisions. The audio session currently computes auxiliary features
at completion and publishes matching downgrades; this implementation therefore
does not establish timely streaming vetoes. Metadata survives ordinary evidence
serialization without changing primary phone or posterior values.

## Validation and limits

The export utility checks eight successive windows, including all primary cache
outputs, for bitwise parity between the original graph and its feature-exposing
copy with graph optimizations disabled. It generates an auxiliary descriptor with
file sizes and SHA-256 hashes, verifies normal SDK loading, and ties the readouts
to the primary graph hash. This verification must run successfully before a
model selection is declared.

Added unit tests cover soft phone-distribution balancing, text-independent
grouping, second-order gradients into the representation, repeated-phone
occupancies, speaker folds, recording eligibility, exact-window certificates, finite strict reversals,
complete-pronunciation scoring, streaming feature geometry, asset corruption,
audio-session downgrades, graph exemptions and insertion preservation.

Local Python syntax checks and `git diff --check` pass. Modal request 1 (data/model
audit), request 2 (the supplied six-shard build/test harness), request 3 (a smaller
audit), and request 4 (training/export) each failed with
`NotFoundError: No such file or directory` before command execution. The returned
results contain no shard logs and report zero compute spend. Request 5 is a
corrected training/export retry, explicitly installing CPU Torch, NumPy, ONNX and
ONNX Runtime in the default Node sandbox. It is pending. These infrastructure errors
establish no acoustic result, target recovery, test pass or score.

No practice score has been used. There is no verified trained bundle and no
`.autolab/model.json` declaration. Primary-only execution would not evaluate this
experiment. The mechanism has not been killed: unavailable infrastructure is not
evidence that recording-condition shortcuts are absent or that fewer than three
target warnings are removed.

Do not evaluate this branch with the primary model alone. Once the supervisor is
restored, finish request 5, verify normal SDK loading and all declared bundle
files, rerun the six-shard tests, and inspect all four targets using the generic
SDK comparison CLI on their actual evidence and audio. Declare the verified
`bundle` through untracked `.autolab/model.json` only then. Fewer than three
actual target wins requires reverting the implementation and recording a kill;
three or more requires the authorized practice score before a decision. No
infrastructure failure is reported as a zero-target acoustic measurement.

Limited none-labelled coverage, discarded classes without common group support,
alignment errors and a potentially uninformative invariant representation remain
risks. Speaker-grouped validation can reject poor fits but cannot guarantee that
true substitutions are preserved. No scorer, lab tooling, supplied manifest,
test data or evidence was changed; no fold was performed.
