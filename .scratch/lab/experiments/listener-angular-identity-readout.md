# listener-angular-identity-readout

Status: needs-info

Prediction: listener −1 and overall −1 on practice, with no predicted losses.
The proposed cause is recording-dependent native encoder-vector magnitude
changing phone identity despite useful feature directions. This remains a
hypothesis; the target acoustics have not yet been measured.

## Implemented comparison

`modal/student_angular_export.py` exposes the float input to the frozen CTC
linear layer, before dynamic quantization, in a separate ONNX graph. It retains
the full native width: no random projection, centering, standardization or
nonlinear hidden layer is applied. The utility checks the primary file hash and
requires bitwise parity of all original outputs and caches over eight sequential
windows against ordinary deployment optimizations. The primary ONNX, manifest
and any insertion-teacher files are copied unchanged into the resulting bundle.
This export check has not yet run on the shipped listener.

The SDK extraction utility accepts help-train labels, permits only training
speakers, and excludes every speaker appearing in a non-training split. None
takes with events or repeat annotations are rejected. None-labelled transcripts
supply exact CTC forward/backward soft phone occupancies from the primary
posteriors. Both heads receive the same conditional nonblank targets, the same
frame draws, and the same recordings with equal total loss mass per recording;
occupied-frame posterior mass weights supervision. Scripted substitutions supply
only baseline-engine comparison windows for weak preservation diagnostics; their
planted text is never forced onto the acoustic frames or used for phone training.
The supplied help-train volume excludes release-held-out speakers; the utility
also rejects excluded-speaker provenance explicitly.

`modal/student_angular_phone.py` fits one unit class vector per nonblank phone
and one positive shared learned concentration. The matched ordinary affine head
has the same native width and output classes, with raw feature vectors and class
biases. Parameter counts are C×D+1 and C×D+C respectively; the small difference
comes from the angular arm's shared concentration instead of class biases,
while both arms have one linear phone readout. Both use 18 epochs, seed 1729,
AdamW learning rate .002 and matched minibatches. Shared regularization is
selected from {.0001, .01} by the mean of both arms' conditional soft NLL in
three speaker-grouped training partitions. Final fitting happens once with the
selected recipe. No practice inputs enter training or model selection.

Features are cached per recording using hashes of its WAV bytes, labels and
extractor. A persistent one-day wall-time budget bounds extraction and fitting
across resumed runs. The fitted JSON records native dimensions, class vectors,
concentration and the minimum/maximum feature norm at training frames with
nonblank occupancy above 1e-5. Bundle declarations hash both the fitted heads
and feature graph. The ordinary Node loader rejects missing, corrupt or
primary-mismatched declared assets; it does not silently replace a missing side
scorer with the primary listener. The complete bundle must pass ordinary SDK
loading before it is eligible for any acoustic diagnostic or scoring.

## Runtime and premise diagnostic

`packages/sdk/src/angular-phone.ts` computes separate identity distributions for
both arms. Each row retains the primary blank probability and distributes
exactly its existing nonblank mass, with no speech floor. Complete written and
already-accused pronunciations use the ordinary CTC forward sum over identical
existing string-path support windows. No new word candidate or boundary model
is introduced. A certificate requires a strict angular preference for written
and a strict affine preference for rival, both finite.

The checker accepts a certificate only on the identical word, full
pronunciations and frame endpoints. Graph confirmations are exempt. Internal
issue confirmation and ownership bookkeeping stay on the baseline path; the
public supported wrong-word verdict becomes uncertain and is omitted from
reported mistakes. This keeps downstream insertion decisions and candidate
creation unchanged. The SDK's final replay transports certificates with
phoneme metadata and checks removal identities against the original live replay,
before gap splicing or auxiliary insertion enrichment can change that context.
The primary graph, greedy tokens, posterior rows, tracker input and tracker
weights are unchanged.

An analysis-only probe preserves vector directions and blank probabilities,
sweeping 33 common radii logarithmically between the recorded training minimum
and maximum. It reports whether the affine written/rival margin crosses zero
within that interval and whether the frozen angular arm prefers written. The
probe is not a warning condition or selection objective. Rescaled features are
a mathematical premise diagnostic, not evidence that a recording physically
underwent that scalar distortion.

The three preregistered targets are the openings at 26:106:1 (104.32 s) and
55:48:1 (282.64 s), and the interior-noon warning at 92:13:1 (36.44 s).
No target references, session identifiers or reader identities occur in engine
or training code. The card's premise requires at least two magnitude-sensitive
windows with an angular written preference. The user's continuation rule also
requires three actual target warnings won. Fewer than three measured wins calls
for rollback and a measured kill; three wins requires an official practice
score before deciding. Neither condition can be decided from unit tests.

## Validation and current blocker

The five Python unit checks pass, including exhaustive CTC path/occupancy
comparisons for genuine repetition, conditional soft-target normalization,
unchanged blank/nonblank mass, speaker partitioning and provenance rejection.
Both core and SDK TypeScript builds pass. The local SDK suite passes all 35
cases, with one optional help integration skipped, using one worker and a
60-second command-line test timeout. Its eight math/native-feature tests check
positive-scale invariance, radial control reversals, complete-sequence scoring,
exact primary speech mass and matching streaming inputs/cache state. Other new
SDK tests exercise certificate publication and rejection of incomplete bundles.
The six new core cases all pass on a one-worker focused run. The full default
core run passed 707 of 772 cases and failed 65 on timeouts, with seven worker RPC
deadline errors. A one-worker rerun of its 28 affected existing files passed 351
of 352 cases; together with the six new cases, 771 of 772 core cases have passed
across runs. The remaining existing startup-persistence case takes about 16.9
seconds against an explicit 15-second test limit. A final focused rerun
confirmed that timeout at about 20.5 seconds; its assertions and limit were
left unchanged. The complete core suite and required Modal harness have not
passed.

Supervisor requests 1 and 2 were read-only data/environment checks. Request 3
was the required six-shard `bash tools/autolab_tests.sh`. Request 4 selected the
supervisor's training image and attempted the complete build, Python checks,
feature extraction, speaker-grouped fitting, export and normal SDK-load check.
All four returned `NotFoundError: No such file or directory` before command
execution and reported zero compute cost. This is an infrastructure blocker,
not a premise failure. The actual none-speaker coverage, soft alignments,
training norm range, trained concentration, selected heads and export parity
remain unmeasured. No complete fitted bundle exists and no `.autolab/model.json`
is declared. No practice score has been used. Target wins and overall deltas are
unmeasured, not zero; no killed outcome is asserted.

The authorized resumption command is:

```sh
python3 modal/student_angular_export.py \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-angular-identity-readout/bundle \
  --labels /lab/data/help-train/labels.jsonl
```

After successful export and ordinary SDK loading, the SDK analysis CLI accepts
`--model`, `--audio`, `--evidence` and `--out`. It discovers actual baseline
warning windows, records both scores and the frozen radial diagnostic, and
asserts identical tracker events and added-word decisions in before/after
replays. Those measurements must precede a model declaration or practice score.
Scoring with only the main model would not evaluate this experiment.

This implementation buffers full-session audio and performs a separate native
feature pass at session completion when supported accusations exist. Live
primary warnings may precede the final downgrade; immediate warning precision,
latency, memory use and device readiness are not established. Small-data soft
supervision and an angular reading error could suppress genuine substitutions.
Unit tests establish mathematical and transport behavior, not acoustic
separation, three target recoveries or fold-gate safety.

No experiment-card, supplied data, evidence, scorer or lab-tooling edits were
made. No fold was performed. The engine draft remains for resumption; this
experiment is incomplete until the authorized supervisor can train and measure
its complete side bundle.
