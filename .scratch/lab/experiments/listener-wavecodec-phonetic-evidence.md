# Waveform-code phonetic evidence

Status: blocked before training and runtime validation. The implementation is a
draft, not an evaluated candidate. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two false substitutions removed and one caught substitution
lost: listener −1 and overall −1. Its premise is that reconstruction-trained
waveform codes retain acoustic distinctions discarded by the recognition
encoder. The final task instructions require at least three registered target
wins before proceeding to practice scoring; that overrides the card's
two-window premise threshold.

The draft convolutional autoencoder consumes uniformly sampled 25 ms waveform
patches. Two convolution/tanh layers feed a 16-dimensional discrete codebook;
a small decoder reconstructs the original samples. Reconstruction and VQ
commitment are its only training losses. No text, recognition outputs, or
engine boundaries enter reconstruction training. It uses at most 100,000
parameters, 1,200 updates per fit, two CPU threads, and a 23-hour training
deadline. Every waveform must come from help-train training speakers; any
speaker with a non-training label or an explicit exclusion is rejected.

After freezing the codec, the draft fits full-Gaussian phone emissions on
none-labelled takes without mistake or repeat events. A second, equal-capacity
16-dimensional Gaussian arm uses a fixed projection of the shipped encoder's
pre-head features. Both arms share the observation grid and complete-pronunciation
CTC topology. Their phone and blank boundaries are marginalized, with the
timing-only completion partition divided out. Uniform transcript occupancy
initializes EM without recognition targets.

Codebook size, code hop, and emission shrinkage are selected using three
speaker groups within training data. Reconstruction is refitted without each
validation group's speakers. Selection compares canonical pronunciations with
deterministic single-phone alternatives, rather than comparing raw likelihood
units across representations. These choices have not yet been fitted or selected.

The SDK draft replays evidence to discover already-supported string-path
wrong-word comparisons. It scores the complete original written and accused
pronunciations over their identical existing frame intervals. A token-carried
certificate requires a strict finite written win on waveform codes and a
strict rival win on the matched shipped-feature control. Certificates must
match the word, both complete strings, and both frame bounds exactly.

The checker keeps its original internal confirmations and ownership ledger;
only the corresponding public wrong-word verdict changes. Graph confirmations
are exempt. The SDK compares baseline and certificate replays before retracting
an exact live occurrence, preserving later confirmations and insertion decisions.
The primary model, decoded phones, posterior rows, and live tracker are unchanged.

## Validation and limits

Local Python syntax compilation passed for the three new training/export
modules. `git diff --check` passed. These are static checks, not acoustic or
runtime validation.

Draft unit tests cover exhaustive small-chain boundary sums, completion and
repeated phones, opposing finite preferences, exact certificate identity,
unchanged primary evidence, native shipped-feature state/clock handling,
graph exemptions, and loader rejection of missing or corrupt declared assets.
They have not run successfully. The original insertion-loader mock was adjusted
to explicitly represent an absent waveform-code descriptor.

The only authorized remote runner failed before producing container output:

| Supervisor request | Requested work | Result |
| --- | --- | --- |
| 1 | Read-only help-train and dependency probe | `NotFoundError: No such file or directory` |
| 2 | Repeat of that probe | Same startup error |
| 3 | Supplied six-shard build/test harness | Same startup error |
| 4 | Minimal `pwd` command, five-minute timeout | Same startup error |

All four result files report zero compute spend. No help-train recordings or
practice evidence were inspected through these runs. No codec, scorer, or
extractor was trained or exported; no target preferences or target wins were
measured. No practice score was requested, so all three allowed scores remain.
The startup errors provide no evidence for or against the acoustic premise.
There is no scientific killed outcome.

The exporter is designed to add only the frozen pre-head graph output and
verify primary outputs and cache states against the original ONNX graph over
eight streaming windows. That verification has not executed. Python/SDK codec
encoding parity and the actual warning-certificate path still need validation
on a completed bundle before any target result can be claimed.

The Node loader expects `wavecodec-asset.json` to declare the sizes and hashes
of `wavecodec-emissions.json` and `wavecodec-control.onnx`, tied to the primary
model checksum. No asset descriptor or `.autolab/model.json` has been created:
there is no complete bundle to select. Running the main listener would not
evaluate this experiment and must not be treated as a measured zero change.
The existing folded teacher assets must also be preserved when preparing the
eventual bundle.

The draft buffers whole-session audio and applies retractions at `end()`.
It therefore does not yet meet an immediate streaming-veto latency requirement.
Small-data phone modelling and a discrete reconstruction bottleneck can still
erase genuine substitutions; none of the draft invariants establishes acoustic
precision or a release gain.

To resume, restore the authorized supervisor runner, inspect the available
training layout/dependencies, run the updated build/tests, export and verify
the frozen control, prepare none-labelled training features, and complete the
speaker-grouped training selection. Then validate native codec parity and the
actual SDK certificate path on the registered targets. Fewer than three target
wins requires the requested engine rollback and measured killed write-up;
three or more requires practice scoring before deciding. The current draft is
not ready for scoring or fold-in.
