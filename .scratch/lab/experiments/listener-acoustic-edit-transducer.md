# Acoustic edit transducer

Status: implementation retained; training and lab validation blocked by
supervisor execution. No replacement model is declared. The card is unchanged.

## Prediction

Listener −2 and overall −2, recovering fragmented added words, with substantial
zero-gain risk. This forecast is inside the noise band and below the fold floor.
The proposed missing capability is repairing an already recognized phone
sequence using its acoustic evidence, including adding missing events and
deleting fragments. No target's acoustic premise has been inspected here.

## Implemented mechanism

`modal/student_edit_transducer.py` defines a monotone lattice with state
`(acoustic frame, consumed source events, emitted target phones)`. The source
is the shipped greedy CTC sequence with its original frame spans. HOLD advances
the acoustic clock; DELETE consumes a source event without advancing time;
COPY and SUBSTITUTE consume a source event and an acoustic frame; INSERT
emits a phone at an acoustic frame without consuming the source. Source-consuming
actions stay inside the source event's span. A frame can emit at most one phone.

The training loss sums every legal script yielding the performed target,
including substitution versus deletion followed by insertion and alternative
emission times. Adjacent identical output phones remain separate articulations.
Inference uses the highest-probability complete timed script. It evaluates
source-cursor alternatives globally within each streaming chunk, rather than
choosing each action greedily. It does not marginalize output hypotheses at
inference. Transcript timing fixes chunk enrollment; the edit marginalization
does not span alternative target alignments across chunk boundaries.

The head receives frozen native encoder features and a cumulative embedding of
the consumed source prefix plus the next source event. A learned residual
corrects the acoustic representation. Both the edit distributions and an
independent framewise CTC support head train that representation. There is no
expected passage, lexicon, verse identity, output language model, or tracker
state in the inference inputs.

The SDK's `edit-transducer.ts` executes the lattice. `listener.ts` uses its
phones as primary evidence and the auxiliary CTC rows as support posteriors.
Copy/substitute spans retain source duration, shortened when a later edit
occupies those frames. Carried source runs are consumed once; blank-separated
repetitions remain separate. The original alphabet, 40 ms clock, encoder
caches, frontend, tracker, checker, and insertion verification are inherited.
Ordinary models retain their existing CTC path.

The exported graph also accepts the number of valid acoustic rows in a call.
Phone events arising only from final padding are excluded from the neural
source context as well as from SDK lattice decoding. The final source-cache
value comes from the last valid row.

## Training and control

`student_edit_data.py` enrolls authorized, verified performed phone transcripts,
help-train training-speaker none takes, and recorded pause negatives. Additional
JSONL enrollment requires speaker, `split=train`, `authorized=true`,
`verified=true`, `transcriptFormat=ctc-articulations`, phones, audio path, and
audio SHA-256. An empty transcript requires `kind=silence`. Preparation rejects
synthetic zero-waveform pause negatives. None takes require empty mistake,
repeat/non-error and unsupported lists; scripted mistakes are not performed
transcripts. Canonical duration notation is converted to articulation counts
without erasing genuine gemination.

Nontraining speakers across both corpora and explicitly enrolled release
exclusions are removed before admission. Audio hashes reject duplicate records.
Whole speakers are deterministically partitioned into five folds; fold zero
is selection only. Both fit and selection require both speech sources and
real pause negatives. These enrollment contracts have not been checked against
the supplied remote schema because data inspection did not execute.

`student_edit_train.py prepare` caches exact streaming encoder features and
actual shipped recognition outputs, under original audio and two fixed views
using the shipped noise, RIR, gain and SpecAugment family. It checks feature
exposure against the ordinary optimized graph, including greedy identities
and recurrent states. A CTC alignment of the performed transcript supplies
chunk enrollment timestamps; it supplies no text during inference.

Empirical corruption channels are estimated from actual source/target
recognition errors. For each recipient fold, its speakers and every selection
speaker are excluded from channel estimation. The channel includes copies,
missing phones, substitutions and fragments, conditioned on the phone and
its preceding phone, with a phone-only fallback. Generated events stay in the
aligned acoustic interval; fragments that cannot fit separate frame positions
are dropped. Each edit update also retains its recording's actual shipped
decode, so the cross-fit draw never replaces real error supervision entirely.
The particular minimum-edit alignment used to estimate channel observations
is not imposed on the transducer's forward-sum loss.

Edit training combines the marginalized recognition objective, full-recording
auxiliary CTC, a KL pull to shipped evidence, and COPY supervision on matching
stretches. Actual pauses have empty phone targets. These are preservation
constraints, not evidence that a globally changed listener will stay quiet.

The direct-CTC control evaluates the same neural network with source context
zeroed, on the same fixed acoustic corpus and views. Both start from the same
native phone-head weights and zero acoustic residuals. The default allocation
is 600 training seconds for the edit arm; the control receives at least its
actual training time, including the edit arm's final-recording overrun. It can
take more updates because its sequence loss is cheaper. Compute matching is
wall-clock training allocation on one machine, not exact FLOP equality. Actual
seconds, parameter counts and update counts are recorded. Selection/export
costs are separate. Both arms choose checkpoints using primary phone error
averaged over selection speakers, including pause emissions; epoch zero is
eligible. No practice data enters selection or hyperparameter decisions.

The learned CTC heads use float projections of native weights; the source
decoder remains the original quantized projection. Removing activation
rounding can therefore change the direct arm's initial decisions. A real
control comparison is still required before crediting any target improvement
to edits.

## Export and validation

`student_edit_native.py` reuses native-graph inspection and integrity helpers
from the unconditional-count experiment branch. `student_edit_export.py`
grafts the head onto the shipped graph and generates a new bundle manifest.
The source posteriors remain available separately from corrected support.
The existing teacher and every external weight file must be present, copied
and checksum-verified. Missing assets fail export. `fit` produces both arms;
`verify` checks them against fitted heads on selection views, verifies source
greedy identities and the valid frame clock, and writes `verified.json` only
after both complete bundles pass. It never writes a main-model fallback or
model selection file.

Focused local validation passed with no skips:

- 17 Python tests: exhaustive script probability sums, normalization, equivalent
  edits, Viterbi optimality, timing, repetitions, cross-fit exclusion, transcript
  admission and identity, held notation, full-recording/partial-drain losses,
  gradients against finite differences, and both ONNX exports with preserved
  native caches and complete teacher assets. Padding-only source phones are
  explicitly exercised.
- 7 SDK tests: insertion, deletion, substitution, copy duration, carried runs,
  global script choice, final padding, malformed output rejection, primary
  token versus CTC-support routing, and valid-frame feed/drain behavior.
- Strict TypeScript checking of the changed SDK modules and their tests;
  Python compilation and git whitespace checks.

The small synthetic checks ran in workspace-local ignored environments with
Torch 2.8.0, ONNX 1.19.0, ONNX Runtime 1.22.1, NumPy 2.4.4, TypeScript 5.9.3
and Vitest 3.2.7. No training, corpus decoding, practice audio or other heavy
computation ran on the Mac. These tests do not establish real-model export
parity, recognition quality, device latency, or warning preservation.

Five authorized supervisor requests all failed before command execution with
`NotFoundError: No such file or directory`:

1. Data and scratch availability probe.
2. Training-image/data-schema probe.
3. Six-shard core/SDK build and test suites.
4. Numerical/ONNX tests on the training image.
5. Six-shard core/SDK build and test suites after implementation.

The supervisor reports USD 0 spent and the full USD 180 GPU budget remaining.
The complete existing core/SDK suites could not be verified. There is no
inspected remote corpus, prepared acoustic cache, trained edit arm or control,
verified real model bundle, full-session target decode, or practice score.
All six registered target spans remain uninspected. Target wins and component,
overall, studio and clean-session changes are unknown. Zero practice-score
invocations were used.

## Decision and remaining work

This is an infrastructure block, not zero measured wins or a falsified premise.
The engine implementation is retained and `.autolab/model.json` is absent.
Scoring the main model would not evaluate these changes. The untracked outcome
records a blocked state with no invented measured values.

Once supervisor execution works, inspect and enroll the supplied training
schema and release exclusions; run prepare, fit and verify under the experiment
scratch directory; then check the two complete bundles through full-session
SDK decoding. The registered incidents require previously absent planted
consonants in the correct interval and complete added-word hypotheses reaching
the exported engine path, not merely better cropped phone strings.

The user's final rules require at least three target wins before practice
scoring. Fewer than three wins or a false premise means an acoustic kill,
engine rollback and a measured write-up. At least three means running the
authorized practice score before deciding or narrowing. Collateral losses
belong to the fold gate, with a new warning on a fully clean session a hard
stop. No scientific keep/kill decision is supported yet.
