# Reverse sequence supervision

Status: training implementation present; corpus audit, matched training, model
export and acoustic evaluation incomplete. The supplied supervisor failed to
start the first four requests, including a plain `pwd`. The latest regression
request is pending. This is an
unmeasured experiment, not an acoustic kill. The card is unchanged.

## Prediction and premise

The card predicts two recovered added-word misses and one collateral wrong call:
listener −1 and overall −1. Weak earlier consonants may become identifiable in
following speech. A decoder trained to transcribe the recorded phone sequence
in reverse order may teach the streaming primary encoder to preserve that
evidence. The five registered incident keys are evaluation targets only; none
is encoded in the training implementation or used for checkpoint selection.

## Implemented mechanism

`modal/student_train.py` exposes the existing encoder output through `encode()`.
Its ordinary forward path still applies the same CTC head to the same frames.
Parameter names, vocabulary, streaming masks and inference graph are unchanged.

`modal/student_sequence.py` supplies a direct training entry point for the
supervisor's train image. A causal Transformer label decoder cross-attends to
all valid streaming encoder frames without detaching them. Reverse transcription
reverses each recording's phones independently and retains repeated occurrences.
Forward transcription is the matched capacity/control arm. Ordinary continuation
constructs the same decoder but applies no auxiliary gradient. Previous-phone
dropout masks label history, preserving BOS and padding; its private RNG cannot
advance the primary model's subsequent random draws. Primary and decoder
gradients are clipped separately so decoder gradient magnitude cannot rescale
the continuation control's primary update.

All three arms require one identical frozen recipe and initial checkpoint.
The original Quran training mixture retains its teacher/spliced CTC targets,
cached KD, RIR, noise, gain, raw Kaldi fbank and SpecAugment chain. Item/epoch
augmentation and batch/step random seeds are matched across arms. Full-frame KL
against a frozen initial primary model supplies ordinary-reading preservation,
including blanks and pauses, with deterministic deployed-mask passes that restore
the model's previous flags and mask settings afterward.

Auxiliary labels require an explicit audited performed transcript; teacher
greedy strings and planted mistake scripts are not automatically accepted as
such transcripts. New help recordings require `kind="none"`, the original
`train` speaker split, the matching audited speaker, scored/nonexcluded status,
and empty events, repeats and unsupported lists. They retain the entire mono
16 kHz recording, use their authorized ayah transcript for CTC and auxiliary
supervision, and receive full-output preservation without fabricated cached KD.
Impossible new CTC alignments, including repeated-phone blank requirements,
raise an error instead of silently zeroing the new supervision.

The recipe requires disjoint training, training-speaker validation and excluded
speaker sets, both sources in both partitions, complete original Quran-mixture
coverage, exact checkpoint/corpus hashes, fixed training configuration and the
selection policy `minimum-primary-validation-objective`. `audit_corpus()`
fingerprints the selected audio, identities, partitions, transcripts and cached
teacher arrays before the same recipe is frozen for all arms. These checks
detect drift and incomplete declarations; they do not prove the factual speaker
identities, release-held-out aliases or performed-transcript audit. No recipe
with invented provenance or training hyperparameters has been created.

Checkpoint selection uses only primary CTC/KD plus preservation on the frozen
training-speaker validation partition. It excludes auxiliary loss and practice
results. Resume rejects a changed recipe or arm. `training.pt` retains decoder
and optimizer state; selected `ckpt.pt` contains only the original primary model
keys/configuration and can be consumed by the existing loader and streaming
exporter. No selected weights or export actually exist yet. A scored primary
bundle must preserve the existing insertion teacher assets beside the exported
model. SDK inference, tracker, checker and insertion verification are unchanged.

## Validation and limits

Read AGENTS.md and `docs/agents/audio-review.md`. Small synthetic tests exercise
per-recording reversal, causal label attention, gradients to following acoustic
frames, padding exclusion, history dropout and private RNG, continuation's zero
auxiliary gradient, blank-preserving detached KL, restoration after exceptions,
joint primary/decoder gradients, missing-KD handling, held-out/script exclusions,
explicit help-none selection and primary-only checkpoint round trips.

The final local rerun passed all 33 tests:

```
.scratch/sequence-test-env/bin/python -m pytest -q modal/test_student_sequence.py tools/test_student.py
```

An earlier combined run had 27 passes and two existing tests blocked by missing
`rapidfuzz`; installing their declared dependency resolved both. The passing
suite also checks selection, production forward equivalence with encoder
doubles, exact corpus retention and hash drift, and impossible new CTC
alignments. Only tiny tensors and synthetic waveforms run locally; heavy
acoustic work remains on Modal. Python compilation and `git diff --check` pass.

Actual Zipformer training/backpropagation, corpus schema and speaker audit,
decoder loss balance, memory/runtime, checkpoint selection, quantization,
streaming export parity and full-session consonant recovery remain unverified.
Unit tests do not establish weak-consonant recovery or warning safety.

## Execution service and result

Every remote request used the user-supplied supervisor. No alternate Modal
access, release exam or outside-worktree corpus search was used.

| Run | Request | Result |
| --- | --- | --- |
| 1 | CPU checkpoint/help inventory | Startup `NotFoundError: No such file or directory`; no command output |
| 2 | Train image, CPU checkpoint/s1/help inventory | Same startup failure; no inventory ran |
| 3 | Six CPU shards, core/SDK regression suites | Same startup failure; no tests ran |
| 4 | CPU, `pwd`, two-minute timeout | Same startup failure; even `pwd` did not run |
| 5 | Six CPU shards, core/SDK regression suites, five-minute timeout | Pending |

The completed requests report $0 GPU spend and $75 remaining. No GPU
training request, trained checkpoint, exported bundle or `.autolab/model.json`
exists. No practice score has been requested; all three opportunities remain.
Scoring the unchanged main model would not evaluate this training mechanism.
No target-win count or score delta is asserted, and no killed outcome is written.

After service restoration, independently audit authorized Quran performed
transcripts, speaker exclusions and help-none provenance; freeze the corpus and
training recipe; run all three matched arms and select within training-speaker
partitions. Export and verify complete int8 primary bundles with unchanged
inventory/frame clock and existing insertion verification assets. Compare
full-session exported primary target decodes with both controls. Fewer than two
targets regaining missing planted consonants fails the card's acoustic criterion;
the user's latest rule also kills a built change winning fewer than three target
incidents. A measured kill requires reverting engine changes and recording the
result. Three or more target wins require practice scoring before any narrowing;
the supervisor's fold gate decides whether gains and collateral calls qualify.
