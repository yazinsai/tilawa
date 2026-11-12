# Unconditional collapsed-count phone-head training

Status: implemented; training and target evaluation blocked by supervisor
execution failures. No replacement model is declared. The card is unchanged.

## Prediction and premise

Listener −1 wrong call and overall −1, recovering a weak consonant in an added
word. This prediction is within the noise band and below the fold floor.
The premise is that a good transcript-conditioned CTC path can coexist with
too little consonant evidence in the unrestricted distribution used by the
ordinary decoder. Count supervision is intended to narrow that gap.

## Built mechanism

`modal/student_count.py` computes, for each nonblank phone k,

`E[N_k] = p[0,k] + Σ(t=1..T−1) p[t,k] × (1 − p[t−1,k])`.

The statistic includes every CTC frame-label path, without conditioning on the
transcript or a forced alignment. Blank is a different preceding label; a
blank-separated repetition is another occurrence. Deterministic sustained
identical labels contribute one occurrence. Uncertain sustained labels can
still have several expected occurrences because unrestricted paths can switch
away and back. Padding is excluded, including its gradients.

The auxiliary loss is the mean, over all nonblank phones, of squared expected
count error divided by transcript count plus one. Absent phones participate,
so verified real silence has zero phone targets and penalizes invented sounds.
No count constraint is added at inference, and matching counts alone does not
establish correct phone identity, order, placement, or safe warnings.

`modal/student_count_train.py` caches native frozen-encoder features from the
shipped streaming ONNX graph. It refits only the existing affine phone head,
starting from its weights and bias. Both the ordinary-CTC control and count
arms remove that head's activation quantization; the quantized encoder remains
frozen. That common export difference cannot be credited to the count loss.

Both arms receive the same recordings, three fixed views per recording
(original plus two shipped waveform/SpecAugment views), seeded recording order,
head initialization, learning rate, gradient bound, and update count. Both use
transcript CTC and a KL pull to shipped posteriors over real frames. The count
arm adds the auxiliary term; the ordinary arm uses strength zero. CTC is
normalized by transcript length, or valid frame length for empty silence.
Views are averaged equally and each recording contributes one update per epoch.

The fixed schedule is twelve epochs of SGD at 0.02, complete-gradient clipping
at 5, and count strengths 0.1, 1, and 10. Selection compares the pair's common
CTC + unit count loss + preservation objective, averaged equally over selection
speakers. It chooses one shared checkpoint epoch and the count strength.
Epoch zero is eligible. Both exported arms therefore have identical selected
update counts. Speech PER and emitted silence phones are also reported. No
practice evidence enters preparation, fitting, regularization, or selection.

## Training provenance and export

Preparation requires authorized, verified performed-transcript recordings,
help-train training-speaker `none` takes, and verified real silence recordings.
Additional recordings are explicitly enrolled JSONL with speaker, split,
authorization, verification, CTC-articulation transcript format, audio path,
and audio SHA-256. Empty transcripts require `kind=silence`; synthetic zero
waveforms are rejected. Help none takes require empty events, non-errors and
unsupported lists. Mistake scripts and repeats are not performed transcripts.

Reference notation is converted using the SDK's `collapseHeld` convention:
madd and ikhfa duration runs become one occurrence, duration-expanded
mushaddad noon/meem become two, and genuine consonant doubles remain doubled.
Already enrolled CTC-articulation transcripts are never run-collapsed.

All speakers in nontraining splits across either corpus, plus explicit release
exclusions, are excluded before enrollment. Whole speakers are partitioned by
a deterministic hash into fit and selection. Both partitions must contain
both speech sources and real silence. Audio hashes prevent duplication across
partitions; cache hashes protect identical views. These contracts have not
been checked against the supplied remote training files because inspection
never executed.

`modal/student_count_export.py` exposes the native feature tensor and replaces
only the phone projection. It inherits the ordered alphabet, frontend, frame
clock, streaming geometry, state inputs and outputs. Decoder, tracker, checker,
and insertion verification are unchanged. Every existing teacher and external
weight file must be present, copied, and checksum-verified. Preparation checks
that feature exposure preserves original outputs and recurrent states exactly.
Both exports must reproduce their fitted heads over all selection views
within 2e-4 probability tolerance. The final report is written only after both
complete bundles pass these checks.

The runner exposes `prepare --model --help --transcribed --exclusions
--augmentation --out` and `fit --prepared --out`. Cache and output directories
must be supplied under `/scratch/listener-unconditional-count-training` through
the authorized supervisor. It does not substitute practice recordings,
teacher guesses or scripted mistake text when an enrolled input is absent.

## Validation and result

Python compilation and whitespace checks pass. The numerical/contract suite
in `modal/test_student_count.py` covers exact enumeration of all tiny CTC paths,
gradients against enumerated finite differences, blank-heavy missing-phone
gradients, sustained labels, real repetitions, padding, zero-count silence,
held notation, enrollment and cross-corpus holdouts, frozen feature gradients,
matched training order and checkpoints, native float/int8 head extraction,
state preservation, and complete teacher/external-file export.

All 21 synthetic tests passed locally with no skips in 42.2 seconds:
`.autolab/count-venv/bin/python -m unittest discover -s modal -p
test_student_count.py -v`. The isolated workspace environment uses Torch 2.8.0,
ONNX 1.19.0 and ONNX Runtime 1.22.1 (matching the supervisor training image),
NumPy 2.5.4 and Python 3.13. No audio or training computation ran on the Mac.
The original system Python's NumPy had an architecture mismatch; a preliminary
Rosetta run passed only the enrollment tests and skipped unavailable numerical
dependencies. Those skipped checks are superseded by the complete local run.

Four supervisor requests failed before command execution with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Run 1: supplied data/model and dependency inspection.
2. Run 2: training-image inspection, including help labels and training roots.
3. Run 3: new numerical/contract tests on the training image.
4. Run 4: required six-shard core/SDK build and test suites.

There is no remotely inspected model or training corpus, prepared cache,
trained matched control, trained count head, verified real export, or target
decode. There were zero practice-score invocations. Target wins, collateral
losses, and component/overall changes are unknown. This is an infrastructure
blocker, not a measured zero gain or a falsified acoustic premise.

## Remaining validation

Restore supervisor execution, inspect the authorized training schema, prepare
the shared cache, fit and select matched heads entirely within training
speakers, and verify both complete bundles. Then compare their full-session
decodes on the six target spans, checking recovery of previously absent planted
consonants in correct order and the resulting engine warnings.

The user's final rules require at least three target wins before practice
scoring; fewer than three or a false premise means an acoustic kill and engine
rollback. If at least three are won, run the supplied practice score before
deciding or narrowing. Collateral losses belong to the fold gate. No
`.autolab/model.json` is present: decoding with the main listener would not
evaluate this implementation. Synthetic correctness does not establish target
recovery, ordinary-reading preservation, or clean-session safety.
