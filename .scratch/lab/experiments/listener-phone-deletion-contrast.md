# Consonant occurrence deletion contrast

Status: training implementation present; real-data training and acoustic validation
blocked by the supplied supervisor. Unmeasured. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two recovered missed additions and one collateral wrong call:
listener −1 and overall −1. The premise is that ordinary utterance-level CTC can
recognize surrounding phones well while leaving an actually spoken consonant
dispensable. A frozen-encoder phone-head refit directly contrasts each complete
performed transcript against every alternative deleting exactly one consonant
occurrence, retaining all its surrounding phones. Identical alternatives from
repeated consonants retain their occurrence multiplicity.

`modal/student_deletion.py` computes full CTC forward sums, including blank
separation for repeated labels. For occurrence i, its margin is
`m_i = log Z(complete) − log Z(delete_i)`. The added objective is
`log(1 + mean_i(exp(−m_i)))`, evaluated with log-sum-exp and softplus. It is
normalized by consonant occurrence count and gives weak comparisons greater
gradient weight; a strong shared vowel cannot cancel a negative consonant margin.
The installed consonants and nasal consonant realizations qualify for deletion;
vowel, duration and pause markers remain in each alternative.

The small affine head starts at zero and adds a correction to the shipped log
posteriors before log-softmax. Encoder features and shipped posteriors are detached.
Both arms retain ordinary CTC per transcript phone and KL preservation against
the shipped distribution on every frame, including speech, waqf tails and pauses.
This preservation term constrains change; it does not guarantee unchanged decodes
or quiet clean sessions.

## Matched training and export

`modal/student_deletion_train.py prepare` requires the shipped model bundle,
help-train labels, enrolled performed-transcript JSONL, an excluded-speaker JSON
list and the shipped noise/RIR banks. Additional transcripts must explicitly
identify their speaker, have `split=train`, `authorized=true`, `verified=true`,
performed `phones`, relative `wav` and the audio's `sha256`. Help takes must have
kind `none`, training split, `scored=true`, and empty events, non-errors and
unsupported labels. Scripted mistakes and repeats are not converted to canonical
performed transcripts. Cross-corpus held-out speaker discovery precedes enrollment;
duplicate audio, changed audio identities and paths leaving the corpus fail.

Speaker hashing separates fit and checkpoint-selection groups; each group must
contain both enrolled corpora. Preparation caches frozen native encoder features,
shipped posteriors, complete transcripts and filterbanks for three shared views:
the original recording and two deterministic views using the shipped reverb,
noise, gain and SpecAugment distributions. Cache files, model, auxiliary assets
and augmentation banks have recorded hashes.

`fit` uses the same initialization, cached views, recording order, SGD optimizer
and fixed recipe for both ordinary and contrast arms: seed 1729, 12 epochs,
learning rate 0.02, and preservation/contrast weights of one. Each arm independently
selects its lowest speaker-balanced CTC-plus-preservation objective on the
training-speaker selection partition, including epoch zero. Practice audio and
labels play no role in training or checkpoint selection.

`modal/student_deletion_export.py` exposes native pre-quantization encoder features
and grafts the selected affine correction onto the frozen shipped ONNX graph.
It preserves the ordered inventory, blank, frame clock, streaming inputs and caches.
The existing SDK can load the resulting ordinary model bundle without acceptance
rule changes. Its insertion teacher and every external weight file are required,
copied and checksum-verified. Both selected arms must pass optimized-runtime
posterior parity on held-out training-speaker recordings before any bundle is
declared for scoring. Tracker, checker and SDK decoding code are unchanged.

## Validation and limits

Local synthetic tests cover exhaustive CTC likelihoods and gradients, empty and
repeated-label transcripts, occurrence multiplicity, surrounding-vowel retention,
weak-comparison weighting, normalization and numerical stability. They also cover
frozen inputs, the matched ordinary objective, speaker-balanced read-only selection,
cross-corpus exclusions, audio identity and duplicate rejection, float/int8 graph
exports, streaming clock/cache preservation, unchanged teacher assets and a complete
one-epoch synthetic fit/export of both arms. All 25 tests passed locally in 154.30
seconds with torch 2.2.2, NumPy 1.26.4, ONNX 1.23.2 and ONNX Runtime 1.23.2:
`.scratch/deletion-venv/bin/python -m pytest modal/test_student_deletion.py -q`.
Python syntax compilation also passed. Synthetic examples are not a target
recovery measurement.

The supplied supervisor's input-inspection request (run 1, train image) and required
six-shard `bash tools/autolab_tests.sh` request (run 2) both failed before their
commands executed with `NotFoundError: No such file or directory`, spending zero.
A corrected lightweight train-image input check (run 3) failed with the same
pre-execution error. All three requests therefore produced no command output or
acoustic evidence. The local test environment is ignored scratch material.

No authorized real recording has been enrolled, no real head has been trained,
and no complete real model bundle is available. `.autolab/model.json` is therefore
absent. Training-data schema compatibility, real shipped graph extraction/parity,
SDK full-session decoding, target recovery, clean-session behavior and studio
regression remain unvalidated. Core and SDK suites have not run successfully.

Zero practice scores have been requested; listener and overall changes and target
wins are unknown. The card requires two previously missing planted consonants
recovered in order relative to the matched ordinary head. The user's later rule
also requires at least three target wins before practice scoring. Neither condition
has been measured. Infrastructure failure does not establish a failed premise,
so this experiment is not labelled killed and has no fabricated zero-target result.
It needs successful supervised enrollment, matched training and full-session target
comparison before the scoring declaration and fold decision can proceed.
