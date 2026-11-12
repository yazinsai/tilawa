# Relative-articulation training of the frozen listener's phone head

Status: implementation and synthetic validation complete; real-data experiment
blocked and unmeasured. No model is declared for scoring. Recorded 2026-11-12.
The experiment card is unchanged.

## Prediction and stopping rule

The card predicts listener −1 and overall −1 wrong calls on lab-v1.5u. Brief
consonants beside sustained vowels may be underrepresented in training, and
whole-recording tempo perturbation preserves that relative duration imbalance.
The prediction is inside the two-call noise band and below the fold threshold.

The registered sites are p07 `37:36:1` at 144.12 s, p48 `12:76:1` at 39.98 s,
p56 `32:21:1` at 106.76 s, and p57 `51:18:1` at 95.26 s and `51:20:1` at
112.26 s. These identifiers appear only in this report. Engine/training code
contains no practice-specific selection or routing.

The card asks whether the training-selected head restores an absent planted
consonant, in order, relative to both matched controls in full-session decoding.
The user's final instruction requires at least three target wins before practice
scoring; fewer than three requires rollback and a killed result. Neither check
has been measured. No practice score has been requested, and an unavailable
training run is not reported as zero target wins or a disproved premise.

## Implemented mechanism

`modal/student_articulation.py` computes exact transcript-conditioned CTC
forward/backward probabilities. It records separate state-occupancy, onset and
last-occupation distributions for every transcript occurrence, including equal
adjacent labels. It does not substitute a Viterbi path or greedy peak for these
distributions.

Consonant cores extend from the 90th-percentile onset to the 10th-percentile
last-occupation endpoint, with another 10 ms protected at each boundary. Cores
shorter than 40 ms, overlapping cores, vowels, madd and sustained nasal markers
are not duration targets. A separate training-only blank-posterior pass protects
runs of at least 200 ms whose blank probability is at least 0.95. Cores crossing
these pause boundaries are left unchanged. These fixed constants specify a
conservative training-view policy, not inference duration tolerance.

Pitch-preserving waveform-similarity overlap/add uses 20 ms source grains, 5 ms
synthesis steps and a 5 ms correlation search. Grains are copied rather than
sample-rate converted. First/final grains retain source boundaries; identity
lengths use the same synthesis path and reproduce the source samples exactly.
The relative arm replaces only consonant cores, copying all intervening audio
in order. Vowel durations, transition guards and real pauses therefore retain
their original sample counts. Only following timestamps in synthetic training
audio shift; inference audio and its clock are unchanged.

`modal/student_articulation_train.py` provides preparation and fitting commands.
Preparation requires help-train training-speaker none takes with explicit empty
event/non-error/unsupported lists, a separately enrolled authorized and verified
performed-transcript JSONL, an explicit excluded-speaker list and the existing
noise/RIR banks. It never derives a performed transcript from a planted mistake
script. Speakers appearing in a non-training split in either corpus are excluded
from both; duplicate audio and mismatched transcript/audio checksums are rejected.
Both fit and selection partitions must contain both corpora. The real schemas
and availability of these inputs remain unverified because the supervisor never
executed the inspection commands.

Speaker hashing assigns approximately 20% of training speakers to selection.
Two preregistered distributions independently draw each consonant's factor
uniformly in log duration: `[0.8, 1.25]` and `[2/3, 1.5]`. Duration draws use a
random stream separate from channel augmentation. For each distribution:

- Relative views independently change consonant cores.
- Identity views use the same cores and synthesis with factors of one.
- Tempo controls apply one uniform factor to each speech island, chosen to
  match the relative view's net sample-count change in that island. They retain
  the same protected pauses while also changing neighbouring vowel durations.

Every arm includes the original recording and two synthetic views with matched
channel/SpecAugment draws. The existing reverb, noise, gain and masking
distributions are retained. Common original/ordinary augmented views provide
the same output-preservation objective and checkpoint-selection evidence for
every arm. Source, auxiliary, audio, transcript, augmentation and cached-view
hashes record provenance.

Only a zero-initialized affine correction to the installed phone projection is
trainable; native encoder features and installed outputs are detached. Each
recording contributes mean transcript CTC over its three arm views plus mean
shipped-posterior KL over its three ordinary views. All six refits use the same
initialization, recording order, plain SGD at 0.02, 12 epochs and unit
preservation weight. Each selects the lowest equal-speaker ordinary-view
CTC-plus-preservation objective, including epoch zero. The best relative
distribution and its two matched controls are fixed from these training-only
metrics before any target decode. Preparation and refitting share a one-day
wall-time limit that fails before proceeding to another recording/update.

`modal/student_head_export.py`, adapted from the prior conflict-filtered head
experiment's generic export helper, exposes pre-quantization native features
and grafts the selected affine correction onto the installed graph. It preserves
encoder nodes/initializers, recurrent states, alphabet and 40 ms clock. Export
creates a separate bundle and new manifest, and copies/checks the existing
insertion teacher graph and external weights. Missing or truncated auxiliary
files stop export. The fitting command verifies the optimized exported graph
against cached head predictions on original selection-speaker recordings.
No SDK decoding, tracker, checker or insertion-verification rule is changed.

## Validation completed

Python compilation passed for all five added Python files. **26 synthetic tests
passed**, including exhaustive-path comparisons of CTC state and boundary
marginals, repeated occurrences, impossible alignments, diffuse-boundary refusal,
vowel/nasal protection, identity sample equality, voiced-tone pitch retention,
protected waveform copying, matched tempo length/pause retention, speaker
exclusion, authorized transcript/checksum admission, read-only equal-speaker
selection and identical-arm objective equality.

Stateful floating-point and dynamically quantized ONNX integration tests check
native-feature extraction, selected-head parity, unchanged encoder initializers,
streaming state/clock preservation, zero-head parity and complete external
teacher-asset copying. A synthetic cache-to-fit-to-export run produces all six
complete bundles, selects a relative distribution and its matched controls
without target data, and rejects corrupted cached evidence. The one-day limit
also has a fail-before-work test.

Local command, using an isolated environment under ignored `.autolab/`:

```sh
.autolab/unit-venv/bin/python -m pytest modal/test_student_articulation.py modal/test_student_head_export.py -q
```

The local environment used Python 3.13, NumPy 2.5.4, Torch 2.15.0, ONNX 1.23.2
and ONNX Runtime 1.31.1. Only small synthetic tests ran locally; no real audio
feature extraction, acoustic training or practice evaluation ran on the Mac.

## Supervisor blocker and remaining validation

All three requests through the supplied supervisor failed before execution
with `NotFoundError: No such file or directory`; each result reports zero spend:

- Run 1: available data/dependency probe.
- Run 2: retry inspecting authorized data and shipped bundle members.
- Run 3: required six-shard `bash tools/autolab_tests.sh`.

The required core/SDK suites therefore did not run. No alternative Modal route,
exam files or training files outside the authorized worktree/provider were
accessed. No real recording was enrolled/prepared, no real head was trained,
no real bundle was exported and none of the five full-session target decodes
ran. Listener/overall changes, target wins, clean-session warnings, losses and
studio behavior are all unmeasured.

After the supervisor is restored, resolve/audit the authorized input paths and
speaker exclusions. Run `prepare` with `--model`, `--help-labels`, `--transcribed`,
`--exclusions`, `--augmentation` and `--out`, then `fit` with `--prepared` and
`--out`, inside the experiment's persistent scratch directory. Inspect actual
phone-boundary uncertainty, eligible consonant durations, synthesized speech,
pause preservation and complete exported bundles. Compare the frozen selected
relative head and both controls in full-session decoding at the five sites;
labels must only select/report diagnostics, never choose the distribution,
checkpoint or inference behavior.

If fewer than three targets are won, remove the engine implementation and record
a killed outcome. If at least three are won, point untracked `.autolab/model.json`
at the complete verified relative bundle, run the required suites and request
practice scoring before deciding whether to narrow the change. Three practice
scores remain available. No fold was performed.

There is currently no `.autolab/model.json` or killed outcome. Running/scoring
the main model would not evaluate this experiment. The synthetic checks establish
algorithmic and export invariants, not the acoustic premise: CTC emission-state
boundaries are uncertain proxies for articulation boundaries, very brief phones
may have no conservative core, and a frozen encoder may already have lost the
information that a head refit needs. Tone pitch tests do not establish natural
speech quality. Preservation and unchanged warning rules do not establish clean
session safety or a population precision gain.
