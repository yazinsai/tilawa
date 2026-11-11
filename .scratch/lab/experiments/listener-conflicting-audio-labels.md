# Contradictory-audio phone-head refit

Status: implementation retained; corpus audit and model evaluation remain
unvalidated. Supervisor execution has not produced a training result. No real
candidate bundle or practice result exists. Infrastructure failure does not
establish a premise kill. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −1 and overall −1. The proposed cause is incompatible complete
phone targets assigned to copies of the same performed recording, rather than
item-pairing errors, annotation timing, spelling differences or lawful alternate
pronunciations. The forecast is below the fold threshold; the task requires at
least three registered target recoveries before practice scoring.

`modal/student_source_families.py` identifies exact decoded waveforms and
verifies delayed/gain-scaled re-encodings. The refit uses at least 99% temporal
coverage, at least 0.995 correlation globally and in every supported 20 ms block,
and at most 0.1 relative residual error. Unmatched edges must contain only
silence; a truncated audible onset or ending cannot become a complete-recording
identity certificate. A full-recording energy envelope only screens candidates.
Complete-link grouping prevents weak matches from chaining distinct performances.
Silence alone cannot establish identity. A separate pairwise audit rejects any
verified cross-partition alias, even if complete-link grouping separates it.

`modal/student_target_conflicts.py` compares complete targets after the existing
`per_breakdown.canon` representation equivalences: repeated runs collapse, ikhfa
nasal spelling becomes plain nasal spelling, and qalqala markers disappear.
Direct connected/isolated/plain-onset renderings of the same generic reference
word are also admitted. The comparison consumes both entire strings, with no
free edge edits, acoustic confusable-phone costs or transitive closure between
different words. Timing and orthographic annotations do not enter the comparison.
Every pair of copies in a verified family is checked. If any pair is incompatible,
the entire family is excluded from CTC, teacher distillation and shipped-output
preservation, including every copy and augmentation; no label is chosen as correct.

`modal/student_conflicting_head.py` admits original clean-match training records
with speaker-cluster provenance and help-train's scored none-labelled training
takes without events, repeats or unsupported annotations. It bans original
clusters appearing outside training or in the gold exclusion list, and bans help
speakers with held-out/excluded entries. Recordings are limited to 30 seconds.
A deterministic subset of at most 4,096 originals is selected before acoustic
comparisons; this is a bounded refit population, not a census of all supervision.
A fifth of eligible training speakers per source is reserved before fitting.
Release-held-out speakers and practice sessions are never used for fitting or
checkpoint selection.

Preparation captures the frozen shipped ONNX encoder's deployment-streaming
features and states. Integer-head features retain the original activation
rounding. Feature capture must preserve all original output/cache tensors, and
reconstructed logits must reproduce shipped log probabilities. Final padding
rows beyond the SDK's real output length are removed. Each recording has an
ordinary view and two views from the existing student augmentation helpers.
Original records retain their spliced CTC targets and renormalized, frame-mapped
source teacher top-five distributions. None-labelled help takes use their
canonical CTC targets and shipped-output preservation, with no invented teacher.
Ordinary speech and in-recording pauses provide preservation examples.

Both arms start at the same shipped linear head and use length-normalized CTC,
teacher cross-entropy and shipped-output KL, the same minibatch order, SGD
settings and optimizer-call count. Fixed population weights preserve original
relative recording/view mass; the filtered population renormalizes only globally
after excluding contradictory families, never within individual minibatches.
The encoder remains frozen. Both arms use the same noncontradictory validation
population, averaging families within speakers. The filtered validation minimum
selects an epoch; the original-supervision control uses that exact epoch and
update count. No incident result selects a checkpoint.

Export replaces only classifier constants. Encoder nodes, other tensors,
inventory, frame timing, streaming caches, tracker, checker and warning rules
remain unchanged. Both arms are exported and their state parity is verified.
Export requires and checks every SDK-declared auxiliary asset, including the
inherited insertion teacher and external weights. Missing or corrupt assets
abort export. The existing SDK model override can consume the resulting bundle;
no additional inference rule is introduced.

## Validation

All 21 acoustic identity, complete-target, synthetic ONNX export and matched-refit
tests passed in isolated local environments under ignored `.autolab/`:

- Exact copies and registered gain/delay copies group; separate performances,
  altered speech, missing audible edges and silence do not establish identity.
  Complete-link grouping and the cross-partition pairwise alias audit are checked.
- Full target comparison distinguishes lawful representation/reference variants
  from incompatible consonants/vowels, avoids transitive closure and prefix-only
  acceptance, ignores annotations and excludes every copy/view of a contradiction.
- Float and integer feature capture reproduces original head outputs. Head
  replacement preserves graph nodes and streaming states, and cached streams
  trim final padding to the deployment output count.
- Export rejects missing/corrupt inherited assets, verifies state parity, and
  produces both complete synthetic bundles with the original inference contract.
- A tiny fit with incompatible targets and nonzero teacher supervision produces
  different selected heads. Instrumented loss calls confirm that the filtered
  arm receives no transcript, teacher or preservation loss from any contradictory
  copy. Both arms retain the same selected epoch, minibatches and update count.

Python compilation and whitespace checks passed. These generated fixtures
contain no recitation or practice audio and establish no target recovery. Logs
are retained as `.autolab/conflicting-head-identity-parity.log` and
`.autolab/conflicting-head-matched-fit.log`. Numerical-library startup was slow;
the completed assertions passed after loading, without an import workaround in
the implementation.

Supervisor Modal requests 1–4 all returned `NotFoundError: No such file or
directory` before container execution, with no command logs and no measured
compute. They included training-data inventory, the required six-shard core/SDK
harness, and two new-unit-test/corpus-audit requests, one using the training image.
These failures do not establish that core/SDK tests pass or that contradictions
are absent.

## Limits and continuation

No real corpus inventory, verified contradiction count, encoder feature cache,
head fit, training-speaker selection, complete export, paired full-session target
replay or practice gain/loss has been measured. The acoustic thresholds have
synthetic checks but no measured false-join rate or recall on authorized speech.
Complete-link grouping and silence-edge checks are conservative and may miss
true codec copies. Original-input schema, training-image dependencies and the
installed auxiliary bundle still require verification in Modal.

The existing CTC target representation collapses held consonant runs as well as
vowel runs. This experiment cannot diagnose duration or gemination contradictions
that are already erased by that representation. Direct same-word alternatives
are checked in the complete phone strings; ambiguous renderings may conceal a
conflict. Conversely, acoustic verification must be calibrated on training-only
known copies and distinct performances before exclusion is trusted. Absence of
contradictions in this bounded subset would not prove their absence corpus-wide.

Once supervisor execution works, verify input inventory and calibrate acoustic
identity on authorized training speakers, then run `audit`, `prepare`, `fit` and
`export` in a fresh persistent directory under
`/scratch/listener-conflicting-audio-labels`. The training image needs ONNX and
ONNX Runtime for extraction/export in addition to its student dependencies.
Inspect the frozen selection and complete bundle reports, then compare both
training-selected heads in paired full-session replay of the registered targets.
Only a candidate with at least three target wins proceeds to the authorized
practice score; losses are assessed by the gate rather than an unsupported
local safety claim.

Target wins remain unknown, not zero. All three practice-score uses remain
available. No `.autolab/model.json` is declared because there is no complete real
bundle. No killed outcome is written because infrastructure failure does not
test the premise. The retained utility supports continuation but does not yet
change what the scored engine hears. No scorer, lab tooling, supplied data,
evidence or manifest was changed, and no fold was performed.
