# Mined real-phone false accusations in listener fine-tuning

Status: building and validating the fixed recipe. Experiment card unchanged.

## Prediction

Listener −2 and overall −2 on practice: three of seven target false warnings
removed, approximately one caught swap lost. Continue to official practice
scoring only if the full-session int8 listener recovers at least three target
slots as the written word with zero hard edits under the existing soft costs.
No training setting is selected against practice.

## Mechanism

The training-only SDK miner runs the shipped listener and unchanged free
tracker/checker on help-train's training speakers. It replays the final evidence,
including the shipped context-free gap replacements, before collecting final
confirmed wrong-word accusations. Negative supervision is limited to phone
recordings and correctly labelled slots. Labelled errors, repeats, multiword
skips, audit exclusions, unscored recordings and unsupported pronunciation
labels cannot become negative slots. Explicitly labelled substitutions from
every eligible training device are protected positives.

Corrected ayah transcripts preserve original word ownership through omissions,
substitutions and repetitions. They are phonetized with ref-v2's existing
profile and word-offset estimator. Duration runs collapse as in shipped CTC,
retaining shared word ownership. A frozen shipped-parent full streaming pass
force-aligns the corrected transcript. Selected words receive local CTC over
their aligned frames with five 40 ms frames (200 ms) of margin on both sides.
A multiword graph accusation supervises every written word it names, rather
than just the first word. Impossible alignments and unsupported phone symbols
are rejected.

The fixed fine-tune starts from spliced-clean, uses AdamW at 0.0003 for 3,000
steps and KL-to-shipped weight 10. Each step contains one selected learner item
and four clean anchors: one other clean learner item when available and three
original clean-match items, otherwise four original items. Original items use
the existing split, gold exclusion, noise/RIR/gain and SpecAugment filters.
The same augmentation chain applies to learner audio. Both models see the same
augmented features. Training uses the shipped parallel causal chunk mask at
the deployment settings (16/128). Full streaming inference supplies alignment
and the deployment check. The architecture and dynamic MatMul int8 exporter
are unchanged.

Original anchors now read only FLAC audio columns after selecting the same
20-hour clean-match/train pool (seed zero, 30-second cap, shipped holdout
filter). Teacher arrays and transcripts are unused by the KL objective. Audio
uses the same noise/RIR/gain/SpecAugment functions with checkpointed randomness.
The pool cache records its source and exclusion hashes.

CTC is computed only inside selected windows. KL covers every other frame and
all frames of each clean anchor. This makes the CTC gradient zero outside the
windows and removes the anchor inside them. Shared weights can still change
outputs elsewhere; the anchor reduces drift but does not guarantee unchanged
strings or clean-session behavior. Tracker/checker implementation and costs
are unchanged.

## Validation so far

Supervisor run 1: 489 training recordings, including 228 none, 60 word skips,
48 substitutions, 44 ayah skips, 41 vowel errors, 36 repeats and 32 tajweed
errors. Devices include 284 phones and 131 laptops. Spoken substitution words
are available in `observed`; pronunciation labels do not provide complete
corrected transcripts.

Run 2 passed core/SDK builds and three SDK mining-admission tests. Its initial
raw-session pass admitted 225 phone recordings from 65 speakers, with 19
labelled substitutions and three confirmed false accusations. Two accuse nasal
endings (`عذابا` and `إذن`); the third names a multiword span. That corpus is
substantially narrower evidence than the hypothesis suggests. Run 2 predates
the final-evidence replay, admission of non-phone protected positives and
multiword-span correction; it is an implementation check, not the complete
recipe or a practice score.

Run 4 completed the final-evidence mining pass: 241 recordings from 72 training
speakers (225 phone, 11 laptop, five headset), 35 labelled substitutions and
the same three false accusations. Non-phone recordings contribute protected
substitutions only. The final miner and three admission tests passed both
package builds. Run 5 passed all six core/SDK validation shards: 734 core tests
and 19 SDK tests, plus one existing SDK test skipped. The unchanged wrapper
accepted the known Vitest RPC timeout after every test in that shard passed.

The supervisor's training image lacks the reference-building phonemizer;
the entry point installs `quran-transcript==0.6.4`, matching `build_index.py`.
Checkpoints preserve optimizer and augmentation-generator state so a timed run
can resume to the same preregistered 3,000 steps. No partial checkpoint is
selected by looking at practice.

Run 6's full-stream alignments admitted 234 recordings and rejected seven
unsupported or unalignable labels, leaving eight correct-word negative slots
and 29 protected substitutions. Six Python tests passed. The first optimization
attempt failed before checkpointing on a non-finite loss. Run 8 passed seven
Python tests, including mixed-length EOS padding and zero loss gradient on
padding, but failed on a non-finite gradient at step three. Both are numerical
implementation failures, not target measurements. Anomaly tracing is being
used to identify the backward operation before proceeding with the fixed
recipe.

Run 7 replayed each of the seven stored baseline sessions through the passive
slot inspector. Its temporary diagnostic checker produced byte-identical
events to the original checker. Every baseline target still has at least one
hard edit (counts 2, 2, 5, 2, 2, 1, 1); none qualifies as recovered. This
validates the continuation check on the shipped model, not an adapted result.

## Limits

Scripted labels are supervision, not an independent listening review. The
shipped acoustic model supplies correlated timing evidence. No reviewed tlog
word-correctness asset is available in the mounted corpus or this worktree, so
unreviewed tlog clips are not used as correction labels. Original clean-match
training items are admissible KL anchors only. Held-out learners, practice and
studio audio do not enter training. The seven target sessions are for the
continuation check only; a full official score is required if that check passes.
