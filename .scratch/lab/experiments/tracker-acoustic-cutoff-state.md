# Acoustic interruption versus incomplete lexical matching

Status: implementation drafted; blocked on supervisor execution. No fitted model or measured target result.

## Prediction

Tracker −1, overall 0. The target is the phone false skipped-passage warning at
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. Correct localization may expose
the documented nearby listener false warning. The experiment card is unchanged.

## Mechanism

The prior pause-gate audit reported that the restart followed a source visit covering only
2 of 11 phones. That establishes an incomplete text match, not an acoustic interruption.
This worktree's baseline has neither the experimental pause gate nor its cutoff state;
it also strips posterior rows in `trackerInput`. The draft therefore supplies the gate
and keeps its acoustic state separate from lexical completeness.

`WordAttemptState` represents lexical completeness and a distinct acoustic judgment:
interrupted, continuous, or unavailable. Only positive acoustic evidence satisfies the
cutoff exception. The draft preserves the checker's existing pause criterion: the same
300 ms gap, sustained-source-spike trimming, blank ownership, and labelled pause tokens.
Local jump arcs onto mid-ayah words require that pause or positive interruption evidence.
The next spoken word and single-word-deletion continuation remain available, including
when acoustic rows are missing. Ayah-start entries retain their existing eligibility.
Candidate costs, jump prices, confirmation threshold, commit margin and checker rules
are unchanged.

`trackerInput` preserves a separate interruption-posterior channel so acquisition retains
its decoded ranking. The tracker retains a bounded posterior history and never fills
missing intervals with invented rows. Thirteen causal seam features summarize the previous
spike's tail, intervening rows and current spike's head: gap length, blank mass, entropy,
nonblank peaks, and distribution overlap. Reference position, word identity, lexical
coverage, recording identity and speaker identity are not scorer features.

## Training construction, frozen before target inspection

`packages/sdk/src/training/interruption-examples.ts` selects only help-train speakers'
none-labelled takes without scripted events or repeats, excluding every speaker with
any non-training label. The shipped primary streaming listener decodes each recording.
Transcript-constrained phone alignments locate candidate word spans; tracker visits do
not supply supervision. Require 80% exact phone coverage, a source span of at least
400 ms, and a gap shorter than the existing 300 ms pause rule. Use at most the first
six qualifying transitions per recording, without selecting locations by practice fit.

Natural complete speech supplies negative examples at its observed phone seams. Additional
matched controls retain the full source through the next word's onset, including its
hidden acoustic tail. Positive examples truncate the source at one third or two thirds
of its aligned span and resume on the next spoken word. Complete and truncated examples
use the same concatenation operation, resumed audio and inserted gap (the original gap,
capped at 80 ms). Each constructed audio stream is decoded anew; posterior arrays are
not cut or fabricated. Output contains derived feature records and audio/model-manifest
hashes, leaving source recordings, labels and stored evidence untouched.

`packages/core/src/training/interruption_fit.py` orders speakers by SHA-256 and assigns
five groups round-robin. Groups 2–4 supply fitted weights and normalization statistics;
speaker-fold cross-validation within those groups selects L2 strength from 0.01, 0.1,
and 1.0. Group 1 alone fits temperature and an interruption threshold above the 99th
percentile of its complete controls, with a probability floor of 0.5. Group 0 is final
validation and cannot influence the model. No refit combines calibration or validation
speakers into the weights. Tests explicitly alter validation examples and require the
generated model to remain identical.

These synthetic truncations are training supervision, not independent listening evidence
that a natural target was interrupted. Transcript-derived span uncertainty, the imbalance
between natural controls and constructed interruptions, and generalization from artificial
truncations must be assessed on the reserved speakers before interpreting the target.

## Validation and limits

Read `AGENTS.md`, the card and audio-review instructions. Reused the related boundary
experiment's posterior-feature design, with interruption supervision replacing its
word-boundary labels. No scorer, lab tooling, card, manifest, test data or existing
evidence was edited.

Added core unit cases for independent lexical/acoustic states, genuine interruption,
unavailable evidence, pause eligibility, posterior summaries, adjacent spikes without
invented blanks, invalid models, separate tracker input, gated restart arcs, continued
next-word progress, and restoration after rewind. Added Python checks for training
partition isolation and rejection of non-training examples.

Lightweight local source-module assertions passed for state separation, posterior
features, local restart eligibility and ordinary continuation. A synthetic tracker replay
was identical with decoded-only input and with posterior-bearing input routed through
the separate channel. Python syntax compilation and `git diff --check` passed. The local
Python mathematical tests could not import NumPy: its installed binary is x86_64 while
this Python requires arm64. No environment changes or heavy local training were attempted.

Supervisor run 1 (training-data/target-file discovery) and run 3 (minimal data-access
retry) both returned `NotFoundError: No such file or directory`, without any execution
log, labels or evidence. Run 2, requesting `bash tools/autolab_tests.sh` on six shards,
returned the same infrastructure error before producing a build or test result. Run 4,
requesting the Python training tests in the training image, returned that same error.
All four requests finished without a usable command log or output; no package build,
Vitest run, Python mathematical test, training extraction or target probe ran remotely.

No target audio or posterior trajectory has been obtained, no model has been fitted,
and no target wins have been measured. No practice score was requested (0 of 3).
`TrackerOptions.interruptionModel` remains unconfigured by default; this is a draft,
not a completed or scored experiment. A neutral or invented model would not test the card.

## Remaining authorized work

Recover supervisor execution, rerun package and Python tests, and extract help-train
examples on the node image. Fit the frozen model on the training image using the shared
experiment scratch directory, inspect its reserved-speaker validation, and install the
generated model as the product default only when that validation supports interruption
classification. Verify the listener/model provenance matches the evidence under review.

Then evaluate the exact source-to-restart target seam with the frozen scorer. Positive
interruption evidence triggers the card's premise kill. Otherwise replay the changed
engine and count actual target warnings removed. The supplied fewer-than-three-wins
rule triggers a kill even if this single target is recovered; revert engine changes,
retain the result write-up, and write the specified killed outcome. If three or more
targets are won, request a practice score before any gate decision. Do not tune against
practice or hide the already predicted listener warning.

Infrastructure failure is not evidence for or against the premise and must not be
reported as zero measured wins. No killed outcome is written without that measurement.
