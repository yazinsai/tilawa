# Listener warning-event posterior recheck

Status: implementation draft; blocked on supervisor Modal infrastructure. Not calibrated or practice-scored. The experiment card is unchanged.

## Prediction

The card predicts one false warning removed with no predicted losses: listener −1 and overall −1. This is below the fold threshold and within the stated noise band. The registered targets are p11 at 26:106:1, p36 at 55:48:1, p55 at 26:174:8 and sob-049 at 92:13:1. The task requires at least three stable target vetoes before running a practice score.

## Implementation

The SDK constructs a bounded reference-free CTC prefix lattice on the acoustic window already used to verify a pending wrong word. Prefixes distinguish blank and nonblank endpoints, merge identical collapsed strings, and retain the highest acoustic masses. After selection, each complete retained string is rescored by the ordinary CTC forward recurrence over all the original window rows, recovering its full acoustic path sum rather than using a single path score. A retained maximum-probability path supplies token timing and confidence for replay.

The core exposes an isolated word-judgement replay. It restores the saved pre-judgement observation, open-issue, decided-word and read-ayah state before every transcription; runs the existing alignment, issue generator, lexical candidates and parallel candidates; and applies existing confidence and acoustic support rules. The original tracker position, reference window and posterior rows are fixed. The acoustic verification interval remains the primary interval. Replay state and seam caches are restored in a finally block, and replay emits no public events. The primary transcription must reproduce a wrong-word verdict before alternatives are consulted.

Each retained transcription contributes its summed acoustic probability once, either to quiet or to any warning at the occurrence. Alternative rivals and different mistake classes remain warning evidence. An unlocated replay abstains conservatively. A veto requires quiet mass to exceed warning mass in both the base and enlarged lattices. Direct live SDK judgements use this callback; the recording path also transports exact-window certificates through ordinary decoded-token metadata, binding the word, primary observed spelling and acoustic interval. Primary tokens, posterior values and tracker inputs are preserved. The checker retains its primary verdict internally and filters only the public confirmation and mistakes list, so a veto cannot alter the generator’s later ownership or merging of nearby issues. Graph-origin confirmations and duration-based madd decisions are exempt.

The draft limits are 64 prefixes / 5 labels and an enlarged 256 prefixes / 10 labels, with a 200-frame resource bound. **These are uncalibrated implementation defaults, not frozen help-train results.** The speaker-grouped help-train selection required by the card remains to be performed before target inspection or scoring. No practice data selected these limits.

## Validation completed

Local Node smoke checks exercised only small synthetic computations and existing synthetic checker fixtures:

- Exhaustive enumeration of all 27 raw paths over three frames and a three-class inventory agreed with all nine collapsed-string probabilities; total mass was one. Blank-separated repeated phones remain distinct.
- Several individually weaker quiet strings pooled to 0.55 against a 0.45 warning-producing maximum-probability string.
- Repeated divergent judgement replays preserved the complete primary checker event sequence and reproduced the primary wrong word.
- Exact-window certificates suppressed only their matching confirmation; a shifted window and an enlarged quiet probability of exactly 0.5 retained it.

Unit tests were added for these behaviours, different warning classes, discarded lattice mass, resource-bound abstention and loss of the quiet majority under enlargement. Full TypeScript builds and Vitest runs have not executed successfully.

## Infrastructure blocker and remaining work

Supervisor requests 1 and 3 were read-only environment/data checks. Request 2 was the required six-shard `bash tools/autolab_tests.sh`. All three returned exactly `NotFoundError: No such file or directory`, with zero compute spend, before command execution and without shard logs. No authorized help-train recordings or practice evidence could be inspected. This differs from a model or test failure inside a container; the required remote commands never started.

No target posterior has been measured. No target-win count, acoustic kill conclusion or practice score is claimed. All three practice-score opportunities remain unused. The implementation is committed as a reviewable draft, not a validated experiment ready to fold.

After the supervisor's input/storage bundle is restored: run the full build/test harness, freeze lattice limits on speaker-grouped help-train training-speaker partitions, inspect all four registered targets with the primary verdict reproduction guard, and verify ordinary certificate serialization and primary tracker equality on the actual target streams. Apply the task's target-win stopping rule, then score if at least three targets are won.

## Limits

The posterior is conditional on retained complete strings; omitted strings can change a majority even when one enlargement is stable. Retained mass is reported separately. One representative timing per string approximates timing-dependent word judgements rather than marginalizing those judgements over every raw CTC path. Recognition errors can be correlated across all alternatives. Replay reruns alignment within the same fixed reference and acoustic window, so it can still redistribute phones among word visits inside that window. CPU cost, speaker transfer, planted-swap preservation and aggregate wrong-call gain remain unmeasured.

No scorer, lab tooling, supplied evidence, manifest, test data or experiment-card changes were made. No fold was attempted.
