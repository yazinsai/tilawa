# Closed formula slots under the listener channel, with madd and skip merging

Status: practice-scored once. Overall and checker both improve by 3. Left in place for the supervisor’s fold gate.

## Prediction

Experiment `checker-formula-slot-channel-madd`, checker, exploit, lab-v1.5 practice with listener `exp-listener-heard-word-must-be-lexical-41570706`. Preregistered checker −2 and overall −2, with upside −3 if the formula-slot decision and both riders survive together. The card is unchanged.

The intended calls are the wrong-word misses at p76 9:47:17 (بالظالمين read as بالمتقين) and p79 20:43:1 (اذهبا read as اذهب), plus removal of the duplicate seam skip at p24 26:131:2.

## Mechanism

Build a generic mushaf index of ayah-final words sharing the identical two-word letter lead. Final occurrences supply pausal phonemes. At a locked final word, with accepted visits of the two lead words, rank the slot's own greedy tokens before its labelled pause against the written word and indexed alternatives using the already folded `LISTENER_CHANNEL` and `spanLogP`. Held sounds and waqf edges are normalized identically on both sides. An alternative must be strictly more probable than the written word; a tie leaves the checker unchanged. The new candidate must then pass affirmative local posterior support, the existing doubled support margin on weak evidence, and the existing confidence and confirmation gates. No channel counts or numeric thresholds are changed.

The slot uses its own visit, including trailing tokens before the pause. A pause before the slot does not make its lead word's tail part of the slot. Support uses the normal word-and-neighbor acoustic window; the index is structural context, not an override of that window.

Reapply the connected-madd rule from `e94ff55c` and the ledger/seam merge from `d7b7106b`, preserving their code and added tests. Their merge conflicts concern the current alignment's step repairs and the returned bound-word set; both are retained. The listener forward-sum rider is absent.

## What was validated

Modal run 8 replayed the three complete phone sessions through both `runEngine` and `runTruePosition`, then repeated them with only the experiment's checker changes reversed. It used the existing session scorer and changed neither labels nor scoring code.

| Case | Before | Experiment, both arms |
| --- | --- | --- |
| p76 9:47:17 | missed | wrong word confirmed at 206.60 s |
| p79 20:43:1 | missed | wrong word confirmed at 164.08 s |
| p24 26:131:2 | false skip confirmed at 267.84 s | absent |
| p24 26:130:1 | real skip confirmed at 256.76 s | still confirmed at 256.76 s |

Session missed/false counts, identical for engine and true-position arms: p76 1/0 → 0/0; p79 2/0 → 1/0; p24 0/1 → 0/0. These are three calls won, with no calls lost in those sessions. They are case checks, not a practice-set score.

The current p76 slot is exactly `بِلمُتتَقِۦۦن`, with mean confidence 0.91071. Under the engine's held-collapse and waqf-edge normalization:

| Pausal candidate | Viterbi log P |
| --- | ---: |
| Written `بِظظَاالِمِۦۦن` | −38.38349 |
| Indexed `بِلمُتتَقِۦۦن` | −3.25847 |
| Indexed `حَلِۦۦم` | −66.21268 |
| Indexed `حَكِۦۦم` | −66.29322 |

The planted filler beats the written word by 35.12501 nats, so the card's premise kill criterion does not fire. The real streamed candidate passes the unchanged support and confidence gates.

Unit tests: 8 formula tests plus the 6 original madd tests and 8 ledger tests passed in Modal run 8. They cover soft consonant swaps, written-word controls, ties, missing text, mid-ayah and short-ayah exclusions, locks, labelled pauses before and after the slot, affirmative posterior support, confidence, and the doubled weak-evidence margin. The madd module and its tests match their original commit byte for byte; the ledger's added tests match its original patch. The final full core/SDK builds and tests passed on all six Modal shards (run 9): 715 core tests, 16 SDK tests, and one skipped SDK test. No worker RPC exception was needed on that final run.

## Practice result

One practice score, on implementation commit `749780da`. No constants were adjusted and no second score was requested.

| Wrong calls | Main line (missed + false) | Experiment (missed + false) | Change |
| --- | ---: | ---: | ---: |
| Overall | 36 (24 + 12) | 33 (22 + 11) | −3 |
| Listener | 32 (22 + 10) | 32 (22 + 10) | 0 |
| Tracker | 1 (0 + 1) | 1 (0 + 1) | 0 |
| Checker | 3 (2 + 1) | 0 (0 + 0) | −3 |

The compare verdict is better. The predicted upside holds in the aggregate: two fewer misses and one fewer false warning. The premise is validated and the built change wins all three intended calls, so it is retained for the fold gate.

## Limits

This is a closed choice over attested final fillers. Unindexed slots are unchanged. The channel remains the folded estimator with its original smoothing and training sample. A slot without accepted lead visits, a completed lock, a labelled pause, or affirmative posterior support still abstains on this path.

The connected-madd rider compares with the shortest earlier matched madd; it abstains without such a baseline. The skip rider can merge a second skip across bound refrain text; unique intervening words preserve separate skips. Both limitations are inherited unchanged.

The aggregate practice compare does not expose incident losses, reader leave-one-out, studio behavior or fully clean-session warnings. Matching net totals to the three target wins does not prove that no other incidents changed. Those remaining decisions belong to the supervisor's fold gate; this experiment was not folded by hand.
