# Unique anchors for refrain skips

Status: oracle arm on the six target sessions recovers five skips and drops the misplaced seams. 26:109 stays a miss. Practice score not folded here; the supervisor scores the branch.

## Prediction

Skip detection anchored only on words whose phonemes occur once in the surah, with repeated refrain words floated onto the copy that keeps recitation order. Expected on practice: checker −4, overall −4. The six checker `skipped_passage` misses sit on word 1 beside a refrain that repeats in the surah (26:103, 26:108, 26:131, 55:28/30, 77:34).

## Step 0 (baseline oracle arm)

Same arm the scorer uses: perfect phonemes, oracle positions, default checker policy. Baseline seams:

| session | target | baseline |
|---|---|---|
| lab-v1.5-p24 | 26:103:1 | miss, false `skipped_passage` at 26:102:7 (المؤمنين / مؤمنين) |
| lab-v1.5-p73 | 26:103:1 | miss, false `skipped_passage` at 26:102:7 |
| lab-v1.5-p24 | 26:130:1 | miss, false `skipped_passage` at 26:131:2 (inside فاتقوا الله وأطيعون) |
| lab-v1.4-p10 | 26:109:1 | miss, no seam. Later phonemes of 26:108:3 are bound back to 26:103:8 |
| lab-v1.5-p15 | 55:29:1 | miss, no false warning |
| lab-v1.5-p16 | 77:33:1 | miss, false `wrong_word` at 77:33:1 |

Five of the six are a refrain copy bound so the skip either vanishes or starts one word off. That clears the kill line (need at least three). 26:109 is the same bind, and the repair below does not catch it.

## Change

`duplicateSpans` marks a word unique when its collapsed phonemes occur once in its surah, and a refrain when it sits in a 3-word sequence that occurs again. Skip spans are taken only between heard unique words. Duplicate words between those anchors are laid on the copy that stays in order. A run is emitted only when something in the span was not heard, so saying a refrain twice, with the unique words on either side still contiguous, stays quiet.

The alignment window commits before both anchors of a refrain skip are in it, so counted visits are kept in order past the commit. A return to words that were actually skipped rewrites the trail from there. A visit to a word already on the trail is the other refrain copy and does not.

A wrong word outside the gap does not cancel the skip. A seam the alignment places on non-unique text between the gap and the next unique anchor (26:102:7, 26:131:2) is dropped.

## Oracle arm after the change

| session | baseline misses / false warnings | after |
|---|---|---|
| p24 | 3 / 2 | 1 / 0 (26:98 unchanged; 26:103 and 26:130 caught; both false seams gone) |
| p73 | 3 / 1 | 2 / 0 (26:98 and 26:123 unchanged; 26:103 caught; 26:102:7 gone) |
| p10 | 1 / 0 | 1 / 0 (26:109 still missed) |
| p15 | 1 / 0 | 0 / 0 |
| p16 | 1 / 1 | 0 / 0 |

No new miss and no new false warning on these five sessions. Net on the oracle arm: five skips caught, four false warnings removed.

## Limits

26:109 is still missed. The aligner binds 26:108:3 back onto 26:103:8, and treating that as a return deletes the unique anchor in front of the skip. Keeping that anchor reopened a false omission at 26:103:8, so the bind is left to the alignment.

Phone and studio overall are not in this note. The fold gate is the supervisor's practice score. A repeated refrain with nothing unique missing stays quiet in the unit test; a clean session that only repeats a refrain is the gate's check, not something this oracle slice contains.
