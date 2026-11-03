# Inner waqf seam, proclitic onset

Status: practice-scored once. Left for the fold gate.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. 3 of 3 checker added-word misses won: checker −3, overall −3.

## Mechanism

The inner-waqf seam is the killed run-end seam, plus one visit rule on that path only.

A boundary carries a mushaf pause sign (ۖ ۗ ۚ, 4257 marks, the word the sign follows). The previous word may be its pausal form. The pause is the blank run after that word, or after a waqf-tail phone sitting just outside the visit, measured from the end of that token's own argmax run. The run still has to be blank-owned (`log 0.5`) and at least `minGapFrames` long. The spelled-surplus test, the ±3 back-run guard and the non-Quran phrase test are unchanged.

On that path only, the next word's visit counts as complete when the phones from its stem on match in order and the only missing reference phone is a word-initial proclitic (و ف ب ك ل in the corpus segmentation) whose heard residue is blank or a vowel. Phones glued onto the front of that visit stay in the surplus. Ayah seams, unmarked boundaries, and the ordinary wrong-word and omission paths are unchanged. No new threshold.

## What was checked before the score

Engine and true position, phone, the three targets:

| Target | Engine | True position | Heard |
| --- | --- | --- | --- |
| `lab-v1.4-p15` 38:4:6 | hit | hit | `ءِننن` |
| `lab-v1.5-p50` 37:113:5 | hit | hit | `وَقِفُۥۥهُم` |
| `lab-v1.5-p77` 9:65:8 | hit | hit | `فِۦۦهَاا` |

38:4:6's checker visit is `نننَقَاالَ`, not `َقَاالَ`: the alignment glues the tail of إن onto وقال. The stem `َقَاالَ` still matches once و is set aside, and those ن's go back into the surplus, which spells إن. 9:65:8's ڇ sits outside the pausal visit of ونلعب; the silence is after that phone. 37:113:5 was already a complete next-word visit and stays one.

Unit tests: proclitic gap confirms, a complete next word confirms, a short gap stays quiet, one held frame of the previous class still counts as silence, a later non-blank frame blocks, a consonant in the proclitic vowel's frames blocks, a missing stem consonant does not, a neighbour and an unmarked boundary and a non-pausal left word stay quiet. Full core and sdk tests passed (one shard hit Vitest's RPC deadline after every test passed).

## Measured practice result

One practice score, worktree `9f523ab2-dirty`. No second score.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 46 (31 missed + 15 false) | 43 (28 missed + 15 false) | −3 |
| checker | 8 (7 missed + 1 false) | 3 (2 missed + 1 false) | −5 |
| listener | 34 (22 missed + 12 false) | 36 (24 missed + 12 false) | +2 |
| tracker | 4 (2 missed + 2 false) | 4 (2 missed + 2 false) | +0 |

False warnings are unchanged. The compare calls the listener move noise. Studio, clean sessions, and leave-one-reader are not in this compare.

## Limits

The stem match is exact after held-sound collapse, so a vowel substituted on the stem does not fire. A consonant whose argmax sits on the proclitic vowel's frames blocks the cut, even when the greedy token is that vowel. The two extra listener misses are not narrowed here.
