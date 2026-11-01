# Ayah-boundary residue as an added word

Status: practice gate fails. One practice score used. No constant was changed after that score.

## Prediction

Test set lab-v1.5, practice, against main `abb918b` (89 wrong calls = 64 missed + 25 false warnings). The card predicted checker −5 and overall −4. Kill criterion: a new added-word warning at an ayah boundary on a fully clean session, or the ayah-start omissions recovered by ayah-start-omissions coming back.

## What the traces showed

Oracle evidence for the eleven word-1 added-word misses (`adhoc-42b744d-dirty-20261101T041412Z`) marked none of those phonemes as a non-Quran phrase. The alignment spends the extra word on a jump back to an earlier word, or on a first visit of the ayah's first word that the real word then corrects. Both are cheaper than an insertion, and the existing addition repairs leave that pattern alone (self-correction, or a backward repeat).

The phrase insertion cost stays at 1/64. Raising it would undo the tie-break that keeps an ayah-start omission on the written word when that word also fits آمين. The omission tests still pass.

## Change

Between the latest intact visit of the previous word and the earliest intact visit of the next ayah's first word, non-junk phonemes are residue.

- A span that covers at least half of a known phrase, with edit distance per phrase phoneme at most half the old junk ceiling, stays quiet. The phrase asset now contains صدق الله with and without العظيم alongside isti'adha, آمين, and the other supported phrases; basmala remains derived from al-Fatiha as before. The checker consumes those supplied phoneme strings without Quran-coordinate exceptions.
- A span within one hard consonant edit of the ayah's first word, the previous word, or an earlier word the alignment jumped back onto (connected or paused form) stays quiet. A jump forward is still an addition.
- Anything else of at least three phonemes and two consonants is an added word on the ayah's first word. Confirmation, confidence, and posterior support are unchanged.

A one-consonant slip of كريم against آمين fitted the old 0.3 junk ceiling (about 0.29). The phrase test now requires half the phrase and half that ceiling. That gate was set from the synthetic case, before any practice score.

## Validation

- Synthetic unrelated words before the non-target boundary 77:16:1 confirm as added words, including an addition sharing the previous word's final consonant. A one-consonant slip of the ayah's first word, a real آمين, and the phrase-asset صدق الله العظيم stay quiet. Existing ayah-start omission tests stay green.
- The paused repeat قبل before 51:18:1 was flagged, then exempted: that visit is the earlier word said again, including its isolated form. `adhoc-42b744d-dirty-20261101T043148Z` failed that assertion; `adhoc-42b744d-dirty-20261101T043505Z` passed the addition and boundary tests.
- Core shards 0–2 and the sdk suite: `adhoc-42b744d-dirty-20261101T043543Z`, exit 0. Shard 3 reported 153 passed, then Vitest's 60s worker RPC deadline. Checker tests re-run under that deadline: `adhoc-42b744d-dirty-20261101T043810Z`, exit 0 (all 49 checker tests plus 13 addition tests).
- Review correction: final boundary/phrase regressions passed in `adhoc-08df619-dirty-20261101T050242Z`; the SDK suite passed in `adhoc-08df619-dirty-20261101T050052Z`. A full core rerun reported all 611 assertions passing in each worker, then hit Vitest's known 60-second worker RPC deadline.

## Measured practice result

Scored code `d357270`. Run `exp-checker-boundary-phrase-absorption-d357270-20261101T044037Z`, baseline `main-abb918b-20261101T034859Z`. One score. Nothing was retuned.

| Phone practice | Baseline | Experiment | Change | Prediction |
| --- | ---: | ---: | ---: | ---: |
| Wrong calls | 89 | 90 | +1 | −4 |
| Missed mistakes | 64 | 58 | −6 | |
| False warnings | 25 | 32 | +7 | |
| Listener | 47 | 56 | +9 | |
| Tracker | 5 | 7 | +2 | |
| Checker | 37 (33 missed + 4 false) | 27 (23 missed + 4 false) | −10 | −5 |
| Checker, perfect inputs | 51 (46 missed + 5 false) | 41 (36 missed + 5 false) | −10 | |

Listener phoneme errors stayed 9039. Tracker false jumps stayed 37. Studio wrong calls 58 → 51 (misses 45 → 37, false warnings 13 → 14).

Phone recovered six of the eleven targets, all still word 1: 44:12:1, 2:11:1, 54:39:1, 51:20:1, 37:71:1, 37:139:1. Zero new misses, so the ayah-start omissions did not return. Four of the ten oracle recoveries never reached the phone engine: listener misses 28 → 30 and tracker misses 3 → 5. 44:10:1, 26:123:1, 37:99:1, 51:18:1, and 37:69:1 stay phone misses. 51:18:1 is the paused repeat of an earlier قبل; that case is the same sequence as an intentional repeat, and the unit test keeps it quiet.

Seven new phone warnings, every one an added word at word 1, every one charged to the listener (perfect inputs still have 5 false warnings). One is on a fully clean session, 55:45:1 in lab-v1.4-p02, and studio warns there too. That is the kill criterion. The other six are on sessions that already contain planted mistakes. None of the old warnings went away.

Fold gate on this report: fail, not a tuning diff. Overall +1 (needs −3). Leaving out the reader who gained most leaves +2. Collateral: 1 new false warning on a fully clean session, 7 incidents lost against 6 won. Studio −7. Hygiene passes.

After diff review, the benchmark-target regression coordinate and Quran-coordinate phrase construction were removed. The regressions now use an unrelated boundary, and the phrase builder owns the two صدق الله entries. This review-only revision was not rescored: the measured version had already hit the clean-session kill criterion and failed the fold gate.
