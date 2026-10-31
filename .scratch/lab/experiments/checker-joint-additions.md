# Joint neighboring-word fit for missed additions

Status: practice measured; 10 fewer wrong calls with unchanged false warnings. Awaiting the user's fold or discard decision.

## Preregistered prediction

Test set lab-v1.5, practice only. Relative to the stored main-line practice result (107 wrong calls = 82 misses + 25 false warnings), expect 8 fewer misses, unchanged false warnings, 8 fewer overall wrong calls and 6 fewer checker-attributed wrong calls. Predictions were registered in `lab-experiment.json` before implementation.

## Mechanism

An inserted word can be aligned as a second, incorrect visit to the preceding word. Because that word already has a correct visit, the checker suppresses the second visit as a failed repeat; the added speech disappears. The synthetic regression inserts قَلِۦۦلَاا between two intact neighboring words and is missed by the baseline.

The experiment compares complete fits of two chronologically adjacent correct word visits, with and without an intervening insertion. It only repairs consecutive reference words, preserves correctly recited intervening visits (including single-word and multiword repeats), requires distinct consonant evidence, and uses the existing noise discounts, hard-edit margin and wrong-word tolerance. The resulting bounded path is used for both judgment and acoustic confirmation. Existing confidence and posterior-support thresholds stay unchanged.

This differs from the already tried one-sided edge re-splitting and contiguous insertion DP: both neighboring words must remain explained, and the repair can recover speech that took a detour through a wrong reference visit.

## Validation

- Baseline regression: `additions-red-12c277d-20261031T081715Z`, 1 failure and 2 passing clean/repeat controls on Modal.
- Temporary path probe: `additions-debug-12c277d-dirty-20261031T081746Z`; revealed a correct 2:7:12 visit, a second wrong visit carrying the addition, then a correct 2:8:1 visit. Debug instrumentation removed.
- A second synthetic regression after the same preceding word in a different surah is also red on baseline: `additions-cross-red-12c277d-dirty-20261031T082242Z`.
- Initial implementation falsely turned corrected attempts into additions; unchanged self-correction tests caught this. Repair now preserves a wrong visit to the following word and an already open wrong-word attempt.
- Independent review found a possible loss of trailing insertion steps; visit path bounds now cover only evidence represented by the reconstructed word fit. An adjacent-addition control checks that existing second evidence is preserved.
- Very long held-m speech is already misclassified by the baseline inner-insertion rule; the experiment does not change that rule. The ordinary held-sound control remains quiet. Short noisy additions and multiple neighboring insertions can remain missed; they are not newly counted as gains.
- Final validation: `additions-final-controls-12c277d-dirty-20261031T082240Z`, all three Modal shards exited 0. TypeScript build and 59 tests passed (8 focused, 6 noise, 45 existing checker). Full checker assertions were split into complementary test-name groups because a single worker can exceed Vitest's fixed 60-second RPC deadline on Modal; assertions were unchanged.
- A further static review found the leading-edge analogue: a visit can begin with a deletion and then inserted evidence before its first diagonal match. A repair now requires its replaced steps to consume exactly every observed index in the reconstructed range, once and in order. Otherwise it retains the original path. This invariant covers both edges without changing confidence thresholds.
- Full practice run `exp-checker-joint-additions-2213713-20261031T082400Z` (Modal app `ap-5fKVDEQM5nMafITIUwlGpE`) was aborted after that static review, before any score or partial case results were read. It is invalidated. The original hypothesis and prediction remain unchanged; the coverage guard is a correctness fix, not measured tuning.
- Coverage-guard validation: `additions-exact-coverage-2213713-dirty-20261031T082704Z`, all three Modal shards exited 0; TypeScript build and all 59 checker tests passed again. Independent review confirmed that exact ordered index coverage proves no observed evidence can be lost or duplicated by replacement.
- A temporary synthetic missing-onset/leading-vowel probe in another passage (`additions-prefix-guard-probe-2213713-dirty-20261031T082753Z`) did not reach an otherwise unsafe repair. It is not claimed as a regression test for the rare leading-edge path; the conservative guard and its exact-coverage invariant address that path. Probe instrumentation was removed.
- The parent agent ran the fresh full practice score reported below. No benchmark case or speaker IDs are in production behavior.


## Measured practice result

Scored code: `5601949`. Run: `exp-checker-joint-additions-5601949-20261031T084117Z`, compared with `main-fe4daad-20261031T062538Z` on lab-v1.5 practice only. The original prediction remains 8 fewer overall wrong calls, with 8 fewer misses, no false-warning increase and 6 fewer checker-attributed wrong calls. No implementation was tuned after this measurement.

| Phone practice measure | Baseline | Experiment | Change |
| --- | ---: | ---: | ---: |
| Wrong calls | 107 | 97 | -10 |
| Missed mistakes | 82 | 72 | -10 |
| False warnings | 25 | 25 | 0 |
| Checker-attributed wrong calls | 58 | 48 | -10 |
| Frozen 20 sessions: wrong calls | 11 | 8 | -3 |
| New 80 sessions: wrong calls | 96 | 89 | -7 |
| Fully clean sessions: false warnings | 5 | 5 | 0 |

All 10 changes are recovered added-word mistakes. There are no new misses, no added or removed false warnings, and no changes to other mistake classes. Recoveries span 9 sessions, 7 readers and 6 surahs. They include old-session `lab-v1.4-p19/51:30:1@81.17` and new-session `lab-v1.5-p17/2:8:1@73.23` and `lab-v1.5-p35/54:51:1@239.95`.

The gain survives two concentration checks: excluding surah 26 leaves 6 fewer wrong calls; excluding the reader with the largest gain leaves 7 fewer. Both the frozen and new sessions improve. Studio audio also improves: 78 to 67 wrong calls, comprising 65 to 54 misses and unchanged 13 false warnings. This supports the mechanism across these practice subsets; it does not establish performance on unseen recitation.

Verdict: measured improvement beyond the two-call noise threshold, exceeding the preregistered overall and checker-attribution predictions. False warnings on fully clean recitation remain a limitation: 5 across the 26 fully clean practice sessions. The short/noisy-addition and rare-path limitations above remain; this result does not justify relaxing the conservative guards. Folding or discarding remains the user's decision.

The result was checked against the sanitized practice score and comparison files supplied by the parent agent. The aborted `2213713` run above remains invalidated and contributed no results to this decision.
