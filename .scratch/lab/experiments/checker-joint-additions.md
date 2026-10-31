# Joint neighboring-word fit for missed additions

Status: implementation validated; practice score pending. Fold or discard belongs to the user.

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
- Parent agent will run a fresh full practice score, then report frozen/new subsets and gain concentration. No benchmark case or speaker IDs are in production behavior.
