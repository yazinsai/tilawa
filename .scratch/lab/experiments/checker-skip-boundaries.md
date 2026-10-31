# Compare skip placements against both spoken boundaries

Status: implemented and validated; practice scoring pending. Fold or discard belongs to the user.

## Prediction recorded before scoring

`checker-skip-boundaries`, checker component, lab-v1.5 practice only. The card predicts three fewer misses and three fewer false warnings: six fewer checker-attributed and overall wrong calls. Compare with main-line practice run `main-fe4daad-20261031T062538Z` (107 wrong calls: 82 misses plus 25 false warnings). The current main-line engine source is unchanged from that run.

The clearest targets are the neighboring-word skip warnings in new p24 and p73. Six potential wrong calls are concentrated in these two sessions; a gain there alone does not establish broad generalization. Report frozen20/new80, clean controls, and gain concentration separately.

## Mechanism and implementation

A gap may be placed on a similar word on the wrong side of a skip. The existing tie rule only compares exact consonant skeletons. An article-bearing word and its bare counterpart are not equal skeletons, although the heard speech can distinguish them.

For passage skips of at least three words, consider shifting the gap by one word left or right. Require correct visits outside both candidate placements and preserve their temporal order. Score all candidates against the same complete heard span and the same outer reference boundaries, deleting only the candidate gap. Fully consume both written flanks and heard evidence, using the existing sound-edit costs. Change placement only for a strictly lower cost; equal fits preserve the existing decision. Candidates remain inside the judged/recited span.

The older exact-repeat tie behavior remains. No article stripping, forced ayah-start placement, case lookup, confidence change, listener/tracker change, scorer change, or test-set change. Single-word omissions retain their existing behavior.

This deliberately tests adjacent boundary drift. The more complex p24 skip at26:130 may remain unresolved; it is not a reason to tune after the first full score.

## Validation

- Baseline red run: `skip-boundary-red-2a57ee0-20261031T081717Z`; one of four assertions failed: the synthetic26:103 skip was incorrectly reported at26:102:7.
- New focused suite covers that failure, a second article-difference family in surah36, an unambiguous within-ayah skip, a repeat control, clean recitation, and a repeated short-word skip boundary.
- Existing checker and noise suites cover self-corrections, real skips, declared/free modes, acoustic confirmation and clean recordings.
- TS build and all57 assertions passed in early validation runs, but the unchanged checker file exceeded Vitest3.2.7's fixed60-second worker RPC deadline. Those runs are not recorded as clean passes.
- Final validation runs unchanged assertions in complementary name selections to keep each worker below that deadline. No dependency or assertion patches are used.
- Final validation: `skip-boundary-validation-2a57ee0-dirty-20261031T082158Z`, exit0 in69.3s. TS build passed; focused/noise12 passed, then complementary checker selections34+11 passed (all45 checker tests). Total57 assertions, no runner errors.

## Pre-score review correction

The existing exact-repeat rule can move the gap before the new acoustic comparison runs. Its original `after` visit can then lie inside the relocated gap. The right-shift candidate now re-reads the visit immediately after the current gap end, preventing a candidate from accidentally shortening the skip.

Added a full-span regression for the repeated ولا boundary in surah109 at chunk sizes2,8,13. It asserts the entire expected skipped span as well as its location and class. Validation `skip-boundary-relocated-validation-57ef12b-dirty-20261031T082620Z` passed the TS build and22 affected assertions (focused/noise15, existing skip7), exit0 in16.7s. The previously validated full checker coverage is unchanged. No practice measurement has occurred yet and the original prediction remains unchanged.
