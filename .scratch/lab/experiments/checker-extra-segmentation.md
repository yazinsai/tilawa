# Extra-word detours

Status: passes the full phone/studio and perfect-evidence practice gates; ready for user fold decision. Not folded.

## Scope and prediction

Main `da14389`, lab-v1.5 practice: 92 wrong calls = 67 misses + 25 false warnings. Before implementation, the card predicted 6 fewer checker-attributed wrong calls and 5 fewer overall wrong calls. Accept only fewer misses, no new missed case, no new false-warning incident, and recoveries across several chapters. Exam remains unopened.

The supplied 19 examples are end-to-end misses of additions that also fail with perfect evidence. There are 23 oracle-missed additions in total: four are nevertheless caught with listener evidence. Both groups were traced separately. Traces are saved in `checker-extra-segmentation-traces.json`; no case identifiers enter engine behavior.

## Diagnosis and change

The alignment can consume added speech through cheap jumps to another Quran word, then jump back. Its old addition repair only considered consecutive visits it called correct. That definition tolerated one-consonant differences, and therefore treated acoustically different detours as legitimate repeats. A real, isolated forward word also interrupted the neighboring pair.

For addition repair only, whole-word pronunciation matching now establishes intact visits. One isolated forward detour can be bypassed if it is beyond the next word and has not already been pronounced earlier in the current window. All existing fit costs, acoustic confirmation, confidence thresholds, exact observed-index coverage and protections for attempts at the following word stay unchanged.

Backward whole-word detours remain protected: adding a word that exactly repeats earlier speech is observationally indistinguishable from an intentional isolated repeat. An early prototype bypassed these too; a paused backward-word regression exposed that ambiguity before scoring, and the broader behavior was removed. Multiword excursions and restarts remain protected. Short additions that reach `uncertain` and attempted following-word substitutions remain outside this fix.

## All 19 targeted misses

The stored traces split the 19 end-to-end misses into two exact forward-word detours (chapters 2 and 9), one tolerated one-consonant detour (chapter 35), three exact backward-word detours (chapters 9 and 51), eleven cases involving an attempted following word or a fragmented return that triggers the correction/repeat protections (chapters 2, 26, 37, 44 and 54), and two short additions (chapters 11 and 38). The latter two reach an uncertain decision and remain outside this segmentation change. The three backward matches stay protected because the same observed word sequence can be an intentional repeat. This candidate targets the first three cases; it does not claim to solve all nineteen.

## Validation

- Baseline red: `extra-red-480f613-dirty-20261031T140307Z`: both original forward/backward extra-word regressions failed. The backward case was subsequently reclassified as a non-error protection for the ambiguity above.
- Focused tests: `extra-protected-controls-480f613-dirty-20261031T140457Z`: both shards pass, including existing correction, repeat and noise assertions.
- Initial test source (superseded before measurement): `0884670`.
- Full core tests and build: `extra-full-controls-0884670-20261031T140542Z`: all six shards pass. Four run the core suite excluding checker.test.ts; two complementary test-name groups run checker.test.ts, retaining all assertions while avoiding the established Vitest worker deadline problem.
- Temporary instrumentation and diagnostic executables were removed before the full measurement.

## Pre-measurement correction

The initial full practice run on `0884670` was invalidated and stopped before any score or partial case outcome was read. Static review identified that replacing the old tolerant-neighbor candidate list could discard previous repairs. The final implementation attempts all original pairs first and uses stricter candidates only in non-overlapping regions; replacements are applied in reverse source order. A noisy-neighbor regression preserves the older repair. The original predictions are unchanged.

A further adversarial repeat test (a preceding word repeated with one tolerated consonant change) passed on main but failed on `08c9b25`. The in-flight `08c9b25` practice run was stopped without reading its scores or case outcomes. Strict new candidates now preserve such counting, tolerated repeats of the preceding word; original candidates retain their behavior. Baseline control: `extra-repeat-baseline-08c9b25-dirty-20261031T141229Z`, pass. Red: `extra-repeat-variation-08c9b25-dirty-20261031T141131Z`.

## First complete measurement

Source `b11dd0c`, run `exp-checker-extra-segmentation-b11dd0c-20261031T141340Z`. Full core build and 607 tests passed (`extra-final-core-b11dd0c-20261031T141544Z`).

Phone: 67 → 64 misses, 25 → 25 warnings; three recovered additions in chapters 2, 9 and 35, zero new misses and zero new warning incidents. Studio: 48 → 45 misses, 13 → 14 warnings; the same three recoveries, no new misses, one new added-word warning at 37:102:16. Perfect-evidence checker: 50 → 46 misses, 6 → 7 warnings. This is a failed safety gate despite the CLI's aggregate “better” verdict. Paired incidents are in `checker-extra-segmentation-comparison-v1.json`.

## Regression diagnosis and revision

The new warning was a real planted addition rediscovered about 12 seconds late after a second addition caused the older window to realign. It was not a new sound in clean speech, but it is still an unacceptable late alarm. New strict repairs now have to establish an addition while the following word is current or just completed (at most one later intact visit). An already open addition retains evidence through its normal confirmation delay. Original repair candidates are unchanged. This temporal guard is defined by streaming word progress, not case IDs, seconds from the scoring matcher or confidence tuning. The failed measurement remains recorded above; the original prediction is unchanged.

Revision validation: `extra-timely-controls-b2fe98d-20261031T142251Z` passes the 13 addition and six noise controls plus build. `extra-timely-core-b2fe98d-20261031T142332Z` passes build and all 607 core tests across six complementary shards. Full revised practice measurement is pending.

## Revised complete measurement

`exp-checker-extra-segmentation-b2fe98d-20261031T142310Z`: 100 unchanged practice sessions, phone 67 → 64 misses and 25 → 25 false warnings; studio 48 → 45 misses and 13 → 13 warnings. In both variants, exactly three recovered added-word mistakes (2:283:1, 9:61:1, 35:22:1), zero new missed cases, zero new warning incidents and no removed warnings. Full paired comparison: `checker-extra-segmentation-comparison.json`.

Perfect-evidence checker: 50 → 46 misses, 6 → 5 warnings. The additional oracle recovery is 37:151:1, already caught with listener evidence on main. The late warning introduced by the first candidate is absent. Predictions (-6 checker-attributed/-5 overall) were not reached; measured attribution and overall gains are both -3. Exam remains unmeasured.

The independent paired oracle audit (`extra-oracle-pairs-b2fe98d-dirty-20261031T143318Z`, all 100 shards pass) confirms 6 → 5 false-warning incidents with **zero new incidents**. The removed warning is the old late report of the 37:151:1 addition, now detected in time. Its paired result is `checker-extra-oracle-pairs.json`; the reconstructible audit driver is `checker-oracle-pairs-driver.ts`. Temporary baseline code was removed. Production source is unchanged from measured `b2fe98d`.
