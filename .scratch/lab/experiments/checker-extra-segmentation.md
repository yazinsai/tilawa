# Extra-word detours

Status: first measured candidate fails the no-new-false-warning gate; regression diagnosis in progress. Not folded.

## Scope and prediction

Main `da14389`, lab-v1.5 practice: 92 wrong calls = 67 misses + 25 false warnings. Before implementation, the card predicted 6 fewer checker-attributed wrong calls and 5 fewer overall wrong calls. Accept only fewer misses, no new missed case, no new false-warning incident, and recoveries across several chapters. Exam remains unopened.

The supplied 19 examples are end-to-end misses of additions that also fail with perfect evidence. There are 23 oracle-missed additions in total: four are nevertheless caught with listener evidence. Both groups were traced separately. Traces are saved in `checker-extra-segmentation-traces.json`; no case identifiers enter engine behavior.

## Diagnosis and change

The alignment can consume added speech through cheap jumps to another Quran word, then jump back. Its old addition repair only considered consecutive visits it called correct. That definition tolerated one-consonant differences, and therefore treated acoustically different detours as legitimate repeats. A real, isolated forward word also interrupted the neighboring pair.

For addition repair only, whole-word pronunciation matching now establishes intact visits. One isolated forward detour can be bypassed if it is beyond the next word and has not already been pronounced earlier in the current window. All existing fit costs, acoustic confirmation, confidence thresholds, exact observed-index coverage and protections for attempts at the following word stay unchanged.

Backward whole-word detours remain protected: adding a word that exactly repeats earlier speech is observationally indistinguishable from an intentional isolated repeat. An early prototype bypassed these too; a paused backward-word regression exposed that ambiguity before scoring, and the broader behavior was removed. Multiword excursions and restarts remain protected. Short additions that reach `uncertain` and attempted following-word substitutions remain outside this fix.

## Validation

- Baseline red: `extra-red-480f613-dirty-20261031T140307Z`: both original forward/backward extra-word regressions failed. The backward case was subsequently reclassified as a non-error protection for the ambiguity above.
- Focused tests: `extra-protected-controls-480f613-dirty-20261031T140457Z`: both shards pass, including existing correction, repeat and noise assertions.
- Final source: `0884670`.
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
