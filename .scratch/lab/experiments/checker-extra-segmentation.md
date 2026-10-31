# Extra-word detours

Status: full practice measurement pending; not folded.

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
