# Bounded lexical consonant candidates

Status: implemented; practice evaluation pending. Leave open for the user's fold/discard decision.

## Preregistered prediction

Registered with `tools/lab.py new checker-lexical-consonants` before implementation, based on main `b5bcc95`.
Prediction unchanged: four fewer misses, no additional false warnings; checker attribution −4 and overall −4.
Verified ledger baseline: `exp-checker-joint-additions-rebased-27ddced-20261031T110941Z`, practice only,
97 wrong calls = 72 misses + 25 false warnings, checker attribution 48. Expected result: 93 = 68 + 25, attribution 44.

| Practice slice | Baseline phone misses / warnings | Baseline studio misses / warnings |
| --- | ---: | ---: |
| All 100 | 72 / 25 | 54 / 13 |
| Old 20 | 8 / 0 | 7 / 1 |
| New 80 | 64 / 25 | 47 / 12 |
| Clean 26 | 0 / 5 | 0 / 1 |

## Implementation

The existing alignment, `wrong(v)`, hard-edit threshold, and confidence/acoustic margins are unchanged.
A separate path requires a complete reference-vocabulary alternative bounded by two complete, correctly matched,
chronological neighboring visits. It rejects incomplete coverage, jumps, junk, overlapping additions, and ambiguous
equal-sound boundaries. Actual complete correct visits retain repeat and self-correction protection; the old soft
`right` classification alone cannot suppress a candidate.

The reference index retains each word's connected and isolated forms. The lexical vocabulary uses only those
forms and the existing held-sound collapse. A consonant-only weighted alignment uses the existing confusable and
length costs, retains true doubling, and chooses fewer hard edits on ties. Every supported expected form must
require at least two hard consonant edits. This protects the two historical one-consonant warnings, including
the real Quran word خليلا, without word-specific runtime exceptions.

Candidates need affirmative finite posterior support over the exact same neighboring-word frames and context,
using the existing margins (including the doubled margin for uncertain changed sounds). Missing support abstains.

## Validation log

- Red: `lexical-red-target-8095018-20261031T123005Z` on Modal. The actual streaming checker returns no event for
  صالح replacing نوح; the regression expects `wrong_word 26:106:5` and fails.
- Focused smoke: `lexical-smoke-8095018-dirty-20261031T124737Z`, exit 0.
- Initial checker run: `lexical-unit-8095018-dirty-20261031T124614Z`; all 73 assertions passed, but the runner
  exited 1 on a Vitest worker `onTaskUpdate` timeout. A single-worker rerun is required below.
- The tests cover both substitution directions, historical warnings, confusables, held sounds, real doubling,
  missing/neutral acoustics, complete vocabulary forms, missing neighbors, clean speech, repeats, self-corrections,
  and neighboring additions.

All builds, tests, traces, and scoring run on Modal. No exam data is inspected, requested, or scored.
