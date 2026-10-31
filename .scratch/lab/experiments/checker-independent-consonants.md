# Independent consonant changes and lexical context

Status: evaluated. Both new false alarms removed and all end-to-end gains retained. Component prediction missed
by one; broader generalization remains unproven. Open for the user's fold/discard decision.

## Preregistration

Follow-up to `checker-lexical-consonants`, authorized by the user after its two false alarms were explained.
Registered with `tools/lab.py new` before copying or changing engine code. Main remains `b5bcc95`; the earlier
experiment stays untouched and open. Practice only, all heavy work on Modal; no fold or discard is authorized.

Main baseline verified in the live ledger: `exp-checker-joint-additions-rebased-27ddced-20261031T110941Z`,
97 wrong calls = 72 misses + 25 warnings, checker attribution 48. Prior lexical result:
`exp-checker-lexical-consonants-2f14277-20261031T125042Z`, 94 = 67 + 27, checker attribution 39.

Prediction versus main: checker attribution −9 and overall −5, giving 92 = 67 misses + 25 warnings. Preserve all
five prior phone recoveries, remove both new warnings, and introduce no new miss or warning. Generalization
remains a separate question; removing the false alarms cannot itself prove gains beyond the original passages.

## Implementation

- Align individual consonants with the existing weighted costs, independently of vowels. Count connected hard
  edits sharing an original contiguous consonant run as one change. A five-state frontier preserves whether
  the active reference/observed run components are separate or joined; minimize components among minimum-cost
  alignments. This avoids charging twice for one doubled identity change without inventing edits when a vowel
  splits an unchanged run. Genuine gemination changes, confusable discounts and held-sound conventions remain.
- Retain connected-form provenance. Isolated forms are context-free; connected forms require the source's
  preceding-final and following-first sounds at the candidate's intact neighboring visits, with explicit ayah edges.
- Retain all existing candidate boundaries, positive identical-frame acoustic support, thresholds, and protections
  for repeats, corrections and additions. No runtime word-specific rules or changes to global alignment.

## Validation log

- Red: `independent-red-765a3c2-dirty-20261031T130802Z`, exit 1. Seven regression failures include the two
  doubled-consonant patterns, incompatible connected contexts, and the reproduced p66 streaming warning.
- Initial run-granularity version: `independent-green-765a3c2-dirty-20261031T130841Z`, exit 0. Independent review
  identified vowel-only run splits and partial-run changes as missing edge cases before any full scoring.
- New edge-case red: `independent-vowel-red-765a3c2-dirty-20261031T130931Z`, exit 1; both added cases fail.
  Replaced run-granularity alignment with the exact five-frontier individual-consonant implementation above.
- Focused green: `independent-frontier-green-765a3c2-dirty-20261031T131007Z`, exit 0; build and all 57 lexical,
  independent-consonant, addition and noise tests pass.
- Independent exhaustive alignment/union-find oracle over 104 short pairs:
  `independent-oracle-765a3c2-dirty-20261031T131121Z`, exit 0.
- Full core suite: `independent-core-765a3c2-dirty-20261031T131017Z`, all four shards exit 0;
  `independent-checker-halves-765a3c2-dirty-20261031T131021Z`, both shards exit 0. Checker halves use the
  earlier documented workaround for Vitest's 60-second worker RPC limit without modifying assertions or timeouts.
  Together with the final oracle run, all 602 core tests pass; the core build passes.
- Both actual false-warning replays: `independent-failure-replays-765a3c2-dirty-20261031T131024Z`, both shards
  exit 0. Neither p66 nor p67 confirms the former wrong-word warning under the default policy. The existing
  transient p66 addition still resolves. No evidence or labels changed.

All builds, tests, traces and scoring run on Modal. No post-score engine tuning was performed.

## Practice result

Standard run: `exp-checker-independent-consonants-4a4c2f5-20261031T131228Z`, source `4a4c2f5`; all 100 shards
and merge exit 0. One worker was preempted and resumed. Same listener, test set, 100 sessions, 429.475 recited
minutes, speakers, durations and planted mistake keys as both comparisons. Evaluation and case analysis were
practice only.
Detailed paired changes, metadata checks and component-only losses are in
[the result JSON](checker-independent-consonants-results.json).

| Measure | Main | Previous lexical experiment | This experiment | Prediction |
| --- | ---: | ---: | ---: | ---: |
| Phone wrong calls | 97 | 94 | 92 | 92 |
| Phone misses | 72 | 67 | 67 | 67 |
| Phone warnings | 25 | 27 | 25 | 25 |
| Checker attribution | 48 | 39 | 40 | 39 |
| Checker with perfect inputs | 64 (58 + 6) | 52 (46 + 6) | 56 (50 + 6) | |
| Studio wrong calls | 67 (54 + 13) | 61 (48 + 13) | 61 (48 + 13) | |

| Slice | Phone misses / warnings: main → previous → this | Studio misses / warnings: main → previous → this |
| --- | --- | --- |
| Old 20 | 8 / 0 → 6 / 0 → 6 / 0 | 7 / 1 → 5 / 1 → 5 / 1 |
| New 80 | 64 / 25 → 61 / 27 → 61 / 25 | 47 / 12 → 43 / 12 → 43 / 12 |
| Clean 26 | 0 / 5 → 0 / 5 → 0 / 5 | 0 / 1 → 0 / 1 → 0 / 1 |

Both previous new phone warnings disappear: p66 `9:3:10` (39.76 s) and p67 `24:60:20` (205.88 s).
The historical p13 `20:53:9` and p17 `28:58:14` warnings remain absent on phone and studio.
There are **no new user-facing misses or warnings**, versus main or the previous experiment.

All five phone recoveries remain: old p10 `26:106:5`, old p20 `26:142:5`, new p06 and p22 `26:161:5`, and new
p73 `26:106:5`. Studio retains these plus new p11 `26:106:5`; the latter remains missed on phone.

## Every new component-only failure

The more conservative checker gives back four perfect-input recoveries from the previous experiment, on both
variants. Three remain caught with listener evidence; one remains missed and changes attribution.

| Session | Location / time | Expected → planted | End-to-end result, previous → this | Perfect-input result |
| --- | --- | --- | --- | --- |
| Old p20 | 26:123:2 / 109.005 s | عاد → ثمود | caught → caught | caught → missed |
| New p41 | 26:123:2 / 185.545 s | عاد → ثمود | caught → caught | caught → missed |
| New p73 | 26:123:2 / 210.525 s | عاد → ثمود | caught → caught | caught → missed |
| New p54 | 26:141:2 / 129.325 s | ثمود → عاد | missed → missed | caught → missed |

p54 shifts from listener to checker attribution. This explains checker attribution 40 instead of predicted 39;
the component target was not met even though the actual overall and zero-new-warning targets were met.
These are not additional end-to-end misses. No separate ablation isolates the two new safeguards' effects.

## Generalization and verdict

| Exclusion | Phone wrong calls, main → this | Studio wrong calls, main → this |
| --- | ---: | ---: |
| Strongest reader | 87 → 83 (−4) | 60 → 55 (−5) |
| Surah 26 | 71 → 71 (0) | 49 → 49 (0) |
| نوح/صالح pair | 92 → 90 (−2, noise) | 62 → 60 (−2, noise) |
| Reader + surah + pair | 62 → 62 (0) | 43 → 43 (0) |

Five phone readers tie with one gain each: `mp3quran-10905`, `25`, `314`, `42`, `72`; studio also includes `38`.
The table uses `10905` as the deterministic representative. The gain survives removal of a strongest reader,
but not removal of surah 26; excluding the familiar word pair leaves only two catches, within lab noise.

**Verdict:** the follow-up repairs both false alarms while preserving all end-to-end gains, and the ledger calls it
better (overall −5). It misses the checker attribution prediction by one and does not establish general
improvement beyond the concentrated passages. Main and the previous experiment remain untouched; this
experiment is neither folded nor discarded.
