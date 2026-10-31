# Independent consonant changes and lexical context

Status: implemented; validation and practice evaluation pending. Leave open for the user's decision.

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
- Both actual false-warning replays: `independent-failure-replays-765a3c2-dirty-20261031T131024Z`, both shards
  exit 0. Neither p66 nor p67 confirms the former wrong-word warning under the default policy. The existing
  transient p66 addition still resolves. No evidence or labels changed.

All builds, tests, traces and scoring run on Modal. Full evaluation will compare main and prior results, old/new/
clean/studio slices, every new failure, and the same reader/surah/pair exclusions. No post-score tuning is planned.
