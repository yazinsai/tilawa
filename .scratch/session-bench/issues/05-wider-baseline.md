# 05 Wider QUD baseline at every context level

Status: ready-for-agent
Type: task
Blocked by: 01, 02, 03

Spec: `../spec.md` (Plan steps 4 and 6). Commands: `docs/benchmarks/session-bench-v1.md`.

## Acceptance

- Run sessions-cli over every converted QUD recording (from 01) with teacher evidence, at each context level from `weakerContexts`, with surah/page now supported (03) and the new slices (02). For QUD, synthesise contexts the app would plausibly know (surah of the first Quran word; page of the first word; assignment = first..last word) — never a true-start hint beyond that; document it.
- Where fresh student evidence exists (`.scratch/session-bench/qud/evidence`) score it too; don't fabricate missing decodes — list what's missing.
- Update `docs/benchmarks/session-bench-v1.md` tables and remaining-work list; add a scoreboard row with `--sessions` and a `--note`.
- Measures stay reported only. Do not set thresholds (that is ticket 08).
