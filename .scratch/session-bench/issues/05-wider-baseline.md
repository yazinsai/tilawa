# 05 Wider QUD baseline at every context level

Status: resolved
Type: task
Blocked by: 01, 02, 03

Spec: `../spec.md` (Plan steps 4 and 6). Commands: `docs/benchmarks/session-bench-v1.md`.

## Acceptance

- Run sessions-cli over every converted QUD recording (from 01) with teacher evidence, at each context level from `weakerContexts`, with surah/page now supported (03) and the new slices (02). For QUD, synthesise contexts the app would plausibly know (surah of the first Quran word; page of the first word; assignment = first..last word) — never a true-start hint beyond that; document it.
- Where fresh student evidence exists (`.scratch/session-bench/qud/evidence`) score it too; don't fabricate missing decodes — list what's missing.
- Update `docs/benchmarks/session-bench-v1.md` tables and remaining-work list; add a scoreboard row with `--sessions` and a `--note`.
- Measures stay reported only. Do not set thresholds (that is ticket 08).

## Comments

Implemented on branch `session-bench-05` (2026-10-13).

- The duplicated replay is gone. `runSession` now returns `{ score, events }`, `session-dump.ts` no longer has `sessionEvents`, and `sessions-cli` replays each context once, building the dump from those events. The disagreement check was dropped because nothing can disagree now. A test pins `run.score == scoreSessionEvents(run.events)`.
- `qudContexts(session)` gives four contexts for QUD sessions whose own context is `none`: assignment over the recited words, lowest to highest in Quran order; the page of the first recited word; its surah; and none. `sessions-cli --qud-contexts own` turns this off.
- `--student-evidence <dir>` pairs current `qudSession` truth with `<dir>/<id>.evidence.json` and lists recordings without a decode as `missingEvidence`. The old `manifest.json` predates ticket 01's rules and is no longer used here.
- Reports: `docs/benchmarks/session-qud-teacher-v1.json` (16 recordings × 4 contexts, 61 min) and `session-qud-int8-v1.json` (3 recordings × 4 contexts). Dumps: `.scratch/scoreboard/clips/sessions-f6098d6-{teacher,spliced-clean-int8}.json`. The teacher audio sits under `.scratch/bench/qud`, outside the served `/audio/sessions/` root, so its review items can't play yet.
- Headline false warnings per 5 clean minutes:

  | Context | Teacher | int8 |
  |---|---:|---:|
  | none | 3.03 | 19.54 |
  | surah | 3.03 | 20.24 |
  | page | 3.04 | 19.54 |
  | assignment | 26.25 | 57.22 |

  Located time is about 97.4% (teacher) and 85.6% (int8) at every context.
- Finding: declared-passage mode (assignment) produces far more false warnings on fluent recitation (taraweeh-maryam 2 → 965). The min..max assignment on multi-surah prayer recordings exaggerates this, but single-surah recordings rise too. This needs a checker investigation before ticket 08 sets thresholds.
- Missing student decodes: 13 recordings (listed in `session-bench-v1.md`). No thresholds were set. Two scoreboard rows were added at f6098d6; both are `= unchanged` on the release gates.
