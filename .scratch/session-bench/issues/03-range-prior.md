# 03 Tracker range prior for surah and page context, plus Madani page map

Status: ready-for-agent
Type: task
Blocked by: none

Spec: `../spec.md` (Engine interface table). Code: `packages/core/src/tracker.ts` (`prior`, `priorBonus`, `MAX_PRIOR_BONUS`), `packages/core/src/reference.ts`, `packages/core/assets/`, `sessions.ts` `contextAdapter`.

## Acceptance

- Tracker option accepts a range prior: `{ from: WordRef-ish ayah, to: ayah, weight }` (or a list of ayah ranges). Bonus applies uniformly inside the range, still capped at `MAX_PRIOR_BONUS` so a prior alone never separates. The existing single-ayah `prior` keeps identical behaviour (existing tests unchanged).
- Madani (Hafs, 604-page) page→first/last ayah table added as a reference asset with its source recorded (e.g. Tanzil `quran-data` page boundaries) and a checksum test (604 pages, contiguous, covers 1:1..114:6).
- `contextAdapter`: `surah` → range prior over the surah; `page` → range prior over the page; `supported: true` for both; `effective` reflects the real context.
- SDK: if the SDK exposes context options (check `packages/sdk`), wire surah/page through; otherwise note it in the ticket comments.
- Tests: tracker unit tests for range prior (bonus inside, none outside, cap); context adapter tests.
- This moves metrics: add a scoreboard row per AGENTS.md with a `--note`.
