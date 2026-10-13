# 03 Tracker range prior for surah and page context, plus Madani page map

Status: resolved
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

## Comments

- `TrackerOptions.priorRange: { ranges: { from: AyahRef; to: AyahRef }[]; weight }`: full `MAX_PRIOR_BONUS × weight` for any candidate touching an ayah inside a range (inclusive; `to.ayah: Infinity` = through surah end), none outside, max'd with the opening bonus and `prior`, so still capped and can't break a tie. `prior` unchanged.
- `packages/core/assets/madani-pages-v1.json`: 604 pages `[fromS, fromA, toS, toA]`, generated from https://tanzil.net/res/text/metadata/quran-data.js (`QuranData.Page`, Tanzil Quran Metadata 1.0, CC BY 3.0), source sha256 `9e9930c592aaa34b5c7a79ff4e99ffdc782263a266090bfe2c49e4ed4bac4ca4`. Page end = ayah before next page start. `pageRange()` in `reference.ts`; `test/pages.test.ts` pins the asset hash, 604 contiguous pages covering every ref-v2 ayah, and pages 2/50/604.
- `contextAdapter`: surah → range over the surah, page → range over the page; `supported: true`, `effective` = context kind for all.
- SDK: `packages/sdk` exposes no context/prior options, so nothing wired; follow-up if the SDK grows a session context option.
- Scoreboard: teacher row at fd98856 `= unchanged` (gate evals don't use a range prior; QUD sessions are context `none`). No surah/page-context sessions exist yet, so the prior's effect is unmeasured.
- Under heavy machine load the default 5 s vitest timeout trips on 8 slow tests; all 492 pass with `--testTimeout=60000`.
