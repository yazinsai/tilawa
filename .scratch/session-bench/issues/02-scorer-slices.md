# 02 Scorer: delay distributions, per-class and mutashabihat slices, twin ambiguity

Status: ready-for-agent
Type: task
Blocked by: none

Spec: `../spec.md` (Scoring, per track). Code: `packages/core/src/bench/sessions.ts` (`scoreSessionEvents`), `sessions-cli.ts`, session section of `scoreboard-cli.ts`.

The scorer records raw delays and a single recall; the spec table needs more.

## Acceptance

- Per class (`wrong_word`, `omitted_word`, `added_word`, `skipped_passage`) errors/hits/recall in `SessionScore`; mutashabihat slice (`forms_other_ayah`) reported separately, same denominators rule (abstentions count as misses).
- Aggregation (sessions-cli report + scoreboard): suspected and confirmed delay p50/p90 with counts; censored-by-miss stated.
- Abstentions split by cause: twin ambiguity (tracker `ambiguous` where the competing candidates are identical-text twins, e.g. 94:5/94:6 — use the reference index to decide) vs. general ambiguity vs. checker `uncertain`; each as share of labelled errors and per clean session.
- Scoreboard session section emits these as reported (non-gating) metrics; per track, mark which measures the spec designates as **gate** in the metric note (still non-gating).
- Unit tests with synthetic sessions/events cover each new field.
