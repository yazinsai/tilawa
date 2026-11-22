# 02 Scorer: delay distributions, per-class and mutashabihat slices, twin ambiguity

Status: resolved
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

## Comments

- `SessionScore` gains `byClass` (per error class errors/hits/recall), `mutashabihat` (`forms_other_ayah` slice), `abstentions` (missed errors by cause, priority checker `uncertain` > tracker twin ambiguity > other ambiguity) and `ambiguousEpisodes` (entries into `ambiguous`, twin vs general). Abstentions stay misses in every denominator.
- Twin = every ambiguous candidate ayah is near-identical to the others in reference phonemes: edit distance at most 10% of the longer (`isTwin`). 94:5/94:6 are not byte-identical (leading fa-), so exact equality would miss the motivating case. Threshold is a judgement call; revisit with the baseline.
- `summarizeSessions(rows)` groups by track.context.source: per-class/mutashabihat recall, suspected/confirmed delay p50/p90 (nearest rank) with `n` and `censoredByMiss`, abstention shares of labelled errors, and per-clean-session (zero-error sessions) uncertain warnings and twin/general ambiguity entries. `sessions-cli` writes it as `summary` and prints confirmed delays; the scoreboard session section is built from it.
- Scoreboard metrics stay `info`; notes mark spec **gate** measures per track (recitation at `none`; memorization at `assignment`/`resume`, including confirmed-delay p90), saying "not gating yet" and the gating context when the row is a different context.
- Open: old session reports lack the new row fields; regenerate (ticket 05) before passing `--sessions`. Full QUD CLI replay of one recording exceeded 5 minutes locally, so no end-to-end smoke run here.
