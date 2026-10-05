# 0001: Adopting tracker relabels of tlog as training labels

**Status:** accepted, 2026-10-05

## Context

tlog filename labels are often wrong, and ticket 06 proposes a relabel for every clip. Ticket 07 had to decide, with evidence, which proposals can become distillation labels.

The evidence is a blind review of shard `clean-00000-of-00411` (970 clips, all filename-labelled surah 100), using index `ref-v2` and the decoded-only tracker. A stratified sample of 302 clips was drawn in 10 strata (boundary confidence, posterior disagreement, then outcome). The reviewer listened to each clip and marked the recited span without seeing the tracker's proposal or the filename ayah. Answers: 251 spans, 33 no recitation, 18 unsure (left out). Rates are weighted back to each stratum's share of the shard. Intervals are 95% Wilson per stratum and Agresti–Coull for weighted rates. Data and tooling: `.scratch/bookmark-tracker/review/`, `packages/core/src/review/`.

## What counts as correct

A lock is correct when both span edges are within 1 word of the reviewer's span, counted across ayah boundaries. Exact-word agreement isn't a usable bar, for two reasons:

- The teacher often drops a clip's first or last word, so the tracker can't see it.
- The reviewer's half-word edges vary.

At exact-word level, weighted precision is about 75–80%. Nearly all of that gap is edge disagreement.

## Results (decoded run, ±1 word)

| | value |
|---|---|
| filename label wrong | ~47% of the shard |
| locked and correct, every lock | 97.0% (93.5–100%) |
| false locks, share of the shard | 2.2% (0–5.2%) |
| excluded (not locked) | 27.6% |
| excluded clips the reviewer could place | about half |

The posterior-reranked run gives the same precision (97.0%) and excludes slightly fewer clips (27.0%). It isn't adopted, because it needs top-k posteriors and gains nothing measurable.

Confidence sweep:

| min confidence | precision (low end) | share of shard kept |
|---|---|---|
| 0.08 | 97.0% (93.5%) | 72% |
| 0.2 | 98.2% (94.7%) | 66% |
| 0.3 | 98.7% (95.1%) | 60% |
| 0.5 | 99.6% (95.3%) | 47% |

## Decision

- Adopt decoded-only tracker locks with **confidence ≥ 0.2**, scored at the ±1-word bar. The point estimate is 98.2%; the 95% interval's low end is 94.7%.
- Reviewed clips use the reviewer's answer instead: their span where they marked one (including clips the tracker didn't lock), and nothing where they heard no recitation.
- `pnpm review eligible --min-confidence 0.2` writes `eligible.jsonl`: 717 of 970 clips (466 from the tracker, 251 from review).

The target was 98%. Showing that the *low end* of the interval clears 98% would need roughly 3× the review. Even 106 correct reviewed locks out of 106 at confidence ≥ 0.7 only reach 93.5%. The decision uses the point estimate, and the interval is recorded here.

## Consequences

- Shards other than 0, and other surahs, are adopted on this evidence until a second shard is reviewed. Shard 0 is all surah 100, so other surahs are the main unmeasured risk.
- Re-score after any change to the index or tracker, because answers describe the audio, not a tracker version.
- The biggest recall gain available is the excluded-but-clean half of the 28% the tracker won't lock. After that come teacher edge drops and repeated ayahs. Reviewed misses are pinned as pending fixtures in `packages/core/test/fixtures.test.ts`.
