# 04: Confidence, ambiguous and lost states

**What to build:** The tracker reports `locked` only when the evidence supports one location. It reports `ambiguous` when several genuinely different locations fit, and `lost` when nothing fits.

- **Deduplicate first.** Overlapping or adjacent candidates that describe the same location (shifted spans of the same ayah, or a span contained in another) are merged before any margin is computed.
- **`locked` requires all of the following:**
  1. **Absolute fit:** the top candidate's normalised cost is under a threshold.
  2. **Sufficient evidence:** a minimum number of observed phonemes and words.
  3. **Separation:** a margin over the best genuinely different candidate after deduplication.
- **`ambiguous`:** the fit is acceptable but the separation fails. All fitting distinct candidates are returned.
- **`lost`:** the fit fails for every candidate.
- **Prior:** the prior adds a bounded bonus to its passage and the nearby ones. It never overrides failed fit or separation.
- **Thresholds:** exposed as options with documented defaults, to be tuned later.
- **Confidence:** each candidate's confidence is derived from fit and separation, and is monotone in both.

**Blocked by:** 03.

**Status:** done

- [x] Shifted or overlapping spans of the same ayah never cause `ambiguous` by themselves.
- [x] A shared phrase across surahs gives `ambiguous`, and lists every occurrence.
- [x] Random phonemes or noise give `lost`.
- [x] Too-short evidence that matches uniquely still isn't `locked`.
- [x] A strong prior on the wrong ayah doesn't produce `locked` there when the audio fits elsewhere.
- [x] A wrong word that exactly matches another verse is tested, and the outcome is documented (`ambiguous` or a correct lock).

## Comments

- Implemented in `createTracker` (`packages/core/src/tracker.ts`); tests in `packages/core/test/states.test.ts`.
- Defaults: `maxPer` 0.35, `maxCost` 0.35, `minPhonemes` 10, `minWords` 2, `minSeparation` 1 edit, `separationScale` 30 (required margin is `max(1, 30 / n)` edits), `minLockConfidence` 0.08, prior bonus at most half an edit (neighbouring ayahs get half of that).
- Separation scale came from shard 0: under 20 phonemes, locks outside surah 100 led the runner-up by under one edit at p50, while true locks led by 2–4.
- `minLockConfidence` 0.08 sits between the lowest verified lock (875 at 0.11) and the bogus locks it removes (≤ 0.04). That gap is thin; retune it on the 07 sample.
- Unique but insufficient or poorly fitting evidence reports `acquiring`. Ambiguous returns every fitting distinct candidate within the margin of the best, not all fitting candidates.
- Confidence is fit × separation: fit is `(1 − per/maxPer)(1 − cost/maxCost)`, separation maps the lead over the best other fitting candidate from −margin..+margin onto 0..1.
- Shard 0 (970 clips): 665 locked in surah 100, 21 locked elsewhere (all look genuine: 97:1 ×9, 2:285, 102, 99:8, 2:1–3), 56 ambiguous, 60 acquiring, 164 lost.
- Follow-up (875, low volume): shortened ghunna (`ں` runs, and 4-long `ن`/`م` runs) now costs 2 like madd; a plain noon or meem dropped still costs a full edit. DP ties now resolve to the path with fewer plain edits, and a mid-word start costs 10 (was 15), since the cheaper ghunna otherwise let a start inside the preceding noon run beat it and inflated 875's PER to 0.33. 875 confidence 0.11 → 0.25. Shard 0: 675 locked in 100, 21 elsewhere (unchanged set), 153 lost.
- Tried and rejected: confidence fit from weighted cost only. It re-admitted bogus locks with high confidence (147 → 10:57 at 0.41, 769 → 87:1 at 0.46, both PER ≈ 0.32). PER stays in the fit term.
