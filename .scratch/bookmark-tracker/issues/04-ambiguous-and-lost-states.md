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

**Status:** ready-for-agent

- [ ] Shifted or overlapping spans of the same ayah never cause `ambiguous` by themselves.
- [ ] A shared phrase across surahs gives `ambiguous`, and lists every occurrence.
- [ ] Random phonemes or noise give `lost`.
- [ ] Too-short evidence that matches uniquely still isn't `locked`.
- [ ] A strong prior on the wrong ayah doesn't produce `locked` there when the audio fits elsewhere.
- [ ] A wrong word that exactly matches another verse is tested, and the outcome is documented (`ambiguous` or a correct lock).
