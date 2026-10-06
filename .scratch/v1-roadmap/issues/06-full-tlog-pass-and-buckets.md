# 06: Full tlog teacher pass and bucketing

**What to build:** PLAN.md's tlog processing over the whole corpus: teacher pass, relabel, buckets, speaker splits.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] Teacher pass over a 10% stratified shard sample, then all 411 shards: greedy phonemes, per-phoneme confidence, and top-5 fp16 posteriors per frame. Log the cost against the $150–400 estimate.
- [ ] Relabel with the threshold from ticket 03 (confirmed: 0.2).
- [ ] Text-only labels for repeated text: an `ambiguous` clip whose candidates all share the same canonical phonemes (the ar-Rahman refrain, for example) gets that text as its label, without a location. See ADR 0001.
- [ ] Buckets: `clean-match`, `deviant`, `ambiguous`, `reject`. Cutoffs tuned on about 300 hand-checked clips.
- [ ] Speaker clusters from ECAPA/WavLM-SV embeddings; splits by cluster.
- [ ] Remove Quran-Lab benchmark IDs, including `tlog_holdout`.
- [ ] Manifest per bucket and split, versioned.
