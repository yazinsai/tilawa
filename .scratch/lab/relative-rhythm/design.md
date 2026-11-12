Training plan frozen before target inspection:
- Only help-train none recordings with split=train, no labelled events or repeats; no release speakers.
- Existing 16-observation local horizon; log emitted-run lengths centered on their geometric mean. This makes the observation invariant under common tempo scaling.
- Eight structural phone classes: background insertion, short vowel, held vowel, ordinary nasal, lengthened nasal, other sonorant, fricative, stop/other. Pronunciation run length distinguishes duration obligations without changing deletion costs.
- Lognormal duration marginals shared by joint and independent models. Gaussian Markov copula captures adjacent shortening/lengthening; rho=0 is the matched independent control.
- Transcript-conditioned Gibbs edit alignment with existing edit costs, reference-only deletions, and ordinary insertions; forward/backward expectations marginalize token correspondence. No tracker visits used as truth.
- Three deterministic speaker-grouped folds choose copula pooling strength from {10, 100, 1000}; duration marginals are identical for every candidate. Freeze final model before practice target observations.
- Local comparison uses the identical retained observations and identical original movement/edit graph in both partitions. Only -log(Z_joint/Z_independent) is added to existing costs, in one-nat-per-existing-edit units.
