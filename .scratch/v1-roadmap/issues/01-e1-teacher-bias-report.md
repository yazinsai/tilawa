# 01: E1 teacher-bias report

**What to build:** A per-class report of how often the teacher shows a recitation error that is really there, using public data with labelled errors. It decides which pronunciation classes are possible at all.

**Status:** ready-for-agent

- [ ] Run the teacher on Modal over `obadx/qdat_bench` and `azizbekphd/ikhlas_recitations`, storing greedy phonemes and top-5 posteriors.
- [ ] Map each dataset's labels onto the v1 classes: short-vowel substitutions, ح/ه ع/ء ص/س ض/د ط/ت ق/ك ذ/ز ث/س, and missing shadda.
- [ ] For each class, report the share of errors that show in the greedy path and the share that show in the top-5 posteriors, plus the false-positive rate on correct tokens.
- [ ] Report the canonicalization rate per class (the error was said but the canonical phoneme was emitted).
- [ ] Commit a versioned report under `docs/benchmarks/`, with a verdict per class: viable, posterior-only, or `unsupported`.
