# 01: E1 teacher-bias report

**What to build:** A per-class report of how often the teacher shows a recitation error that is really there, using public data with labelled errors. It decides which pronunciation classes are possible at all.

**Status:** done

- [x] Run the teacher on Modal over `obadx/qdat_bench` and `azizbekphd/ikhlas_recitations`, storing greedy phonemes and top-5 posteriors. (`modal/e1_pass.py`)
- [x] Map each dataset's labels onto the v1 classes: short-vowel substitutions, ح/ه ع/ء ص/س ض/د ط/ت ق/ك ذ/ز ث/س, and missing shadda. (`tools/e1_report.py`)
- [x] For each class, report the share of errors that show in the greedy path and the share that show in the top-5 posteriors, plus the false-positive rate on correct tokens.
- [x] Report the canonicalization rate per class (the error was said but the canonical phoneme was emitted).
- [x] Commit a versioned report under `docs/benchmarks/`, with a verdict per class: viable, posterior-only, or `unsupported`. (`docs/benchmarks/e1-teacher-bias-v1.md`)

## Answer

- Short vowels: `posterior-only`, provisional. 21 positives from only 5 sites. 43% show in greedy and 29% are canonicalized. A CTC score comparison catches 57% at 1.1% FP.
- Every letter pair: `unsupported`. Neither dataset labels a single pair error except one ع/ء case. ذ/ز and ث/س letters don't occur at all.
    - Ticket 11 (E1b, `docs/benchmarks/e1b-spliced-v1.md`) leaves every pair `unsupported`. Its `dead` directions (ت→ط, ذ→ز, س→ث) don't hold up. On listening, many splices are jarring, and the teacher's pull toward the expected letter fills in the canonical over an unintelligible splice. The controls can't tell that apart from real deafness. Only real recordings of the error can settle these.
- Missing shadda: `unsupported`. There are 4 positives, all from one site (unassimilated noon in يَكُن لَّهُۥ), and 3 of the 4 are canonicalized.
- Top-5 presence can't serve as a detector: a confusable sits in the top-5 of 55–91% of correct tokens.
- FP floors on correct speech worth knowing for the checker: ص→س 4.5% greedy (10% posterior), ق/ك 2.1% greedy, missing shadda 5.6% greedy. All other pairs are ≤ 0.2%.
- Public data can't settle the pair classes. Ticket 05's scripted recordings are the real E1 for them, so re-run `tools/e1_report.py` once those exist.
