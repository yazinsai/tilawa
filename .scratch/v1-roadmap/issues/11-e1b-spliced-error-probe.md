# 11: E1b spliced-error probe

**What to build:** Synthetic audio positives for the classes E1 couldn't judge (every letter pair and missing shadda), made by splicing phoneme segments between clean recitations. Then re-run the E1 report on them. Spliced errors are cleaner than real learner errors, so a pass is an upper bound on what the teacher can hear. A fail rules the class out before ticket 05's recordings arrive.

**Status:** needs-triage

- [ ] Letter boundaries come from teacher CTC forced alignment of the canonical phonemes (the Viterbi in `tools/e1_report.py`). QUD as exported has word timestamps only, not the letter timestamps PLAN.md assumes.
- [ ] Source audio is clean single-reciter ayahs (everyayah or QUD murattal). Take donor segments from the same reciter where possible, so the splice tests the phoneme, not the voice.
- [ ] Letter pairs: replace a canonical consonant plus its vowel with the partner consonant plus the same vowel, cut from another word by the same reciter. Crossfade 5–10 ms at zero crossings. For each of the 8 pairs, aim for ≥200 positives from ≥30 distinct sites, in both directions.
- [ ] Missing shadda: shorten a geminate's closure or hold to a single consonant's median duration for that reciter.
- [ ] Controls: splice the *same* phoneme from another word, with an identical seam and no error. If the teacher flags these, it is reacting to the seam, and the class's numbers are void.
- [ ] Produce the clips as an E1-format parquet so `tools/e1_report.py` scores them unchanged. Add a dataset entry plus a spliced/control flag, and report control FP next to each class.
- [ ] Commit `docs/benchmarks/e1b-spliced-v1.md` with per-class verdicts, labelled as upper bounds. Update the 01 verdicts only from `unsupported` to "dead" (fails even on clean splices), never upward.
- [ ] Spot-listen to ~20 splices per class and record a pass/fail on whether each sounds like the target error.

## Notes

- Speculative: the splicer is reusable for ticket 08's E3 training data, so build it as a library, not a one-off.
- Iqra_TTS (E4) is an alternative positive source. It's left out of scope here, because synthetic voices add a second domain gap.
