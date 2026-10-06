# 11: E1b spliced-error probe

**What to build:** Synthetic audio positives for the classes E1 couldn't judge (every letter pair and missing shadda), made by splicing phoneme segments between clean recitations. Then re-run the E1 report on them. Spliced errors are cleaner than real learner errors, so a pass is an upper bound on what the teacher can hear. A fail rules the class out before ticket 05's recordings arrive.

**Status:** ready-for-human

Everything but the spot-listen is done.

- [x] Letter boundaries come from teacher CTC forced alignment of the canonical phonemes (the Viterbi in `tools/e1_report.py`). QUD as exported has word timestamps only, not the letter timestamps PLAN.md assumes. (`splice.boundaries` splits each blank gap at its midpoint)
- [x] Source audio is clean single-reciter ayahs (everyayah or QUD murattal). Take donor segments from the same reciter where possible, so the splice tests the phoneme, not the voice. (Husary and Minshawy murattal, juz 29–30; donors always same reciter)
- [x] Letter pairs: replace a canonical consonant plus its vowel with the partner consonant plus the same vowel, cut from another word by the same reciter. Crossfade 5–10 ms at zero crossings. For each of the 8 pairs, aim for ≥200 positives from ≥30 distinct sites, in both directions. (8 ms; 284–300 positives from 182–287 sites per pair)
- [x] Missing shadda: shorten a geminate's closure or hold to a single consonant's median duration for that reciter. (300 positives, 281 sites)
- [x] Controls: splice the *same* phoneme from another word, with an identical seam and no error. If the teacher flags these, it is reacting to the seam, and the class's numbers are void. (letters: host cut points identical, donor is the same consonant plus vowel; shadda: the same kind of midpoint cut, removing 15%, so the seam is similar but not identical)
- [x] Produce the clips as an E1-format parquet so `tools/e1_report.py` scores them unchanged. Add a dataset entry plus a spliced/control flag, and report control FP next to each class. (`e1b-spliced`, `kind` = spliced/control/clean; `tools/e1_report.py --e1b`. Scoring logic is E1's `score_clip`, now with an optional site filter so a clip is scored only where it was spliced)
- [x] Commit `docs/benchmarks/e1b-spliced-v1.md` with per-class verdicts, labelled as upper bounds. Update the 01 verdicts only from `unsupported` to "dead" (fails even on clean splices), never upward.
- [ ] Spot-listen to ~20 splices per class and record a pass/fail on whether each sounds like the target error. (Pack built at `.scratch/e1b/listen/`: 20 spliced + 20 controls per class, with a pass column in its README. Needs a human.)

## Answer

Upper bounds, Husary and Minshawy, judged against seamed controls:

- `viable`: ح/ه, ع/ء, ص/س, ض/د, ق/ك, missing shadda. Greedy recall 52–72%, control FP ≤ 1%. Canonicalization is still 16–43%, so on real learners expect well under these.
- `posterior-only`: ط/ت, carried by ط→ت (70% greedy). Treat ط→ت with care: only 80.7% of its controls stay intact, just over the void line. ت→ط is `dead` (23% greedy, 53% canonicalized).
- `dead` in the report: ت→ط, ذ→ز (13% greedy, 82% canonicalized) and س→ث (13%). **Not carried into ticket 01.** On listening, the splices are often jarring. When a splice is unintelligible, the teacher's pull toward the expected letter writes the canonical, and a control can't tell that apart from deafness, because there the pull and the ear agree.
    - Check: hit rate vs whether the donor's neighbouring phonemes match the host's (0/1/2 neighbours matched) is flat. 53% / 53% / 43%; 2-match n = 72. ت→ط stays ~21–29% at 0–1 matches, while ط→ت holds ~70%. So the asymmetry isn't caused by coarticulation mismatch. But that rules out only one source of jarring: pitch, loudness and the seam itself are unchecked.
- `mixed`: ذ/ز and ث/س pool to `dead`, but their other directions aren't settled. ز→ذ is `inconclusive` (54% recall, but 7% control FP against a 0.2% clean floor). ث→س is `void`, because only 73% of ث controls decode as ث. The weak interdentals don't survive the seam.
- Asymmetry is the rule. ح→ه 45% vs ه→ح 69%, and ص→س 55% vs س→ص 65%. The checker should gate per direction, not per pair.
- Control FP tracks clean FP closely everywhere except ز/ث, so the seam is not what's driving the passes.

## Notes

- Speculative: the splicer is reusable for ticket 08's E3 training data, so build it as a library, not a one-off. (`tools/splice.py` is DSP only: `boundaries`, `replace`, `shorten`)
- Iqra_TTS (E4) is an alternative positive source. It's left out of scope here, because synthetic voices add a second domain gap.
- Rebuild: `tools/e1b_build.py clean`, `modal run modal/e1b_pass.py --stage clean`, `tools/e1b_build.py splice`, `modal run modal/e1b_pass.py --stage spliced`, `tools/e1_report.py --e1b`. The Modal client needs pyarrow: `uv run --with modal --with pyarrow modal run ...`.
