# 03: Fuzzy locate with learner errors and multi-ayah spans

**What to build:** Clips with learner mistakes, timing variation, or several ayahs still locate to the right span. Each result reports both how well the clip fits and how far the spoken phonemes differ from the reference.

- **Retrieval:** n-gram anchor voting, then substring Levenshtein DP constrained to word boundaries.
- **Two separate quantities:**
  - The **weighted alignment cost** is used only for ranking candidates. Its class-aware costs make confusable pairs and short vowels cheaper, and madd extension cheap.
  - The **PER** is plain, unweighted phoneme error against the matched span, computed from the same alignment path. Cheap madd in the cost must not hide differences in the PER.
- **Edit path:** the alignment/edit path for the chosen span is preserved and exposed through trace, so the checker can reuse it later.
- **Matched span:** start and end words, which may cross ayah boundaries.

**Blocked by:** 01, 02.

**Status:** ready-for-agent

- [ ] Synthetic fixtures with single and multiple substitutions, deletions and insertions locate to the correct span.
- [ ] A madd-length change lowers the alignment cost penalty but still shows up in the PER.
- [ ] A ض→د substitution locates to the correct ayah, with nonzero PER and the substitution visible in the edit path.
- [ ] Real tlog fixtures (100:6 containing 100:6 plus 100:7, 100:7 apparently containing 100:8, and 100:1 with ض→د) are added. Their expected labels are asserted only after the audio has been independently checked; until then they are regression snapshots.
- [ ] The alignment cost and PER are exposed as separate fields.
- [ ] A synthetic substitution shows in the traced alignment path (the `it.todo` carried over from 02).
