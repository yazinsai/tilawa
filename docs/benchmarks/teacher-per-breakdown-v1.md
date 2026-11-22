# Teacher PER breakdown v1

The teacher's 8.5% PER against the reference on tlog-v2 `clean-match` test is mostly representation and clip-edge mismatch, not mishearing. Clip edges account for 4.25 points, madd and ghunna run length for 2.15, and nasal spelling plus qalqala for 0.20. Edits that look like real mishearing add up to about 1.2 points, with another 0.7 points that could go either way.

Eval set: the same 3,000 cluster-held-out clips as `student-v1.md`, 167,210 reference phonemes. Student: `e2-libri-full`, greedy decode, chunk 16 / left 128. It was re-decoded on CPU in fp32 with `modal/student_decode_cpu.py`, which reproduces the logged numbers exactly (9.87% vs ref, 5.55% vs teacher). Script: `tools/per_breakdown.py`. Data: `teacher-per-breakdown-v1.json`.

## Method

Edits are rapidfuzz Levenshtein editops of (reference → hypothesis). An edit counts as **edge** if it falls before the first or after the last exact match of at least 3 phonemes. Interior edits are classified as follows:

| kind | rule |
|---|---|
| madd_length | ins/del of ا ۥ ۦ next to the same letter |
| ghunna_length | ins/del of ن م ں ۾ next to the same letter, or of و/ي inside a run of 3 or more (idgham with ghunna) |
| assimilation | ں↔ن, ۾↔م, ں↔۾; or an idgham/iqlab pair (ن↔م/ل/ر, م↔ب, ...) inside a run |
| qalqala | ins/del of ڇ |
| gemination | ins/del of any other consonant next to the same consonant |
| vowel_insdel, madd_insdel, madd_vowel_sub | lone short vowel or madd ins/del; madd↔vowel substitution |
| vowel_sub, confusable_sub, consonant_sub, consonant_insdel | everything else, with PLAN's 8 confusable pairs counted separately |

The groups are **representation** (edge, madd_length, ghunna_length, assimilation, qalqala), **ambiguous** (gemination, vowel/madd ins/del, madd↔vowel) and **mishearing** (the rest).

Normalisations:
- **collapse**: merge each run of one phoneme into a single phoneme.
- **canon**: ں→ن, ۾→م, drop ڇ, then collapse.
- **interior**: count only interior edits. This one is a diagnostic, not a metric, because it depends on the hypothesis.
- **spliced reference**: keep the reference between the teacher's first and last 3-phoneme anchors and use the teacher's own phonemes outside them. It is fixed per clip and does not depend on the hypothesis.

## PER under each normalisation

| pair | raw | collapse | canon | raw interior | canon interior |
|---|---|---|---|---|---|
| teacher vs ref | 8.51% | 6.38% | 6.18% | 4.26% | 2.33% |
| student vs ref | 9.87% | 8.01% | 7.81% | 5.07% | 3.34% |
| student vs teacher | 5.55% | 5.01% | 4.91% | 3.55% | 2.94% |
| teacher vs spliced ref | 4.38% | 2.61% | **2.44%** | — | — |
| student vs spliced ref | 7.22% | 5.57% | **5.40%** | — | — |

Clip-level exact match: 14.3% raw and 23.8% canon for the teacher, 13.8% and 21.6% for the student. 1,660 of 3,000 clips (55%) have no canon interior edit between teacher and reference.

## Teacher vs reference by group

| group | kind | edits | PER points |
|---|---|---|---|
| representation | edge | 7,111 | 4.25 |
| | madd_length | 2,222 | 1.33 |
| | ghunna_length | 1,368 | 0.82 |
| | assimilation | 217 | 0.13 |
| | qalqala | 125 | 0.07 |
| | **total** | | **6.60** |
| ambiguous | vowel_insdel | 907 | 0.54 |
| | gemination / madd_insdel / madd_vowel_sub | 287 | 0.17 |
| | **total** | | **0.71** |
| mishearing | consonant_sub | 997 | 0.60 |
| | consonant_insdel | 448 | 0.27 |
| | confusable_sub | 371 | 0.22 |
| | vowel_sub | 182 | 0.11 |
| | **total** | | **1.19** |

For the student, the same groups come to 7.10 (edge 4.81), 0.80 and 1.97 points. Its extra 1.4 points over the teacher are mostly mishearing (+0.78, of which consonant_sub is +0.54) and edge (+0.56). Madd and ghunna length is slightly lower than the teacher's (2.03 vs 2.15).

By operation, the teacher has 9,653 deletions (5.77 points), 2,538 substitutions (1.52) and 2,044 insertions (1.22). By zone: interior 4.26, end 2.54, start 1.71. The deletion skew is mostly the edges: the reference is longer than the audio there.

### Edge edits (teacher)

| region | edits | points |
|---|---|---|
| end, reference longer | 3,405 | 2.04 |
| start, reference longer | 2,066 | 1.24 |
| start, teacher longer | 498 | 0.30 |
| end, waqf (reference keeps the final short vowel, teacher stops on sukun) | 409 | 0.24 |
| end, teacher longer | 317 | 0.19 |
| start, hamzat wasl (teacher adds ء) | 103 | 0.06 |
| start, basmala in the audio but not in the reference | 95 | 0.06 |
| other | 218 | 0.13 |

"Reference longer" means the label span covers phonemes the teacher doesn't hear. In 214 of 3,000 clips the teacher's output starts on a vowel or madd letter, so the audio begins mid-syllable. Those are segmentation cuts, not mishearing. I didn't listen to the clips, so cases where the reciter really dropped words can't be ruled out. The 77/80 exact rate from the ticket 06 review says they are rare.

### Top interior phoneme pairs

| op | ref → hyp | kind | teacher | student |
|---|---|---|---|---|
| del | ا → ∅ | madd_length | 913 | 910 |
| del | ن → ∅ | ghunna_length | 443 | 411 |
| del | َ → ∅ | vowel_insdel | 382 | 288 |
| del | ں → ∅ | ghunna_length | 368 | 341 |
| ins | ∅ → ۦ | madd_length | 330 | 253 |
| del | م → ∅ | ghunna_length | 311 | 317 |
| del | ۦ → ∅ | madd_length | 299 | 282 |
| ins | ∅ → ا | madd (mostly length) | 252 | 219 |
| del | ۥ → ∅ | madd_length | 230 | 207 |
| del | ِ → ∅ | vowel_insdel | 219 | 243 |
| ins | ∅ → ۥ | madd_length | 216 | 200 |
| del | ُ → ∅ | vowel_insdel | 192 | 209 |
| del | و → ∅ (وووَ → ووَ) | ghunna_length | 130 | 145 |
| del | ڇ → ∅ | qalqala | 116 | 106 |
| del | ي → ∅ (يييَ → ييَ) | ghunna_length | 100 | 109 |
| sub | ۾ → م | assimilation | 99 | 134 |
| del | ء → ∅ | consonant_insdel | 83 | 85 |
| ins | ∅ → َ | vowel_insdel | 73 | 116 |
| sub | ں → ن | assimilation | 66 | 57 |
| sub | ل → ن | consonant_sub / idgham | 66 | 112 |

Top mishearing edits for the teacher: del ء 82, ط→ت 61, ع→ء 55, del ه 54, َ→ُ 53, ص→س 51, ل→ن 47, ء→ع 43, ُ→َ 42, ح→ه 34. For the student: َ→ُ 101, ل→ن 89, ُ→َ 86, del ء 84, del ه 75, ط→ت 62, ع→ء 61, ِ→َ 57, ن→ل 57. ط→ت, ع→ء, ص→س and ح→ه are the teacher's known light/heavy and throat-letter biases (E1). Since 77/80 of these recitations were judged exact, the reference is the likelier side on these. The student copies them and adds vowel-quality and ل/ن confusions of its own. Many of the deleted ء are dropped hamzas inside a word (قُلءَعُۥۥذُ → قُلَعُۥۥذُ), which may be the reciter's connected speech rather than a teacher error (speculative).

Madd lengths go both ways. The teacher hears ا shorter than the reference (913 deletions vs 252+ insertions) but ۦ and ۥ longer at about the same rate. That fits the reference using fixed lengths from the riwaya profile (munfasil/muttasil 4, aarid 2) while reciters vary. Run length is a duration estimate on both sides. It isn't phoneme identity.

## Conclusion: is `clean-match` text a better CTC target than the teacher path?

Not as is. Raw text CTC would be worse. 4.25 of the 8.5 points are at clip edges, where the reference names phonemes the audio doesn't contain or cuts mid-syllable. A text CTC target would train the student to hallucinate them at edges, which is the worst thing a streaming model can learn. Another 2.15 points are canonical madd and ghunna counts that the reciter didn't produce, and the teacher's run lengths follow the audio. In the interior, after canon, text and teacher disagree on only 2.3% of phonemes, and about half of that (1.2 points) is the teacher's known letter biases on recitations that are about 96% exact. There the text is the better label. So the target for the planned run should be the **spliced reference** (teacher phonemes outside the anchored span, text inside it), with soft KD on the teacher posteriors carrying the actual timing and duration. Text inside the span still forces canonical madd counts, so either collapse runs in the CTC target or keep KD weight high enough that duration comes from the teacher. My guess (speculative): the student's ceiling vs the reference moves by at most the teacher's ~1.2 mishearing points. The student's current interior gap to the teacher is 1.0 point canon (3.34% vs 2.33%), so most of the remaining gain has to come from capacity and data, not from the target.

## Normalisation for student-v1 PER

Headline metric: **canon PER vs the spliced reference**, corpus level. The teacher floor on it is 2.44%, and `e2-libri-full` scores 5.40%.

1. `spliced = teacher[:t0] + ref[r0:r1] + teacher[t1:]`. `(r0, t0)` is the start of the first and `(r1, t1)` the end of the last `equal` opcode of length ≥ 3 in `rapidfuzz.distance.Levenshtein.opcodes(ref, teacher)`. If there is no such opcode, use `teacher`.
2. `canon(s)`: replace ں→ن and ۾→م, delete ڇ, then collapse every run of one phoneme into a single phoneme. Apply it to both the hypothesis and `spliced`.
3. PER = Σ Levenshtein(canon(hyp), canon(spliced)) / Σ len(canon(spliced)).

Keep raw PER vs ref and canon PER vs teacher as secondary columns for continuity. Canon drops shadda and duration by design, consistent with PLAN's rule to grade madd from alignment timing, so missing-shadda and madd errors need their own scored checks (ticket 07). It also makes PER depend on the teacher at the edges. That is fine for distillation but should be stated wherever the number is quoted.
