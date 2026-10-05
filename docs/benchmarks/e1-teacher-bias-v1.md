# E1 teacher-bias report v1

Report v1, teacher `obadx/muaalem-model-v3_2`, commit 7b28629. Data: `obadx/qdat_bench` (5:109 fragment) and `azizbekphd/ikhlas_recitations` (112:1–4).

| dataset | clips | scored | skipped |
|---|---|---|---|
| `obadx/qdat_bench` | 159 | 159 | 0 |
| `azizbekphd/ikhlas_recitations` | 1503 | 1453 | 50 teacher decode too far from label |

## Per class

Positives are labelled errors at sites where the dataset's usual label matches the canonical. Negatives are correctly said tokens of the class.

| class | positives | sources | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | negatives | greedy FP | top-5 FP | posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| short vowel | 21 | 5 | 9/21 42.9% | 6/21 28.6% | 6/21 28.6% | 20/21 95.2% | 12/21 57.1% | 13408 | 63/13408 0.5% | 8528/13408 63.6% | 145/13408 1.1% | posterior-only |
| ح/ه | 0 | 0 | — | — | — | — | — | 2167 | 2/2167 0.1% | 131/2167 6.0% | 4/2167 0.2% | unsupported |
| ع/ء | 1 | 1 | 1/1 100.0% | 0/1 0.0% | 0/1 0.0% | 1/1 100.0% | 1/1 100.0% | 635 | 1/635 0.2% | 351/635 55.3% | 2/635 0.3% | unsupported |
| ص/س | 0 | 0 | — | — | — | — | — | 730 | 33/730 4.5% | 641/730 87.8% | 73/730 10.0% | unsupported |
| ض/د | 0 | 0 | — | — | — | — | — | 1821 | 1/1821 0.1% | 191/1821 10.5% | 1/1821 0.1% | unsupported |
| ط/ت | 0 | 0 | — | — | — | — | — | 158 | 0/158 0.0% | 49/158 31.0% | 0/158 0.0% | unsupported |
| ق/ك | 0 | 0 | — | — | — | — | — | 1395 | 29/1395 2.1% | 1270/1395 91.0% | 40/1395 2.9% | unsupported |
| ذ/ز | 0 | 0 | — | — | — | — | — | 0 | — | — | — | unsupported |
| ث/س | 0 | 0 | — | — | — | — | — | 0 | — | — | — | unsupported |
| missing shadda | 4 | 1 | 1/4 25.0% | 3/4 75.0% | 0/4 0.0% | — | 1/4 25.0% | 1764 | 98/1764 5.6% | — | 111/1764 6.3% | unsupported |

## Definitions

- **in greedy**: the greedy decode at the aligned site is the phoneme actually said.
- **canonicalized**: the error was said, but the greedy decode emitted the canonical phoneme.
- **greedy other**: the site was deleted or decoded as some third phoneme.
- **in top-5**: the said phoneme is in the top-5 of at least one frame of its CTC forced-aligned span.
- **posterior prefers error**: the CTC Viterbi score of the labelled sequence beats the same sequence with the site corrected to canonical. For negatives (posterior FP), some confusable variant beats the labelled sequence.
- **FP** columns are measured on negatives, against the class's confusables: the pair partner, or either other short vowel. For shadda, the FP is the doubled consonant decoded or scored as single.
- Clips whose greedy decode is more than 50% phoneme edits from the label are skipped, because many Ikhlas clips are truncated.

## Verdict rule

- Fewer than 20 positives: `unsupported` until scripted-error recordings (ticket 05) supply positives.
- `viable`: greedy recall ≥ 50% and greedy FP ≤ 2%.
- `posterior-only`: posterior-preference recall ≥ 50% and posterior FP ≤ 5%. Top-5 presence is reported but not used: a confusable sits in the top-5 of most correct tokens, so it can't separate errors from correct speech.
- Otherwise `unsupported`.

## Caveats

- Both datasets are a single passage each (one 5:109 fragment, al-Ikhlas), so **sources** (distinct canonical sites behind the positives) is the real sample size, not the positive count. A verdict built on a handful of sites is provisional until ticket 05's scripted recordings re-run it.
- Neither dataset labels letter-pair errors. qdat_bench labels madd, ghunna and qalqalah; Ikhlas labels mostly qalqalah, madd and a few vowel and idgham errors. Ikhlas's one "wrong makhraj" symbol never occurs in its transcripts.
- The only missing-shadda positives are an unassimilated noon in يَكُن لَّهُۥ (idgham dropped), not a dropped shadda inside a word.
- Negatives still measure each pair's false-positive floor on correct speech, which the checker needs whatever the recall turns out to be.

## Positives

| class | clip | ayah | canonical | said | greedy |
|---|---|---|---|---|---|
| short vowel | a298aab2 | 5:109 | ُ | ِ | ِ |
| short vowel | ID86V3F | 112:3 | ِ | َ | َ |
| short vowel | ID85V3F | 112:3 | ِ | َ | َ |
| short vowel | ID69V3F | 112:3 | ِ | َ | َ |
| short vowel | ID97V4F | 112:4 | َ | ُ | (deleted) |
| short vowel | ID57V3F | 112:3 | ِ | َ | َ |
| short vowel | ID82V4F | 112:4 | َ | ُ | (deleted) |
| short vowel | ID165V3F | 112:3 | ِ | َ | َ |
| short vowel | ID174V3F | 112:3 | ِ | َ | َ |
| short vowel | ID100V3F | 112:3 | ِ | َ | ِ |
| short vowel | ID105V4F | 112:4 | َ | ُ | و |
| short vowel | ID63V4F | 112:4 | َ | ُ | (deleted) |
| short vowel | ID64V3F | 112:3 | ِ | َ | َ |
| short vowel | ID178V3F | 112:3 | ِ | َ | ِ |
| short vowel | ID61V3F | 112:3 | ِ | َ | َ |
| short vowel | ID62V4F | 112:4 | َ | ُ | َ |
| short vowel | ID58V4F | 112:4 | َ | ُ | (deleted) |
| short vowel | ID143V4F | 112:4 | ُ | َ | ُ |
| short vowel | ID257V4F | 112:4 | َ | ِ | َ |
| short vowel | ID257V4F | 112:4 | َ | ُ | َ |
| short vowel | ID270V4F | 112:4 | ُ | َ | ِ |
| ع/ء | 289cbb94 | 5:109 | ء | ع | ع |
| missing shadda | ID114V4F | 112:4 | لل | نل | لل |
| missing shadda | ID194V4F | 112:4 | لل | نل | نل |
| missing shadda | ID196V4F | 112:4 | لل | نل | لل |
| missing shadda | ID273V4F | 112:4 | لل | نل | لل |
