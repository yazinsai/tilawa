# E1b spliced-error probe v1

Report v1, teacher `obadx/muaalem-model-v3_2`, commit 005fc6d. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 6 spliced, 3 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 297 | 263 | 170/297 57.2% | 85/297 28.6% | 42/297 14.1% | 255/297 85.9% | 194/297 65.3% | 298 | 288/298 96.6% | 0/298 0.0% | 2/298 0.7% | 0/1910 0.0% | 1/1910 0.1% | viable |
| ع/ء | 300 | 287 | 217/300 72.3% | 48/300 16.0% | 35/300 11.7% | 294/300 98.0% | 239/300 79.7% | 300 | 294/300 98.0% | 0/300 0.0% | 2/300 0.7% | 0/2968 0.0% | 1/2968 0.0% | viable |
| ص/س | 299 | 227 | 180/299 60.2% | 90/299 30.1% | 29/299 9.7% | 286/299 95.7% | 203/299 67.9% | 300 | 290/300 96.7% | 2/300 0.7% | 2/300 0.7% | 3/918 0.3% | 2/918 0.2% | viable |
| ض/د | 289 | 195 | 207/289 71.6% | 51/289 17.6% | 31/289 10.7% | 275/289 95.2% | 225/289 77.9% | 291 | 280/291 96.2% | 0/291 0.0% | 0/291 0.0% | 0/778 0.0% | 0/778 0.0% | viable |
| ط/ت | 300 | 232 | 140/300 46.7% | 91/300 30.3% | 69/300 23.0% | 259/300 86.3% | 182/300 60.7% | 300 | 268/300 89.3% | 0/300 0.0% | 1/300 0.3% | 0/1438 0.0% | 0/1438 0.0% | posterior-only |
| ق/ك | 300 | 276 | 156/300 52.0% | 130/300 43.3% | 14/300 4.7% | 299/300 99.7% | 166/300 55.3% | 300 | 286/300 95.3% | 3/300 1.0% | 5/300 1.7% | 6/1840 0.3% | 7/1840 0.4% | viable |
| ذ/ز | 286 | 182 | 99/286 34.6% | 155/286 54.2% | 32/286 11.2% | 257/286 89.9% | 115/286 40.2% | 286 | 253/286 88.5% | 11/286 3.8% | 20/286 7.0% | 2/936 0.2% | 1/936 0.1% | mixed |
| ث/س | 284 | 193 | 80/284 28.2% | 167/284 58.8% | 37/284 13.0% | 234/284 82.4% | 99/284 34.9% | 283 | 239/283 84.5% | 13/283 4.6% | 27/283 9.5% | 1/144 0.7% | 1/144 0.7% | mixed |
| missing shadda | 300 | 281 | 199/300 66.3% | 99/300 33.0% | 2/300 0.7% | — | 201/300 67.0% | 300 | 297/300 99.0% | 2/300 0.7% | 3/300 1.0% | 7/1860 0.4% | 6/1860 0.3% | viable |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 147 | 121 | 66/147 44.9% | 55/147 37.4% | 26/147 17.7% | 124/147 84.4% | 82/147 55.8% | 148 | 145/148 98.0% | 0/148 0.0% | 1/148 0.7% | — | — | posterior-only |
| ه→ح | 150 | 142 | 104/150 69.3% | 30/150 20.0% | 16/150 10.7% | 131/150 87.3% | 112/150 74.7% | 150 | 143/150 95.3% | 0/150 0.0% | 1/150 0.7% | — | — | viable |
| ع→ء | 150 | 141 | 98/150 65.3% | 31/150 20.7% | 21/150 14.0% | 147/150 98.0% | 110/150 73.3% | 150 | 148/150 98.7% | 0/150 0.0% | 1/150 0.7% | — | — | viable |
| ء→ع | 150 | 146 | 119/150 79.3% | 17/150 11.3% | 14/150 9.3% | 147/150 98.0% | 129/150 86.0% | 150 | 146/150 97.3% | 0/150 0.0% | 1/150 0.7% | — | — | viable |
| ص→س | 149 | 98 | 82/149 55.0% | 54/149 36.2% | 13/149 8.7% | 147/149 98.7% | 91/149 61.1% | 150 | 146/150 97.3% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| س→ص | 150 | 129 | 98/150 65.3% | 36/150 24.0% | 16/150 10.7% | 139/150 92.7% | 112/150 74.7% | 150 | 144/150 96.0% | 2/150 1.3% | 2/150 1.3% | — | — | viable |
| ض→د | 148 | 70 | 91/148 61.5% | 39/148 26.4% | 18/148 12.2% | 139/148 93.9% | 103/148 69.6% | 150 | 145/150 96.7% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| د→ض | 141 | 125 | 116/141 82.3% | 12/141 8.5% | 13/141 9.2% | 136/141 96.5% | 122/141 86.5% | 141 | 135/141 95.7% | 0/141 0.0% | 0/141 0.0% | — | — | viable |
| ط→ت | 150 | 88 | 105/150 70.0% | 11/150 7.3% | 34/150 22.7% | 145/150 96.7% | 126/150 84.0% | 150 | 121/150 80.7% | 0/150 0.0% | 1/150 0.7% | — | — | viable |
| ت→ط | 150 | 144 | 35/150 23.3% | 80/150 53.3% | 35/150 23.3% | 114/150 76.0% | 56/150 37.3% | 150 | 147/150 98.0% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| ق→ك | 150 | 135 | 77/150 51.3% | 62/150 41.3% | 11/150 7.3% | 150/150 100.0% | 88/150 58.7% | 150 | 145/150 96.7% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ك→ق | 150 | 141 | 79/150 52.7% | 68/150 45.3% | 3/150 2.0% | 149/150 99.3% | 78/150 52.0% | 150 | 141/150 94.0% | 3/150 2.0% | 5/150 3.3% | — | — | viable |
| ذ→ز | 136 | 114 | 18/136 13.2% | 111/136 81.6% | 7/136 5.1% | 115/136 84.6% | 18/136 13.2% | 136 | 133/136 97.8% | 0/136 0.0% | 0/136 0.0% | — | — | dead |
| ز→ذ | 150 | 68 | 81/150 54.0% | 44/150 29.3% | 25/150 16.7% | 142/150 94.7% | 97/150 64.7% | 150 | 120/150 80.0% | 11/150 7.3% | 20/150 13.3% | — | — | inconclusive |
| ث→س | 150 | 69 | 63/150 42.0% | 66/150 44.0% | 21/150 14.0% | 147/150 98.0% | 78/150 52.0% | 150 | 110/150 73.3% | 13/150 8.7% | 27/150 18.0% | — | — | void |
| س→ث | 134 | 124 | 17/134 12.7% | 101/134 75.4% | 16/134 11.9% | 87/134 64.9% | 21/134 15.7% | 133 | 129/133 97.0% | 0/133 0.0% | 0/133 0.0% | — | — | dead |

## How the clips are made

- Each clean ayah's canonical phonemes are CTC force-aligned on the teacher's own posteriors (`viterbi` in `tools/e1_report.py`). Each blank gap between two tokens is split at its midpoint, giving contiguous per-phoneme segments.
- **Letter pairs**: a pair consonant plus its short vowel is replaced by the partner consonant plus the same vowel, cut from another ayah by the same reciter (segment duration within 0.6–1.6×). Cuts snap to zero crossings and join with an 8 ms raised-cosine crossfade, loudness-matched.
- **Missing shadda**: the middle of a doubled consonant is cut out until it lasts as long as that reciter's median single consonant of the same letter. Only geminates at least 1/0.7 that long are used.
- **Controls** carry the same seam with no error. Letters: the same consonant plus vowel from another ayah. Shadda: the same kind of internal cut at the hold's midpoint, removing only 15%, so the consonant stays doubled.
- **Clean** is the unspliced ayah scored at the same kinds of sites: the teacher's floor without any seam.
- Clips are trimmed at pauses at least 1.5 s from the splice and scored only at the spliced site.

## Verdict rule

- Positives come from splices; negatives come from **controls**, so the FP side carries the same seam as the positives.
- `void`: fewer than 80% of controls decode as the canonical phoneme. The seam garbles the sound, so the class's numbers mean nothing.
- `viable`: greedy recall ≥ 50% and control greedy FP ≤ 2%.
- `posterior-only`: posterior-preference recall ≥ 50% and control posterior FP ≤ 5%.
- `dead`: neither, with ≥ 20 positives from ≥ 30 sites, and both greedy and posterior recall under 50%. The teacher can't hear the class even on clean splices.
- `mixed`: a pair that pools to `dead` but has a direction that isn't. Read its directions instead.
- `inconclusive`: the recall is there, but control FP is over the bound. The teacher reacts to the seam, so the splice can't separate the error from the artifact either way.
- Clean FP for ث/س counts only ث sites: a clean س is filed under ص/س. Its FP still counts both of س's confusables (ص and ث), as in E1.
- Definitions of the other columns match `e1-teacher-bias-v1.md`. **control intact** is the share of controls whose greedy decode keeps the canonical phoneme (for shadda, both copies).
