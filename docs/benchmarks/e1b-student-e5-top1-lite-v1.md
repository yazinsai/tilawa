# E1b spliced-error probe v1

Report v1, teacher `student:e5-top1-lite`, commit 3e91183. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 98 spliced, 85 control, 4 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 288 | 257 | 166/288 57.6% | 38/288 13.2% | 84/288 29.2% | 268/288 93.1% | 215/288 74.7% | 288 | 249/288 86.5% | 0/288 0.0% | 2/288 0.7% | 0/1909 0.0% | 1/1909 0.1% | viable |
| ع/ء | 289 | 276 | 224/289 77.5% | 38/289 13.1% | 27/289 9.3% | 285/289 98.6% | 254/289 87.9% | 293 | 277/293 94.5% | 1/293 0.3% | 3/293 1.0% | 2/2966 0.1% | 5/2966 0.2% | viable |
| ص/س | 285 | 218 | 154/285 54.0% | 98/285 34.4% | 33/285 11.6% | 271/285 95.1% | 175/285 61.4% | 286 | 267/286 93.4% | 4/286 1.4% | 7/286 2.4% | 13/915 1.4% | 19/915 2.1% | viable |
| ض/د | 280 | 190 | 119/280 42.5% | 52/280 18.6% | 109/280 38.9% | 243/280 86.8% | 181/280 64.6% | 280 | 206/280 73.6% | 1/280 0.4% | 2/280 0.7% | 2/778 0.3% | 6/778 0.8% | void |
| ط/ت | 290 | 230 | 142/290 49.0% | 74/290 25.5% | 74/290 25.5% | 261/290 90.0% | 189/290 65.2% | 293 | 238/293 81.2% | 1/293 0.3% | 3/293 1.0% | 5/1437 0.3% | 8/1437 0.6% | posterior-only |
| ق/ك | 290 | 266 | 158/290 54.5% | 97/290 33.4% | 35/290 12.1% | 286/290 98.6% | 179/290 61.7% | 294 | 254/294 86.4% | 13/294 4.4% | 18/294 6.1% | 78/1840 4.2% | 99/1840 5.4% | inconclusive |
| ذ/ز | 279 | 176 | 102/279 36.6% | 106/279 38.0% | 71/279 25.4% | 230/279 82.4% | 143/279 51.3% | 283 | 187/283 66.1% | 36/283 12.7% | 66/283 23.3% | 53/936 5.7% | 66/936 7.1% | void |
| ث/س | 268 | 190 | 100/268 37.3% | 92/268 34.3% | 76/268 28.4% | 246/268 91.8% | 135/268 50.4% | 266 | 158/266 59.4% | 33/266 12.4% | 54/266 20.3% | 40/143 28.0% | 54/143 37.8% | void |
| missing shadda | 294 | 277 | 242/294 82.3% | 50/294 17.0% | 2/294 0.7% | — | 239/294 81.3% | 293 | 244/293 83.3% | 35/293 11.9% | 19/293 6.5% | 301/1860 16.2% | 158/1860 8.5% | inconclusive |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 142 | 119 | 63/142 44.4% | 26/142 18.3% | 53/142 37.3% | 127/142 89.4% | 90/142 63.4% | 148 | 140/148 94.6% | 0/148 0.0% | 1/148 0.7% | — | — | posterior-only |
| ه→ح | 146 | 138 | 103/146 70.5% | 12/146 8.2% | 31/146 21.2% | 141/146 96.6% | 125/146 85.6% | 140 | 109/140 77.9% | 0/140 0.0% | 1/140 0.7% | — | — | void |
| ع→ء | 143 | 134 | 123/143 86.0% | 13/143 9.1% | 7/143 4.9% | 143/143 100.0% | 131/143 91.6% | 146 | 132/146 90.4% | 1/146 0.7% | 3/146 2.1% | — | — | viable |
| ء→ع | 146 | 142 | 101/146 69.2% | 25/146 17.1% | 20/146 13.7% | 142/146 97.3% | 123/146 84.2% | 147 | 145/147 98.6% | 0/147 0.0% | 0/147 0.0% | — | — | viable |
| ص→س | 142 | 94 | 71/142 50.0% | 49/142 34.5% | 22/142 15.5% | 136/142 95.8% | 85/142 59.9% | 142 | 128/142 90.1% | 2/142 1.4% | 4/142 2.8% | — | — | viable |
| س→ص | 143 | 124 | 83/143 58.0% | 49/143 34.3% | 11/143 7.7% | 135/143 94.4% | 90/143 62.9% | 144 | 139/144 96.5% | 2/144 1.4% | 3/144 2.1% | — | — | viable |
| ض→د | 148 | 71 | 65/148 43.9% | 40/148 27.0% | 43/148 29.1% | 135/148 91.2% | 92/148 62.2% | 148 | 111/148 75.0% | 1/148 0.7% | 2/148 1.4% | — | — | void |
| د→ض | 132 | 119 | 54/132 40.9% | 12/132 9.1% | 66/132 50.0% | 108/132 81.8% | 89/132 67.4% | 132 | 95/132 72.0% | 0/132 0.0% | 0/132 0.0% | — | — | void |
| ط→ت | 141 | 87 | 92/141 65.2% | 11/141 7.8% | 38/141 27.0% | 138/141 97.9% | 121/141 85.8% | 143 | 106/143 74.1% | 1/143 0.7% | 3/143 2.1% | — | — | void |
| ت→ط | 149 | 143 | 50/149 33.6% | 63/149 42.3% | 36/149 24.2% | 123/149 82.6% | 68/149 45.6% | 150 | 132/150 88.0% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| ق→ك | 142 | 127 | 91/142 64.1% | 32/142 22.5% | 19/142 13.4% | 140/142 98.6% | 105/142 73.9% | 147 | 117/147 79.6% | 9/147 6.1% | 11/147 7.5% | — | — | void |
| ك→ق | 148 | 139 | 67/148 45.3% | 65/148 43.9% | 16/148 10.8% | 146/148 98.6% | 74/148 50.0% | 147 | 137/147 93.2% | 4/147 2.7% | 7/147 4.8% | — | — | posterior-only |
| ذ→ز | 130 | 108 | 22/130 16.9% | 88/130 67.7% | 20/130 15.4% | 88/130 67.7% | 23/130 17.7% | 135 | 115/135 85.2% | 0/135 0.0% | 2/135 1.5% | — | — | dead |
| ز→ذ | 149 | 68 | 80/149 53.7% | 18/149 12.1% | 51/149 34.2% | 142/149 95.3% | 120/149 80.5% | 148 | 72/148 48.6% | 36/148 24.3% | 64/148 43.2% | — | — | void |
| ث→س | 137 | 69 | 94/137 68.6% | 17/137 12.4% | 26/137 19.0% | 134/137 97.8% | 113/137 82.5% | 133 | 33/133 24.8% | 33/133 24.8% | 51/133 38.3% | — | — | void |
| س→ث | 131 | 121 | 6/131 4.6% | 75/131 57.3% | 50/131 38.2% | 112/131 85.5% | 22/131 16.8% | 133 | 125/133 94.0% | 0/133 0.0% | 3/133 2.3% | — | — | dead |

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
