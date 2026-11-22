# E1b spliced-error probe v1

Report v1, teacher `student:e5-soft5-lite`, commit 3e91183. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 40 spliced, 23 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 295 | 261 | 177/295 60.0% | 41/295 13.9% | 77/295 26.1% | 283/295 95.9% | 242/295 82.0% | 300 | 273/300 91.0% | 1/300 0.3% | 2/300 0.7% | 0/1910 0.0% | 1/1910 0.1% | viable |
| ع/ء | 295 | 282 | 235/295 79.7% | 22/295 7.5% | 38/295 12.9% | 294/295 99.7% | 263/295 89.2% | 298 | 284/298 95.3% | 0/298 0.0% | 0/298 0.0% | 0/2968 0.0% | 2/2968 0.1% | viable |
| ص/س | 293 | 224 | 163/293 55.6% | 71/293 24.2% | 59/293 20.1% | 289/293 98.6% | 203/293 69.3% | 295 | 264/295 89.5% | 3/295 1.0% | 7/295 2.4% | 2/918 0.2% | 5/918 0.5% | viable |
| ض/د | 286 | 194 | 132/286 46.2% | 70/286 24.5% | 84/286 29.4% | 269/286 94.1% | 183/286 64.0% | 286 | 249/286 87.1% | 0/286 0.0% | 0/286 0.0% | 0/778 0.0% | 2/778 0.3% | posterior-only |
| ط/ت | 297 | 232 | 132/297 44.4% | 70/297 23.6% | 95/297 32.0% | 270/297 90.9% | 195/297 65.7% | 297 | 253/297 85.2% | 1/297 0.3% | 2/297 0.7% | 2/1438 0.1% | 3/1438 0.2% | posterior-only |
| ق/ك | 296 | 272 | 166/296 56.1% | 78/296 26.4% | 52/296 17.6% | 292/296 98.6% | 206/296 69.6% | 298 | 254/298 85.2% | 6/298 2.0% | 14/298 4.7% | 48/1840 2.6% | 68/1840 3.7% | posterior-only |
| ذ/ز | 282 | 182 | 98/282 34.8% | 135/282 47.9% | 49/282 17.4% | 232/282 82.3% | 121/282 42.9% | 286 | 226/286 79.0% | 32/286 11.2% | 44/286 15.4% | 28/936 3.0% | 31/936 3.3% | void |
| ث/س | 278 | 189 | 82/278 29.5% | 119/278 42.8% | 77/278 27.7% | 265/278 95.3% | 125/278 45.0% | 279 | 196/279 70.3% | 21/279 7.5% | 39/279 14.0% | 12/144 8.3% | 18/144 12.5% | void |
| missing shadda | 299 | 280 | 257/299 86.0% | 31/299 10.4% | 11/299 3.7% | — | 267/299 89.3% | 299 | 253/299 84.6% | 43/299 14.4% | 41/299 13.7% | 245/1860 13.2% | 224/1860 12.0% | inconclusive |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 147 | 121 | 70/147 47.6% | 27/147 18.4% | 50/147 34.0% | 140/147 95.2% | 114/147 77.6% | 150 | 144/150 96.0% | 1/150 0.7% | 2/150 1.3% | — | — | posterior-only |
| ه→ح | 148 | 140 | 107/148 72.3% | 14/148 9.5% | 27/148 18.2% | 143/148 96.6% | 128/148 86.5% | 150 | 129/150 86.0% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ع→ء | 146 | 137 | 121/146 82.9% | 8/146 5.5% | 17/146 11.6% | 146/146 100.0% | 135/146 92.5% | 149 | 138/149 92.6% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ء→ع | 149 | 145 | 114/149 76.5% | 14/149 9.4% | 21/149 14.1% | 148/149 99.3% | 128/149 85.9% | 149 | 146/149 98.0% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ص→س | 143 | 95 | 64/143 44.8% | 46/143 32.2% | 33/143 23.1% | 140/143 97.9% | 85/143 59.4% | 145 | 129/145 89.0% | 1/145 0.7% | 2/145 1.4% | — | — | posterior-only |
| س→ص | 150 | 129 | 99/150 66.0% | 25/150 16.7% | 26/150 17.3% | 149/150 99.3% | 118/150 78.7% | 150 | 135/150 90.0% | 2/150 1.3% | 5/150 3.3% | — | — | viable |
| ض→د | 148 | 71 | 56/148 37.8% | 56/148 37.8% | 36/148 24.3% | 139/148 93.9% | 77/148 52.0% | 148 | 136/148 91.9% | 0/148 0.0% | 0/148 0.0% | — | — | posterior-only |
| د→ض | 138 | 123 | 76/138 55.1% | 14/138 10.1% | 48/138 34.8% | 130/138 94.2% | 106/138 76.8% | 138 | 113/138 81.9% | 0/138 0.0% | 0/138 0.0% | — | — | viable |
| ط→ت | 147 | 88 | 80/147 54.4% | 14/147 9.5% | 53/147 36.1% | 144/147 98.0% | 124/147 84.4% | 148 | 113/148 76.4% | 1/148 0.7% | 2/148 1.4% | — | — | void |
| ت→ط | 150 | 144 | 52/150 34.7% | 56/150 37.3% | 42/150 28.0% | 126/150 84.0% | 71/150 47.3% | 149 | 140/149 94.0% | 0/149 0.0% | 0/149 0.0% | — | — | dead |
| ق→ك | 148 | 133 | 93/148 62.8% | 28/148 18.9% | 27/148 18.2% | 147/148 99.3% | 114/148 77.0% | 149 | 125/149 83.9% | 2/149 1.3% | 4/149 2.7% | — | — | viable |
| ك→ق | 148 | 139 | 73/148 49.3% | 50/148 33.8% | 25/148 16.9% | 145/148 98.0% | 92/148 62.2% | 149 | 129/149 86.6% | 4/149 2.7% | 10/149 6.7% | — | — | inconclusive |
| ذ→ز | 135 | 114 | 26/135 19.3% | 92/135 68.1% | 17/135 12.6% | 89/135 65.9% | 29/135 21.5% | 136 | 125/136 91.9% | 3/136 2.2% | 3/136 2.2% | — | — | dead |
| ز→ذ | 147 | 68 | 72/147 49.0% | 43/147 29.3% | 32/147 21.8% | 143/147 97.3% | 92/147 62.6% | 150 | 101/150 67.3% | 29/150 19.3% | 41/150 27.3% | — | — | void |
| ث→س | 148 | 69 | 76/148 51.4% | 35/148 23.6% | 37/148 25.0% | 146/148 98.6% | 104/148 70.3% | 147 | 71/147 48.3% | 21/147 14.3% | 39/147 26.5% | — | — | void |
| س→ث | 130 | 120 | 6/130 4.6% | 84/130 64.6% | 40/130 30.8% | 119/130 91.5% | 21/130 16.2% | 132 | 125/132 94.7% | 0/132 0.0% | 0/132 0.0% | — | — | dead |

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
