# E1b spliced-error probe v1

Report v1, teacher `student:spliced-clean`, commit 2d94ea0. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 21 spliced, 13 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 297 | 263 | 136/297 45.8% | 105/297 35.4% | 56/297 18.9% | 290/297 97.6% | 175/297 58.9% | 298 | 283/298 95.0% | 0/298 0.0% | 1/298 0.3% | 0/1910 0.0% | 0/1910 0.0% | posterior-only |
| ع/ء | 297 | 284 | 186/297 62.6% | 70/297 23.6% | 41/297 13.8% | 295/297 99.3% | 225/297 75.8% | 299 | 298/299 99.7% | 0/299 0.0% | 0/299 0.0% | 0/2968 0.0% | 1/2968 0.0% | viable |
| ص/س | 298 | 227 | 131/298 44.0% | 115/298 38.6% | 52/298 17.4% | 296/298 99.3% | 172/298 57.7% | 298 | 282/298 94.6% | 1/298 0.3% | 1/298 0.3% | 1/918 0.1% | 1/918 0.1% | posterior-only |
| ض/د | 287 | 193 | 109/287 38.0% | 119/287 41.5% | 59/287 20.6% | 274/287 95.5% | 142/287 49.5% | 289 | 278/289 96.2% | 0/289 0.0% | 0/289 0.0% | 0/778 0.0% | 1/778 0.1% | mixed |
| ط/ت | 297 | 231 | 107/297 36.0% | 131/297 44.1% | 59/297 19.9% | 275/297 92.6% | 146/297 49.2% | 297 | 278/297 93.6% | 0/297 0.0% | 1/297 0.3% | 0/1438 0.0% | 0/1438 0.0% | mixed |
| ق/ك | 298 | 274 | 125/298 41.9% | 130/298 43.6% | 43/298 14.4% | 298/298 100.0% | 162/298 54.4% | 300 | 277/300 92.3% | 9/300 3.0% | 10/300 3.3% | 9/1840 0.5% | 16/1840 0.9% | posterior-only |
| ذ/ز | 285 | 181 | 47/285 16.5% | 210/285 73.7% | 28/285 9.8% | 253/285 88.8% | 64/285 22.5% | 285 | 267/285 93.7% | 5/285 1.8% | 8/285 2.8% | 3/936 0.3% | 3/936 0.3% | dead |
| ث/س | 281 | 191 | 49/281 17.4% | 181/281 64.4% | 51/281 18.1% | 262/281 93.2% | 80/281 28.5% | 282 | 248/282 87.9% | 8/282 2.8% | 14/282 5.0% | 1/144 0.7% | 2/144 1.4% | mixed |
| missing shadda | 300 | 281 | 255/300 85.0% | 41/300 13.7% | 4/300 1.3% | — | 259/300 86.3% | 300 | 267/300 89.0% | 29/300 9.7% | 31/300 10.3% | 135/1860 7.3% | 145/1860 7.8% | inconclusive |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 147 | 121 | 47/147 32.0% | 65/147 44.2% | 35/147 23.8% | 142/147 96.6% | 69/147 46.9% | 150 | 147/150 98.0% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| ه→ح | 150 | 142 | 89/150 59.3% | 40/150 26.7% | 21/150 14.0% | 148/150 98.7% | 106/150 70.7% | 148 | 136/148 91.9% | 0/148 0.0% | 1/148 0.7% | — | — | viable |
| ع→ء | 148 | 139 | 86/148 58.1% | 47/148 31.8% | 15/148 10.1% | 148/148 100.0% | 105/148 70.9% | 150 | 149/150 99.3% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ء→ع | 149 | 145 | 100/149 67.1% | 23/149 15.4% | 26/149 17.4% | 147/149 98.7% | 120/149 80.5% | 149 | 149/149 100.0% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ص→س | 148 | 98 | 51/148 34.5% | 66/148 44.6% | 31/148 20.9% | 147/148 99.3% | 77/148 52.0% | 148 | 140/148 94.6% | 0/148 0.0% | 0/148 0.0% | — | — | posterior-only |
| س→ص | 150 | 129 | 80/150 53.3% | 49/150 32.7% | 21/150 14.0% | 149/150 99.3% | 95/150 63.3% | 150 | 142/150 94.7% | 1/150 0.7% | 1/150 0.7% | — | — | viable |
| ض→د | 149 | 71 | 54/149 36.2% | 73/149 49.0% | 22/149 14.8% | 144/149 96.6% | 67/149 45.0% | 149 | 144/149 96.6% | 0/149 0.0% | 0/149 0.0% | — | — | dead |
| د→ض | 138 | 122 | 55/138 39.9% | 46/138 33.3% | 37/138 26.8% | 130/138 94.2% | 75/138 54.3% | 140 | 134/140 95.7% | 0/140 0.0% | 0/140 0.0% | — | — | posterior-only |
| ط→ت | 148 | 88 | 59/148 39.9% | 61/148 41.2% | 28/148 18.9% | 146/148 98.6% | 81/148 54.7% | 147 | 134/147 91.2% | 0/147 0.0% | 1/147 0.7% | — | — | posterior-only |
| ت→ط | 149 | 143 | 48/149 32.2% | 70/149 47.0% | 31/149 20.8% | 129/149 86.6% | 65/149 43.6% | 150 | 144/150 96.0% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| ق→ك | 148 | 133 | 70/148 47.3% | 56/148 37.8% | 22/148 14.9% | 148/148 100.0% | 89/148 60.1% | 150 | 132/150 88.0% | 7/150 4.7% | 7/150 4.7% | — | — | posterior-only |
| ك→ق | 150 | 141 | 55/150 36.7% | 74/150 49.3% | 21/150 14.0% | 150/150 100.0% | 73/150 48.7% | 150 | 145/150 96.7% | 2/150 1.3% | 3/150 2.0% | — | — | dead |
| ذ→ز | 135 | 113 | 15/135 11.1% | 115/135 85.2% | 5/135 3.7% | 108/135 80.0% | 18/135 13.3% | 136 | 133/136 97.8% | 0/136 0.0% | 0/136 0.0% | — | — | dead |
| ز→ذ | 150 | 68 | 32/150 21.3% | 95/150 63.3% | 23/150 15.3% | 145/150 96.7% | 46/150 30.7% | 149 | 134/149 89.9% | 5/149 3.4% | 8/149 5.4% | — | — | dead |
| ث→س | 149 | 69 | 40/149 26.8% | 83/149 55.7% | 26/149 17.4% | 149/149 100.0% | 62/149 41.6% | 149 | 119/149 79.9% | 7/149 4.7% | 13/149 8.7% | — | — | void |
| س→ث | 132 | 122 | 9/132 6.8% | 98/132 74.2% | 25/132 18.9% | 113/132 85.6% | 18/132 13.6% | 133 | 129/133 97.0% | 1/133 0.8% | 1/133 0.8% | — | — | dead |

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
