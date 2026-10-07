# E1b spliced-error probe v1

Report v1, teacher `student:combo-clean`, commit 2d94ea0. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 17 spliced, 3 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 295 | 262 | 171/295 58.0% | 64/295 21.7% | 60/295 20.3% | 288/295 97.6% | 217/295 73.6% | 300 | 284/300 94.7% | 0/300 0.0% | 0/300 0.0% | 0/1910 0.0% | 0/1910 0.0% | viable |
| ع/ء | 296 | 283 | 245/296 82.8% | 31/296 10.5% | 20/296 6.8% | 296/296 100.0% | 263/296 88.9% | 298 | 296/298 99.3% | 0/298 0.0% | 0/298 0.0% | 0/2968 0.0% | 0/2968 0.0% | viable |
| ص/س | 298 | 226 | 164/298 55.0% | 106/298 35.6% | 28/298 9.4% | 297/298 99.7% | 186/298 62.4% | 300 | 294/300 98.0% | 1/300 0.3% | 1/300 0.3% | 0/918 0.0% | 0/918 0.0% | viable |
| ض/د | 289 | 195 | 160/289 55.4% | 76/289 26.3% | 53/289 18.3% | 276/289 95.5% | 196/289 67.8% | 290 | 279/290 96.2% | 0/290 0.0% | 0/290 0.0% | 0/778 0.0% | 0/778 0.0% | viable |
| ط/ت | 298 | 232 | 145/298 48.7% | 109/298 36.6% | 44/298 14.8% | 276/298 92.6% | 174/298 58.4% | 300 | 279/300 93.0% | 0/300 0.0% | 0/300 0.0% | 0/1438 0.0% | 0/1438 0.0% | posterior-only |
| ق/ك | 300 | 276 | 157/300 52.3% | 113/300 37.7% | 30/300 10.0% | 298/300 99.3% | 182/300 60.7% | 300 | 280/300 93.3% | 6/300 2.0% | 8/300 2.7% | 10/1840 0.5% | 11/1840 0.6% | viable |
| ذ/ز | 285 | 181 | 83/285 29.1% | 184/285 64.6% | 18/285 6.3% | 263/285 92.3% | 92/285 32.3% | 286 | 256/286 89.5% | 12/286 4.2% | 19/286 6.6% | 10/936 1.1% | 10/936 1.1% | mixed |
| ث/س | 283 | 193 | 73/283 25.8% | 157/283 55.5% | 53/283 18.7% | 271/283 95.8% | 103/283 36.4% | 284 | 249/284 87.7% | 11/284 3.9% | 15/284 5.3% | 2/144 1.4% | 2/144 1.4% | mixed |
| missing shadda | 300 | 281 | 236/300 78.7% | 59/300 19.7% | 5/300 1.7% | — | 240/300 80.0% | 300 | 281/300 93.7% | 15/300 5.0% | 15/300 5.0% | 52/1860 2.8% | 53/1860 2.8% | posterior-only |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 145 | 120 | 61/145 42.1% | 44/145 30.3% | 40/145 27.6% | 140/145 96.6% | 90/145 62.1% | 150 | 147/150 98.0% | 0/150 0.0% | 0/150 0.0% | — | — | posterior-only |
| ه→ح | 150 | 142 | 110/150 73.3% | 20/150 13.3% | 20/150 13.3% | 148/150 98.7% | 127/150 84.7% | 150 | 137/150 91.3% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ع→ء | 147 | 138 | 119/147 81.0% | 18/147 12.2% | 10/147 6.8% | 147/147 100.0% | 129/147 87.8% | 148 | 146/148 98.6% | 0/148 0.0% | 0/148 0.0% | — | — | viable |
| ء→ع | 149 | 145 | 126/149 84.6% | 13/149 8.7% | 10/149 6.7% | 149/149 100.0% | 134/149 89.9% | 150 | 150/150 100.0% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ص→س | 148 | 97 | 55/148 37.2% | 76/148 51.4% | 17/148 11.5% | 147/148 99.3% | 71/148 48.0% | 150 | 149/150 99.3% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| س→ص | 150 | 129 | 109/150 72.7% | 30/150 20.0% | 11/150 7.3% | 150/150 100.0% | 115/150 76.7% | 150 | 145/150 96.7% | 1/150 0.7% | 1/150 0.7% | — | — | viable |
| ض→د | 150 | 71 | 83/150 55.3% | 43/150 28.7% | 24/150 16.0% | 144/150 96.0% | 101/150 67.3% | 150 | 148/150 98.7% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| د→ض | 139 | 124 | 77/139 55.4% | 33/139 23.7% | 29/139 20.9% | 132/139 95.0% | 95/139 68.3% | 140 | 131/140 93.6% | 0/140 0.0% | 0/140 0.0% | — | — | viable |
| ط→ت | 148 | 88 | 95/148 64.2% | 33/148 22.3% | 20/148 13.5% | 147/148 99.3% | 109/148 73.6% | 150 | 134/150 89.3% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ت→ط | 150 | 144 | 50/150 33.3% | 76/150 50.7% | 24/150 16.0% | 129/150 86.0% | 65/150 43.3% | 150 | 145/150 96.7% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| ق→ك | 150 | 135 | 84/150 56.0% | 53/150 35.3% | 13/150 8.7% | 149/150 99.3% | 95/150 63.3% | 150 | 137/150 91.3% | 2/150 1.3% | 3/150 2.0% | — | — | viable |
| ك→ق | 150 | 141 | 73/150 48.7% | 60/150 40.0% | 17/150 11.3% | 149/150 99.3% | 87/150 58.0% | 150 | 143/150 95.3% | 4/150 2.7% | 5/150 3.3% | — | — | posterior-only |
| ذ→ز | 135 | 113 | 16/135 11.9% | 115/135 85.2% | 4/135 3.0% | 116/135 85.9% | 17/135 12.6% | 136 | 133/136 97.8% | 0/136 0.0% | 0/136 0.0% | — | — | dead |
| ز→ذ | 150 | 68 | 67/150 44.7% | 69/150 46.0% | 14/150 9.3% | 147/150 98.0% | 75/150 50.0% | 150 | 123/150 82.0% | 12/150 8.0% | 19/150 12.7% | — | — | inconclusive |
| ث→س | 149 | 69 | 67/149 45.0% | 63/149 42.3% | 19/149 12.8% | 148/149 99.3% | 85/149 57.0% | 150 | 120/150 80.0% | 10/150 6.7% | 14/150 9.3% | — | — | inconclusive |
| س→ث | 134 | 124 | 6/134 4.5% | 94/134 70.1% | 34/134 25.4% | 123/134 91.8% | 18/134 13.4% | 134 | 129/134 96.3% | 1/134 0.7% | 1/134 0.7% | — | — | dead |

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
