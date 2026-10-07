# E1b spliced-error probe v1

Report v1, teacher `student:combo-libri-soft5-full`, commit 15f6443. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 27 spliced, 18 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 296 | 262 | 178/296 60.1% | 62/296 20.9% | 56/296 18.9% | 292/296 98.6% | 216/296 73.0% | 298 | 283/298 95.0% | 0/298 0.0% | 0/298 0.0% | 0/1910 0.0% | 0/1910 0.0% | viable |
| ع/ء | 297 | 284 | 236/297 79.5% | 26/297 8.8% | 35/297 11.8% | 296/297 99.7% | 264/297 88.9% | 299 | 293/299 98.0% | 0/299 0.0% | 1/299 0.3% | 0/2968 0.0% | 1/2968 0.0% | viable |
| ص/س | 297 | 225 | 149/297 50.2% | 104/297 35.0% | 44/297 14.8% | 295/297 99.3% | 181/297 60.9% | 298 | 284/298 95.3% | 2/298 0.7% | 2/298 0.7% | 1/918 0.1% | 1/918 0.1% | viable |
| ض/د | 286 | 194 | 149/286 52.1% | 88/286 30.8% | 49/286 17.1% | 273/286 95.5% | 179/286 62.6% | 287 | 278/287 96.9% | 0/287 0.0% | 0/287 0.0% | 1/778 0.1% | 1/778 0.1% | viable |
| ط/ت | 296 | 231 | 137/296 46.3% | 99/296 33.4% | 60/296 20.3% | 266/296 89.9% | 178/296 60.1% | 298 | 271/298 90.9% | 0/298 0.0% | 1/298 0.3% | 0/1438 0.0% | 0/1438 0.0% | posterior-only |
| ق/ك | 299 | 275 | 160/299 53.5% | 109/299 36.5% | 30/299 10.0% | 297/299 99.3% | 187/299 62.5% | 299 | 277/299 92.6% | 8/299 2.7% | 9/299 3.0% | 12/1840 0.7% | 16/1840 0.9% | posterior-only |
| ذ/ز | 285 | 182 | 76/285 26.7% | 172/285 60.4% | 37/285 13.0% | 243/285 85.3% | 97/285 34.0% | 285 | 250/285 87.7% | 10/285 3.5% | 23/285 8.1% | 9/936 1.0% | 11/936 1.2% | mixed |
| ث/س | 280 | 190 | 57/280 20.4% | 174/280 62.1% | 49/280 17.5% | 267/280 95.4% | 84/280 30.0% | 281 | 249/281 88.6% | 5/281 1.8% | 8/281 2.8% | 0/144 0.0% | 0/144 0.0% | dead |
| missing shadda | 298 | 280 | 229/298 76.8% | 67/298 22.5% | 2/298 0.7% | — | 231/298 77.5% | 298 | 279/298 93.6% | 16/298 5.4% | 17/298 5.7% | 62/1860 3.3% | 53/1860 2.8% | inconclusive |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 147 | 121 | 81/147 55.1% | 30/147 20.4% | 36/147 24.5% | 143/147 97.3% | 104/147 70.7% | 149 | 147/149 98.7% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ه→ح | 149 | 141 | 97/149 65.1% | 32/149 21.5% | 20/149 13.4% | 149/149 100.0% | 112/149 75.2% | 149 | 136/149 91.3% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ع→ء | 149 | 140 | 122/149 81.9% | 11/149 7.4% | 16/149 10.7% | 149/149 100.0% | 137/149 91.9% | 150 | 145/150 96.7% | 0/150 0.0% | 1/150 0.7% | — | — | viable |
| ء→ع | 148 | 144 | 114/148 77.0% | 15/148 10.1% | 19/148 12.8% | 147/148 99.3% | 127/148 85.8% | 149 | 148/149 99.3% | 0/149 0.0% | 0/149 0.0% | — | — | viable |
| ص→س | 148 | 97 | 46/148 31.1% | 74/148 50.0% | 28/148 18.9% | 146/148 98.6% | 68/148 45.9% | 148 | 147/148 99.3% | 0/148 0.0% | 0/148 0.0% | — | — | dead |
| س→ص | 149 | 128 | 103/149 69.1% | 30/149 20.1% | 16/149 10.7% | 149/149 100.0% | 113/149 75.8% | 150 | 137/150 91.3% | 2/150 1.3% | 2/150 1.3% | — | — | viable |
| ض→د | 149 | 71 | 68/149 45.6% | 60/149 40.3% | 21/149 14.1% | 143/149 96.0% | 81/149 54.4% | 149 | 145/149 97.3% | 0/149 0.0% | 0/149 0.0% | — | — | posterior-only |
| د→ض | 137 | 123 | 81/137 59.1% | 28/137 20.4% | 28/137 20.4% | 130/137 94.9% | 98/137 71.5% | 138 | 133/138 96.4% | 0/138 0.0% | 0/138 0.0% | — | — | viable |
| ط→ت | 147 | 88 | 83/147 56.5% | 35/147 23.8% | 29/147 19.7% | 144/147 98.0% | 110/147 74.8% | 149 | 128/149 85.9% | 0/149 0.0% | 1/149 0.7% | — | — | viable |
| ت→ط | 149 | 143 | 54/149 36.2% | 64/149 43.0% | 31/149 20.8% | 122/149 81.9% | 68/149 45.6% | 149 | 143/149 96.0% | 0/149 0.0% | 0/149 0.0% | — | — | dead |
| ق→ك | 150 | 135 | 96/150 64.0% | 39/150 26.0% | 15/150 10.0% | 150/150 100.0% | 111/150 74.0% | 149 | 134/149 89.9% | 5/149 3.4% | 5/149 3.4% | — | — | posterior-only |
| ك→ق | 149 | 140 | 64/149 43.0% | 70/149 47.0% | 15/149 10.1% | 147/149 98.7% | 76/149 51.0% | 150 | 143/150 95.3% | 3/150 2.0% | 4/150 2.7% | — | — | posterior-only |
| ذ→ز | 136 | 114 | 14/136 10.3% | 111/136 81.6% | 11/136 8.1% | 101/136 74.3% | 17/136 12.5% | 135 | 131/135 97.0% | 1/135 0.7% | 1/135 0.7% | — | — | dead |
| ز→ذ | 149 | 68 | 62/149 41.6% | 61/149 40.9% | 26/149 17.4% | 142/149 95.3% | 80/149 53.7% | 150 | 119/150 79.3% | 9/150 6.0% | 22/150 14.7% | — | — | void |
| ث→س | 149 | 69 | 43/149 28.9% | 86/149 57.7% | 20/149 13.4% | 148/149 99.3% | 59/149 39.6% | 149 | 125/149 83.9% | 4/149 2.7% | 7/149 4.7% | — | — | dead |
| س→ث | 131 | 121 | 14/131 10.7% | 88/131 67.2% | 29/131 22.1% | 119/131 90.8% | 25/131 19.1% | 132 | 124/132 93.9% | 1/132 0.8% | 1/132 0.8% | — | — | dead |

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
