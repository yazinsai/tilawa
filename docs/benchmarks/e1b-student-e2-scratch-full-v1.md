# E1b spliced-error probe v1

Report v1, teacher `student:e2-scratch-full`, commit 3e91183. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 54 spliced, 40 control, 3 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 289 | 257 | 165/289 57.1% | 45/289 15.6% | 79/289 27.3% | 267/289 92.4% | 222/289 76.8% | 296 | 245/296 82.8% | 3/296 1.0% | 3/296 1.0% | 5/1909 0.3% | 7/1909 0.4% | viable |
| ع/ء | 293 | 280 | 220/293 75.1% | 33/293 11.3% | 40/293 13.7% | 289/293 98.6% | 260/293 88.7% | 295 | 274/295 92.9% | 3/295 1.0% | 2/295 0.7% | 4/2964 0.1% | 8/2964 0.3% | viable |
| ص/س | 290 | 220 | 185/290 63.8% | 70/290 24.1% | 35/290 12.1% | 284/290 97.9% | 213/290 73.4% | 290 | 274/290 94.5% | 2/290 0.7% | 2/290 0.7% | 14/918 1.5% | 20/918 2.2% | viable |
| ض/د | 286 | 195 | 114/286 39.9% | 59/286 20.6% | 113/286 39.5% | 252/286 88.1% | 185/286 64.7% | 284 | 204/284 71.8% | 0/284 0.0% | 0/284 0.0% | 4/778 0.5% | 7/778 0.9% | void |
| ط/ت | 299 | 232 | 174/299 58.2% | 47/299 15.7% | 78/299 26.1% | 266/299 89.0% | 220/299 73.6% | 299 | 242/299 80.9% | 1/299 0.3% | 1/299 0.3% | 5/1437 0.3% | 6/1437 0.4% | viable |
| ق/ك | 296 | 272 | 162/296 54.7% | 84/296 28.4% | 50/296 16.9% | 288/296 97.3% | 197/296 66.6% | 298 | 247/298 82.9% | 10/298 3.4% | 15/298 5.0% | 70/1836 3.8% | 93/1836 5.1% | inconclusive |
| ذ/ز | 282 | 180 | 105/282 37.2% | 130/282 46.1% | 47/282 16.7% | 262/282 92.9% | 131/282 46.5% | 285 | 218/285 76.5% | 30/285 10.5% | 43/285 15.1% | 53/936 5.7% | 56/936 6.0% | void |
| ث/س | 276 | 191 | 107/276 38.8% | 93/276 33.7% | 76/276 27.5% | 250/276 90.6% | 144/276 52.2% | 277 | 162/277 58.5% | 39/277 14.1% | 68/277 24.5% | 31/144 21.5% | 60/144 41.7% | void |
| missing shadda | 296 | 278 | 265/296 89.5% | 30/296 10.1% | 1/296 0.3% | — | 266/296 89.9% | 297 | 241/297 81.1% | 39/297 13.1% | 21/297 7.1% | 336/1860 18.1% | 131/1860 7.0% | inconclusive |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 140 | 116 | 64/140 45.7% | 16/140 11.4% | 60/140 42.9% | 126/140 90.0% | 109/140 77.9% | 148 | 137/148 92.6% | 3/148 2.0% | 3/148 2.0% | — | — | posterior-only |
| ه→ح | 149 | 141 | 101/149 67.8% | 29/149 19.5% | 19/149 12.8% | 141/149 94.6% | 113/149 75.8% | 148 | 108/148 73.0% | 0/148 0.0% | 0/148 0.0% | — | — | void |
| ع→ء | 145 | 136 | 119/145 82.1% | 14/145 9.7% | 12/145 8.3% | 144/145 99.3% | 131/145 90.3% | 148 | 128/148 86.5% | 3/148 2.0% | 2/148 1.4% | — | — | posterior-only |
| ء→ع | 148 | 144 | 101/148 68.2% | 19/148 12.8% | 28/148 18.9% | 145/148 98.0% | 129/148 87.2% | 147 | 146/147 99.3% | 0/147 0.0% | 0/147 0.0% | — | — | viable |
| ص→س | 144 | 94 | 66/144 45.8% | 54/144 37.5% | 24/144 16.7% | 138/144 95.8% | 84/144 58.3% | 145 | 139/145 95.9% | 0/145 0.0% | 0/145 0.0% | — | — | posterior-only |
| س→ص | 146 | 126 | 119/146 81.5% | 16/146 11.0% | 11/146 7.5% | 146/146 100.0% | 129/146 88.4% | 145 | 135/145 93.1% | 2/145 1.4% | 2/145 1.4% | — | — | viable |
| ض→د | 146 | 71 | 41/146 28.1% | 54/146 37.0% | 51/146 34.9% | 124/146 84.9% | 68/146 46.6% | 148 | 116/148 78.4% | 0/148 0.0% | 0/148 0.0% | — | — | void |
| د→ض | 140 | 124 | 73/140 52.1% | 5/140 3.6% | 62/140 44.3% | 128/140 91.4% | 117/140 83.6% | 136 | 88/136 64.7% | 0/136 0.0% | 0/136 0.0% | — | — | void |
| ط→ت | 149 | 88 | 90/149 60.4% | 20/149 13.4% | 39/149 26.2% | 135/149 90.6% | 116/149 77.9% | 149 | 109/149 73.2% | 1/149 0.7% | 1/149 0.7% | — | — | void |
| ت→ط | 150 | 144 | 84/150 56.0% | 27/150 18.0% | 39/150 26.0% | 131/150 87.3% | 104/150 69.3% | 150 | 133/150 88.7% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ق→ك | 147 | 132 | 91/147 61.9% | 31/147 21.1% | 25/147 17.0% | 141/147 95.9% | 107/147 72.8% | 149 | 112/149 75.2% | 7/149 4.7% | 11/149 7.4% | — | — | void |
| ك→ق | 149 | 140 | 71/149 47.7% | 53/149 35.6% | 25/149 16.8% | 147/149 98.7% | 90/149 60.4% | 149 | 135/149 90.6% | 3/149 2.0% | 4/149 2.7% | — | — | posterior-only |
| ذ→ز | 134 | 112 | 33/134 24.6% | 90/134 67.2% | 11/134 8.2% | 125/134 93.3% | 35/134 26.1% | 136 | 116/136 85.3% | 6/136 4.4% | 8/136 5.9% | — | — | dead |
| ز→ذ | 148 | 68 | 72/148 48.6% | 40/148 27.0% | 36/148 24.3% | 137/148 92.6% | 96/148 64.9% | 149 | 102/149 68.5% | 24/149 16.1% | 35/149 23.5% | — | — | void |
| ث→س | 144 | 69 | 97/144 67.4% | 22/144 15.3% | 25/144 17.4% | 143/144 99.3% | 121/144 84.0% | 143 | 39/143 27.3% | 37/143 25.9% | 62/143 43.4% | — | — | void |
| س→ث | 132 | 122 | 10/132 7.6% | 71/132 53.8% | 51/132 38.6% | 107/132 81.1% | 23/132 17.4% | 134 | 123/134 91.8% | 2/134 1.5% | 6/134 4.5% | — | — | dead |

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
