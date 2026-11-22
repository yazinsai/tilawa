# E1b spliced-error probe v1

Report v1, teacher `student:e2-libri-full`, commit 3e91183. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.

**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.

Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). 2661 spliced clips, 2661 controls, 1990 clean ayahs; skipped as too far from the label: 38 spliced, 20 control, 0 clean.

## Per class

| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح/ه | 296 | 261 | 180/296 60.8% | 58/296 19.6% | 58/296 19.6% | 288/296 97.3% | 224/296 75.7% | 297 | 278/297 93.6% | 1/297 0.3% | 2/297 0.7% | 1/1910 0.1% | 1/1910 0.1% | viable |
| ع/ء | 294 | 281 | 243/294 82.7% | 22/294 7.5% | 29/294 9.9% | 289/294 98.3% | 269/294 91.5% | 298 | 284/298 95.3% | 1/298 0.3% | 2/298 0.7% | 1/2968 0.0% | 1/2968 0.0% | viable |
| ص/س | 294 | 223 | 166/294 56.5% | 95/294 32.3% | 33/294 11.2% | 288/294 98.0% | 191/294 65.0% | 298 | 283/298 95.0% | 1/298 0.3% | 2/298 0.7% | 1/918 0.1% | 1/918 0.1% | viable |
| ض/د | 287 | 194 | 135/287 47.0% | 96/287 33.4% | 56/287 19.5% | 268/287 93.4% | 168/287 58.5% | 289 | 254/289 87.9% | 0/289 0.0% | 0/289 0.0% | 0/778 0.0% | 2/778 0.3% | posterior-only |
| ط/ت | 294 | 232 | 152/294 51.7% | 96/294 32.7% | 46/294 15.6% | 273/294 92.9% | 176/294 59.9% | 296 | 259/296 87.5% | 0/296 0.0% | 1/296 0.3% | 0/1438 0.0% | 0/1438 0.0% | viable |
| ق/ك | 295 | 271 | 171/295 58.0% | 92/295 31.2% | 32/295 10.8% | 288/295 97.6% | 191/295 64.7% | 295 | 253/295 85.8% | 8/295 2.7% | 9/295 3.1% | 41/1840 2.2% | 47/1840 2.6% | posterior-only |
| ذ/ز | 285 | 182 | 94/285 33.0% | 145/285 50.9% | 46/285 16.1% | 247/285 86.7% | 113/285 39.6% | 286 | 232/286 81.1% | 22/286 7.7% | 37/286 12.9% | 27/936 2.9% | 30/936 3.2% | mixed |
| ث/س | 280 | 191 | 100/280 35.7% | 122/280 43.6% | 58/280 20.7% | 252/280 90.0% | 130/280 46.4% | 282 | 197/282 69.9% | 26/282 9.2% | 42/282 14.9% | 24/144 16.7% | 30/144 20.8% | void |
| missing shadda | 298 | 280 | 242/298 81.2% | 55/298 18.5% | 1/298 0.3% | — | 240/298 80.5% | 300 | 278/300 92.7% | 17/300 5.7% | 7/300 2.3% | 126/1860 6.8% | 46/1860 2.5% | posterior-only |

## Per direction

A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.

| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ح→ه | 147 | 120 | 73/147 49.7% | 33/147 22.4% | 41/147 27.9% | 141/147 95.9% | 106/147 72.1% | 149 | 147/149 98.7% | 0/149 0.0% | 1/149 0.7% | — | — | posterior-only |
| ه→ح | 149 | 141 | 107/149 71.8% | 25/149 16.8% | 17/149 11.4% | 147/149 98.7% | 118/149 79.2% | 148 | 131/148 88.5% | 1/148 0.7% | 1/148 0.7% | — | — | viable |
| ع→ء | 146 | 137 | 126/146 86.3% | 8/146 5.5% | 12/146 8.2% | 146/146 100.0% | 133/146 91.1% | 148 | 138/148 93.2% | 1/148 0.7% | 2/148 1.4% | — | — | viable |
| ء→ع | 148 | 144 | 117/148 79.1% | 14/148 9.5% | 17/148 11.5% | 143/148 96.6% | 136/148 91.9% | 150 | 146/150 97.3% | 0/150 0.0% | 0/150 0.0% | — | — | viable |
| ص→س | 145 | 95 | 64/145 44.1% | 61/145 42.1% | 20/145 13.8% | 140/145 96.6% | 79/145 54.5% | 148 | 142/148 95.9% | 0/148 0.0% | 0/148 0.0% | — | — | posterior-only |
| س→ص | 149 | 128 | 102/149 68.5% | 34/149 22.8% | 13/149 8.7% | 148/149 99.3% | 112/149 75.2% | 150 | 141/150 94.0% | 1/150 0.7% | 2/150 1.3% | — | — | viable |
| ض→د | 150 | 71 | 53/150 35.3% | 75/150 50.0% | 22/150 14.7% | 137/150 91.3% | 67/150 44.7% | 150 | 139/150 92.7% | 0/150 0.0% | 0/150 0.0% | — | — | dead |
| د→ض | 137 | 123 | 82/137 59.9% | 21/137 15.3% | 34/137 24.8% | 131/137 95.6% | 101/137 73.7% | 139 | 115/139 82.7% | 0/139 0.0% | 0/139 0.0% | — | — | viable |
| ط→ت | 144 | 88 | 81/144 56.2% | 39/144 27.1% | 24/144 16.7% | 139/144 96.5% | 97/144 67.4% | 147 | 121/147 82.3% | 0/147 0.0% | 1/147 0.7% | — | — | viable |
| ت→ط | 150 | 144 | 71/150 47.3% | 57/150 38.0% | 22/150 14.7% | 134/150 89.3% | 79/150 52.7% | 149 | 138/149 92.6% | 0/149 0.0% | 0/149 0.0% | — | — | posterior-only |
| ق→ك | 148 | 133 | 98/148 66.2% | 35/148 23.6% | 15/148 10.1% | 145/148 98.0% | 108/148 73.0% | 148 | 117/148 79.1% | 3/148 2.0% | 4/148 2.7% | — | — | void |
| ك→ق | 147 | 138 | 73/147 49.7% | 57/147 38.8% | 17/147 11.6% | 143/147 97.3% | 83/147 56.5% | 147 | 136/147 92.5% | 5/147 3.4% | 5/147 3.4% | — | — | posterior-only |
| ذ→ز | 136 | 114 | 24/136 17.6% | 99/136 72.8% | 13/136 9.6% | 106/136 77.9% | 27/136 19.9% | 136 | 125/136 91.9% | 4/136 2.9% | 4/136 2.9% | — | — | dead |
| ز→ذ | 149 | 68 | 70/149 47.0% | 46/149 30.9% | 33/149 22.1% | 141/149 94.6% | 86/149 57.7% | 150 | 107/150 71.3% | 18/150 12.0% | 33/150 22.0% | — | — | void |
| ث→س | 149 | 69 | 94/149 63.1% | 33/149 22.1% | 22/149 14.8% | 147/149 98.7% | 113/149 75.8% | 148 | 71/148 48.0% | 26/148 17.6% | 42/148 28.4% | — | — | void |
| س→ث | 131 | 122 | 6/131 4.6% | 89/131 67.9% | 36/131 27.5% | 105/131 80.2% | 17/131 13.0% | 134 | 126/134 94.0% | 0/134 0.0% | 0/134 0.0% | — | — | dead |

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
