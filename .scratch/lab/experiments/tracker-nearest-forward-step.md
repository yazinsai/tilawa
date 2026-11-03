# Nearest in-lock forward step

Status: implemented. Not practice-scored here (one target, predicted below the gate). The supervisor runs the fold gate.

Test set: lab-v1.5 practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`.

## Prediction

Tracker −1, overall −1. The only counted target is the false skip `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism

On the current-main p46 phone trace the tracker is locked and walking in order: `54:33:1` through `54:36:4` at 120.44 s, one run. At 121.84 s that run restarts on `54:37:9` (`ونذر`). There is no `relocated` event and no excursion. The ayah ledger then confirms `skipped_passage` at `54:36:1` (122.48 s). The planted omission is `54:36:4`; the recited word is `54:36:5` (`بالنذر`), heard as `فِلنُذُر`.

A local replay from the end of `54:36:3` stays on `54:36:4` through `فِلنُذ` and flips its origin to `54:37:9` on the following `ُ` (121.68 s). That step's marginal cost is 0.01 edits, the jump's distance term. The two paths are tied apart from that.

Word cost of `فِلنُذُر` (`anchorFitCost`, the tracker's word cost): `54:36:5` `بِننننُذُر` is 2.40 edits, `54:37:9` `وَنُذُر` is 2.50. The commit margin for 8 phones is 3.75. The nearer word is the cheaper fit, and the farther one does not clear the margin. `wordSubstringCost` goes the other way (2.10 vs 1.50); 0.60 is still inside the margin. Neither alignment is exact. The kill criterion does not fire: this is an in-lock forward step, and the nearer position fits at a lower anchor cost.

In `tracker.ts`, a forward run restart is scored with `forwardStepChoice` over the spoken words ahead of the cursor, out to the local window (`NEAR_AYAHS`). Cost is `anchorFitCost`. A candidate fits only under the existing `maxCost`. The nearest fit wins unless a farther fit beats it by `commitMargin`. The cursor moves only when that choice costs no more than the landing the search already took; a strictly cheaper landing stays. The heard span has to be at least as long as the landing word before it is scored, so a one-phone prefix cannot lock a short neighbour. Nothing is added to the skip or relocation margins.

Clean in-order recitation does not restart its run, so it never takes this path. A real skip's landing is cheaper than the text it jumped over, so the search's landing stays.

## What was checked

Phone trace after the change, same session and listener:

- 121.72 s locks `54:36:5` (new run), then 122.24 s continues to `54:37:1` and walks `54:37` in order, including the real `54:37:9` at 129.56 s.
- `skipped_passage` at `54:36:1` is gone. `54:36:4` is still confirmed as `skipped_passage` at 136.88 s; that confirmation was already on the baseline trace.

Unit tests: the nearer of two fits wins when the gap is under the commit margin, and the farther wins when the gap clears it. A recitation that drops `54:36:4` and says `فِلنُذُر` locks `54:36:5` before `54:37:1` and does not open a run on `54:37:9`. The same block read in order does not restart on the rhyme. Existing local skips, repeats, the memorization jump, and the excursion rebind still pass.

An earlier cut of the rule moved whenever the farther word failed the margin, including when it was strictly cheaper. That pulled the skipped-ayah fixture back onto `2:256` (four words of the skipped ayah) and let a memorization jump freeze its expected bookmark at `2:259` instead of `2:255`. Keeping a strictly cheaper landing removed both failures. p46 does not use that slack.

## Limits

Practice was not scored. The fold gate has not run. One target is below the gate on its own. `54:36:4` remains a confirmed skip at 136.88 s, as before. A landing that is strictly cheaper by less than the commit margin is left where the search put it.
