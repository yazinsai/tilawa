# Unique anchors for refrain skips

Status: engine change in place. The supervisor scores practice. This build was not practice-scored.

## Prediction

Skip detection anchored only on words whose phonemes occur once in the surah, with repeated refrain words floated onto the copy that keeps recitation order. Expected on practice: checker −4, overall −4. The six checker `skipped_passage` misses sit on word 1 beside a refrain that repeats in its surah (26:103, 26:108, 26:131, 55:28/30, 77:34).

## Step 0

Baseline arm (perfect phonemes, oracle positions, default checker). Five of the six bind a refrain copy so the skip either vanishes or starts one word off. That clears the kill line (need at least three).

| session | target | baseline |
|---|---|---|
| lab-v1.5-p24 | 26:103:1 | miss, false `skipped_passage` at 26:102:7 (المؤمنين / مؤمنين) |
| lab-v1.5-p73 | 26:103:1 | miss, false `skipped_passage` at 26:102:7 |
| lab-v1.5-p24 | 26:130:1 | miss, false `skipped_passage` at 26:131:2 (inside فاتقوا الله وأطيعون) |
| lab-v1.4-p10 | 26:109:1 | miss, no seam. 26:108:3 is bound back onto 26:103:8 |
| lab-v1.5-p15 | 55:29:1 | miss, no false warning |
| lab-v1.5-p16 | 77:33:1 | miss, false `wrong_word` at 77:33:1 |

## Mechanism

`duplicateSpans` marks a word unique when its collapsed phonemes occur once in its surah, and a refrain when it sits in a 3-word sequence that occurs again. Skip spans are taken only between heard unique words. Duplicate words between those anchors are laid on the copy that stays in order. A run is emitted only when the span still contains text that was not heard, so saying a refrain twice, with the unique words on either side still contiguous, stays quiet. A span with no refrain in it is left to the alignment.

The alignment window commits before both anchors of a refrain skip are in it. The last unique word counted before the cut is kept, and so is every counted word after it. A backward bind onto an earlier copy does not rewrite that anchor. Words already matched stay heard, so the cut is not a hole.

The alignment window stops on the refrain, and the next unique word is then bound to an earlier copy (55:31 heard as 55:26). A word that occurs once, is at least six phonemes, and sits in the phonemes within three ayahs of the last forward visit is still an anchor. Exact phonemes are what the clean tests have; phone audio rarely matches a whole word, so this does not fire on a near miss.

A refrain seam can confirm a word early (26:102:7) before the next unique word shows the gap. That omission waits for a unique word past its end. When the anchor run is emitted it replaces the seam; a seam on the word before an ayah-start run, or on non-unique text between the run and the next anchor, is dropped only then.

## What was rejected

Keeping every counted visit past the commit, and splicing the trail when a word index went backwards, scored 89 → 110 wrong calls (checker 37 → 48). Committed words fell out of the trail and the next unique word turned the hole into a skip.

Searching the raw phoneme string for exact unique-word spellings recovers the skips on perfect phonemes and does not survive the phone channel. An unconditional drop of any alignment gap that sits on a non-unique word immediately before a refrain deletes a real omission when the anchor pass does not replace it. Neither is in this build.

The session JSON and the dump harness keyed to the six targets are not part of the change.

## Limits

Unit coverage is clean phonemes: the six refrain shapes, a repeated refrain, a surah-55 stretch longer than the window with nothing missing, and a 26:130 skip whose left anchor has left the window. Phone posteriors and the clean-session slice are the supervisor's score.

A restart onto an earlier unique word drops the carried words between the two copies out of this pass. The phoneme anchor only sees three ayahs past the last forward visit, and only an exact word, so a skip whose next unique word is further than that, or was heard with a phoneme changed, still depends on the alignment visiting it.
