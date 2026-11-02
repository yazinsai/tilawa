# Learned seam token tagger

Status: implemented, not practice-scored in this worktree. The supervisor scores the branch.

## Prediction

Checker −3, overall −3, from 3 of the 6 `checker/added_word/missed` seam targets. The operating point is the highest tagger score on a real clean gold-excluded seam, so it is not a constant fitted on practice.

## Mechanism

At an ayah seam where the previous ayah's last word and the next ayah's word 1 both have a complete in-order visit (full coverage, and either a lexical form or the word's consonant skeleton), the phones strictly between those visits are a seam span. If that span already passes the added-word shape (3 phones, 2 consonants) and is not a non-Quran phrase, a small transformer scores it. The features are the greedy phones and their confidence, whether a pause precedes the phone, who the alignment gave the phone to (left neighbour, right neighbour, some other word, nobody), whether the tracker stepped on that phone, and the span length. A score above the operating point confirms an added word in place of the 10-nat support margin. A score below it leaves the margin in charge. The margin only exists when posteriors do, so with none the checker is unchanged. Shape and phrase handling are unchanged. An existing omission or wrong word on that word is not replaced.

The model is 2 pre-norm layers, width 32, 4 heads. Weights are `packages/core/src/seam-tagger-weights.ts`. The threshold stored there is **0.432**, the score of the one real clean gold-excluded seam that passed the shape test (800 phone-degraded clips, gold-excluded tlog train, shipped listener and the lab phone degradation). Unspliced clips almost never leave a shape-passing span, so the training negatives that are not that one seam are spliced with the same 10 ms crossfade: a copy of the audio before the seam (a back-read) and a mid-word fragment. Positives are one pause-bounded mushaf word from the same clip, inserted at the seam, then phone-degraded. Calibration did not use those splices.

## What the offline replay showed

Listener `exp-listener-context-free-gap-decode-62ac8b5`, phone. A seam is "won" when the checker confirms `added_word` there. Threshold 0.432.

| Seam | Span | Score | Confirmed added word |
| --- | --- | ---: | --- |
| 37:69:1 وقفوهم (p63) | 11 | 0.118 | no |
| 37:71:1 وقفوهم (p63) | 11 | 0.453 | yes |
| 37:139:1 وقفوهم (p64) | 11 | 0.574 | yes |
| 44:12:1 قبلهم (p14) | 9 | 0.007 | no |
| 54:39:1 (p35) | 10 | 0.697 | yes |
| 51:20:1 وابل (p57) | 5 | 0.173 | no |
| 38:7:1 lab-v1.5-p15 | — | — | no seam |
| 38:7:1 lab-v1.4-p15 تبع | 6 | 0.949 | no (the word already has an omission; the tagger does not replace another class) |
| 26:88:1 p02 restart | 29 | 0.001 | no |

3 of 6 targets confirm. The p02 restart scores at the floor. The p15 noise spelling scores high, and the only thing that keeps it quiet is an omission that was already confirmed with the tagger disabled. That is the weak point: the operating point does not by itself reject تبع.

## Limits

The threshold sits on one clean seam. Training separation on a 10-example holdout peaked at 8/10 insertions above every held-out negative, then overfit. No practice score was taken. Studio audio was not replayed.
