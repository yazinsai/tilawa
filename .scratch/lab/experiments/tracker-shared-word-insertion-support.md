# Relocation support ignores a word one insertion also explains

Status: targets confirmed on the phone traces; practice score belongs to the supervisor.

Test set: lab-v1.5, practice only. Component: tracker.

## Prediction, before scoring

Two fewer tracker-attributed phone wrong calls, and the same two fewer overall. The calls are the missed additions `lab-v1.5-p63/phone/mistake:added_word/37:69:1/243.87` and `lab-v1.5-p14/phone/mistake:added_word/44:10:1/78.53`. No new constant. Clean in-order recitation never reaches a relocation commit, and a hold with no added word never takes the new edit, so those paths stay byte-identical.

## Cause and change

A relocation commits on a raw count of heard words that fit the destination. At p63 the planted وقفوهم is only 37:24:1, but the next heard word إنهم is both 37:24:2 and the anchor's 37:69:1, and both were counted. The anchor was never offered an insertion, so the tracker moved to 37:24 and the checker sealed the addition uncertain.

At each relocation commit the tracker now scores one extra hypothesis with the existing word costs: the anchor continuation with exactly one inserted heard word. A heard word counts only when the destination fit beats that hypothesis by the commit margin. The anchor start is whichever of the local cursor and its neighbouring words that hypothesis explains cheapest. إنهم then contributes nothing, وقفوهم is the only discriminative word, and the move waits. The following words ألفوا آباءهم are the anchor, not 37:24:3, so the insertion is left to the existing added-word confirmation. A real jump still moves once several destination-only words have accumulated. A word the destination reads exactly counts once it leads by the margin's floor; a partial reading has to clear the wide short-span margin, which is what keeps a shared word heard in part from voting.

The p14 addition is the other path. The open added word is sealed by the lock-drop rule, and the written-word cut makes the following phones fit worse (gain −5.5). Deleting the 9 heard phones and keeping the 8 that follow restores the anchor: gain 7.5 against the commit margin of the full 17-phone span, 1.76. That edit is the one that keeps the issue open.

## Validation

On current main, the p63 commit at 246.84 s is a lock on 37:24:1–2 over `وَقِفُۥۥهُمءِننننَه`. Against anchor 37:69:1 the insertion hypothesis costs 9 and has one discriminative word: وقفوهم gains 9 over a margin of 2.73, and the إنهم prefix gains 0 over a margin of 3.75. The audio just after it aligns to 37:69 at cost 0 and to 37:24:3 at about 10. The kill criterion does not fire.

On current main the p14 hold at 80.92 s has an open added word of 9 heard phones and 8 phones of فارتقب after them. Heard-phone deletion gains 7.5 against 1.76 with an edited cost of 0. The written-word cut gains −5.5 and does not explain the drop.

With the change, p63 never leaves 37:69 and confirms the addition at 250.52 s. p14 still holds at 80.92 s, keeps the issue, and confirms it at 83.80 s. A synthetic insertion of 2:205:8 before 2:71:12 (the same shape: one distant word plus a shared next word, then an anchor-only continuation) stays on 2:71 and confirms the addition. Existing jump, hold and checker tests pass, including the jump-persistence latency gap.

## Limits

No practice score was run here. Collateral on other sessions, studio audio and clean sessions is the fold gate's measurement. The anchor is whichever of the cursor and its neighbours the insertion hypothesis explains cheapest, because the cursor sits on the word before the shared word as often as on the shared word itself.
