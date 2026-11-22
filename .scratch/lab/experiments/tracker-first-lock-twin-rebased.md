# Re-land the first-lock twin guard on the folded seam rule

Status: practice-scored once. Left for the fold gate.

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-context-free-gap-decode-62ac8b5`. Tracker −4, overall −4, 0 lost. The two p25 name warnings plus the issue-explains-drop pair (54:36:4, 9:3:21). The parenthetical tracker −6 (the surplus fold's two added-word misses moving back off the tracker) was not counted.

## Mechanism

The discarded rebase scored `8e35a600`. The second fold resumed that Modal call because the source key matched, so the scored tree was the pre-surplus branch, not current main plus the guard. Checker went 13 → 24 and listener 41 → 38, the surplus fold undone in full, and the six lost incidents were that fold's wins.

This rebuilds the stack on current main. `surplusWord` and the seam block that calls it are untouched. The guard is the one that already measured the p25 flip: hold and continuation resolution arm only before the first lock hears a location-unique word.

What changed in delivery: while that hold is armed the checker used to skip `commit`, so `lead` stayed set and the opening stayed one alignment. The spelled-surplus rule reads the phones between the previous ayah's last complete visit and word 1. That is the gap a locked step leaves after `commit`. Commit now runs during the hold, same cut as a locked step. A slot accusation that scrolls out of that window is frozen with the clarity it had there. The other copy drops it; the same copy confirms it. Relocation slides the anchor onto the parallel text and does not slice the window or mark it as a fresh lock, and it does not reopen a decision that is not a held slot.

## What was checked

Unit tests, local vitest, 661 core tests passing. The new one confirms قبلهم before a word-1 inside the Noah span while `spanHold` is set, after the window has committed. The existing Noah/Lut continuation tests still move only a lock that has not yet heard a location-unique word.

One-session replay on the practice evidence, before the score:

- p25 phone locks 26:106:5 at 8.40 s with the hold armed, relocates 26:104:1 → 26:159:1 at 35.04 s onto 26:165:1, and confirms neither 26:105:3 nor 26:106:5. Its three planted mistakes still hit. Studio's first lock is already 26:161:5; those two warnings are absent there too.
- p46 phone and studio hit the 54:36:4 omission.
- p66 phone and studio hit 9:3:21. The 9:5:15 added word stays missed; it is not a word-1 seam.

## Measured practice result

One practice score, worktree `2458b8aa-dirty`. No second score.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 60 (39 missed + 21 false) | 56 (37 missed + 19 false) | −4 |
| tracker | 6 (4 missed + 2 false) | 2 (2 missed + 0 false) | −4 |
| listener | 41 (24 missed + 17 false) | 41 (24 missed + 17 false) | 0 |
| checker | 13 (11 missed + 2 false) | 13 (11 missed + 2 false) | 0 |

Checker and listener are unchanged, so the surplus fold's calls stayed put. The four tracker calls are the two p25 false warnings and the two misses above. Studio, clean-session, and leave-one-reader cuts are not in this compare; the fold gate has them.

## Limits

The two tracker misses the surplus fold added are still tracker misses (the −6 parenthetical did not happen). A held slot that scrolls out of the committed window is judged from the snapshot at the cut, not from a later re-alignment of those phones. A real name swap on the copy the continuation keeps is confirmed from that snapshot.
