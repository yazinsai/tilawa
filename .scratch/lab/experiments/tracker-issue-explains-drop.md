# Keep an open issue only when its edit explains the lock drop

Status: replayed on practice phone. No practice score from this worktree; the supervisor scores the branch.

Test set: lab-v1.5 practice. Listener evidence `spliced-clean-int8` (the folded context-free gap decode).

## Prediction, before any replay

Tracker −2, overall −2. The planted omission 54:36:4 (p46) and the swap 9:3:21 (p66) are open when the lock drops, and each one's own edit puts the following words back on the anchor. The false words 26:221:5 (p33) and 54:37:2 (p46) do not, so they stay uncertain. Honest about the gate: −2 is short of −3 on its own.

## Mechanism

A misfit hold used to end every open issue `uncertain`. Time of birth cannot split these four: the real ones and the false ones are both already open at the drop. What splits them is whether the issue is why the anchor stopped fitting.

At `hold: "start"`, for each open issue, the heard phones after that issue and before the next open issue are aligned to the anchor twice, with the tracker's existing word costs (`subCost`, `insertionCost`, `deletionCost`). Unedited, the anchor still starts at the issue word. Edited, that one word is removed: an omission or a widened skip loses the word, and a wrong word's continuation is the text after the substituted word, which is the same removal. The issue stays open only when the edited fit beats the unedited fit by the commit margin (`max(1, 30 / n)` plain edits) and the edited cost per phone falls below the misfit cost (0.35). Every other issue is downgraded as before. A later real `lost` still downgrades whatever is left. No new threshold.

## What the replay showed

No-engine replay of stored phone evidence, every practice session, baseline hold against this rule.

Practice holds where any open issue passes, and the final state after the rule:

- p46 drop at 125.04 s, anchor still 54:33:4, local cost 0.463. The open issue at the labelled omission is a `skipped_passage` 54:36:4–54:37:1, not a one-word omission. Removing 54:36:4 makes the 13 phones before the false 54:37:2 fit (0.22 per phone, gain 5 against a margin of 2.31). The same phones through the drop stay a misfit (0.39) because 54:37:2 is garbage, and deleting the whole skip fits worse (0.52). 54:37:2 itself gains 3.4 against a margin of 3.75, so it is downgraded. 54:36:4 then confirms as `skipped_passage` at 136.88 s. 54:37:2 stays `uncertain`.
- p66 drop at 56.48 s, local cost 0.468. 9:3:21 is a `wrong_word` (said قدير). The 16 phones after it are توليتم فاعلم, cost 0 once وإن is removed, gain 4.6 against 1.88. It confirms as `wrong_word` at 57.60 s.
- p33 drop at 229.20 s, local cost 0.563. 26:221:5 has 3 phones after it, gain 1 against a margin of 10, and stays `uncertain`. The 26:222 skip still confirms at 230.96 s. 26:221:2 is the only other practice issue that passes (14 phones, 0.18 per phone): it is kept, then the existing confirmation rules end it `uncertain` at 230.88 s, so it is not a warning.

No other open issue at a practice hold passes the test.

## Limits

The edit that saves 54:36:4 is a one-word deletion of the first word of a skip the checker had widened through 54:37:1, scored up to the next open issue. The labelled class is `omitted_word`; the confirmation is `skipped_passage` on the same word, which the scorer still counts as a catch. Studio audio was not replayed here. No exam replay. No practice score from this worktree.
