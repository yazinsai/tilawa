# Hold the lock's tokens, downgrade only the issues already open

Status: replayed on practice phone and studio. No practice score from this worktree; the supervisor scores the branch.

Test set: lab-v1.5 practice. Listener evidence `spliced-clean-int8` (the folded context-free gap decode).

## Prediction, before any replay

Tracker −2, overall −3. The held tokens should catch the p33 26:222 skip that is born after the lock drops, without confirming the two false words that were already open (26:221:5, 54:37:2). The first-lock veto should take p06 and p54 off 25:57. 9:3:21 and 54:36:4 stay missed: those issues are open at the drop, and this change downgrades them on purpose.

## Mechanism

`follow` in `tracker.ts` used to emit `lost` once a weak window fit nothing and local cost exceeded `maxCost`. The checker then dropped its anchor, so phones heard in the gap never entered a comparison window. The hold keeps the last locked event and keeps feeding those phones. `lost` is emitted only when the search then locks off that anchor, or the session ends still uncommitted. The first held event is marked `hold: "start"`. The checker runs the same uncertain downgrade a lock drop uses — open issues are sealed and never reopened — and leaves the window in place, so only evidence that arrives after the drop is judged.

The first lock is still the suffix-window commit, except when that winner does not already start on an ayah's first word. Each fitting candidate is then aligned over the contiguous ayahs back to the first speech onset. A leading basmala or isti'adha is removed by the same phrase matcher that cuts those phrases out of acquisition. A candidate beaten by the commit margin (`max(minSeparation, separationScale / n)` edits) is barred. 25:57 cannot match 26:126 with its own preceding ayah, so it loses; an opening that is already on word 1 of its ayah is left as ranked, which is what keeps a clean ayah-start lock.

## What the replay showed

No-engine replay of stored phone and studio evidence, baseline tracker against this one, every practice session's first lock. Modal: the hold, reacquisition, startup-prefix, and checker tests, then the full core and sdk suites (6 shards, all passed).

First locks. The only practice sessions whose first lock moved are p06 and p54, phone and studio, from 25:57 onto 26:128. No clean session's first lock moved, including p36 (55:2) and p62 (26:192). p33's first lock stays 26:181.

Warnings on the new engine, phone:

- p33 confirms `skipped_passage 26:222:1` at 230.96 s. There is no confirmed `26:221:5`.
- p46 confirms neither `54:37:2` nor `54:36:4`. Studio still has the `54:36:4` omission, which this change does not try to save.
- p06 and p54 are not on 25:57.

## Limits

The veto refuses a tail match; it does not choose among tied refrain copies. p06 and p54 first lock at 26:128, after the opening they actually recited, not on 26:126. Whether that late lock adds a false warning of its own is what the practice score has to count. The hold still judges phones that arrive during the misfit, so a stretch of non-speech can confirm a new issue; the issues that were already open cannot. Studio audio of the same sessions was replayed for first locks only. No exam replay.
