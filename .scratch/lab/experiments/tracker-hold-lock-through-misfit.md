# Hold the tracker lock through a misfit search

Status: implemented, not practice-scored in this worktree. The supervisor scores the branch.

Test set lab-v1.5, practice, listener `spliced-clean-int8`.

## Prediction, before scoring

Tracker −2, overall −3. The two tracker calls are p66 `9:3:21` and p46 `54:36:4`. The third is a checker-attributed added word whose tokens were dropped in the same kind of gap. No change to commit margin, skip rules, or global-search scoring. Clean sessions and studio should not gain warnings.

## Cause and change

On a misfit the weak-window search reaches `lost` before any other place has held the commit margin. `goLost` emitted that state immediately and the following search emitted `ambiguous`. The checker treats every non-`locked` event as dropping the anchor: open issues are closed `uncertain`, and phonemes heard until the next lock are not aligned.

The search is the same. While it runs, the tracker keeps the last locked anchor and does not emit `lost` or `ambiguous`. A lock within two ayahs of that anchor (the existing near-passage rule) is the continuation, still `locked`. A lock anywhere else emits `lost` and then the new place, as a relocation. If the session ends with the search still uncommitted, it emits `lost`.

## What was validated

Phone traces, then a replay that deletes the non-locked events in the gap (and, when the relock is not near, puts `lost` back at the end of the gap):

- `lab-v1.5-p66` `wrong_word` `9:3:21`. Locked on the word at 51.88, then on `9:3:12`, `lost` at 56.48, `ambiguous` through 61.40, relock `9:3:28`. Oracle and true position both confirm the wrong word; the engine marks it `uncertain` at 56.32. The replay confirms it at 57.60. Tracker miss.
- `lab-v1.5-p46` `omitted_word` `54:36:4`. `lost` at 125.04 from anchor `54:38:2`, relock `54:37:5` (one ayah back). The engine closes a `skipped_passage` as `uncertain` at 124.96. The replay confirms that `skipped_passage` at 136.88, which is also what the true-position arm confirms. Any-class matching counts it caught. Oracle confirms `omitted_word`. Tracker miss.
- `lab-v1.5-p14` `added_word` `44:10:1`. `lost` at 80.92, relock `44:10:3`. Oracle misses, so the miss is charged to the checker. The engine goes `uncertain` at 80.88; the replay confirms the added word at 83.80. This is the third call.
- The other ten practice word-1 added words the checker misses have no non-locked tracker event in the fifteen seconds after the label. Holding the lock does not touch them.
- All 26 fully clean practice sessions, phone and studio: the tracker never leaves `locked`. The replay therefore adds no warning.

A unit test inserts the same 60-phoneme noise that, as its own segment, goes `lost` and relocks on another surah. Inside 2:255 the anchor stays locked, every lock stays within two ayahs, and the ayah finishes. The distant-jump test still emits `lost` before the new surah.

## Limits

No practice score here. A gap that relocks off the passage still aligns the intervening phones at the old anchor and only then emits `lost`, so a warning confirmed during that gap would stick. Clean sessions had no such gap. p46 is caught as `skipped_passage`, matching the true-position arm, not as `omitted_word`. Studio was replayed only on the clean sessions.
