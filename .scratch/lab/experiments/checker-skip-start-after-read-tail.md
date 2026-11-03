# Skip start after a read tail

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`. The p26 pair flips (false `skipped_passage` at 41:19:7, missed skip at 41:20:1): checker 0, overall −2. p31 4:162:15 is speculative and was not won.

## Mechanism

Skip emission still starts at the first source word with no counting visit, and only an already-proposed skip of at least three words is touched. Nothing is created or removed, and no constant is added.

The landing side of that skip can own the source ayah's tail. The onset used here is the earliest zero-hard-cost fit of a later ayah's word 1 plus word 2's first consonant: every consonant in that needle matches in order, which is the existing exact-match cost, and the match has to fall inside a counting visit of that word 1. Tokens from the landing visit up to that onset are aligned to the unread trailing words of the source ayah, right edge against right edge, with the checker's soft and hard costs. A trailing word whose tokens keep at least two of its consonants in order (the same two-consonant shape an added word uses) counts as read, and the skip key moves to the next word. A heard skeleton that is the written one plus or minus a single leading consonant stays with the pause seam.

Words before that last read word are not re-filed. فهم in p26 is only ه, and putting it back on the warning that was already opened as the early skip makes that same id confirm.

## What the phone replay showed

Listener above, oracle position, before the change. p26's planted skip is 41:20:1–12, then 41:21:1. Greedy phones between النار and وقالوا are `هَسَعُۥۥن`. ع then ن are two consonants of يوزعون in order. فهم's ف and م are absent. The alignment's first visit after 41:19:6 is 41:20:12 on those phones; 41:21:1's visit starts at و, and `وقلل` (وقالوا plus ل of لجلودهم) is the zero-hard-cost bigram. ج of لجلودهم is not in the greedy string, so the fit stops at word 2's onset consonant.

p31 4:162:15 is the decode of والمقيمين with ق→ء, which is والمؤمنون (4:162:19) exactly. There is no unread trailing word of a source ayah sitting in front of a later ayah's word 1+2 fit. The suspicion still collapses to a wrong word on 4:162:15 and the skip does not confirm.

## Validation

- Unit: deleting 41:19:7 through 41:20 with no inserted tail still confirms `skipped_passage` at 41:19:7. The same cut with `هَسَعُۥۥن` inserted confirms it at 41:20:1 and not at 41:19:7. The pause-less المؤمنين/مؤمنين seam stays at 26:102:7; the paused one still moves to 26:103:1.
- Phone, both arms (oracle position and the engine tracker), after the change:
  - p26: `skipped_passage` 41:19:7 is suspected at 282.48 s and resolved at 283.28 s. `skipped_passage` 41:20:1 is suspected at 283.28 s and confirmed at 307.76 s. Nothing else at 41:19 confirms.
  - p31: `skipped_passage` 4:162:15 is still only suspected (145.28 s), then the same word is a confirmed `wrong_word` at 155.68 s. That confirmation was already the way this id closed; the skip key does not move.
- Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0. Checker tests alone: 51 passed.

## Limits

The fit is word 1's consonants plus word 2's first consonant, not the rest of word 2. p26's لجلودهم has no ج in the greedy string, so a full word-2 skeleton never matches. A tail that is only one consonant (فهم as ه) does not count as read and is not given its own warning. Practice was not scored here.
