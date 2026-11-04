# In-ayah skip continuation veto

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. Listener −1, overall −1. One target: the false `skipped_passage` at `lab-v1.5-p31/phone/warning:skipped_passage/4:162:15/145.28`. Below the fold gate on purpose; a zero-loss rider if the rest of the set does not move.

## Mechanism

Skip emission in `checker.ts` still starts at the first written word with no counting visit. On p31 the greedy phones of 4:162:15 are `وَلمُءمِنِۥۥنَ` (written والمقيمين is `وَلمُقِۦۦمِۦۦنَ`, written والمؤمنون is `وَلمُءمِنُۥۥنَ`). That visit first lands on 4:162:19, so 15–18 are an in-ayah skip. A few hundred milliseconds later the same phones are a back-repeat of 4:162:6 and 16–18 align in order, leaving only 15 unread; the open skip then becomes a wrong word and confirms. The scored class is the first state, `skipped_passage`.

Before that gap is emitted, and only when the start and the bound word are in the same ayah, the visit's phones are scored against both words with the existing hard and soft charges. A fit within one hard edit (`MIN_HARD` is 2) is a look-alike. The next two counting visits are then priced with `anchorFitCost` against the start word's continuation and against the bound word's continuation. If the in-order pair wins by `commitMargin`, the visit is moved onto the start word and the gap is not emitted. The ordinary `wrong()` test, then the lexical and 2× `SUPPORT` gate, judge that word. Fewer than two following words and the session is still open: the gap waits, so the early skip is not opened and then relabelled. No new constant. A skip is never created. A genuine in-ayah skip continues from its destination, so the continuation test does not move it.

## What the p31 trace showed

Oracle checker, listener evidence, phone, before the change.

| Moment | What the alignment did |
| --- | --- |
| 143.16 | First counting visit after 4:162:14 is 4:162:19, heard `وَلمُءمِنِۥۥنَ` (hard 0, cost 3 against والمؤمنون). Gap 4:162:15–18, `skipped_passage`. |
| 144.76 | الصلاة starts and is briefly stuck on 4:162:20 (`صصَلَاا`, hard 1). The skip is still proposed. Only one following word, so the two-word test cannot run yet. |
| 145.28 | Shipped checker suspects `skipped_passage` at 4:162:15. That is the incident key. |
| 145.72 | الصلاة completes as 4:162:16. The 143.16 phones rebind to a repeat of 4:162:6. The gap shrinks to an omission of 15. |
| 145.92–148.28 | والمؤتون, الزكاة, then the real والمؤمنون at 4:162:19 (`وَلمُءمِنُۥۥنَ`). |
| 155.96 | The same issue confirms as `wrong_word`, observed `وَلمُءمِنِۥۥنَ`. |

Charge alignment of `وَلمُءمِنِۥۥنَ` against 4:162:15 is cost 24, hard 0, five consonants kept; against 4:162:19 it is cost 3, hard 0. That is within one hard edit. The two heard words `صصَلَااتَ` + `وَلمُءتُۥۥنَ` cost 0 against 4:162:16–17 and 115 against 4:162:20–21 (`anchorFitCost`). Gain 11.5 plain edits, `commitMargin(21)` is 1.43. The continuation test passes. `wrong()` on the rebinding is false (24/87 < 0.3 and hard 0), and `lexical.alternative` is false for this string, so the support gate is not asked to confirm a substitution.

A real deletion of 4:162:15–18 lands on the true 4:162:19 and the next words are 20–21, so the in-order continuation loses and the skip stays.

## Validation

- Unit: reciting 4:162:12–23 with 4:162:15 replaced by `وَلمُءمِنِۥۥنَ` emits nothing at 4:162:15. The same span with 15–18 deleted still confirms `skipped_passage` at 4:162:15.
- `bash tools/autolab_tests.sh` on six shards: every shard exit 0. Core and sdk tests passed, including the new case.
- Oracle phone trace of p31 at 145.28 after the change: no checker event within 11 s, on the shipped policy and with `minConfidence` 0, the local window off, and both off. The `skipped_passage` suspicion and the later confirmed `wrong_word` are both gone. Practice was not scored here.

## Limits

The hard/soft alignment can call a consonant-to-vowel swap soft, so the one-hard-edit gate is wide. The continuation margin is what keeps a real in-ayah skip. The veto waits out the first two words after a look-alike visit, so a genuine in-ayah skip is emitted once those words are in, not on the first frame of the jump. A skip the alignment never proposes is not invented.
