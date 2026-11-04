# Cover a far step's ayah without rebinding it

Status: implemented. Not practice-scored here. Autolab paused before a changed-code p46 trace could run. The supervisor scores the branch.

Test set: lab-v1.5 practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`.

## Prediction

Tracker −1, overall −1. The only target is the false skip `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. No new constant.

## Mechanism

After the planted omission of 54:36:4, the in-lock step at 121.84 s binds the heard word to 54:37:9 and opens a new run there. 54:36 never finishes, so the ayah-absence ledger lists it between the settled neighbours 54:35 and 54:37 and emits the skip. Rebinding that run onto 54:36 removes the skip and also lets 54:37:2 confirm; on main that wrong word ends uncertain because the lock drops.

The far lock is left where it is. The tracker keeps the nearest one-deletion landing (the word after the one unread word) as a shadow. While that run lives, the ledger may suspect the shadow's ayah but does not confirm it. Over the hold lag (`LOST_AFTER`, 24 phonemes, commit margin 1.25) the two in-order continuations are scored with the existing word costs. If the run ends first, the phones already heard are scored against that shorter span's own commit margin, which is wider. When the shadow wins, the tracker attaches `coverage` listing the ayahs the shadow read in order. The checker drops those ayahs from the ledger candidate list and resolves an open ledger skip on them without confirming. A loss clears the hold and the skip confirms as before. The position, the window, the hold and every other issue stay on main's path.

## What was checked

Baseline p46 phone trace, before this change:

- Ledger: `54:36` between `54:35` and `54:37`, margin 803.7 nats, at 121.84 s. `skipped_passage 54:36:1` suspected then, confirmed at 122.48 s. Expected length 55 is the whole ayah. The kill criterion does not fire: this is the ayah-absence ledger, and 54:36 is the candidate.
- The lock walks 54:36:3, enters 54:36:4 at 120.44 s, and at 121.84 s restarts the run on 54:37:9. That run is still the run at the hold (125.04 s). 24 phonemes after 121.8 s land at 125.04 s.
- `54:37:2` is suspected at 124.00 s and uncertain at 124.96 s. It is not confirmed.
- The planted `54:40:1` skip still confirms at 146.04 s.

Unit, clean phonemes, the same jump shape:

- Omit 54:36:4 and hear 54:37:9 where 54:36:5 is written, then 54:37 in order. The shadow opens at base 54:36:3, near 54:36:5, far 54:37:9 (10 words unread). The continuation wins by 2.00 edits against a margin of 2.00 on 15 phonemes. `coverage` includes 54:36, the lock still visits 54:37:9, and `54:36:1` is not confirmed.
- An in-order reading of 54:33–42 emits no coverage and no skip.
- A jump whose next heard word is 54:38:2 loses by 8.00 edits against a margin of 1.25. 54:36 is not witnessed. The skip at 54:36:4 still confirms, and the lock still visits 54:38:2.

Excursion rebind and the ayah-ledger units still pass.

## Limits

The changed engine was not run on p46. Autolab paused at 04:47 UTC (`the test data on the volume changed`) before that trace started, so the 3.10-edit win measured by continuation-revokes-far-step was not re-measured on these alignments. The comparison here is the continuation after the landing word, not the landing word itself. A run that dies on one or two phonemes cannot clear its margin and does not witness. Practice was not scored here.
