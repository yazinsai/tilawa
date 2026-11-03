# Revoke a far in-lock step when the continuation stays near

Status: implemented. Not practice-scored here. The supervisor runs the fold gate.

Test set: lab-v1.5 practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`.

## Prediction

Tracker −1, overall −1. One target: the ledger false skip `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism

After the planted omission of فتماروا (54:36:4), local search jumps at 121.72 s from 54:36:4 onto ونذر (54:37:9), a new run. The heard word is بالنذر (54:36:5). The jump is cheaper than deleting فتماروا, so the rhyme landing wins on that one word. The checker then never finishes 54:36, and the ayah ledger confirms a skip at 54:36:1 (122.48 s on the unchanged tracker).

`tracker.ts` withholds an in-lock jump that leaves 2 or more words unread. The shadow is the next spoken word, the one-deletion landing. Phonemes after the landing word are aligned, with the existing word costs, to the text after the shadow and to the text after the landing. The commit margin is the one a lock already uses (`max(1, 30/n)`), so a short shared opening cannot decide it. If the shadow wins, this push locks only the next word; the continuation locks go out on the next push, because the checker notes the last cursor in a batch. The local search restarts at the end of that word and replays the continuation. If the landing wins by the same margin, or a hold or the session end arrives while they still tie, the withheld jump is emitted and stands.

A one-word step does not open this. A same-run deletion does not either: the origin has to move. Clean in-order recitation never jumps. A real skip onto the text that follows it keeps winning the continuation, which is what the existing skip tests still check. No new threshold.

## What was checked

Phone trace of p46, before any practice score. Continuation after 121.84 s, text after 54:36:5 against text after 54:37:9:

| phones | through | margin | near cheaper by |
| --- | --- | ---: | ---: |
| 8 | 123.00 s | 3.75 | 0.00 |
| 16 | 123.96 s | 1.88 | 0.20 |
| 24 | 125.12 s | 1.25 | 3.10 |

The shadow clears the margin. After the change the tracker locks 54:36:5 at 125.28 s and walks 54:37 in that run. `skipped_passage 54:36:1` is not emitted. The planted 54:40 skip still confirms, at 146.04 s. The 54:36:4 skip still confirms at 136.88 s, as it did before.

Unit: deleting 54:36:4 out of 54:33–38 rebinds onto 54:36:5 and does not keep a run that starts at 54:37:9; the same passage read in order steps 54:36:4, 54:36:5, 54:37:1. Existing local-tracking skips, the excursion rebind, and the core suite (695 tests) pass.

## Limits

Practice was not scored. On the changed trace, `wrong_word 54:37:2` confirms at 146.88 s. Before the change the misfit hold at 125.04 s left that suspicion uncertain at 124.96 s, so it was not a confirmed warning. Staying locked removes the false skip and also removes the hold that was downgrading this one. If that confirmation is charged to the tracker, it cancels the win. The gate has not run.
