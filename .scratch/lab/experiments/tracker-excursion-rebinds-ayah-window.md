# Rebind an excursion window when the in-order continuation returns

Status: implemented. One practice score: overall −3, tracker −3 (expected −3 / −3). The supervisor runs the fold gate.

Test set: lab-v1.5 practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`.

## Prediction

Tracker −3, overall −3. The rider is `tracker-shared-word-insertion-support`, re-landed as it was: it had already scored tracker −2, overall −2, with no other gate failure. At least one of the two ledger skips should go with the rebind. p46 was the speculative one.

## Mechanism

The ledger lists an ayah between two settled neighbours when that ayah never received a window. A lock onto the other member of a parallel span opens the window there. The ayah that sits in the gap on that copy — 26:144 between 26:143 and 26:145 — is then reported skipped, even though the speech was the in-order member and the real skip is 26:126.

On p41 the tracker is on 26:123, jumps to 26:141 at 187.32 s (عاد was said as ثمود, donor 26:141:2), walks the Salih copy through 26:145, and returns to 26:128 at 206.60 s. 26:144 is suspected at 202.32 s. 26:126 is confirmed at 204.72 s, before the return. Phones from 187.3–206.6 s cost 61.0 edits against 26:123–127 and 65.2 against 26:141–145, so the in-order text fits better. The continuation lock is already the commit-margin decision.

p46 does not have that shape. 54:36:1–5 is recited inside a window that opened on 54:33, which is the ayah just repeated, and the cursor walks 54:33 → 54:36 in order. There is no jump onto another member followed by a return to that member's continuation. The false skip stays.

The change, in `tracker.ts`, records an excursion only when a committed jump leaves one member of a parallel span (or an exact copy of the next ayah) for the other. While that copy is held, the phones are kept. When a later lock lands on the in-order member's continuation, the excursion's phones are aligned to both texts with the existing word costs. Exact copies tie. A one-word variant pays that word, so the copy that was heard is cheaper. If the in-order text is no more expensive, the tracker emits a `locked` event with `relocated` from the destination member to the in-order member before the continuation lock. The checker's existing rebase slides the anchor, drops a duplicate skip that would land on a word already decided, and widens the window so the continuation lock is not a fresh drop. Clean in-order recitation never jumps, so it emits nothing new. A jump onto text the in-order span fits worse is left where it locked.

The rider is the shared-word insertion branch, byte-identical: relocation support ignores a heard word an anchor-plus-one-insertion also explains, and an added word is kept through a lock drop when deleting its heard phones restores the anchor.

## What was checked

Phone traces, before any practice score:

- p41: `rel=26:140:1->26:122:1` at 206.52 s. `skipped_passage 26:126:1` stays confirmed at 204.72 s. `26:144:1` is resolved at 206.44 s and not confirmed.
- p46: no rebind. `54:36:1` is still confirmed at 122.48 s. The planted `54:40` skip is still confirmed.

Unit: a noise burst plus عاد said as ثمود, then Hud's own continuation, rebinds 26:140 to 26:122 and does not confirm 26:144. The same shape with both name slots said as Salih does not rebind. An in-order reading of 26:120–130 emits no cross-member relocate and no skip. The rider's insertion and relocation-support tests pass with it.

## Practice score

One score, against the current main line (43 overall, tracker 4):

| | main | this | change |
| --- | ---: | ---: | ---: |
| overall | 43 | 40 | −3 |
| tracker | 4 | 1 | −3 |
| listener | 36 | 36 | 0 |
| checker | 3 | 3 | 0 |

Misses 28 → 26, false warnings 15 → 14. The compare verdict was better. That is the rider's two added-word misses and p41's false skip, with nothing else moving in the headline. Leave-one-reader, clean sessions and studio are the supervisor's gate, not this score.

## Limits

p46 is not this mechanism. The rebind refuses once the phone buffer is trimmed (480 phones), so a very long excursion stays on the copy it locked. Practice was scored once here; the fold gate has not run.
