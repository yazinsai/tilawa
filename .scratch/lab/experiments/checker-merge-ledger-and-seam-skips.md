# Merge a seam skip into the ledger skip it continues

Status: implemented. Not practice-scored. One target, predicted checker −1, overall −1, which is below the fold gate. The supervisor scores the branch.

## Prediction

Test set lab-v1.5, practice, phone. The false skip at 26:131:2 goes away. The ledger skip at 26:130:1 stays. Checker −1, overall −1.

## Mechanism

The ledger and the seam emitter both describe one unread run. The ledger keys word 1 of the ayah the with/without fit called absent. The seam keys the first written word the alignment left unbound. A bound refrain word between those two starts splits the run, so the seam key sits one word later.

After both emitters, a seam skip that starts after a ledger ayah is the same event when every word between them is empty, or a counting visit of refrain text (an ayah whose letter text occurs again) or of a shared word in a near-duplicate span. The differing slot of a span is not shared text. The seam is dropped and the ledger key kept. A unique word between them, read or unread, keeps both. A seam that starts inside the ledger ayah, or before it, stays.

The ledger span is remembered after the skip confirms. On this case the anchor moves to the refrain word before the seam opens, so the two are not in one window.

No constant. A session that emits no skip is unchanged, because there is no second skip to drop.

## What the p24 trace showed

Oracle position, listener `spliced-clean-int8`, phone, shipped policy. Truth is a skip of 26:130:1–4 at 247.2 s.

Before the change, at 256.76 s the ledger confirms `skipped_passage` at 26:130:1. At 257.56 s the anchor is 26:131:1 and the seam opens `skipped_passage` 26:131:2–26:132:1, confirmed at 267.84 s. 26:131:1 فاتقوا has a counting visit (7/7 phones, three consonants kept). Its ayah text is copied eight times and the word is not a span slot. 26:132:1 واتقوا is inside the seam span, unique and unbound. Nothing unique sits between the ledger ayah and the seam start.

After the change, shipped and `minConfidence 0` still confirm 26:130:1 and do not open 26:131:2. With the posterior window off, the ledger never emits, so the seam at 26:131:2 remains. The scorer runs with the window on.

## Validation

- Unit: reciting 26:129, then only 26:131:1, then from 26:132:2, confirms 26:130:1 and not 26:131:2. Leaving 26:132:1 unread and skipping 26:135:2–6 confirms both 26:130:1 and 26:135:2. Reading 26:129–133 confirms no skip.
- `test/ayah-ledger.test.ts` and `test/checker.test.ts`: 58 passed. Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0.

## Limits

A real second skip whose first word comes right after a bound refrain is dropped with the seam, including when the seam span itself holds unique unread text. Overlapping the ledger ayah does not merge, so a seam that starts on the word before the ledger ayah is left where it is. Practice was not scored. The predicted gain is one wrong call, under the gate.
