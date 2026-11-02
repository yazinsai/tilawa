# Skip edge, whole-ayah fit

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. The p24 miss and its false warning flip together: checker −2, overall −3. The p26 41:19:7 edge is the speculative third call; the floor if it does not move is −2.

## Mechanism

Skip emission takes its start from the first word the alignment left unbound. On p24 the lock has already jumped to 26:131, فاتقوا is a counting visit of 26:131:1 (`تتَقُ`, the ف gone), and 26:130 has no visit, so it is not in the gap. The skip is 26:131:2–26:132:1.

For a skip of three or more words, the frames between the nearest committed words whose letter skeleton is unique in the checker's window are scored two ways. The previous ayah is a candidate only when the window still holds it, none of its words was visited, and one bound word sits between it and the skip — the refrain, kept as the destination. The ayah that holds the start is the other candidate, and only when that previous ayah was visited. Both paths are the in-order text between those anchors, one with the ayah and one without, scored by the CTC forward sum plus the checker's contiguous repeat arcs (a jump back onto text just read, no arc cost, no filler). The start moves only when one path leads by `PHRASE_SWITCH` (4 nats). A path that has more phones than frames cannot be emitted; that is a loss, not an abstention. No skip is created. A skip that already starts on an ayah's first word is not moved. Without posteriors nothing moves.

## What was validated

Oracle checker, listener `spliced-clean-int8`, phone, p24 at 247.21.

Shipped policy, after the change: `skipped_passage` suspected at 26:130:1 at 257.56 s and confirmed at 267.84 s. The same run with `localWindow` off still confirms 26:131:2, so the move is the posterior fit. The margin on the 45-frame landing from 26:131:1 to 26:132:2 is infinite: 26:130 does not fit in those frames, the landing without it does. The reported span still ends at 26:132:1, so the refrain that was said stays inside the one warning instead of opening a second gap at 26:131:2.

An uncertain `wrong_word` at 26:130:1 still fires a second earlier (256.76 s). It is not a confirmed warning. The skip replaces that word for anything confirmed afterwards.

Unit: deleting 26:130 and locking the way the oracle does (the run restarts at 26:131) confirms `skipped_passage` at 26:130:1 only. Peaked frames of فاتقوا + الذي prefer that landing over the same landing with 26:130 forced in, by more than 4 nats. The look-alike pause seam (26:102:7 / 26:103:1) is unchanged. Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0.

## Limits

The end of the skip is the alignment's end, not the end of 26:130. Scoring keys on the start word. p26 41:19:7 was not traced. Practice was not scored here.
