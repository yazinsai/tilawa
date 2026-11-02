# Ayah absence ledger

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. Three of the four unique ayahs skipped between refrain copies are caught: checker −3, overall −3. The fourth stays missed. Re-keying p16's wrong-word warning into the skip was the speculative −4 and did not happen.

## Mechanism

Skip detection only judges text the alignment bound. A heard refrain can bind to some copy, so the skipped ayah sits inside the search window and still gets no visit. The ledger does not move alignment, anchors or placement.

After the visited frontier advances, and when the tracker jumps the anchor, the checker lists mushaf ayahs between those two words that still have no counting visit and whose phonemes are not the neighbours' text. A hole of more than two such ayahs is a lost lock, not a refrain gap, and is left alone. Each remaining one is a whole ayah of at least three words, the same bar as a passage skip. A CTC keyword decode places that ayah's phone sequence anywhere on the stored posteriors between the neighbours and compares it with a phone-loop filler. The filler pays nothing: every frame keeps its best token. The keyword's extra cost is how much worse its best path is. The decode has to actually run (the span has to be long enough to hold the sequence). Above `SUPPORT` (10 nats) the filler wins, and a `skipped_passage` at the ayah's first word goes through the confirm rules. The warning time is the first later ayah the tracker locked inside that same audio, so a refrain bound backward cannot push the time past the omission horizon. With no tracker locks (a passage replay) any time in the span is enough for the confirm rules.

The neighbour already being committed is enough to confirm. Waiting two further ayahs never arrives when every later refrain binds backward.

## What the replay showed

Listener `spliced-clean-int8`, practice, compared with the ledger off. Phone engine is the overall arm. Oracle evidence with true positions is the checker arm.

| Ayah | Phone engine | Phone oracle | Studio engine |
| --- | --- | --- | --- |
| p10 26:109 | won, filler 514 nats | won | won |
| p15 55:29 | won, filler 428 nats | won (168.36 s, inside the window) | not ledgered (a visit) |
| p16 77:33 | won, filler 198 nats; the wrong word at 77:33:1 stays | won | won, and the late false skip at 136.64 s goes away |
| p16 77:39 | won (was a separate miss) | already hit | unchanged |
| p24 26:130 | not ledgered (word 2 is a visit) | won, filler 149 nats | won |

On this replay that is overall −3 (−4 misses, +1 false warning) and checker −4. The false warning is `lab-v1.5-p63` phone and studio engine, `skipped_passage` at 37:25:1. Oracle evidence binds 37:25, so the checker arm does not raise it; the listener posteriors between 37:24 and 37:26 do not, and the filler wins at 169 nats. One loss against four wins.

26 fully clean practice sessions, phone and studio, true position and engine: no recited ayah was ledgered with the filler winning. A tracker jump across a stretch of unbound ayahs (p25, 40 of them; p74, nine) is not accused. A span shorter than the ayah (2:24, 2:283, 4:15, 19:65, 25:58) is not a decode and is not a skip.

## Validation

- Unit: deleting 26:109 between the surrounding refrains, with posteriors and no tracker locks, confirms `skipped_passage` at 26:109:1. The same span recited whole confirms nothing. A sequence on its own frames costs under 1 nat; a different sequence on those frames costs far more.
- Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0. One shard hit Vitest's RPC deadline after every test in it had passed.

## Limits

26:130 on phone is visited, so the ledger never asks. p63 37:25 is a recited ayah the listener did not bind and the keyword did not find; that one false warning is the loss. Practice was not scored here.
