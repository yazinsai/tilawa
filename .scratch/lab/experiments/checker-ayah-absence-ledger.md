# Ayah absence ledger

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. Three of the four unique ayahs skipped between refrain copies are caught: checker −3, overall −3. The fourth stays missed. Re-keying p16's wrong-word warning into the skip was the speculative −4 and did not happen.

## Mechanism

Skip detection only judges text the alignment bound. A heard refrain can bind to some copy, so the skipped ayah sits inside the search window and still gets no visit. The ledger does not move alignment, anchors or placement.

After the visited frontier advances, and when the tracker jumps the anchor, the checker lists mushaf ayahs between those two words that still have no counting visit and whose phonemes are not the neighbours' text. Each one is a whole ayah of at least three words, the same bar as a passage skip. A CTC keyword decode places that ayah's phone sequence anywhere on the stored posteriors between the neighbours and compares it with a phone-loop filler. The filler pays nothing: every frame keeps its best token. The keyword's extra cost is how much worse its best path is. Above `SUPPORT` (10 nats) the filler wins, and a `skipped_passage` at the ayah's first word goes through the ordinary confirm rules.

The warning time is the start of the last frame matched to an ayah before the hole. That frame is still the previous word. The end of the frame can sit past the labelled word, which is too late for the scorer's window.

## What the replay showed

No-engine replay, listener `spliced-clean-int8`, phone. A recited ayah on its own audio costs under 1 nat (worst 0.9). The unsaid ayah on the neighbour's audio costs hundreds of nats, or cannot be placed because the gap is shorter than the sequence.

| Ayah | Bound | Filler | Warning time | Label window | Result |
| --- | --- | --- | --- | --- | --- |
| p10 26:109 | never | 514 nats | 40.80 s | 35.63–43.45 | inside |
| p15 55:29 | never | 428 nats | 179.52 s | 167.88–179.54 | inside |
| p16 77:33 | never | 198 nats | 136.08 s | 127.68–136.41 | inside |
| p24 26:130 | word 2 at 256.76 s | not ledgered | — | — | stays missed |

All four were inside the lookahead window. Three were never bound, which is the list the ledger uses. p16 still confirms its separate `wrong_word` at 77:33:1. The same pass also confirms the planted skip at 77:39 (153.00 s, window to 154.44). Skips the alignment already raised (26:115, 26:136, 77:36) are not raised again.

26 fully clean practice sessions, phone and studio: no recited ayah was ledgered with the filler winning.

## Validation

- Unit: deleting 26:109 between the surrounding refrains, with posteriors, confirms `skipped_passage` at 26:109:1. The same span recited whole confirms nothing. A sequence on its own frames costs under 1 nat; a different sequence on those frames costs far more. `test/keyword-spot.test.ts` and `test/checker.test.ts`: 50 passed.
- Core and sdk tests: `bash tools/autolab_tests.sh` on six shards. One shard hit Vitest's RPC deadline after every test in it had passed. The keyword threshold in that file was then raised; the re-run above covers it.

## Limits

26:130 is visited on word 2, so the ledger never asks about it. A skip the alignment binds any word of is left to the existing rules. Practice was not scored here.
