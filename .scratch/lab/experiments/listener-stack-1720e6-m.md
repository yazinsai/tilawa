# Stack: madd duration and ayah-final manner

Status: practice-scored. Left for the fold gate. Overall −3, listener −2, checker −1, on sha `794730e3`.

## Preregistered prediction

Test set lab-v1.5, practice. Stack `checker-connected-madd-vowel-drops` (−1, nothing lost) and `listener-final-slot-manner-rekey` (−2, nothing lost). Different incidents, so together listener −2 and overall −3.

## Mechanism

No new rule. Both branches' checker changes, byte-for-byte, with this card's `lab-experiment.json` kept.

The manner pass runs first, on accusations already raised, and only at an ayah-final word before a labelled pause. The last consonant run in that slot, after the waqf-tail strip, is nasal, liquid, obstruent, or glide. Same manner is never a contradiction. A wrong word already on that slot keeps `SUPPORT` and the weak-evidence gate; the mismatch replaces the lexical doubling and the 0.6 confidence cut. An omission already on that word becomes a wrong word when the slot holds at least two non-blank tokens, one of them a consonant, and that manner contradicts. A spliced-out omission has no speech before the pause and stays an omission.

The madd pass runs after that. It only adds a wrong word where no issue exists yet, so it cannot replace a manner rekey, and the manner pass never sees the issue it adds. The written word ends in a long vowel, the next word starts with hamza, and no labelled pause sits between them. Dropping the vowel yields an attested mushaf word whose 4-word phrase occurs at another ayah. The span from the stem's last consonant onset to the next hamza has to be strictly shorter than every earlier matched madd munfasil in the session. A flagged drop does not enter that baseline.

The two sites do not meet. 20:43:1 is ayah-initial, so the manner slot returns nothing. 11:11:10 and 37:84:5 are ayah-final consonant swaps, not a vowel drop, so the madd index does not contain them.

## Validation

- Unit tests already on the two branches (`checker-manner.test.ts`, `checker-madd.test.ts`). `bash tools/autolab_tests.sh` on 6 shards, exit 0. Shard 5 hit Vitest's worker RPC deadline after every test had passed; the script treats that as a pass.
- Practice score 1 of 3, against the main line:

| | main | this | change |
| --- | --- | --- | --- |
| overall | 40 (26 missed + 14 false) | 37 (23 missed + 14 false) | −3 |
| listener | 36 (24 + 12) | 34 (22 + 12) | −2 |
| checker | 3 (2 + 1) | 2 (1 + 1) | −1 |
| tracker | 1 | 1 | 0 |

Three fewer misses, false warnings unchanged. That is the sum of the two earlier measurements: manner's two listener-charged misses (11:11:10, 37:84:5) and madd's one checker-charged miss (20:43:1).

## Limits

The score is totals, not incident keys. It matches the only total those two zero-loss cards add, and it does not re-list the keys. Leave-one-reader, studio, and the loss ratio are the fold gate's, not this compare. Listener −2 sits in the noise band on its own row; overall −3 is the gain line. Madd's single target previously failed leave-one-reader by itself (`mp3quran-10905` carried the whole −1). The two manner wins have to sit on other readers for the stacked −3 to survive that check; this score does not say which reader moved.
