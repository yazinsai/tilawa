# Stack the manner rekey on the seam-skip merge

Status: practice-scored. Left for the fold gate. Overall −3, listener −2, checker −1, nothing lost.

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. The two parents won different incidents and lost none, so the stack should win all three:

- `lab-v1.5-p78/phone/mistake:wrong_word/11:11:10/143.13` and `lab-v1.5-p50/phone/mistake:wrong_word/37:84:5/50.27` from listener-final-slot-manner-rekey (listener −2)
- `lab-v1.5-p24/phone/warning:skipped_passage/26:131:2/257.56` from checker-merge-ledger-and-seam-skips (checker −1)

Listener −2, checker −1, overall −3. Nothing lost.

## Mechanism

No new rule. `exp/listener-final-slot-manner-rekey` is merged onto the branch that already contains checker-merge-ledger-and-seam-skips. Git conflicted only on the return of `judge` in `packages/core/src/checker.ts`. The resolution keeps the manner block and returns `bound` with it. This card's `lab-experiment.json` is unchanged.

The manner rule is the one measured on that branch. Ayah-final words before a labelled pause only. The manner of the last consonant run, after the waqf-tail strip, is compared with the written word's isolated pausal final: nasal (ن م ں ۾), liquid (ر ل), obstruent (stop or fricative, so ف/ك are one class), glide (و ي). Same manner is not a contradiction.

- A wrong word already raised on that slot keeps `SUPPORT` and the weak-evidence gate. The mismatch replaces the lexical doubling and the 0.6 confidence cut. That is 11:11:10 (مَںں, nasal against pausal ر, 28.6 nats, confidence 0.5997).
- An omission already raised on that word becomes a wrong word when the phones from the previous word's last in-order visit up to the first labelled pause hold at least two non-blank tokens, one a consonant, and the final manner contradicts. That is 37:84:5 (مِل, liquid against pausal م). A spliced-out omission has no speech and stays an omission.

The seam rule is the one measured on the other parent. After both skip emitters, a seam skip that starts after a ledger ayah is the same event when every word between them is empty or a counting visit of refrain text (an ayah whose letter text occurs again) or of a shared word in a near-duplicate span. The seam is dropped and the ledger key kept. On p24 the ledger keys 26:130:1 and the seam keys 26:131:2, with only the bound refrain word فاتقوا between them.

The two predicates do not share a slot. Manner rewrites `wrong_word` and `omitted_word` on an ayah-final word before a pause. The seam drop deletes a `skipped_passage` that continues a ledger skip. `bound` is the set of counting visits and is computed after the manner rewrite, from the same visits.

Line check against the merge base `03bb17f1`: manner parent +98 lines, seam parent +61, resolved file +159. Every line either parent adds is in the result. Every line either parent deletes is gone.

## What was checked

Unit tests from both parents are on the branch: `packages/core/test/checker-manner.test.ts` and the seam cases in `packages/core/test/ayah-ledger.test.ts`. `bash tools/autolab_tests.sh` on six shards, every shard exit 0. Shard 4 reported Vitest's worker RPC timeout after its 149 tests had passed; the harness treats that as a pass, same as the seam parent.

One practice score, sha `399095b4`, listener `exp-listener-heard-word-must-be-lexical-41570706`:

| | main | this | change |
| --- | --- | --- | --- |
| overall | 40 (26 missed + 14 false) | 37 (24 missed + 13 false) | −3 |
| listener | 36 (24 missed + 12 false) | 34 (22 missed + 12 false) | −2 |
| tracker | 1 | 1 | 0 |
| checker | 3 (2 missed + 1 false) | 2 (2 missed + 0 false) | −1 |

The two missed calls and the one false warning are the three targets. No other cell moved, so the parents did not interact. Verdict on that score: better.

## Limits

The stack adds nothing the parents did not already measure. 2:284:28 stays under the 10-nat floor, as on the manner branch. A second real skip whose first word sits right after a bound refrain is still dropped with the seam. Stop and fricative remain one obstruent class.
