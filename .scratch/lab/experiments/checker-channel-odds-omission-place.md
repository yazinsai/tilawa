# Channel likelihood for two-candidate omission placement

Status: practice-scored once. Left for the fold gate.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. 1 of 2 pairs flips: checker 0, overall −2 (upside −4 if both).

## Mechanism

Two-candidate omission placement is the alignment's summed hand cost. Hard-edits-first reordered those same costs, fixed 20:25, and flipped 11 correct placements: the hand costs mislabel which edits are common listener blur.

Inside an already-proposed one-word omission, when the previous word was accepted, the next word closes the gap, both words are in the same ayah, and the phones between the previous word's visit and this one are no longer than the two words together, rank the two survivors by the shipped listener's confusion channel. The channel is `LISTENER_CHANNEL`: P(heard token | written phone), counted by minimum-cost alignment of gold-excluded shipped decodes, with the estimator's Laplace 0.5. It is not split by word position or after-pause; that split is not in the table that was measured. The omission moves onto the other word only when the later word's survival has strictly higher Viterbi log probability on that stretch. A tie leaves today's placement. The omission count does not change, and a stretch with no such pair is untouched. No placement margin was added.

## What was checked

True position, phone, both targets. The stretch the rule sees:

| Target | Stretch | log P as the spoken word | log P as the other word | Result |
| --- | --- | ---: | ---: | --- |
| p14 44:5 | `ءُننننَاا` | كنا −7.04 | إنا −7.12 | suspected 44:5:4 at 38.92, confirmed at 50.08 |
| p79 20:25 | `دڇلِۦ` | صدري −12.80 | لي −18.90 | suspected 20:25:4 at 26.40, confirmed at 29.56 |

Both were confirmed on the spoken neighbour before (44:5:5, 20:25:5). `لِۦ` alone still prefers لي by 12 nats, which is why the stretch has to include the `دڇ` that sits between اشرح and the لي visit. p14's margin is 0.07 nat: the table has no count of ك heard as ء (the post-pause cell the card expected), and ُ heard for ِ is one count against 402 of ِ, so the smoothed off-diagonals nearly tie. The real stretch is exactly `ءُننننَاا`, and that tie breaks toward كنا.

Unit tests: the shipped table ranks those two stretches toward the spoken word and ranks `لِۦ` toward لي; with the channel off, the `لِۦ` fixture still omits صدري; an injected channel that makes صدري the survivor moves that omission onto لي; a fully heard لي does not take an omission of صدري. Modal core and sdk tests passed; one core shard hit Vitest's RPC deadline after every test passed.

## Measured practice result

One practice score. No second score.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 40 (26 missed + 14 false) | 36 (24 missed + 12 false) | −4 |
| listener | 36 (24 missed + 12 false) | 32 (22 missed + 10 false) | −4 |
| tracker | 1 (0 missed + 1 false) | 1 (0 missed + 1 false) | 0 |
| checker | 3 (2 missed + 1 false) | 3 (2 missed + 1 false) | 0 |

Misses and false warnings each fell by 2, which is both pairs. Checker attribution is unchanged, as predicted. The compare verdict was better.

## Limits

The table mixes the seam sample's studio and phone decodes (5246 written phones). It was not re-counted on phone only, and not conditioned on onset or pause. A heard string other than `ءُننننَاا` at 44:5 would not inherit this 0.07 nat. The practice compare does not print clean-session warnings, studio, or incidents lost; the fold gate still has those.
