# Wide student, constants from its own decodes

Status: implemented. The supervisor scores the branch.

## Prediction

Test set lab-v1.5, practice. The 2×-width student (`wide-clean`, Zipformer-M depth, every width doubled, 247,690,818 parameters) hears added words the shipped student blanks. Deployed as the decoder with the shipped student's constants it lost 58 incidents, because those constants were fit to the shipped noise. Re-derive them from this model's own gold-excluded decodes. 4 of 6 added-word targets won, losses cut from 58 to at most 3: listener −5, overall −4.

## Mechanism

`tools/listener_calibration.py` decodes 60 gold-excluded training readers (clean-match, gold PER clips removed) under the shipped augmentation: noise probability 0.5 at 5–25 dB, RIR probability 0.2, gain of −10 to +6 dB, fixed seed. SpecAugment is a training mask and is not applied at decode. Both students see the same augmented waves.

The formulas are fixed, and running them on the shipped student reproduces today's constants (the idempotence test):

- A soft pair's cost is the negative log of its confusion rate, in units where the shipped rate costs exactly today's soft cost, floored at that cost and capped at one plain edit. The wide student moves only ث/س, from 5 to 6. Vowel `SOFT` stays 3.
- The weak-evidence confidence cut is the clean-word quantile that keeps the shipped gate's pass rate. It moves from 0.8 to 0.798.
- `SUPPORT` and the tracker's nats-per-edit scale by the ratio of median per-phone CTC nll on words the alignment matched. 10 becomes 9.601 and 3 becomes 2.880. The wide student is slightly sharper on correct words.
- The parallel-phrase switch stays at 4 while the 99th percentile of correct-phrase margins is still under it, and follows a tail only once that tail crosses 4. Both tails are negative (shipped −8.93, wide −5.56), so the penalty stays 4. Adding the 3.4 nat gap between those tails had put it at 7.37, above planted swaps.
- The blank-cost cut keeps the shipped offset from the maximum blank-forcing cost of a clean pause (8 frames or more, between two correct words). That offset is what lands the shipped student on 2× the old support, 20 nats. The wide cut is 18.45. No landed gate compares a gap's blank nll, so the number is derived and tested, and not a new rejection.

4004 correctly matched words, 300 parallel phrases, 190 clean pauses. Idempotence on the merged shipped decode: no constant moved outside the test's noise band.

The int8 export already produced for `wide-clean` is what the supervisor decodes. `.autolab/model.json` points at it.

## What was validated

SDK streaming decode of the wide int8 student (`wide-clean-int8`, 20 ms pushes, the lab's decode path), then the checker at the true position with the derived constants:

| incident | true position | engine |
|---|---|---|
| p31 `4:157:24` 55.47 | hit | caught, misclassified |
| p31 `4:170:12` 288.67 | miss | miss, listener |
| p50 `37:99:1` 146.47 | hit | hit |
| p55 `26:193:1` 269.44 | hit | hit |
| p56 `32:21:1` 106.76 | hit | hit |
| p79 `20:40:25` 136.10 | hit | caught, misclassified |

5 of 6 confirmed at true position. The kill line is 4. The one miss is the second p31 insertion. Unit tests: the formula reproduces today's constants on a synthetic shipped-shaped sample and on the real shipped decode, and the wired numbers are the wide derivation. A more-confused pair stays at the floor; a less-confused pair rises.

## First practice score

Wide int8 student, constants including the phrase-switch penalty at 7.37. Phone practice, main line 43 wrong calls:

| | main | this | change |
|---|---:|---:|---:|
| overall | 43 (28 missed + 15 false) | 50 (33 + 17) | +7 |
| listener | 36 (24 + 12) | 37 (23 + 14) | +1 |
| tracker | 4 | 4 | 0 |
| checker | 3 (2 + 1) | 9 (8 + 1) | +6 |

The checker gained 6 misses and no false warnings. That is the penalty moving from 4 to 7.37: real swaps whose margin sits between those two numbers stop confirming, including on oracle evidence, so they are charged to the checker. The listener's net is one fewer miss and two new false warnings. The switch rule was narrowed as above; the other derived constants are unchanged.

## Limits

The two halves of the 60 readers disagreed on the sign of the phrase-tail shift. Support and the confidence cut barely move. Added-word confirmation uses `SUPPORT` (9.60, was 10) and the ث/س soft cost (6, was 5), not the phrase switch. The first score is before the switch narrow.
