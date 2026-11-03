# Hard edits first in two-candidate omission placement

Status: implemented. Practice was not scored here. The card's expected overall change is −2, which is below the fold gate on its own; the supervisor scores the branch.

Test set: lab-v1.5, practice. Component: listener (the checker places the omission; attribution follows the listener's phonemes).

## Prediction

Listener −2, overall −2. Win the p79 phone false warning `lab-v1.5-p79/phone/warning:omitted_word/20:25:5/26.40` and the paired miss on 20:25:4 (لي).

## Mechanism

Two-candidate omission placement sums hard consonant edits and soft channel charges into one cost and puts the omission on the cheaper candidate. Soft charges are there to absorb phone blur, so a point of them can outweigh a hard consonant mismatch.

At 20:25 the listener hears صدري as `دڇ` plus `لِۦ`. The alignment keeps `لِۦ` on لي and leaves the gap on صدري empty. Omitting صدري costs hard 3 / total 45. Omitting لي costs hard 2 / total 46. Total charge therefore lands the omission on صدري.

The change runs only when the checker is already going to emit a one-word omission, the gap holds no phones, and the previous word in the same ayah is holding that segment between two accepted neighbours. It ranks the two placements by hard-edit count, then by total charge. A zero-charge reading of the owning word is left alone: a neighbour heard in full is that word, and must not absorb the omission of the word after it. The omission moves from one candidate to the other. It is not created or removed, and no constant is added. The alignment steps of that pair are rewritten so confirmation sees the segment on the word that kept it.

## Validation

On a current-main trace of p79 20:25, omitting لي has strictly fewer hard edits than omitting صدري (2 vs 3) in that comparison. The kill criterion does not fire.

After the change, the shipped checker (local window on) suspects and confirms `omitted_word` at 20:25:4 and does not warn at 20:25:5. The same confirmation holds with confidence and the local window forced off.

Unit tests in `checker-omission-place.test.ts`:

- `لِۦ` in the لي/صدري slot moves the omission onto لي.
- لي deleted and صدري spoken in full stays an omission of لي (hard edits already agree).
- الذي heard in full with يشفع deleted stays an omission of يشفع (zero charge on the neighbour).

`bash tools/autolab_tests.sh` on 6 shards exited 0. One core shard hit Vitest's worker RPC deadline after every test in that shard had passed; the test script counts that as a pass.

## Limits

44:5 is not claimed. There both orderings already prefer the wrong word, so hard-first does not move it.

A neighbour whose segment charges zero is not re-ranked. That is what keeps an ordinary deletion, such as 2:255:23, on the deleted word. A blurred neighbour (positive charge, fewer hard edits on the other placement) can still move a real omission; that is the same comparison that fixes p79.

p50 37:113:10 and p65 4:12:40 were not measured. The card's speculative upside is not part of the prediction.
