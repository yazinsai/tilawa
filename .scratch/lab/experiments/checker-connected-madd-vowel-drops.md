# Connected madd, judged by its length

Status: implemented. Not practice-scored. The card predicts overall −1, which is below the fold gate, and the practice set has one target.

## Preregistered prediction

Test set lab-v1.5, practice. A twin-verse swap that only drops a word-final long vowel in connected speech (اذهبا read as اذهب at 20:43:1) is abstained as waqf or held-vowel noise. Decide it by the length of that vowel against this reciter's own madd munfasil. Expect checker −1 and overall −1.

## Mechanism

The checker abstains on a word-final difference that is only a vowel. That is right at a stop, where waqf really drops the vowel, and it is why the suffix checks left 20:43 alone. In اذهبا إلى the ā is madd munfasil, held before the hamza of إلى, and there is no pause. The singular اذهب إلى has no such vowel. Vowel length is still in the phone decode when the consonant burst is not.

The check runs only when three things hold. The written word ends in a long vowel. The next word in the ayah starts with hamza and no labelled pause sits between them (the same blank gap the inner-waqf path already calls a pause). Dropping that vowel letter yields an attested mushaf word, and the 4-word phrase with that word occurs at another ayah whose own word does not still end in a madd. Two sites in the mushaf: 20:43:1 (twin 20:24 and 79:17) and 24:9:5. بها at 7:106 is not one of them: the parallel at 26:31 is بهۦ, still a madd, only written with a diacritic.

At those sites the checker measures frames from the stem's last consonant onset to the next word's hamza onset. It confirms the shorter word only when that span is strictly shorter than every earlier matched madd munfasil in the session. A matched madd is one where a madd letter of the written word was heard and there was no pause. The threshold is that minimum. No new constant. A flagged drop does not enter the baseline. Spans already scrolled out of the window are kept for the rest of the session. The forward sum is not asked to price the vowel; one token is nothing under CTC, which is why the phrase graph never switched this pair.

## Validation

- Unit tests (`checker-madd.test.ts`): the index marks 20:43:1 with the phonemes of 20:24 and leaves 7:106 and an ordinary madd munfasil (20:26:2) out. A synthetic drop shorter than an earlier madd is a wrong word. The same reading held as long as that madd is quiet. No earlier madd is quiet. A 16-frame pause (past the existing 300 ms silence) is quiet even when the span would otherwise be the shortest. A short vowel that is not another mushaf phrase is quiet.
- p79 phone, before wiring the warning. The only practice session that contains either site. Truth gap between 20:43:1 and إلى is 0.080 s. Heard `ب` then `ڇ` then the hamza of إلى, no madd token. Span 6 frames at 40 ms (240 ms). Eight earlier matched madds, shortest 10 frames (400 ms).
- Trace `lab-v1.5-p79` at 160.19 s, oracle positions, shipped policy. Confirmed `wrong_word` at 20:43:1, expected `ءِذهَبَاااا`, observed `ءِذهَبڇ`, confidence 0.95, at 164.08 s (3.9 s after the label, inside the match window). The same confirmation with minConfidence 0 and with the local window off. Nothing else within 7 s.
- `bash tools/autolab_tests.sh` on 6 shards, exit 0. Shard 4 hit Vitest's worker RPC deadline after every test had passed; the script treats that as a pass.

## Limits

One practice target, so the best honest score is overall −1. That does not clear a gate that wants −3. No practice score was spent; the supervisor scores the branch.

The bar is the shortest earlier madd, not a margin under it. A correct madd that is this reciter's shortest so far would flag, and only at the two sites above. Practice contains only the planted drop. A stop shorter than the 300 ms pause test can still be judged if its span is below the reciter's minimum; p79's seam is 80 ms and is the swap, not a stop. The first madd munfasil in a session has nothing to compare with and stays on today's abstention.
