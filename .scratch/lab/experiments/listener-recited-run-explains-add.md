# A recited run explains a support-path addition

Status: left for the fold gate. Not practice-scored. One of the two targets is won.

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. The card predicted 1–2 of 2 added-word false warnings won: listener −2, overall −2, below the gate on its own.

Measured on the two targets, before any practice score: 1 of 2. `7:198:7` is no longer confirmed. `26:5:1` stays confirmed.

## Mechanism

A support-path added word is phones the alignment inserted between two words, confirmed by shape and the 10-nat margin. Repeat forgiveness only covers a neighbour or prefix within ±3, a contiguous back-run that ends on the previous word, and a non-Quran phrase. A re-read from an ayah start or a phrase one or two ayahs back misses all three, and a phone blur means the greedy string is no longer that lexicon word.

Spell-rule emissions (ayah-start and inner-waqf surplus, `lexicon`) are left alone. For every other pending added word the checker takes the heard span with pauses and waqf tails stripped, and the words it has already read correctly in the current ayah and the two before it, up to but not including the insertion site. A gap in that reading breaks the run.

The span is that earlier text when its consonants, adjacent duplicates collapsed, are exactly some contiguous run of at least two consonants there. The phoneme alignment of that slice uses the existing hard and soft judging charges and must cost no more than inserting the span. A different consonant is not that run: one confusable step (`ه` for `ح`) is enough to match a short planted word onto a neighbour it is not, and the synthetic Al-Mulk addition of `هُوَ` did that.

The rule only vetoes. It adds no constant. The main alignment and the tracker input are unchanged.

## What was checked

True-position traces of both targets. That trace still counted a confusable consonant as the same run; neither target is one, and the final rule only drops exact consonant runs, so these two results stand:

| case | heard | recited run | result |
| --- | --- | --- | --- |
| 7:198:7 | `كُم` | `كُم` inside نصركم, 7:197 | was confirmed, now uncertain |
| 26:5:1 | `قَلِۦۦلُمن` | none in 26:3–5; nearest skeleton is two consonants off | stays confirmed |

`كُم` is the tail of the planted wrong word يتبعوكم at 7:198:6 (`يَتتَبِعُۥۥكُم`). That wrong word stays confirmed. The extra added-word warning was the tail, and it is the same consonants as نصركم from the previous ayah, nine words back, which ±3 does not cover.

`قَلِۦۦلُمن` is the planted addition قليلون (`قَلِۦۦلُۥۥن`, donor 26:54:4) with an extra `م`. Ayahs 26:3–5 do not contain those consonants. The kill criterion was that neither span fits. One does.

Unit coverage:

- `كُم` before 7:198:7, after 7:196–197 have been read, is not confirmed. `قُم` and `كَوثَر` at the same boundary still are.
- `قَلِۦۦلُمن` before 26:5:1, after 26:3–4, is still confirmed.
- Existing addition, surplus, inner-waqf and noise tests stay as they were. The synthetic Al-Mulk session still gives the oracle checker no misses; that is what rejected counting a confusable consonant as the same run.

## Limits

A re-read whose consonants actually changed, including a confusable pair, is kept. So is a re-read of text more than two ayahs back, or of a word this session has not yet read correctly. A planted addition whose consonant skeleton already occurred in the last two ayahs is vetoed; the exemplar census said 15 of 16 planted additions were words the session had not recited, which is stronger than sharing a skeleton. `26:5:1` is the case the consonant test cannot reach. No practice score was taken: one target is under the gate's floor of three.
