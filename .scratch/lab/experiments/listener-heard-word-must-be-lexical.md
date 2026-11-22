# A wrong word must spell another mushaf word

Status: practice-scored. Left for the fold gate.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-context-free-gap-decode-62ac8b5`. 3 of 6 phone wrong-word false warnings won, 1 loss: listener −3, overall −2.

## Mechanism

The per-word string path confirms a substitution once the heard phones beat the written word by `SUPPORT` (10 nats), or by twice that when the changed sounds average below 0.8. Those margins do not separate a channel blur from a planted swap. The heard string does: a planted swap is an attested mushaf word, and a one-consonant blur, a swallow, or a truncation almost never is.

A pending wrong word from that path is still confirmed at today's margin when either of these holds, with no new constant:

- Its heard span, phrase-labelled pauses removed and edge waqf marks stripped, equals a lexicon word under soft costs only (the folded surplus spell test: same length after held-sound collapse, vowels may differ, no consonant edit). The word must not be the written word and not a word or prefix within ±3.
- The span has more consonants than the written word. That extra speech is not a blur of this word; the scorer counts a wrong word here as the added word on the next word.

Otherwise the span must also clear twice `SUPPORT`. Parallel-phrase warnings are exempt. The rule only vetoes.

## What was checked before the score

True-position traces of the six targets. Every heard span fails the spell test:

| case | written | heard | result |
| --- | --- | --- | --- |
| 26:106:1 | إذ | يَاا | stays; margin already clears twice `SUPPORT` |
| 7:71:16 | نَزَّلَ | لَمزَلَ | vetoed (12 nats) |
| 22:60:11 | لَيَنصُرَنَّهُ | truncated | vetoed (14 nats) |
| 55:48:1 | ذَوَاتَا | دَوَاافَا | stays; margin already clears twice `SUPPORT` |
| 18:26:20 | يُشْرِكُ | نَصِرِكُ | vetoed (12 nats) |
| 22:9:1 | ثَانِيَ | ذَاالِيَ | vetoed (15 nats) |

4 of 6 targets. A fifth phone false warning, truncated غرورا at 35:40:33 (`رَاا`, 12 nats), goes with them. صالح for نوح in the same session as 26:106:1 still spells a lexicon word and stays confirmed.

The first score, before the extra-consonant clause, was overall −4 and listener −4 against the previous main, with one new miss: the added word قليل at 2:23:13 was caught only by a wrong-word warning on the previous word, whose span `مممِثلِهقَلِ` is not one lexicon word and won by 13 nats. That span has more consonants than مثله. The clause puts it back on today's margin. The same-length and shorter blurs stay vetoed.

Unit coverage, full core and sdk suite on Modal, passing (one shard hit Vitest's worker RPC deadline after its tests had passed):

- A non-word at the one-support band (غ/ق for ش/ف, the same frames that still confirm يسمع) is not confirmed. Halving the configured margin, so twice `SUPPORT` sits on that band, confirms it.
- A spelled word with a trailing madd mark still confirms at that band.
- Replacing a vowel with a consonant, which adds a consonant, keeps today's margin.
- A distant mushaf word is accepted by the spell test; the written word, a ±3 neighbour, and a non-word are not.

## Measured practice result

One score of the merged tree (this change on the folded first-lock tracker), sha `df70b672`. Two earlier scores are notes: the pre-narrow score, and a score taken after main had moved but before this branch merged it. No third interpretation.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 56 (37 missed + 19 false) | 51 (37 missed + 14 false) | −5 |
| listener | 41 (24 missed + 17 false) | 36 (24 missed + 12 false) | −5 |
| tracker | 2 (2 missed + 0 false) | 2 (2 missed + 0 false) | 0 |
| checker | 13 (11 missed + 2 false) | 13 (11 missed + 2 false) | 0 |

Phone false warnings fell by 5. Misses did not move. Tracker and checker are unchanged. Studio and the leave-one-reader cut are not in this compare; the fold gate has them. The rule cannot add a warning.

## Limits

The spell test is exact after held-sound collapse, so a planted swap the listener hears with an extra phone or a shortened skeleton needs twice `SUPPORT` and can be lost when that margin is only one `SUPPORT`. The extra-consonant clause covers the glued-addition case that did that on the first score; it does not cover a shorter garble of a real swap. Two of the six targets already clear twice `SUPPORT` and stay.
