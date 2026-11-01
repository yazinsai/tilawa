# Ayah-seam additions

Status: stopped at step 0. No engine change. Practice was not scored.

## Preregistered prediction

Test set lab-v1.5, practice only. The card predicted 6 of the 11 word-1 added-word targets won, checker −5, overall −4. The kill criterion was to stop if fewer than 6 of those 11 had window-trimmed gap phonemes carrying 2 or more consonants, before any rule was written.

## What step 0 checked

For each target, the decoded phonemes whose start time falls in the truth gap between the previous ayah's last word and word 1 were labeled from the checker's final alignment path, the same path `judge` consumes. A phoneme is window-trimmed when that path drops it as leading junk, as an insertion before the first visit, as trailing speech after the last visit, or never aligns it. Anything matched or inserted on a reference word, or held in the between-word insertion block, is inside a window.

Three arms, one Modal run (`seam-gap-e1e61e9-dirty-20261101T063436Z`), eight shards, all exit 0. The temporary tracer was removed; it did not feed judgment. Per-phoneme rows are in `checker-ayah-seam-additions-step0.json`.

| Arm | Targets with ≥2 window-trimmed gap consonants |
| --- | ---: |
| Oracle evidence, oracle position | 0 / 11 |
| Phone evidence, oracle position | 0 / 11 |
| Phone evidence, tracker | 1 / 11 |

No gap phoneme was marked as a non-Quran phrase. One phone vowel on 44:12:1 sat in the previous word's late half (a waqf tail). The rest are diagonal matches or between-word insertions.

## Where the oracle gap consonants went

This is the checker arm. The inserted word is in the decoded string in every case (`saidConsonants` is 3–5). It is labeled as Quran text.

| Target | Said | Gap consonants |
| --- | --- | --- |
| 44:10:1 | قَبڇلِهِم | 2 on word 1, 4 matched to 44:9:1 and 44:9:2 |
| 44:12:1 | قَبڇلِهِم | 5 matched to 44:17:3 |
| 2:11:1 | قَلِۦۦلَاا | 3 matched or inserted inside word 1 |
| 54:39:1 | فَںںںتَصِر | 2 on word 1, 3 matched to 54:44:5 |
| 26:123:1 | كَرِۦۦم | 2 on word 1, 2 matched to the previous word 26:122:5 |
| 37:99:1 | وَقِفُۥۥهُم | 3 on word 1, 3 matched to 37:102:24–25; an added-word issue opened and resolved |
| 51:18:1 | قَبڇل | 3 matched to 51:16:7 |
| 51:20:1 | قَبڇل | 3 in the between-word insertion block; confirmed only at 126.9 s, after the mistake's match window (ends 118.8 s) |
| 37:69:1 | وَقِفُۥۥهُم | 5 on word 1; a wrong-word issue opened and resolved |
| 37:71:1 | وَقِفُۥۥهُم | 2 on word 1, 3 matched to 37:70:1; an added-word issue opened and resolved |
| 37:139:1 | وَقِفُۥۥهُم | 3 on word 1, 2 matched to 37:139:2 |

The phone true-position arm agrees: 0 / 11 window-trimmed. The same consonants are absorbed into word 1, the previous word, or a third reference word. 51:18:1's phone decode keeps only two between-word consonants (و ب) and drops the rest of قَبڇل, so that one is also thin in the listener string.

The single engine case, 44:10:1, is not a gap the windows never saw. While both seam words were in the alignment those nine phonemes were between-word insertions, and an added-word warning was suspected at 80.5 s. A later tracker re-lock labeled the same frames as lead / before the first visit, and the warning ended uncertain at 80.9 s. True position on the same phone evidence confirms it. Oracle evidence never opens it, because those frames match 44:9:1 and 44:9:2.

## Why the proposed change does not apply

The seam fit was going to price gap residue that neither the previous ayah's last word nor word 1 owns. On these 11 targets that residue is not there. The inserted consonants are diagonal matches on some reference word, so they already belong to a visit. The one between-word block with two or more consonants is the block the current insertion state already prices. Extending the joint fit across the ayah boundary would not move matches off those words, and it would not touch the non-Quran cost or the ayah-start omission recoveries — there is nothing in that shape to repair.

Stopped before any checker edit. No session, reader, or verse literal was added to engine code. Practice scores used: 0.
