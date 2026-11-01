# Pause-anchored skip seams

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. Two of the three surah-26 miss-plus-false-warning pairs flip: checker −4, overall −4. One pair left in place is the registered −2 case, not a retune.

## Mechanism

Skip detection, the alignment and the anchor set are unchanged. An already-proposed skip whose boundary words are the same consonant skeleton still prefers an ayah start, as before.

The المؤمنين/مؤمنين pair is not that equality. The heard body is `مُءمِنِۦۦن`, an exact visit of the skipped ayah's مؤمنين, and the article's ل is only a cut-off fragment of المؤمنين. The alignment's seam sits on that fragment (40–160 ms). The stop is later, on the blank run after the heard word.

A seam may also slide when one skeleton is the other plus a single leading consonant. That slide is kept only when its jump falls on a blank run the checker already treats as a pause, or on a cut-off visit. A pause is `minGapFrames` (`GAP_PAD_S`, 300 ms): shorter than that is neighbour audio. Frames in the run must be blank-owned (`log 0.5`, the same test the phrase scorer uses); with no posteriors the greedy blank run is enough. Among those anchored slides, an ayah start wins. If none is anchored, the lexical placement stands. No skip is created or removed, and a seam that is already on an ayah start is not moved.

## What step 0 showed

Oracle checker, listener `spliced-clean-int8`, phone. Blank-owned runs and greedy gaps agree.

| Case | True seam | Pause at the true seam | What the alignment did |
| --- | --- | --- | --- |
| p24 `26:103:1` @ 77.25 | 26:102:7 المؤمنين ends, 710 ms, then 26:104:1 | 77.24–77.96, 680 ms, ن\|و, blank-owned | Gap 26:102:7–26:103:7. After-visit is 26:103:8 مؤمنين heard exactly. Jump at 75.4 s is 160 ms. 26:102:7 is a cut-off `ل` (coverage 0.09). |
| p73 `26:103:1` @ 99.92 | 26:102:7 ends, 600 ms, then 26:104:1 | 99.92–100.52, 560 ms, ن\|و, blank-owned | Same binding. Jump at 98.0 s is 120 ms. |
| p24 `26:130:1` @ 247.21 | 26:129:4 تخلدون ends, 810 ms, then 26:131:1 | 247.16–248.00, 800 ms, ن\|ف, blank-owned | No skip of 26:130 is proposed. A different suspicion sits at 26:131:2, seam 80 ms. فاتقوا/واتقوا differ by ف/و, not one leading consonant. |

The labelled pause sits at the true seam in all three pairs. The third pair is not a look-alike seam of an already-proposed skip, so the rule does not create one. That is the pair the prediction left unwinnable.

A held madd inside the last word (ۦ\|ن, 640–760 ms) is a blank run too. It is not a seam: the jump the slide keeps is the blank between the heard word and the next word.

## Validation

- Unit: with the article deleted and ayah 26:103 omitted, the warning stays at 26:102:7. The same audio with a 20-frame gap before 26:104:1 (fixture stride 20 ms, above `minGapFrames`) confirms `skipped_passage` at 26:103:1 and nothing else. A full recitation of 26:102–105 with that gap and no deletion confirms nothing.
- Existing skip tests, including the equal-skeleton ayah-start cases, still pass. Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0.
- Oracle phone after the change: p24 and p73 confirm `skipped_passage` at 26:103:1 (they confirmed 26:102:7 before). p73's separate skip at 26:100:1 is still confirmed there. p24 26:130:1 is still not that warning; 26:131:2 stays a suspicion.
- Studio and fully clean phone, every practice session: the only seams that move are p24 and p73 studio, both `26:102:7 → 26:103:1`. Clean sessions do not move a skip. Those two studio moves are the same correction, not a new warning.

## Limits

The slide does not fire on a one-consonant substitution, so 26:130 stays missed. A skip the alignment never proposes is not invented. Practice was not scored here.
