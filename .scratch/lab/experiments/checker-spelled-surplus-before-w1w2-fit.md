# Spelled surplus before a zero-cost word-1 + word-2 fit

Status: implemented, not practice-scored here. The supervisor scores the branch.

## Prediction

Practice, lab-v1.5. Confirm an added word only when the surplus before word 1 spells a non-neighbour mushaf word and ends before the first zero-cost in-order fit of word 1 plus word 2, and the span is not a contiguous back-run onto the previous ayah's last word. Expect 3 of 5 targets, checker −3, overall −3 (floor −2).

Targets: p63 37:69:1, p63 37:71:1, p64 37:139:1, p14 44:12:1, p35 54:39:1.

## Mechanism

In `checker.ts`, at an ayah seam whose previous word already has a complete in-order form in the greedy tokens, take the phones from the end of that form to the first exact phoneme match of word 1 concatenated with word 2. Strip a leading waqf tail and pause marks. Emit an added word, and skip the 10-nat posterior margin, only when that span:

- matches a whole mushaf form under edits cheaper than one consonant, with the existing added-word shape (at least 3 phones and 2 consonants);
- is not a form or a proper prefix of a word within ±3 of the seam;
- is not a non-Quran phrase at the existing phrase cost;
- is not a contiguous back-run of any length ending on the previous ayah's last word, scored with the existing edit costs;
- is not already the pronunciation of a visit the alignment placed on those phones (a repeat, including a pause form such as قَبْل).

No new threshold. A seam with no such span is unchanged.

## What was validated

Step 0 replayed stored greedy tokens (no acoustic model) over every practice ayah seam, phone and studio.

- Phone targets: 5 of 5 pass. Studio: 4 of 5. p63 37:71 studio is `ءِوَقِفُۥۥهُم`; the extra hamza keeps it from being وقفوهم. Phone is `وَقِفُۥۥهُم`.
- p02 26:88:1, phone and studio: the gap is the restart of 26:87 (`وَلَاا…يُبڇعَثُۥۥن`). It is a back-run and does not fire. The session is fully clean.
- Fully clean sessions: 0 seams fire, phone or studio.
- Other fires: 103, every one on a seam that already has a planted word-1 addition. 0 fires on dirty sessions without that plant.
- p15 38:7:1 is an omission of word 1, not an addition. Phone gap `دڇتَبِعَمَااتِهَ` does contain تبع, and there is no zero-cost fit of 38:7 word 1 + word 2 (best fuzzy cost 45 phone, 34 studio). It does not fire. lab-v1.5-p15 does not recite surah 38.

Unit coverage: a وقفوهم insertion before 37:69:1 warns; re-reading 37:68 does not; replacing 38:7:1 with تبع does not; the existing paused قبل repeat before 51:18:1 stays quiet because that visit already owns the phones.

## Limits

The fit is an exact phoneme string, so a madd-length difference in word 1 or word 2 hides the seam. That is why the studio 37:71 hamza does not count, and it is also how p15 is rejected. A repeat the alignment never visits, of a word more than ±3 away and not a suffix of the previous ayah, can still warn; the paused قبل case is owned by a visit and does not. Step 0 did not run the checker, so a visit boundary that differs from the raw token match can move a seam. Practice score is the supervisor's.
