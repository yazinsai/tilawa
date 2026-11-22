# Dropped consonant endings, fit with the next word

Status: practice-scored once. Not folded.

## Preregistered prediction

Test set lab-v1.5, practice only, against the stored main line `4001bd6` (89 wrong calls = 64 missed + 25 false warnings, checker 37). Re-apply the longer attested-suffix check and add the missing shorter direction. Expect checker −5 and overall −4, counting that check's two wins plus the three dropped-suffix targets.

## Mechanism

The longer check only scores an attested form that is longer than the written word. A missing ـهما→ـه is still absorbed: a jump to the next word is cheaper than deleting the suffix, and held-sound tolerance can lend that word's onset to the suffix, so one /m/ pays twice and the next word takes only a soft deletion.

When a shorter attested form exists, both pairs are fit with the same insertion/deletion DP over one span, `[written + next word]` against `[shorter form + next word]`, so each heard phone is consumed once. A wrong word is emitted only when three things hold: the shorter pair wins by a consonant (`UNIT`, the same margin as the longer check); the dropped segment contains a consonant (waqf and اذهبا→اذهب do not); and the frames between the remaining word's last confident phone and the next word's confident onset are too short to hold that consonant plus a long vowel. A brisk murattal phone is two frames of the 40 ms stride and a long vowel is a madd of two counts, so the hold is `(1 + 2) * 2 * 40` = 240 ms. A shorter hole was not spoken. A longer hole can be a phone channel that blanked speech, and the check abstains. No confident edge on either side also abstains. The posterior comparison is the same joint span, so the next word's consonant is inside both CTC hypotheses.

If the alignment never opens a visit on the next word because it already used those phones as the ending, the same fit is run on that one visit. Confirmation then does not wait for a diagonal the main alignment will never produce.

## Validation

- `endings-test4` and `endings-test5` (Modal): the ending tests pass. A dropped consonant with nothing in its place is a wrong word; so is the case where the next word's onset is that consonant. A heard consonant that still accounts for the ending stays quiet. Twelve silent frames plus the boundary frame (260 ms at the fixture's 20 ms stride) abstain. Missing and neutral posteriors abstain. A clean recitation of the long form stays quiet. Vowel-only duals are not candidates.
- `core-sdk-test` (`core-sdk-test-410b09d-dirty-20261101T070012Z`): 625 core tests passed. Vitest then exited 1 on its worker RPC timeout; `checker.test.ts` alone is 75 s, past the 60 s `onTaskUpdate` limit. `sdk-test` exited 0. `reacq` exited 0; an earlier parallel run had timed that file out at 5 s under load.
- Oracle traces `suffix-traces-410b09d-dirty-20261101T070013Z`, shipped policy. Both p64 phone targets confirm (`37:119:2` heard `عَلَيهِ`, `37:122:1` heard `ءِننننَهُۥۥ`). p08 `37:119:2` does not: the decode has a one-frame /m/ (conf 0.73) between the stem and `ف`, and the shorter pair does not win. That abstention is the intended one, not a verse special case.

## Measured practice result

Scored code: `dd9d334`. Run: `exp-checker-dropped-suffix-wordform-dd9d334-20261101T071152Z`, via `python3 tools/lab.py run .`. Autolab is paused (`the test data on the volume changed`), so `practice-score` would not have been served. This is the one practice score. No constant was changed after it. `packages/core` is identical between `4001bd6` and this branch's parent, so the delta is the suffix check.

| Phone practice measure | Baseline | Experiment | Change |
| --- | ---: | ---: | ---: |
| Wrong calls | 89 | 85 | −4 |
| Missed mistakes | 64 | 60 | −4 |
| False warnings | 25 | 25 | 0 |
| Checker-attributed wrong calls | 37 | 30 | −7 |
| Listener-attributed wrong calls | 47 | 50 | +3 |
| Tracker-attributed wrong calls | 5 | 5 | 0 |
| Checker with perfect inputs | 51 | 45 | −6 |
| Studio wrong calls | 58 | 52 | −6 |

Phone recoveries, all `wrong_word`, none lost, no new or removed phone false warning:

- `lab-v1.5-p50/phone/mistake:wrong_word/37:108:2/232.40`
- `lab-v1.5-p64/phone/mistake:wrong_word/37:119:2/85.31`
- `lab-v1.5-p64/phone/mistake:wrong_word/37:122:1/102.86`
- `lab-v1.5-p64/phone/mistake:wrong_word/37:129:2/141.41`

Two of the three dropped-suffix targets are won. The third, `lab-v1.5-p08/phone/mistake:wrong_word/37:119:2/185.82`, stays a miss and moves from checker to listener. The same move happens to `lab-v1.5-p08/.../37:108:2/120.64` and `lab-v1.5-p64/.../37:132:1/157.86`. Studio audio of all three is recovered. The phone decode does not give the checker the clean suffix hole; with perfect inputs the checker now catches them, so the miss is the listener's. That is the listener +3. It is not three new mistakes.

Studio also gains a false warning, charged to the listener: `lab-v1.5-p44/studio/warning:wrong_word/23:55:6/108.28`. Trace `p44-studio-dd9d334-20261101T073413Z`: confirmed `wrong_word`, expected `مممَاالِوو`, heard `مَاالِوو`, and it disappears with the posterior window off, so it is a lexical candidate rather than an ordinary hard edit. The word is مَّالٍ. Its isolated form is `مَاال`. Dropping the idgham meems means the heard tanween no longer matches the connected form, and the longer-suffix check treats the tanween waw as an added consonant against the isolated form. The new shorter-direction candidates for this word only drop that waw (`وو`); they are not what emitted `مَاالِوو`. The session has six planted mistakes, so it is not a fully clean session. Phone false warnings on fully clean sessions did not change. Studio overall is still −6.

Fold gate on `git diff -U0 abb918b HEAD` (the engine diff; a diff against `4001bd6` also contains later autolab files and fails hygiene for that reason): gain, spread, collateral and hygiene pass. Spread without reader `mp3quran-295` is −1, the minimum. Significance and surah spread are non-blocking notes: the 95% upper bound is +0 (2 sessions moved), and leaving out surah 37 leaves +0. Every phone recovery is a surah 37 refrain.

## Limits

The p08 phone target remains a miss. The stray /m/ is real in the decode, the shorter pair loses, and catching it would mean flagging a consonant that was heard. 240 ms was not fit to the score. A vowel-only ending still abstains. The check needs the next word's consonants in the span, so a drop at the end of the recording, with no following visit and no absorbed onset, stays on the ordinary path. The studio warning above is the longer check's answer to a deleted idgham, which is the kill criterion's shape (a listener-deleted /m/, a new studio warning) even though the frame-gap test is not the path that fired. All four phone wins sit in one surah, and three of them in one reader.
