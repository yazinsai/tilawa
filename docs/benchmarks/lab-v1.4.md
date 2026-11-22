# Lab test set v1.4

lab-v1.3 rebuilt so that inserted audio meets host audio only at pauses. You heard 9 bad in lab-v1.3's 116 reviewed items, against a bar of 2: 7 wrong words and 2 added words, 4 of them auto-accepted by Nemotron + FastConformer. Every bad splice so far has put donor audio in. No deletion (omitted word, skipped passage) and no mutashabihat twin span has failed. Everything else is as in `lab-v1.3.md`: same catalog, seed 20261028, slices, targets, phone bank, checks, the lab-v1.2 and lab-v1.3 rules, the class cap and the loop. Counts are in `lab-v1.4.json` beside this file. lab-v1.1 to lab-v1.3 are untouched on the volume.

- Build: `modal run modal/lab_build_sessions.py --stage loop`. A loop that hits `--iterations` resumes with `--first-iteration N`. `--stage dry` plans without writing, and `--stage audit --audit-set lab-v1.3` runs this version's rules over lab-v1.3.
- Logic: `tools/lab_sessions.py`. Test: `uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest --with rapidfuzz pytest tools/test_lab_sessions.py tools/test_lab_triage.py tools/test_review_api.py`.
- Data: `/lab/data/lab-v1.4/` (same layout). The energy frames were copied from lab-v1.3.

## The rule

**Inserted audio is cut pause to pause.** A wrong or added word (not a mutashabihat twin span) may meet host audio only at seams that are a pause on both sides: the host pauses there, and so does the donor in its own recording. A pause is an ayah end or an in-ayah gap of at least 0.3 s, cut at −15 dB or deeper (`Rec.pause_after`).

The planner (`Donors.pause_spans`) takes a stretch of the host's own recording that runs from a pause to a pause and reads as the host's words there, except for the planted word:
- **Wrong word:** the donor stretch replaces host words a..b around word k. Its words equal the host's, same diacritised text, scored and said, in one ayah, and differ only at k. For example, مَلِكِ ٱلنَّاسِ in place of إِلَـٰهِ ٱلنَّاسِ, cut at both ayah ends.
- **Added word:** the donor stretch is the host's a..b with one word put in before host word k. With a = k and b = k − 1, it is a donor word said alone between two pauses, landing on a host pause.
- At most `PAUSE_SIDE` = 6 shared words either side, shortest stretch first.
- Only the differing word is labelled (`wrong` or `added`). The shared words are `ref` tokens with `donor: true`, so the scorer and `revert_text` see one edit.

**Repeats go back to a pause.** A repeat already had to break off at a pause. Now it must also re-enter at one: an in-ayah pause at most 6 words back, or, from an ayah's end, that ayah's start (a short ayah said twice). Restarts already went back to the ayah start.

Both rules are enforced in the planner and in the seam gate. `splice_rule_failures` re-reads each seam side's boundary from the source alignment and fails the splice with "inserted audio joins … not pause to pause" (`insertLinked`) or "re-enters … mid-phrase" (`reenterLinked`). The limits are `SEAM_LIMITS["insertSeams"]` and `["reenterAt"]`, both "pause". Deletions and twin spans keep the lab-v1.3 rules.

**Review clips now cover every seam.** Labels carry `meta.reviewSeamPadS` = 1.0, and `review_splice` widens the clip to every seam of the splice, ±1 s. Without it, a shared stretch could put a seam outside the 2.5 s window. Earlier labels lack the field, so their clips and triage cases are unchanged; this was checked on lab-v1.1 to lab-v1.3.

**Bias.** Every inserted mistake in this set sits at a pause, and its donor was said between pauses (usually waqf form). Inserted mistakes in linked speech, which real slips include, are not represented. Deletions and twin swaps still sit in linked speech too: 153 of the 359 seams are linked on at least one side.

## The rule on lab-v1.3

`--stage audit --audit-set lab-v1.3` → `lab-v1.4-rules-on-lab-v1.3.json`. Planner and gate agree on every splice.

| Class | Splices | Refused |
| --- | ---: | ---: |
| wrong word | 98 | 98 |
| added word | 94 | 84 |
| repeat | 29 | 29 (re-entry) |
| mutashabihat, omitted, skipped, restart | 154 | 0 |
| all | 375 | 211 |

**All 9 bad lab-v1.3 splices are refused**, each by `insertLinked`: e09-x3, e14-x4, e14-x9, e14-x19 (bad_audio), p01-x2, p01-x4, p06-x1, p07-x5, p12-x8. e09-x3 is the p15-x1 pattern: host pause into a donor that was linked in its source.

The rule is blunt. Of the 56 inserted splices you passed, 50 are refused, plus 13 passed repeats. Of the old classes, only added words that already sat alone between pauses survive (10 of 94).

## Counts

| | practice | exam | all |
| --- | --- | --- | --- |
| sessions | 20 | 20 | 40 |
| reciters | 15 | 14 | 29 |
| minutes | 88.8 | 91.4 | 180.2 |
| ayah-1 starts | 4 | 4 | 8 |
| strict starts | 3 | 3 | 6 |
| clean sessions | 5 | 5 | 10 |
| planted mistakes | 134 | 118 | 252 |
| wrong word (plain) | 3 | 2 | 5 (2.0%) |
| mutashabihat swap | 6 | 13 | 19 (7.5%) |
| omitted word | 73 | 62 | 135 (53.6%) |
| added word | 30 | 13 | 43 (17.1%) |
| skipped passage | 22 | 28 | 50 (19.8%) |
| repeats / restarts (not mistakes) | 12 / 8 | 12 / 8 | 24 / 16 |
| events with `forms_other_ayah` | 18 | 25 | 43 |

Sessions run 193–416 s, at 0.24 planted mistakes per ayah (lab-v1.3: 0.32). Every session has exactly one repeat or restart. lab-v1.3 had 335 mistakes: 98 wrong, 21 mutashabihat, 62 omitted, 94 added and 60 skipped.

As expected, wrong words nearly vanish. Within one surah recording, a pause-bounded stretch that reads as another stretch except for one word is rare; the clearly different words `wrong_word` wants mostly appear as mutashabihat swaps instead. With those, 24 events are `wrong_word`. The slots go to omitted words, which need no donor.
- Wrong words: 3 are single words between pauses in host and donor, and 2 are 3-word shared stretches.
- Added words: all 43 are a single donor word said between pauses.
- All 67 donors are from the host recording.
- Seams: 359. 206 are pause to pause, 91 pause to link and 62 link to link. All 96 inserted-audio seams are pause to pause.

## Loop

The loop converged in 6 iterations, in one run.

| Iteration | Rendered | Seam gate | Studio mistakes | Non-errors | Clean | Timing | New refusals |
| ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 1 | 40 | 354/354 | 239/249 | 40/40 | 360/361 | 36/40 | 15 |
| 2 | 13 | 344/344 | 237/238 | 40/40 | 350/350 | 39/40 | 2 |
| 3 | 10 | 361/362 (p13-x4: linked cut −7.9 dB, vowel tail) | | | | | 1 |
| 4 | 1 | 362/362 | 253/255 | 40/40 | 366/366 | 40/40 | 2 |
| 5 | 1 | 361/361 | 252/254 | 40/40 | 365/365 | 40/40 | 2 |
| 6 | 1 | 359/359 | **252/252** | **40/40** | **363/363** | **40/40** | 0 |

Refusal totals are 6 ayahs (timing and clean), 16 sites and 0 donors. Almost all were omitted words the teacher didn't hear cleanly. The new rules failed no splice at the gate in any iteration, because the planner already holds them.

Final gate: 359/359 seams pass every check. 9378/9378 words were found at their labels, 0 of 77 clipped words were over the limit, and 0 of 67 donors were short. `spliceRules` are all 0.

Final checks, studio (phone in brackets):

| | n | pass |
| --- | --- | --- |
| wrong word | 5 | 5 (5) |
| mutashabihat | 19 | 19 (19) |
| omitted word | 135 | 135 (126) |
| added word | 43 | 43 (42) |
| skipped passage | 50 | 50 (46) |
| repeat / restart | 24 / 16 | 24 / 16 (23 / 16) |
| clean stretches | 363 | 363 (343) |
| sessions with timings in 0.3 s | 40 | 40 (29) |

Studio clean PER is 0.88% over 62,250 phonemes (phone 2.34%).

## Decode

80 files per listener (40 sessions × studio and phone, about 21,600 s of audio).
- `teacher`: one run per decoded iteration, 5 runs (iteration 3 stopped at the gate), at 325–348 container-s each.
- `spliced-clean-int8` (`/lab/data/models/spliced-clean-int8`): once after convergence, 3007 container-s on CPU, 663 s wall.

## Triage

Nemotron + Soniox, the reach margin and seam pools: `lab-triage-v1.md`, "lab-v1.4". The queue to hear is `lab-v1.4/review/triage.json` (126 items, maxBad 3).

## Known limits

- **Wrong words: 5.** The pause-to-pause rule leaves almost no plain wrong-word sites; see Counts.
- **Inserted mistakes only at pauses**; see Bias.
- **Phone timing is 29/40**, as in lab-v1.3. The loop refuses on studio checks only.
- No scoreboard row: nothing in the scoreboard reads lab test sets yet.
