# Lab test set v1.5

lab-v1.4 grown to 100 sessions per split (ticket 20). Each half of lab-v1.4 had about 88 recited minutes and 9 practice wrong words, the class that matters most. A count of 14 wrong calls moved by about ±4 depending on which sessions were in it, so experiments that moved practice by 5 couldn't be told from noise. Counts are in `lab-v1.5.json` beside this file. lab-v1.1 to lab-v1.4 are untouched on the volume.

- Build: `modal run modal/lab_build_sessions.py --stage loop`. A loop that hits `--iterations` resumes with `--first-iteration N`. `--stage dry` plans without writing, and `--stage supply` counts wrong-word and twin sites per recording.
- Logic: `tools/lab_sessions.py`. Test: `uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest --with rapidfuzz pytest tools/test_lab_sessions.py tools/test_lab_triage.py tools/test_review_api.py`.
- Data: `/lab/data/lab-v1.5/` (same layout).

## What changed

**lab-v1.4 is a frozen subset.** `seed_frozen` copies each of its 40 sessions into lab-v1.5 byte for byte under its old id (`lab-v1.4-p01`…): label, studio and phone WAVs, render checks, and the teacher and `spliced-clean-int8` evidence with their decode records. The planner reads them back and never replans them (`lab_sessions.FROZEN`). They fill their share of the slices and clean slots, and no new window overlaps them. lab-v1.4's refusals are this set's first ones.

- lab-v1.4's applied flags (`flags-v3.json`, 4 windows, the `lab-v1.4.v3` test set) are `lab-v1.5/flags-v1.json` with the same sessions and times.
- On practice, the main line scores the same on the frozen 20 inside lab-v1.5 as on lab-v1.4.v3 (below).
- New sessions are `lab-v1.5-p01`–`p80` and `lab-v1.5-e01`–`e80`.

**Every rule of lab-v1.2 to lab-v1.4 holds unchanged** for the new sessions: seam limits, linked-cut depths, donor length, weak onsets, pause-form donors, host-recording donors, pause-to-pause inserted audio, repeat re-entry and the 30% class caps. Only the window choice and the class mix changed.

**Supply.** Under those rules few words can take a wrong word or a mutashabihat swap. `--stage supply` counts them over every clean run of every held-out recording, with no session window:

| | recordings | minutes | twin-swap words (recordings) | wrong-word words (recordings) | ayahs with either |
| --- | ---: | ---: | --- | --- | ---: |
| practice | 320 | 4105 | 100 (25) | 229 (84) | 259 |
| exam | 300 | 3761 | 109 (26) | 238 (75) | 264 |

Surah 26 holds a third of practice's sites, and they are uneven across readers (6 to 39 ayahs each in practice).

**Planner changes, all for new sessions only:**

- **Several windows per recording.** Windows never overlap in their source recording. lab-v1.4 allowed one window per reader and surah.
- **Windows ranked by sites.** A session with mistakes takes the window with the most ayahs holding a wrong-word or twin site, up to `SITE_CAP` = 3. A clean session takes the window with the fewest. Surah variety breaks ties, as before.
  - A normal-start session with mistakes may go to the split's reader with the richest window, as long as that reader stays within `READ_SLACK` = 3 sessions of an even share.
  - Mutashabihat-focus slots still rank by twin swaps first.
- **Wrong words first.** In the one-per-2–3-ayahs loop, a wrong-word site always wins its ayah, and the loop leaves room when the next ayah holds one.
- **Slices ×5 per split:** 20 ayah-1 starts, 15 strict, 15 mutashabihat focus and 25 clean, the frozen ones included.

## Counts

| | practice | exam | all |
| --- | --- | --- | --- |
| sessions (frozen + new) | 100 (20 + 80) | 100 (20 + 80) | 200 |
| reciters | 15 | 14 | 29 |
| sessions per reciter | 4–8 | 3–10 | |
| minutes | 434.2 | 434.3 | 868.5 |
| ayah-1 starts | 20 | 20 | 40 |
| strict starts | 15 | 15 | 30 |
| clean sessions | 26 | 26 | 52 |
| planted mistakes | 617 | 533 | 1150 |
| wrong word (plain) | 31 | 37 | 68 |
| mutashabihat swap | 48 | 49 | 97 |
| **wrong word, all** | **79** | **86** | **165** |
| omitted word | 298 | 220 | 518 |
| added word | 118 | 101 | 219 |
| skipped passage | 122 | 126 | 248 |
| repeats / restarts (not mistakes) | 68 / 32 | 61 / 39 | 129 / 71 |
| events with `forms_other_ayah` | 97 | 94 | 191 |

- The frozen sessions are as in `lab-v1.4.md`. Practice adds 80 new sessions, 345.5 min: 28 wrong, 42 mutashabihat, 225 omitted, 88 added, 100 skipped. Exam adds 80, 342.8 min: 35, 36, 158, 88, 98.
- Sessions run 183–419 s, at 0.26 planted mistakes per ayah (lab-v1.4: 0.24).
- Against lab-v1.4 per split: about 4.9× the minutes, and about 9× the wrong-word class on practice (79 against 9).

**Short of the target: the wrong-word class has 79 and 86 per half, not 100.** The dry run planned 89 and 90. The loop's refusals then took about 10 per split, as refused sites and ayahs moved windows elsewhere.

- In the dry run, the new windows held 89 and 97 site ayahs, and the plan used 80 and 75 of them. Edit spacing and the ayah-edge rules lose the rest.
- `SITE_CAP` 6 or 10, and a higher per-session twin-swap cap (6/8 against 3/6), changed nothing.
- Going further needs one of:
  - donors from the reader's other recordings (refused since lab-v1.3);
  - inserted audio at linked seams (refused since lab-v1.4);
  - more than 5× the sessions.

**Biases** on top of lab-v1.4's (inserted mistakes only at pauses):

- **Surah concentration.** Surah 26 hosts 51 of the 200 sessions (practice 28, exam 23), against 8 of 40 in lab-v1.4, because its twin ayahs are the main supply. Surah 37 hosts 13.
- **Reciter imbalance.** Reciters with many sites recite more often: up to 8 sessions in practice and up to 10 in exam.

## Loop

The loop converged in 12 iterations, over three runs: the first stopped at a `lab_check` bug, the second at the iteration cap.

- **The bug.** `lab_check` rebuilt the review queue over every session, while the builder now builds it over its own sessions only. It is fixed in 15ec94d.
- **Refusals.**
  - In iteration 3 (the first one past the gate and the checks), the new sessions failed 39 of 1247 studio edits, 9 clean stretches and 12 timing checks. No frozen session failed any studio check.
  - The refusals moved windows, so later iterations replanned many sessions. New refusals fell to 18, 1–2 from the gate, then 4, 1 and 0.
  - Totals: 69 ayahs, 92 sites and 17 donors, lab-v1.4's included.

Final gate: 1734 of 1734 seams pass every check. 43,066 of 43,066 checkable words were found at their labels, 0 of 320 clipped words were over the limit (worst 15 ms), and 0 of 384 donors were short. `spliceRules` are all 0.

Final checks, studio (phone in brackets):

| | n | pass |
| --- | --- | --- |
| wrong word | 68 | 68 (67) |
| mutashabihat | 97 | 97 (94) |
| omitted word | 518 | 518 (493) |
| added word | 219 | 219 (212) |
| skipped passage | 248 | 248 (241) |
| repeat / restart | 129 / 71 | 129 / 71 (128 / 71) |
| clean stretches | 1731 | 1731 (1640) |
| sessions with timings in 0.3 s | 200 | 200 (148) |

- Studio clean PER is 0.90% over 283,088 phonemes (phone 2.39%).
- Core validation (`lab-validate-cli --dir /lab/data/lab-v1.5`) passes on 200 session files and 200 manifest copies, with 0 failures.

## Decode

400 files per listener (200 sessions × studio and phone, about 104,000 s of audio). The frozen sessions' evidence is copied, not decoded again.
- `teacher`: one run per decoded iteration.
- `spliced-clean-int8`: once after convergence.

## Run time

`tools/lab.py` now gives one shard per session, up to 200 (`MAX_SHARDS`). Scoring the main line on lab-v1.5 practice (100 sessions on 100 shards) took 565 s wall from a clean checkout, against about 11 min on lab-v1.4. The run time is still set by the longest session (419 s of audio).

## Main line on lab-v1.5 (practice)

Main line 15ec94d, `spliced-clean-int8`, phone, with the 3 carried practice flags excluded:

| practice | sessions | recited min | wrong calls | missed | false warnings | per 5 min | listener | tracker | checker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab-v1.4.v3 (ledger, 241c788) | 20 | | 11 | 11 | 0 | | | | |
| lab-v1.5, frozen 20 | 20 | 88.1 | 11 | 11 | 0 | 0.62 | 3 | 0 | 8 |
| lab-v1.5, new 80 | 80 | 341.3 | 96 | 71 | 25 | 1.41 | 41 | 5 | 50 |
| **lab-v1.5, all** | 100 | 429.5 | **107** | 82 | 25 | 1.25 | 44 | 5 | 58 |

By class on all of practice (planted / missed): wrong words 79 / 25 (mutashabihat included), omitted 298 / 5, added 118 / 42 and skipped 122 / 10. The 25 false warnings are 10 wrong word, 7 skipped, 5 omitted and 3 added. The teacher reference has 126 wrong calls (78 + 48).

The new sessions score more than twice as badly per minute as the frozen ones: 25 false warnings against 0, and a third of the wrong words missed. Every experiment folded so far was tuned on the frozen 20 practice sessions. Fitting to them would look like this, and so would bad data in the new sessions. Telling the two apart is what the listening check is for.
