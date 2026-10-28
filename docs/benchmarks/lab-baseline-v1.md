# Lab baseline v1 (PROVISIONAL)

**PROVISIONAL.** The lab-v1 test set is not yet verified by listening (ticket 07 pending). The automated checks failed 13 of 370 planted mistakes and 15 of 450 clean stretches (`lab-v1.md#automated-checks`). These numbers are a first reading of the scorer, not a release baseline.

Listener `spliced-clean-int8` (`model.onnx` `714e98ad…af1b`), engine at `6643c64`, teacher `obadx/muaalem-model-v3_2` @ `01a1ef9f` as the ceiling arm. Mode is free recitation with no context; evidence is from `modal/lab_decode.py` (ticket 05).

- Run: `python3 tools/lab.py run --shards 40 --name baseline --cmd 'pnpm --filter @tilawa/core lab'` → `baseline-6643c64-dirty-20261028T131235Z`. The run is marked dirty only because of uncommitted docs; `packages/` and `tools/` are `6643c64`. 40 shards took 56–335 s each, 5931 container-s in all.
- Merge: `python3 tools/lab.py run --name baseline-merge --cmd 'pnpm --filter @tilawa/core lab --merge /lab/runs/baseline-6643c64-dirty-20261028T131235Z'` → `baseline-merge-6643c64-dirty-20261028T132401Z`.
- Score: `/lab/runs/baseline-6643c64-dirty-20261028T131235Z/score.json`, with 40 sessions scored and 0 excluded.
- An earlier run, `baseline-fb65319-20261028T125825Z`, excluded 6 sessions. Its evidence check was too tight for a conv front end: student evidence ends 65–100 ms short, and float error at exactly 0.1 s failed it. Fixed in `evidenceFits` (100 ms plus one frame, compared in whole ms). Discard that run.

## Headline (exam, phone)

**8.00 wrong calls / 5 min** over 93.8 recited minutes: 91 of 188 mistakes missed, 59 false warnings, none on repeats or restarts.

| attribution | missed | false warnings | total |
| --- | --- | --- | --- |
| listener | 27 | 47 | 74 |
| tracker | 1 | 2 | 3 |
| checker | 63 | 10 | 73 |

| component (exam, phone) | |
| --- | --- |
| listener PER | 8.4% (3485 edits / 41,349 phonemes) |
| tracker on place | 98.9%; 14 false jumps (3 before the disambiguation point) |
| lock delay p50 / p90 / max | 0.07 / 0.75 / 16.8 s (41 entries, 21 re-entries, 0 censored) |
| tracker on oracle evidence | 99.4% on place, 11 false jumps, lock delay p90 0.28 s |
| checker on oracle evidence and position | 5.54 / 5 min (77 missed + 27 false warnings) |
| teacher evidence through the engine | 8.10 / 5 min (78 missed + 74 false warnings); teacher PER 7.0% |

## Splits and slices

| / 5 min (missed + FW) | studio | phone |
| --- | --- | --- |
| exam | 5.22 (73 + 25) | **8.00** (91 + 59) |
| practice | 6.22 (75 + 38) | 8.09 (84 + 63) |
| strict, exam (3 sessions) | 11.11 (19 + 12) | — |
| strict, all (6 sessions) | 7.83 (26 + 18) | — |

Phone vs studio is almost all listener. On exam, listener PER goes from 5.2% to 8.4%, and listener-attributed calls go from 25 to 74. The checker share barely moves (71 to 73). The strict slice is a smoke number: only 6 sessions, and one exam start has a 16.8 s lock delay.

## Per class (exam, phone)

| class | missed / planted | listener | tracker | checker |
| --- | --- | --- | --- | --- |
| wrong word | 36 / 67 | 13 | 1 | 22 |
| mutashabihat (within wrong word) | 5 / 14 | 2 | 0 | 3 |
| omitted word | 7 / 43 | 5 | 0 | 2 |
| added word | **46 / 57** | 9 | 0 | 37 |
| skipped passage | 2 / 21 | 0 | 0 | 2 |

False warnings by class: wrong word 34, omitted 14, skipped 8, added 3.

## What it says

- **Added words are the biggest loss.** 46 of 57 are missed on exam, 37 of them by the checker. On oracle evidence with an oracle position, the checker misses 77 of 110 added words over both splits. Of the 92 missed added words in practice cases, 75 have no warning of any class within 6 s, so the checker is silent rather than mislocating. This is a checker problem first.
- **Class confusion counts twice.** About 10 missed added words in practice have a warning on the same word but of another class (`wrong_word` or `skipped_passage`). The scorer requires the class to match, so each counts as a miss plus a false warning. That is defensible (the user sees the wrong message), but worth an explicit decision in ticket 08.
- **The listener drives phone false warnings.** 47 of 59 are listener-attributed. In studio exam that falls to 13 of 25. Phone robustness is the listener lever.
- **The tracker is not the problem.** It accounts for 3 of 150 calls and is 98.9% on place.
- **The teacher arm is not a ceiling yet.** Run through the engine, teacher evidence scores *worse* than the student (8.10 vs 8.00 on exam phone, 74 false warnings), even though its PER is lower (7.0% vs 8.4%). Speculation: the checker's thresholds are calibrated to the student's posteriors, and the teacher's top-5 bf16 posteriors are sharper and differently calibrated. Treat the "teacher ceiling" as a calibration check, not a bound.
- **Hardest exam sessions**, by misses plus false warnings on phone: e08 17, e18 16, e19 14, and e05, e07, e16 at 13 each. e08, e18 and e19 also hold failed data checks.

## Data checks in the score

Cases on data that failed the automated checks carry `dataCheck` (`lab-v1.md#automated-checks`); counts are not changed. In the practice drill-down, 10 cases are marked:
- 6 are planted mistakes: p07-x5 and p07-x7 (pause-form donors) and p13-x15, each on both copies.
- 4 are false warnings inside failed clean stretches (p09 ×3, p10 ×1).

Exam cases stay hidden by design. Leaving out all 13 failed edits would remove at most 13 of the 188 + 182 mistake cases.

No scoreboard row: the scoreboard doesn't read lab-v1. The lab dashboard and ledger (tickets 10–11) will.
