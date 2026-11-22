# v1 roadmap: the unimplemented parts of PLAN.md

Agreed 5 October 2026.

## Where we are

The tracker is built and runs on teacher evidence (bookmark-tracker tickets 01–12). tlog shard 0 is relabelled and adopted (ADR 0001). The tracker misses its QUD bar: 69% of starts are found within 5 s against a 95% target, and false jumps are 9% against a 2% cap.

Nothing else in PLAN.md exists yet: the checker, the full tlog pass, ground truth, the student listener, the runtimes, or the SDK.

### Update, 9 October 2026

All tickets an agent can finish are done. What's left is human work:
- Ticket 09 and ticket 10's phone and browser runs (`ready-for-human`).
- Ticket 05's recordings (deferred).
- Ticket 07 phase 2, which needs 05.

The shipped listener is `spliced-clean`: `full` int8 is 68 MB and `lite` int8 is 25 MB. `docs/benchmarks/scoreboard.md` and the dashboard track every metric against its bar. As of `0c41345`, shipped `full` int8 on the help eval:
- **Declared passage:** precision 87.5%, recall 87.5%. Recall passes; precision misses the 90% bar.
- **Free recitation with a lead-in and lead-out:** precision 77.9%, recall 82.8%.
- **Listener gold PER:** 4.82%, against a 3.65% bar.
- **False warnings in the lead-out:** 2.7 per 5 min, against a bar of 1. This may come from the short stitched sessions, which end inside the lead-out; that's untested.

### Closed, 12 October 2026

The roadmap is closed with v1's bars not yet met. Agent work is finished; the open tickets carry over as human work under their current statuses:
- 09 and 10 (`ready-for-human`): the phone and browser runs.
- 05 (deferred): the volunteer recordings.
- 07 phase 2: needs 05, and so does the tracker gate on user-like audio.

Final numbers for shipped `full` int8 (scoreboard as of `d9ccc9b`):
- **Declared passage:** precision 87.5% against 90% (fails), recall 87.5% (passes).
- **Gold PER:** 4.82% against 3.65% (fails).
- **False warnings on clean long sessions:** 1.46 per 5 min against 1 (fails), from 747 min of stitched sessions (`docs/benchmarks/stitched-warnings-v1.md`). It's not a session-end artifact: finalisation adds 0 warnings, and the rate in the last 20 s matches mid-session. The fix is in how the checker confirms a warning (the local-window margin or a confidence floor).

## Decisions

1. **Critical path:** prove what the teacher can hear (E1) and build the checker on teacher evidence before spending on distillation. A cheap device probe de-risks the student in parallel.
2. **Tracker gate:** the 95% / 2% bar is measured on user-like audio, meaning scripted-error recordings and ambiguous tlog clips. QUD becomes a regression report, not a gate.
3. **Checker:** a separate consumer of the tracker's locked alignment and the evidence. Memorization errors come from the alignment; pronunciation errors come from the evidence.
   - *Amended 2026-10-07:* a session that declares its passage gets a checker that aligns against that passage, using the tracker lock only to follow a deliberate jump; free recitation is unchanged. See `docs/benchmarks/acceptance-v2.md`.
   - *Amended 2026-10-08:* free recitation latches onto the tracker's first confident lock and treats it like a declared passage. The passage extends forward as the recitation continues. It re-anchors only on a sustained jump, and re-latches when the lock is lost. See ticket 07 Comments.
   - *Amended 2026-10-09:* local-window confirmation. A wrong, omitted or added word is confirmed only if the listener posteriors favour the decode over the written words by ≥ 10 nats, over the word and one neighbour on each side (`CheckerPolicy.localWindow`). Latched free recitation also finalises at session end (`finaliseFree`). See `docs/benchmarks/local-window-v1.md`.
12. **Listener acceptance (added 2026-10-07):** gold PER on human-verified exact recitations (`docs/benchmarks/gold-per-v1.json`; these clips are excluded from training), with the bar calibrated against end-to-end declared-arm precision (`listener-target-v1.md`). End-to-end swap-in metrics are the real gate. Gold PER is the training proxy.
13. **Student (added 2026-10-07):** `spliced-clean` ships, at `full` and `lite` int8. Its CTC target is the teacher's output at clip edges with the verified text in between, plus teacher soft-target KD. E3 and E6 are deferred until after v1, because v1 is memorization-only. Run E3 first if pronunciation work builds on this student: its canonicalised share is 41.7%, against the teacher's 34.5%.
14. **Mutashabihat (added 2026-10-09):** a planted error that turns the passage into another real ayah is tagged `forms_other_ayah` and scored as its own slice. Free recitation is scored with a correct lead-in and lead-out (stitched preceding and following ayahs), because real slips happen mid-session. Ticket 05's generator plants both kinds.
4. **Pronunciation fallback:** classes the teacher can't hear ship as `unsupported`, so v1 can ship memorization-only. The research bet is per-pair classifiers on tracker-aligned segments (speculative).
5. **Self-correction:** it emits `suspected`, then `resolved`. Whether it counts as a mistake is a policy setting; by default it doesn't.
6. **Repeats** aren't errors. A **skip** is a suspected omission. It becomes `resolved` if the skipped words are recited before the next ayah ends, and `confirmed` otherwise.
7. **Teacher-bias evidence (E1):** `obadx/qdat_bench` and `azizbekphd/ikhlas_recitations`, measured per class. Corrupted synthetic evidence tests checker logic only.
8. **Volunteer recordings:** consent covers commercial training and evaluation. Held-out speakers stay out of training for acceptance numbers. The audio lives in a private bucket, never in git.
9. **Scripts:** about one planted error per 2–3 ayahs, from Juz Amma, Juz Tabarak and a few long surahs, with the mix weighted so each class gets at least 50 positives. Each volunteer also reads one clean script, for false warnings per 5 minutes.
10. **Second relabel review:** a shard filename-labelled ar-Rahman (or al-Baqarah), about 150 clips weighted toward low-confidence and disagreement strata, using the same blind protocol.
11. **Device checks:** a throwaway probe page now. The real check is on physical devices with the SDK web sample.

## Tickets

| # | Ticket | Blocked by |
|---|---|---|
| 01 | E1 teacher-bias report | — |
| 02 | Device probe page | — |
| 03 | Second-shard relabel review | — |
| 04 | Checker: memorization events | — |
| 05 | Gap recordings: pairs, shadda, long sessions, self-correction | 12 |
| 06 | Full tlog teacher pass and bucketing | 03 |
| 07 | E0 and acceptance eval on scripted recordings | 04, 12 (phase 2: 05) |
| 08 | Student distillation (E2, E3, E5, E6) | 01, 02, 06 |
| 09 | SDK runtimes and session API | 04, 08 |
| 10 | E7 quantize, export, device measurements | 08, 09 |
| 11 | E1b spliced-error probe for classes E1 couldn't judge | 01 |
| 12 | Import and audit the help-page clips | — |
| 13 | Stitched-session false-jump test | — |
