# v1 roadmap: the unimplemented parts of PLAN.md

Agreed 5 October 2026.

## Where we are

The tracker is built and runs on teacher evidence (bookmark-tracker tickets 01–12). tlog shard 0 is relabelled and adopted (ADR 0001). The tracker misses its QUD bar: 69% of starts are found within 5 s against a 95% target, and false jumps are 9% against a 2% cap.

Nothing else in PLAN.md exists yet: the checker, the full tlog pass, ground truth, the student listener, the runtimes, or the SDK.

## Decisions

1. **Critical path:** prove what the teacher can hear (E1) and build the checker on teacher evidence before spending on distillation. A cheap device probe de-risks the student in parallel.
2. **Tracker gate:** the 95% / 2% bar is measured on user-like audio, meaning scripted-error recordings and ambiguous tlog clips. QUD becomes a regression report, not a gate.
3. **Checker:** a separate consumer of the tracker's locked alignment and the evidence. Memorization errors come from the alignment; pronunciation errors come from the evidence.
   - *Amended 2026-10-07:* a session that declares its passage gets a checker that aligns against that passage, using the tracker lock only to follow a deliberate jump; free recitation is unchanged. See `docs/benchmarks/acceptance-v2.md`.
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
