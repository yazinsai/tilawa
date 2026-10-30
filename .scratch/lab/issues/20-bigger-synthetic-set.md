# 20 Bigger synthetic test set

Status: ready-for-agent
Type: task
Blocked by: none

Each half of the current test set (lab-v1.4.v3) has about 88 recited minutes and about 134 planted mistakes, so a count of 14 wrong calls wobbles by about ±4 from which sessions happen to be in it. Per class it is thinner: practice has only 9 wrong words, the most important class. Experiments moving practice by 5 can't be told from noise.

Build the next test set version with about 4–5 times the sessions per half, using the existing spliced session builder (ticket 03, `modal/lab_build_sessions.py`) and its rules (lab-v1.2 and later).

## Notes

- Only 32 held-out reciters exist (16 practice, 16 exam; `docs/benchmarks/lab-reciters-v1.json`). More sessions means more passages per reciter, not more voices. Keep each reciter in its split.
- Weight the plant mix toward wrong words, with mutashabihat swaps included, aiming at 100 or more of each class per half.
- Keep the current sessions as a subset, so the new version's numbers can be compared on the old sessions too.
- Flags from earlier versions (`flags-v3.json`) must carry over to the sessions they apply to.
- Decoding and scoring the larger set cost Modal time. Check the experiment run time, about 11 min today, set by the longest session.

## Acceptance

- The new version validates, renders, and passes the builder's seam and render checks.
- A spot-check listening queue goes to the user, like ticket 07's v1 check, before the set is trusted.
- `lab apply-flags`-style registration as the current test set, with the main line re-scored on it.
- Write-up with counts per class and split, and the main line's practice numbers on the new version next to the old.
