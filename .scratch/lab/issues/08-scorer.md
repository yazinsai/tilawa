# 08 Lab scorer: overall, components, attribution

Status: ready-for-agent
Type: task
Blocked by: 03

Spec: `../spec.md` goal and decisions 3, 7, 12. Build on `packages/core/src/bench/sessions.ts`.

## Acceptance

- Overall score: wrong calls per 5 min on the phone-degraded audio, plus the missed and false-warning counts, per practice and exam set.
- Listener: PER against what was said (the labels with planted edits applied).
- Tracker: share of recitation time on the right place counted from the disambiguation point; false jumps (confident wrong locks, including before the disambiguation point).
- Checker: wrong calls per 5 min with oracle evidence and the true position.
- Attribution: each overall wrong call assigned to exactly one of listener, tracker or checker; the parts add up to the total. Unit tests cover the additivity.
- Slices: strict, mutashabihat, studio, per mistake class.
- Per-case outcomes written for the drill-down (same keys the review UI uses).
