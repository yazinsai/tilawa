# 13: Stitched-session false-jump test

**What to build:** A long-session benchmark for the tracker's false-jump cap (≤ 2%), built by stitching existing clean audio together. No new recordings. Ticket 05's long sessions were the planned source, and 05 is deferred.

**Blocked by:** —

**Status:** ready-for-agent

## Why

The `acceptance-v3` tracker gate shows 0/388 clips with a false jump. That proves little: clips average 10.7 s, so the tracker almost never changes lock after its first one. A wrong first lock shows up as a missed start, not as a jump. The false-jump bar is untested on real session lengths.

## Tasks

- [ ] Source audio: help-page clean clips (eval split, held-out speakers only) and tlog-v2 `clean-match` test clips (held-out speaker clusters). Never mix speakers within one session.
- [ ] Contiguous sessions: same speaker, consecutive ayahs in mushaf order, 3–10 minutes. Join with 0.3–1.5 s gaps of silence or room tone, varied. Record the source clip IDs and the ayah span per segment.
- [ ] Deliberate-jump sessions: the same, plus 1–3 planted jumps to a far passage (another surah, or > 2 ayahs away). These measure that a true jump is followed (and how fast) as well as false jumps.
- [ ] Hard cases on purpose: sessions across near-duplicate ayahs (e.g. ar-Rahman's refrain, 79:39 / 79:41, al-Kafirun), where a false jump is most likely.
- [ ] Labels per session: the expected ayah timeline, so a false jump means a non-local lock change to an ayah > 2 from the true position (same definition as `acceptance-v3`).
- [ ] Scoring in core's bench (next to `acceptance-cli.ts`). Report false jumps per session, per jump, and per 5 minutes; true-jump follow rate and latency; and correct-ayah-within-5-s at the session start. Report on teacher evidence first. Students come later, from ticket 08's runs.
- [ ] Audio stays out of git (Modal volume or `.scratch/`). Only the manifest and the report go in `docs/benchmarks/`.

## Caveats

- Stitch seams are cleaner than real pauses: no breath, restart or hesitation. A pass here is necessary, not sufficient. Ticket 05's real long sessions remain the final check when they happen.
- tlog clips carry label noise (bucket cutoff PER ≤ 0.12). Use clean-match clips only.
