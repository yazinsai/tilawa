# 13: Stitched-session false-jump test

**What to build:** A long-session benchmark for the tracker's false-jump cap (≤ 2%), built by stitching existing clean audio together. No new recordings. Ticket 05's long sessions were the planned source, and 05 is deferred.

**Blocked by:** —

**Status:** done

## Why

The `acceptance-v3` tracker gate shows 0/388 clips with a false jump. That proves little: clips average 10.7 s, so the tracker almost never changes lock after its first one. A wrong first lock shows up as a missed start, not as a jump. The false-jump bar is untested on real session lengths.

## Tasks

- [x] Source audio: help-page clean clips (eval split, held-out speakers only) and tlog-v2 `clean-match` test clips (held-out speaker clusters). Never mix speakers within one session.
- [x] Contiguous sessions: same speaker, consecutive ayahs in mushaf order, 3–10 minutes. Join with 0.3–1.5 s gaps of silence or room tone, varied. Record the source clip IDs and the ayah span per segment.
- [x] Deliberate-jump sessions: the same, plus 1–3 planted jumps to a far passage (another surah, or > 2 ayahs away). These measure that a true jump is followed (and how fast) as well as false jumps.
- [x] Hard cases on purpose: sessions across near-duplicate ayahs (e.g. ar-Rahman's refrain, 79:39 / 79:41, al-Kafirun), where a false jump is most likely.
- [x] Labels per session: the expected ayah timeline, so a false jump means a non-local lock change to an ayah > 2 from the true position (same definition as `acceptance-v3`).
- [x] Scoring in core's bench (next to `acceptance-cli.ts`). Report false jumps per session, per jump, and per 5 minutes; true-jump follow rate and latency; and correct-ayah-within-5-s at the session start. Report on teacher evidence first. Students come later, from ticket 08's runs.
- [x] Audio stays out of git (Modal volume or `.scratch/`). Only the manifest and the report go in `docs/benchmarks/`.

## Caveats

- Stitch seams are cleaner than real pauses: no breath, restart or hesitation. A pass here is necessary, not sufficient. Ticket 05's real long sessions remain the final check when they happen.
- tlog clips carry label noise (bucket cutoff PER ≤ 0.12). Use clean-match clips only.

## Results, 2026-10-07

Agent note. Report `docs/benchmarks/jumps-v1.md` (+ `.json`, which holds each session's expected ayah timeline), manifest `docs/benchmarks/jumps-v1-manifest.json`. Build with `tools/jumps_build.py`, score with `pnpm --filter @tilawa/core jumps`. Teacher evidence only, stitched at the evidence level from the existing per-clip decodes; no Modal runs.

129 sessions, 747 minutes: 84 contiguous (tlog), 35 with 1–3 planted jumps (tlog), 2 hard-run sessions (tlog), 8 help speakers.

- False jumps: 3 in 235 jumps (1.3% per jump). That's 2 of 129 sessions, 0.02 per session and 0.02 per 5 minutes. Contiguous tlog alone has 1 in 554 minutes.
- Planted jumps followed: 192/216 (88.9%), 70.8% within 5 s, median latency 3.4 s, p90 6.4 s. The tlog planted jumps were all followed (48/48). The 24 misses are help jumps to single short or repeated ayahs (7 in the ar-Rahman refrain, 113:x, 44:1 حم, 86:2).
- Correct ayah within 5 s of session start: 111/129 (86.0%), against the ≥ 95% bar.
- By hand: both help false jumps come just after an unfollowed planted jump, when the next clip's opening pulls the lock to a wrong far ayah (3:2 for al-Fatiha, 53:26 for 10:84). The tlog one is a 4-ayah backward lock (48:29 → 48:26) at the 48/49 surah seam.

Deviations from the tasks above:

- **Speakers are mixed in tlog sessions.** tlog has no speaker IDs and its clusters are coarse. Held-out clips rarely give consecutive ayahs within one cluster (longest about 3.5 minutes), so the stitcher keeps the previous clip's cluster when it can and otherwise changes it (2,952 of 3,861 seams). Help sessions are single-speaker, but help clips are scattered single ayahs, so every help seam is a planted jump.
- The tlog pool is clean-match **test + val**, both held out by cluster. Contiguous steps may skip one ayah (still local).
- Few dedicated hard sessions (2). Most near-duplicate ayahs sit inside long contiguous runs; the report slices sessions with 3+ near-duplicate segments instead.
- Stitched audio exists only for help sessions (`.scratch/jumps/audio/`). tlog audio isn't local, so student runs on tlog sessions first need the clips downloaded.
