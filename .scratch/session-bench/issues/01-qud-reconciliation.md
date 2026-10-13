# 01 QUD truth reconciliation across all 16 recordings

Status: resolved
Type: task
Blocked by: none

Spec: `../spec.md` (Sources §2, "Full QUD annotation reconciliation"). Code: `packages/core/src/bench/sessions.ts` `qudSession`. Data: `.scratch/bench/qud/*.json` (+ `.parquet` teacher evidence).

`qudSession` currently throws on partially overlapping non-Quran truth and other contradictions; only 3 of 16 recordings are scored. Make every one of the 16 convert, or be excluded with an explicit, recorded reason.

## Acceptance

- A conversion audit (CLI flag or script) lists, per recording: converted / excluded, reason, counts of spans, repeats, restarts, non-Quran spans, surah jumps.
- Reconciliation rules are deterministic and documented in `docs/benchmarks/session-bench-v1.md`: e.g. a non-Quran interval partially overlapping a word is clipped to the gap, never stretched over truth; Fatiha restarts in prayer audio become `restart` spans with `nonErrors` entries; backward jumps within an ayah become `repeat`. Contradictory truth is still rejected, never silently counted.
- Restart detection: a return to an earlier ayah (not just same ayah) is a `restart`/`repeat` occurrence and gets its own span; validation still passes.
- Tests in `packages/core/test/sessions.test.ts` pin each rule with a small synthetic `QudRecording`.
- The all-16 teacher replay finishes (it was previously stopped before completion — find and fix why, e.g. runtime on long recordings) and writes a report.

## Comments

- All 16 QUD recordings now convert and validate; none excluded. `node packages/core/dist/bench/sessions-cli.js --qud .scratch/bench/qud --audit` writes `docs/benchmarks/session-qud-audit-v1.json` (per recording: status, reason, spans, repeats, restarts, non-Quran spans, surah jumps). Repeat+restart counts equal QUD `repeat_events` except the extra Fatiha restarts in the 3 prayer recordings.
- Previously only 2 actually failed: Minshawi (final word 36 ms past audio end) and prayer-baqarah-yasin (final non-Quran 92 ms past end). Rule: clip <=0.1 s clock overruns, reject larger.
- New rules in `qudSession` (documented in `docs/benchmarks/session-bench-v1.md`, "QUD reconciliation"): returns to an earlier ayah become `restart` (lands on word 1) or `repeat` spans with `nonErrors` entries, tracked across Basmala/Isti'adha; non-Quran clipped to word gaps and splits the occurrence; non-Quran covering or inside a word is rejected as contradictory. `qudAudit` wraps conversion + validation. 5 new synthetic tests in `packages/core/test/sessions.test.ts`.
- Why the all-16 replay never finished: the checker re-ran its window alignment every 40 ms push with per-cell cost-function calls (~300 ns/cell). Cost tables are now cached per window and phoneme in `checker.ts`; output identical (Nisa rows byte-equal, Diban/Jaber match prior baseline, checker tests pass), ~10x faster. All 16 replay in 15 min -> `docs/benchmarks/session-qud-teacher-v1.json`.
- No scoreboard row (ticket 05).
