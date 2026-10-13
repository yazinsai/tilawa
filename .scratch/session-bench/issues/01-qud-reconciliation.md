# 01 QUD truth reconciliation across all 16 recordings

Status: ready-for-agent
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
