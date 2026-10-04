# 10: Streaming benchmark and performance gate

**What to build:** A benchmark harness reports how the full streaming tracker performs against the v1 acceptance bar on the QUD alignment benchmark and `nurlingo/quran-recitation-detect-split`, plus a performance budget check.

- **Location metrics:**
  - correct-ayah-within-5-seconds rate;
  - false-lock rate;
  - unresolved rate (never locked);
  - false-jump rate;
  - shortlist recall;
  - time to first lock.
- **Shared openings:** for passages whose opening is genuinely shared, lock delay is also measured from the point where the audio becomes distinguishable.
- **Configurations:** run phoneme-only, plus posterior reranking if the release configuration includes it.
- **Performance:** a fixed replay records processing time per second of audio against the real-time budget for the device floor.

**Blocked by:** 09; also 05 if the release configuration includes reranking.

**Status:** ready-for-agent

- [ ] The benchmark runs end to end on both datasets and writes a versioned report.
- [ ] False locks and unresolved cases are reported alongside the 5-second rate.
- [ ] The distinguishable-point lock delay is reported for shared-opening cases.
- [ ] The performance gate fails when replay falls below real time at the budgeted speed factor.
- [ ] The report states which configuration (with or without reranking) was measured.
