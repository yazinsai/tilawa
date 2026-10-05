# 07: E0 and acceptance eval on scripted recordings

**What to build:** The v1 acceptance harness on held-out scripted recordings, plus E0: which component (listener, tracker or checker) loses each planted mistake.

**Blocked by:** 04, 05.

**Status:** needs-triage

- [ ] Run the teacher, tracker and checker on held-out speakers and score against the planted labels.
- [ ] Memorization: precision ≥ 90%, recall ≥ 80%, and ≤ 1 false warning per 5 minutes on clean scripts.
- [ ] Each pronunciation class: precision ≥ 80%, or it's marked `unsupported`.
- [ ] Tracker gate on this audio plus ambiguous tlog clips: correct ayah within 5 s in ≥ 95% of starts, ≤ 2% false jumps. QUD stays a regression report.
- [ ] E0 attribution for each missed error: not visible in the evidence, lost by the tracker, or rejected by the checker.
- [ ] Versioned report under `docs/benchmarks/`.
