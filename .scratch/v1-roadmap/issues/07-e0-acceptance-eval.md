# 07: E0 and acceptance eval on scripted recordings

**What to build:** The v1 acceptance harness on held-out acted-error recordings, plus E0: which component (listener, tracker or checker) loses each planted mistake.

**Blocked by:** 04, 12. Ticket 05 adds classes and long sessions later; each re-run is a new report version.

**Status:** needs-triage

- [ ] Phase 1 (help clips from 12): memorization classes, short vowels, and false warnings on the 464 clean clips. Score repeats as non-errors.
- [ ] Phase 2 (05 recordings): letter pairs, missing shadda, self-correction, and false warnings per 5 minutes on long clean scripts.
- [ ] Run the teacher, tracker and checker on held-out speakers only, scored against the planted labels.
- [ ] Memorization: precision ≥ 90%, recall ≥ 80%, and ≤ 1 false warning per 5 minutes on clean audio. Phase 1 reports false warnings per clean clip and per 5 minutes, but only phase 2's long sessions can pass the per-5-minutes bar.
- [ ] Each pronunciation class: precision ≥ 80% on ≥ 50 positives from ≥ 15 sites, or it's marked `unsupported`.
- [ ] Re-run `tools/e1_report.py` on the same clips (real acted errors), next to E1 and E1b (ticket 11).
- [ ] Tracker gate on this audio plus ambiguous tlog clips: correct ayah within 5 s in ≥ 95% of starts, ≤ 2% false jumps. QUD stays a regression report.
- [ ] E0 attribution for each missed error: not visible in the evidence, lost by the tracker, or rejected by the checker.
- [ ] Versioned report under `docs/benchmarks/`.
