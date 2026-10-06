# 07: E0 and acceptance eval on scripted recordings

**What to build:** The v1 acceptance harness on held-out acted-error recordings, plus E0: which component (listener, tracker or checker) loses each planted mistake.

**Blocked by:** 04, 12. Ticket 05 adds classes and long sessions later; each re-run is a new report version.

**Status:** needs-triage

- [x] Phase 1 (help clips from 12): memorization classes, short vowels, and false warnings on the 464 clean clips. Score repeats as non-errors.
- [ ] Phase 2 (05 recordings): letter pairs, missing shadda, self-correction, and false warnings per 5 minutes on long clean scripts.
- [x] Run the teacher, tracker and checker on held-out speakers only, scored against the planted labels.
- [ ] Memorization: precision ≥ 90%, recall ≥ 80%, and ≤ 1 false warning per 5 minutes on clean audio. Phase 1 reports false warnings per clean clip and per 5 minutes, but only phase 2's long sessions can pass the per-5-minutes bar.
- [ ] Each pronunciation class: precision ≥ 80% on ≥ 50 positives from ≥ 15 sites, or it's marked `unsupported`.
- [x] Re-run `tools/e1_report.py` on the same clips (real acted errors), next to E1 and E1b (ticket 11).
- [ ] Tracker gate on this audio plus ambiguous tlog clips: correct ayah within 5 s in ≥ 95% of starts, ≤ 2% false jumps. QUD stays a regression report.
- [x] E0 attribution for each missed error: not visible in the evidence, lost by the tracker, or rejected by the checker.
- [x] Versioned report under `docs/benchmarks/`.

## To explore: declared-passage mode

On the provisional help-clip run (`docs/benchmarks/help-memorization-v1`), wrong_word recall is 1/29. Most misses are a substitution that forms another valid ayah: the tracker follows it, and the checker (Decision 3: it consumes the tracker's lock) sees no error. Passing the intended ayah as the tracker `prior` (`help-cli.ts --prior`) doesn't change this, because `MAX_PRIOR_BONUS` is capped below `minSeparation` on purpose.

Idea: when the session declares the passage (memorization), the checker aligns against that passage. The tracker lock is used only without a declared passage, or to detect a deliberate jump (several consecutive wrong words that fit elsewhere). This would amend Decision 3, and it needs the session API (09) to accept a declared passage. Free recitation keeps the ambiguity, so report it separately. The help clips already carry their prompted passage, so the idea can be tested on them directly.

## Comments

- **Phase 1 done, 2026-10-06** (`docs/benchmarks/acceptance-v1.md` + `.json`, E1 part in `acceptance-v1-e1.json`). Final ticket-12 labels, eval split only: 388 scored clips, 29 speakers. Run: `uv run --python cpython-3.12-macos-aarch64-none tools/e1_report.py --help-clips`, then `pnpm --filter @tilawa/core exec tsx src/bench/acceptance-cli.ts` (both read `HELP_DIR`). Existing teacher evidence was reused; no Modal run was needed.
  - Memorization, free recitation (default): precision 55.2%, recall 50.0% (32 / 26 FP / 32 missed). wrong_word recall 1/29, omitted_word 31/35. Clean audio: 8 false warnings on 225 clips (0.036 per clip, 0.86 per 5 min, short clips). No false positives on the 23 repeat clips. Fails both bars.
  - Declared-passage arm (prototype, scoring-side only; core unchanged): the checker is anchored once to the prompted passage and ignores the lock. Precision 61.4%, recall 67.2%; wrong_word recall 10/29. Clean false warnings are unchanged (8).
  - E0, free arm, 32 misses: wrong_word: 5 evidence, 9 tracker, 14 checker. omitted_word: 0 / 2 / 2. **All 14 wrong_word checker misses end `uncertain` at clip end**: the clip ends before the 2.5 s confirm delay or the next word. So on these clips the checker's loss is the confirmation rule, not the alignment. Short clips understate recall; phase 2's long scripts won't have this edge.
  - short_vowel: 30 positives at 27 sites. The checker has no pronunciation classes, so it's `unsupported` as feedback. Teacher E1 on the same clips: greedy 19/30, posterior 28/30, greedy FP 0.1%, so it's `viable` at the teacher level. ص/س has 4 planted sites (7 positive phonemes, since الصمد is geminate), so it's unsupported.
  - Tracker gate on help audio (proxy: start = first decoded phoneme): correct ayah within 5 s on 191/225 clean clips (84.9%), 308/388 all. 0 false jumps; 21 error clips locked outside the passage first, all of them substitutions that form another ayah. Fails 95%.
  - **Not done, blocked:** the tracker gate on ambiguous tlog clips. tlog-v2 `ambiguous` has no passage truth (`label: null`), and the bucket review's 30 ambiguous samples are wrong-passage or no-recitation. It needs labelled ambiguous clips first. Memorization and pronunciation bars stay unticked: measured, not met.
  - Declared-passage follow-up: the prototype never re-anchors, so a deliberate jump isn't handled. Amending Decision 3 still needs 09's session API.

