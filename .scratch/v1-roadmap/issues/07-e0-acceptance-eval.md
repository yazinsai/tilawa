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
- **Declared-passage mode in core, 2026-10-07** (`docs/benchmarks/acceptance-v2.md` + `.json`; same evidence and labels, no Modal). `CheckerOptions.passage` aligns against the declared passage. The tracker lock is used only to follow a deliberate jump: a lock of 4+ words more than an ayah off the passage whose words don't also fit it (cost per phoneme ≥ 0.35). Without the fit check, a substitution forming another ayah tripped it and recall fell to 56%. At session end, a declared session confirms a suspected wrong or added word once a later word was recited right. Free recitation is unchanged, and v2 reproduces v1 exactly.
  - Free: precision 55.2%, recall 50.0% (32 / 26 / 32), 8 clean false warnings on 225 clips.
  - Declared: precision 62.6%, recall 89.1% (57 / 34 / 7); wrong_word 24/29, omitted_word 33/35. 13 clean false warnings (0.058 per clip, 1.40 per 5 min, short clips). 0 FP on repeats. Recall passes the bar; precision fails it.
  - Finalisation is scoped to declared sessions. In free recitation it gave +2 hits and +5 FP, and confirmed two decoder-noise errors on a clean tlog fixture. In the declared arm, turning it off gives 43 / 30 / 21.
  - Still open: precision. Jump detection is checked only by a synthetic test; the help clips contain no jumps. 09's session API still has to pass `passage` through.
- **Checker precision fixes, 2026-10-07** (`docs/benchmarks/acceptance-v3.md` + `.json`; one scoreboard row per fix). Recall is unchanged in both arms.
  - Skip placement: when a word recurs across a skip, the alignment had two equally good placements for the gap. An-Nas with 114:2 skipped gave the 9 false omitted_word events seen with oracle lock and oracle evidence, and Al-Kafirun had the same tie the other way. The checker now takes the placement that starts on an ayah's first word.
  - A word with no consonants (ghunna or madd only) needs a hard edit to be a wrong word.
  - An added word between two words needs two consonants, as inside a word.
  - Oracle: 86.8% / 92.2% → 100% / 92.2%. Free: 55.2% / 50.0% → 76.2% / 50.0%, 8 → 2 clean warnings. Declared: 62.6% / 89.1% → 77.0% / 89.1% (57 / 17 / 7), 13 → 6 clean warnings.
  - The 17 declared false positives that remain: about 11 decoder noise (ب decoded as م or ض, sandhi), 4 at a clip edge or next to a planted substitution (possibly incomplete labels), and 2 skip placements a word off where the ayah's first word was restarted. Not judging the passage's first word would give 80.3% precision at the same recall, but it would hide a wrong first word, so it isn't applied. The rest of the gap to 90% is the listener.

**Agent, 2026-10-07: free recitation latches onto its first lock.** Branch `ticket-10-device`, commit db72fff.

- Change: in free recitation the checker takes the tracker's first lock as a declared passage, grown forward as the reciter goes on (across ayah and surah boundaries) and trailing one ayah behind the anchor. It re-anchors only on the declared arm's jump rule (a lock of 4+ words more than an ayah off the passage whose words don't also fit it), or when the tracker loses its lock. The tracker's lock events are unchanged. Confirmation is unchanged too: a free error still needs the tracker locked, and there is no end-of-session finalisation.
- Help clips (teacher, 0.32 s chunks), before → after: free 33 / 10 / 31 (hits / FP / misses; 76.7% / 51.6%) → identical; declared 57 / 17 / 7 → identical; clean false warnings 2 → 2. QUD tracker regression unchanged (76.9% within 5 s, 8/91 false jumps; tracker untouched).
- Why no recall gain: of the 24 misses the free arm makes and the declared arm doesn't, 21 have the **first** lock on the other ayah (26:124 with its fifth word swapped is 26:142; 12:2 ↔ 43:3; 7:111 ↔ 26:36; 26:106 → 26:124/177). Latching keeps that wrong place. On a single-ayah clip the audio is a correct recitation of the other ayah, so free recitation can't tell. 2 never lock (1:5), and 1 has its error in the lock span's first words (80:26:2). None is a correct first lock lost to a later relock, which is the case latching fixes (unit test `free recitation > checks a wrong word that sends the tracker to another ayah`).
- Stitched sessions (ticket 13, 747 min, real jumps): new metric, false confirmed wrong/added words from a free checker (omissions and skips excluded: contiguous runs drop up to an ayah and spans trim edge words). 305 (2.04 per 5 min) → 309 (2.07). Planted-jump sessions 47 → 48. So jumps are still followed with no flood of errors after them.
- What would move the free arm: context before the first lock. In an app session the previous ayah or the last position is usually known. Passing it as a hint (or carrying the latch across sessions) turns these 21 into declared-arm cases, and the declared arm catches all 21.
