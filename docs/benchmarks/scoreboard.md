# Engine scoreboard

One row per commit × listener, recomputed from evidence on disk (no GPU). Bars are PLAN.md's v1 acceptance bars; `unmeasurable` means the data to score it doesn't exist yet, `pending` means the evidence exists in principle but hasn't been produced for this listener.

**How to run / add a row.** From `packages/core`: `HELP_DIR=<repo>/.scratch/help QUD_DIR=<repo>/.scratch/bench/qud pnpm scoreboard --note '<what changed and why>'` (teacher). The note is required by AGENTS.md. Tag changed engine modules with `--modules checker` (or a comma-separated list such as `--modules listener,tracker`). Tags describe the implementation change, not every metric that moved; the dashboard uses them to filter the change history. Existing tags survive reruns that omit this option. For a student, decode the help eval clips with it into a directory of `<clip>.evidence.json` and run `pnpm scoreboard --listener <run-name> --evidence <dir> --note '...'`; without `--evidence` its help metrics are `pending`. Each run replaces the row with the same commit and listener, else appends one; commit the updated `scoreboard.json` and `scoreboard.md` with the change they measure. Listener PER comes from `student-v1.json`, E1/E1b from their reports; rerun those first if the listener changed. Help clips: 388 scored eval clips, 29 held-out speakers; letter-confusion substitutions count as negatives. `Declared passage` gives the checker the prompted passage (`CheckerOptions.passage`), the memorization-session arm; `oracle evidence` replaces the decode with a perfect one synthesised from the label, so its losses are the checker's own.

**Dashboard.** `uv run tools/scoreboard_serve.py` serves the dashboard on http://127.0.0.1:8765/scoreboard.html. Choose a listener model to see end-to-end precision and recall, listener/tracker/checker summaries and trends, and missed-error attribution. The change history shows module tags, key figures and deltas, with filters for Listener, Tracker and Checker. Full notes are expandable; comparisons always use the previous run for that model, even when filtered. Expand “Explore all measurements” for sample counts, targets and backing reports. It reads `scoreboard.json`, so a new row appears on reload.

## Latest: teacher @ 16d6396 (2026-10-07)

Delta is against the baseline row 3e91183 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 76.2% (32/42) | ≥ 90.0% | fail | +21.0 pp | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 50.0% (32/64) | ≥ 80.0% | fail | 0 |  |
| End to end | wrong_word precision | 14.3% (1/7) | ≥ 90.0% | fail | +4.3 pp |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail | 0 |  |
| End to end | omitted_word precision | 96.9% (31/32) | ≥ 90.0% | pass | +21.3 pp |  |
| End to end | omitted_word recall | 88.6% (31/35) | ≥ 80.0% | pass | 0 |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.009 (2/225) |  | info | -0.027 |  |
| End to end | False warnings per 5 min (short clips) | 0.215 | ≤ 1 | pass | -0.645 | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | 0 | perf.ts gate: device floor assumed 20× slower; slowest push 28.0 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Fair PER (canon vs spliced reference) | 2.4% |  | info | 0 | teacher floor; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 8.5% |  | info | 0 | mostly clip edges and madd/ghunna length, not mishearing (teacher-per-breakdown-v1) |
| Listener | Raw PER vs teacher | 0.0% |  | info | 0 |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info | 0 | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | 63.3% (19/30) | ≥ 50.0% | pass | 0 | E1 viable rule; 27 sites; report 66f60a6 |
| Listener | E1 help clips: short vowel canonicalised | 10.0% (3/30) |  | info | 0 |  |
| Listener | E1 help clips: short vowel clean greedy FP | 0.1% (8/6755) | ≤ 2.0% | pass | 0 |  |
| Listener | E1 (qdat_bench + Ikhlas): classes the teacher hears | 1/10: 1 posterior-only, 9 unsupported |  | info | 0 | report 7b28629; most classes have < 20 positives |
| Listener | E1b splices (upper bound): classes the teacher hears | 7/9: 6 viable, 1 posterior-only, 2 mixed |  | info | 0 | report 005fc6d |
| Listener | E1b splices: canonicalised share (all classes) | 34.5% (916/2655) |  | info | 0 |  |
| Tracker | help clean: correct ayah within 5 s | 84.9% (191/225) | ≥ 95.0% | fail | 0 | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass | 0 |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info | 0 |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info | 0 | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 79.4% (308/388) | ≥ 95.0% | fail | 0 | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass | 0 |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info | 0 |  |
| Tracker | help all: median time to correct lock | 2.62 s |  | info | 0 | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | 76.9% (30/39) | ≥ 95.0% | fail | 0 | 16 recordings, 357 min; regression report, not a gate (Decision 2) |
| Tracker | QUD regression: false jumps | 8.8% (8/91) | ≤ 2.0% | fail | 0 |  |
| Tracker | QUD regression: median time to first lock | 4.11 s |  | info | 0 |  |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 77.0% (57/74) | ≥ 90.0% | fail | +15.6 pp | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 89.1% (57/64) | ≥ 80.0% | pass | +21.9 pp |  |
| Checker | Declared passage: wrong_word precision | 66.7% (24/36) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: omitted_word precision | 94.3% (33/35) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 94.3% (33/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.027 (6/225) |  | info | -0.009 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.645 | ≤ 1 | pass | -0.215 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass | +19.1 pp | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass | +32.8 pp |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass | +72.4 pp |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass | +20.5 pp |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info | 0 |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info | -1.6 pp | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/32 |  | info | 0 |  |
| Checker | E0: misses lost to tracker | 25/32 |  | info | +14 |  |
| Checker | E0: misses lost to checker | 2/32 |  | info | -14 | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: e2-libri-full @ 3e91183 (2026-10-07)

Delta is against the baseline row 3e91183 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| End to end | Memorization recall | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| End to end | False warnings per 5 min | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 29.9 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Fair PER (canon vs spliced reference) | 5.4% | ≤ 3.4% | fail |  | bar: teacher floor 2.44% + 1 pp (proposed); canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 9.9% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 5.5% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Tracker | help clean: correct ayah within 5 s | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| Tracker | help all: correct ayah within 5 s | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from e2-libri-full |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Oracle lock: recall | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| Checker | E0 attribution | — |  | pending |  | no help-clip evidence from e2-libri-full on disk (needs a GPU decode of the 388 eval clips) |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## History

| Date | Commit | Listener | End to end: Memorization precision | End to end: Memorization recall | End to end: False warnings per 5 min (short clips) | Listener: Fair PER (canon vs spliced reference) | Tracker: help clean: correct ayah within 5 s | Tracker: QUD regression: correct ayah within 5 s | Tracker: QUD regression: false jumps | Checker: Oracle lock: Memorization recall | Checker: Oracle lock + evidence: Memorization recall | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-10-07 | 3e91183 | teacher | 55.2% (32/58) | 50.0% (32/64) | 0.861 | 2.4% | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 67.2% (43/64) | 59.4% (38/64) | Baseline: engine at 3e91183, offline Muaalem teacher evidence; first scoreboard row |
| 2026-10-07 | 3e91183 | e2-libri-full | — | — | — | 5.4% | — | — | — | — | — | Baseline student: e2-libri-full (E2 LibriSpeech init); PER only, help-clip evidence not decoded yet |
| 2026-10-07 | 8aca172 | teacher | 55.2% (32/58) | 50.0% (32/64) | 0.861 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Declared-passage mode in core (CheckerOptions.passage) + end-of-session finalisation in declared sessions; declared arm 62.6%/89.1%, free unchanged |
| 2026-10-07 | 004986d | teacher | 66.7% (32/48) | 50.0% (32/64) | 0.861 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Skip placement: prefer the equal placement on an ayah start when a word recurs across the skip; oracle false omitted_word 9 -> 0 |
| 2026-10-07 | 46a068e | teacher | 71.1% (32/45) | 50.0% (32/64) | 0.538 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | A word with no consonants (ghunna/madd only) needs a hard edit to be a wrong word |
| 2026-10-07 | 16d6396 | teacher | 76.2% (32/42) | 50.0% (32/64) | 0.215 | 2.4% | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Added word between words needs two consonants, as inside a word; declared 77.0%/89.1%, free 76.2%/50.0%, oracle 100%/92.2% |
