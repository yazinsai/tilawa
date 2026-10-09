# Engine scoreboard

One row per commit × listener, recomputed from evidence on disk (no GPU). Bars are PLAN.md's v1 acceptance bars; `unmeasurable` means the data to score it doesn't exist yet, `pending` means the evidence exists in principle but hasn't been produced for this listener.

**How to run / add a row.** From `packages/core`: `HELP_DIR=<repo>/.scratch/help QUD_DIR=<repo>/.scratch/bench/qud pnpm scoreboard --note '<what changed and why>'` (teacher). The note is required by AGENTS.md. Tag changed engine modules with `--modules checker` (or a comma-separated list such as `--modules listener,tracker`). Tags describe the implementation change, not every metric that moved; the dashboard uses them to filter the change history. Existing tags survive reruns that omit this option. For a student, decode the help eval clips with it into a directory of `<clip>.evidence.json` and run `pnpm scoreboard --listener <run-name> --evidence <dir> --note '...'`; without `--evidence` its help metrics are `pending`. Each run replaces the row with the same commit and listener, else appends one; commit the updated `scoreboard.json` and `scoreboard.md` with the change they measure. Listener PER is gold PER from `gold-per-v1-scores.json` (`tools/gold_set.py score`), gated by the bar in `listener-target-v1.json`; raw and clean-match fair PER come from `student-v1.json` and the PER breakdown. E1/E1b from their reports; rerun those first if the listener changed. Help clips: 388 scored eval clips, 29 held-out speakers; letter-confusion substitutions count as negatives. `Declared passage` gives the checker the prompted passage (`CheckerOptions.passage`), the memorization-session arm; `oracle evidence` replaces the decode with a perfect one synthesised from the label, so its losses are the checker's own.

**Dashboard.** `uv run tools/scoreboard_serve.py` serves the dashboard on http://127.0.0.1:8765/scoreboard.html. Choose a listener model to see end-to-end precision and recall, listener/tracker/checker summaries and trends, and missed-error attribution. The change history shows module tags, key figures and deltas, with filters for Listener, Tracker and Checker. Full notes are expandable; comparisons always use the previous run for that model, even when filtered. Expand “Explore all measurements” for sample counts, targets and backing reports. It reads `scoreboard.json`, so a new row appears on reload.

## Latest: teacher @ adbcafe (2026-10-09)

Delta is against the baseline row 3e91183 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 84.6% (33/39) | ≥ 90.0% | fail | +29.4 pp | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 51.6% (33/64) | ≥ 80.0% | fail | +1.6 pp |  |
| End to end | wrong_word precision | 50.0% (2/4) | ≥ 90.0% | fail | +40.0 pp |  |
| End to end | wrong_word recall | 6.9% (2/29) | ≥ 80.0% | fail | +3.4 pp |  |
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
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | 0 | perf.ts gate: device floor assumed 20× slower; slowest push 29.3 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | pass --latency <pnpm bench:help report> (packages/sdk); device runs: ticket 10 |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 3.39% (233 clips) | ≤ 3.7% | pass |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 7.61% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 9.12% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 1.77% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 5.45% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 11.43% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 5.11% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 3.20% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 5.2% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 7.27% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | 2.4% |  | info | 0 | teacher floor; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
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
| End to end | Stitched sessions: false wrong/added words per 5 min (free) | 2.07 (309 in 747 min) | ≤ 1 | fail |  | clean clips stitched end to end with real jumps (ticket 13); `pnpm jumps` |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 83.8% (57/68) | ≥ 90.0% | fail | +22.4 pp | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 89.1% (57/64) | ≥ 80.0% | pass | +21.9 pp |  |
| Checker | Declared passage: wrong_word precision | 80.0% (24/30) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: omitted_word precision | 94.3% (33/35) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 94.3% (33/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.022 (5/225) |  | info | -0.013 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.538 | ≤ 1 | pass | -0.323 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass | +19.1 pp | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass | +34.4 pp |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass | +75.9 pp |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass | +20.5 pp |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info | 0 |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info | 0 | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/31 |  | info | 0 |  |
| Checker | E0: misses lost to tracker | 24/31 |  | info | +13 |  |
| Checker | E0: misses lost to checker | 2/31 |  | info | -14 | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: e2-libri-full @ bfc52cb (2026-10-07)

Delta is against the baseline row 3e91183 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 73.2% (30/41) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 46.9% (30/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 0.0% (0/7) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 0.0% (0/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 96.8% (30/31) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 85.7% (30/35) | ≥ 80.0% | pass |  |  |
| End to end | added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.031 (7/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 0.757 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | +0 | perf.ts gate: device floor assumed 20× slower; slowest push 29.1 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 5.98% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 12.18% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 11.63% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 4.17% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 10.91% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 15.71% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 9.15% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.91% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 7.7% |  | info |  |  |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | 5.4% |  | info | 0 | secondary to gold PER: teacher-labelled clips; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 9.9% |  | info | 0 | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 5.5% |  | info | 0 |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on e2-libri-full's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.4% (190/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.4% (1/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.52 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 79.1% (307/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.54 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from e2-libri-full |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 67.1% (53/79) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 82.8% (53/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 51.2% (21/41) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 72.4% (21/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 91.4% (32/35) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.080 (18/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 1.946 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 8/34 |  | info |  |  |
| Checker | E0: misses lost to tracker | 23/34 |  | info |  |  |
| Checker | E0: misses lost to checker | 3/34 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: e5-top1-lite @ bfc52cb (2026-10-07)

Delta is against the baseline row bfc52cb (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 53.8% (28/52) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 43.8% (28/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 0.0% (0/16) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 0.0% (0/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 84.8% (28/33) | ≥ 90.0% | fail |  |  |
| End to end | omitted_word recall | 80.0% (28/35) | ≥ 80.0% | pass |  |  |
| End to end | added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.089 (20/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 2.163 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 29.6 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 7.21% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 14.84% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 12.63% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 5.31% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 10.39% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 25.00% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 11.98% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 6.12% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 9.1% |  | info |  |  |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | 11.8% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 7.7% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on e5-top1-lite's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on e5-top1-lite's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on e5-top1-lite's posteriors |
| Tracker | help clean: correct ayah within 5 s | 83.6% (188/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.9% (2/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.50 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.6% (305/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.52 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from e5-top1-lite |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 50.9% (54/106) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 84.4% (54/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 35.4% (23/65) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 79.3% (23/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 83.8% (31/37) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: omitted_word recall | 88.6% (31/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.169 (38/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 4.109 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 7/36 |  | info |  |  |
| Checker | E0: misses lost to tracker | 26/36 |  | info |  |  |
| Checker | E0: misses lost to checker | 3/36 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: e5-soft5-lite @ bfc52cb (2026-10-07)

Delta is against the baseline row bfc52cb (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 65.9% (29/44) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 45.3% (29/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 10.0% (1/10) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 93.3% (28/30) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 80.0% (28/35) | ≥ 80.0% | pass |  |  |
| End to end | added_word precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.053 (12/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 1.298 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 29.1 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 5.94% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 12.18% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 10.73% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 4.31% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 9.61% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 20.71% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 8.88% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.98% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 7.9% |  | info |  |  |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | 10.4% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 6.2% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on e5-soft5-lite's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on e5-soft5-lite's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on e5-soft5-lite's posteriors |
| Tracker | help clean: correct ayah within 5 s | 83.6% (188/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.60 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 76.5% (297/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.2% (20/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.52 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from e5-soft5-lite |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 62.8% (54/86) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 84.4% (54/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 46.8% (22/47) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 75.9% (22/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 91.4% (32/35) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.107 (24/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 2.595 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/35 |  | info |  |  |
| Checker | E0: misses lost to tracker | 26/35 |  | info |  |  |
| Checker | E0: misses lost to checker | 4/35 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: e2-scratch-full @ bfc52cb (2026-10-07)

Delta is against the baseline row bfc52cb (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 43.9% (29/66) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 45.3% (29/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 0.0% (0/27) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 0.0% (0/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 82.9% (29/35) | ≥ 90.0% | fail |  |  |
| End to end | omitted_word recall | 82.9% (29/35) | ≥ 80.0% | pass |  |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.142 (32/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 3.46 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 29.5 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 7.30% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 13.61% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 12.58% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 5.57% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 11.95% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 28.57% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 12.25% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 5.81% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 9.1% |  | info |  |  |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | 12.1% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 7.9% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on e2-scratch-full's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on e2-scratch-full's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on e2-scratch-full's posteriors |
| Tracker | help clean: correct ayah within 5 s | 83.1% (187/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 76.5% (297/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from e2-scratch-full |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 44.7% (55/123) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 85.9% (55/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 28.6% (22/77) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 75.9% (22/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 82.5% (33/40) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: omitted_word recall | 94.3% (33/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.240 (54/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 5.839 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/35 |  | info |  |  |
| Checker | E0: misses lost to tracker | 26/35 |  | info |  |  |
| Checker | E0: misses lost to checker | 4/35 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: combo-libri-soft5-full @ 2d94ea0 (2026-10-07)

Delta is against the baseline row 72d1f04 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 83.8% (31/37) | ≥ 90.0% | fail | 0 | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 48.4% (31/64) | ≥ 80.0% | fail | 0 |  |
| End to end | wrong_word precision | 33.3% (1/3) | ≥ 90.0% | fail | 0 |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail | 0 |  |
| End to end | omitted_word precision | 100.0% (30/30) | ≥ 90.0% | pass | 0 |  |
| End to end | omitted_word recall | 85.7% (30/35) | ≥ 80.0% | pass | 0 |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail | 0 |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.009 (2/225) |  | info | 0 |  |
| End to end | False warnings per 5 min (short clips) | 0.216 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | 0 | perf.ts gate: device floor assumed 20× slower; slowest push 28.4 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.82% (233 clips) | ≤ 3.7% | fail | 0 | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 11.61% (16 clips) |  | info | 0 |  |
| Listener | Gold PER: phone | 10.40% (26 clips) |  | info | 0 |  |
| Listener | Gold PER: tlog | 2.97% (191 clips) |  | info | 0 |  |
| Listener | Gold PER: pharyngeals | 8.05% (385 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: emphatics | 15.00% (140 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: hamza ain | 7.00% (743 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: short vowels | 4.31% (5340 ref phonemes) |  | info | 0 |  |
| Listener | Gold raw PER vs spliced reference | 6.7% |  | info | 0 |  |
| Listener | Gold PER: never in tlog train | 9.19% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | 4.0% |  | info |  | secondary to gold PER: teacher-labelled clips; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 8.7% |  | info | +0.0 pp | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 4.3% |  | info | -0.0 pp |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info | 0 | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1b splices: canonicalised share (all classes) | 34.2% (901/2634) |  | info |  | report 15f6443 |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on combo-libri-soft5-full's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on combo-libri-soft5-full's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on combo-libri-soft5-full's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.4% (190/225) | ≥ 95.0% | fail | 0 | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass | 0 |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info | 0 |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info | 0 | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 79.4% (308/388) | ≥ 95.0% | fail | 0 | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass | 0 |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info | 0 |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info | 0 | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from combo-libri-soft5-full |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 81.8% (54/66) | ≥ 90.0% | fail | 0 | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 84.4% (54/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: wrong_word precision | 74.2% (23/31) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: wrong_word recall | 79.3% (23/29) | ≥ 80.0% | fail | 0 |  |
| Checker | Declared passage: omitted_word precision | 96.9% (31/32) | ≥ 90.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word recall | 88.6% (31/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.022 (5/225) |  | info | 0 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.541 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass | 0 | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info | 0 |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info | 0 | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/33 |  | info | 0 |  |
| Checker | E0: misses lost to tracker | 24/33 |  | info | 0 |  |
| Checker | E0: misses lost to checker | 4/33 |  | info | 0 | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: combo-clean @ 2d94ea0 (2026-10-07)

Delta is against the baseline row 2d94ea0 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 75.7% (28/37) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 43.8% (28/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 20.0% (1/5) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 100.0% (27/27) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 77.1% (27/35) | ≥ 80.0% | fail |  |  |
| End to end | added_word precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.022 (5/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 0.541 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 27.8 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.97% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 11.80% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 10.49% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.13% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 7.79% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 17.14% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 7.40% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.40% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 6.7% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 9.45% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | 4.1% |  | info |  | secondary to gold PER: teacher-labelled clips; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 8.7% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 4.4% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 65.6% (42/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1b splices: canonicalised share (all classes) | 34.0% (899/2644) |  | info |  | report 2d94ea0 |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on combo-clean's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on combo-clean's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on combo-clean's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.4% (190/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.6% (305/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.52 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from combo-clean |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 75.0% (51/68) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 79.7% (51/64) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: wrong_word precision | 65.7% (23/35) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 79.3% (23/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 96.6% (28/29) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 80.0% (28/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.044 (10/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 1.081 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 7/36 |  | info |  |  |
| Checker | E0: misses lost to tracker | 24/36 |  | info |  |  |
| Checker | E0: misses lost to checker | 5/36 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean @ 2d94ea0 (2026-10-07)

Delta is against the baseline row 2d94ea0 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 81.8% (27/33) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 42.2% (27/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 25.0% (1/4) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 100.0% (26/26) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 74.3% (26/35) | ≥ 80.0% | fail |  |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.013 (3/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 0.324 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 28.2 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | needs word onsets and device runs (E7, ticket 10) |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.71% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 10.66% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 9.97% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.02% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 5.45% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 8.57% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 5.92% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.16% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 7.4% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 8.80% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | 3.9% |  | info |  | secondary to gold PER: teacher-labelled clips; canon PER vs spliced reference (teacher-per-breakdown-v1): nasal spelling and qalqala unified, runs collapsed, clip edges from the teacher; ignores shadda and duration |
| Listener | Raw PER vs reference (tlog clean-match test) | 9.4% |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | 5.6% |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1b splices: canonicalised share (all classes) | 41.7% (1102/2640) |  | info |  | report 2d94ea0 |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.66 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.1% (303/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 83.6% (56/67) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 77.4% (24/31) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: omitted_word precision | 97.0% (32/33) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.022 (5/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.541 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (59/59) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 92.2% (59/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (24/24) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 92.2% (59/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 6/37 |  | info |  |  |
| Checker | E0: misses lost to tracker | 29/37 |  | info |  |  |
| Checker | E0: misses lost to checker | 2/37 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean-onnx @ 2bdd2be (2026-10-07)

Delta is against the baseline row 457e2f0 (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 80.0% (28/35) | ≥ 90.0% | fail | -1.8 pp | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 43.8% (28/64) | ≥ 80.0% | fail | +1.6 pp |  |
| End to end | wrong_word precision | 33.3% (2/6) | ≥ 90.0% | fail | +8.3 pp |  |
| End to end | wrong_word recall | 6.9% (2/29) | ≥ 80.0% | fail | +3.4 pp |  |
| End to end | omitted_word precision | 100.0% (26/26) | ≥ 90.0% | pass | 0 |  |
| End to end | omitted_word recall | 74.3% (26/35) | ≥ 80.0% | fail | 0 |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.013 (3/225) |  | info | 0 |  |
| End to end | False warnings per 5 min (short clips) | 0.324 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | +0 | perf.ts gate: device floor assumed 20× slower; slowest push 28.5 ms |
| End to end | Word highlight latency p95 | 495 ms (p50 338 ms, 4706 words) | ≤ 600 | pass |  | onnxruntime-node, 20 ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10 |
| End to end | Suspected latency p95 | 1018 ms (p50 307 ms, 132 errors) | ≤ 1000 | fail |  | from the onset of the word after the error; PLAN: within about 1 s |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.80% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 10.66% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 9.92% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.14% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 5.45% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 8.57% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 6.33% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.21% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 7.4% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 8.80% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | — |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | — |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info | 0 | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean-onnx's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-onnx's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-onnx's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail | 0 | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass | 0 |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info | 0 |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info | -0.02 s | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.1% (303/388) | ≥ 95.0% | fail | 0 | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass | 0 |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info | 0 |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info | 0 | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean-onnx |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 83.6% (56/67) | ≥ 90.0% | fail | 0 | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: wrong_word precision | 77.4% (24/31) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word precision | 97.0% (32/33) | ≥ 90.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.022 (5/225) |  | info | 0 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.541 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass | 0 | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass | +1.6 pp |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass | +3.4 pp |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info | 0 |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info | +1.6 pp | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 6/36 |  | info | 0 |  |
| Checker | E0: misses lost to tracker | 28/36 |  | info | -1 |  |
| Checker | E0: misses lost to checker | 2/36 |  | info | 0 | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean-int8 @ adbcafe (2026-10-09)

Delta is against the baseline row 2bdd2be (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 87.5% (28/32) | ≥ 90.0% | fail | +9.7 pp | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 43.8% (28/64) | ≥ 80.0% | fail | 0 |  |
| End to end | wrong_word precision | 50.0% (2/4) | ≥ 90.0% | fail | +21.4 pp |  |
| End to end | wrong_word recall | 6.9% (2/29) | ≥ 80.0% | fail | 0 |  |
| End to end | omitted_word precision | 100.0% (26/26) | ≥ 90.0% | pass | 0 |  |
| End to end | omitted_word recall | 74.3% (26/35) | ≥ 80.0% | fail | 0 |  |
| End to end | added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.009 (2/225) |  | info | -0.009 |  |
| End to end | False warnings per 5 min (short clips) | 0.216 | ≤ 1 | pass | -0.216 | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | 0 | perf.ts gate: device floor assumed 20× slower; slowest push 28.5 ms |
| End to end | Word highlight latency p95 | 490 ms (p50 332 ms, 4722 words) | ≤ 600 | pass | -0.353 | onnxruntime-node, 20 ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10 |
| End to end | Suspected latency p95 | 1013 ms (p50 294 ms, 136 errors) | ≤ 1000 | fail | -0.002 | from the onset of the word after the error; PLAN: within about 1 s |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.82% (233 clips) | ≤ 3.7% | fail | 0 | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 10.66% (16 clips) |  | info | 0 |  |
| Listener | Gold PER: phone | 10.07% (26 clips) |  | info | 0 |  |
| Listener | Gold PER: tlog | 3.14% (191 clips) |  | info | 0 |  |
| Listener | Gold PER: pharyngeals | 5.97% (385 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: emphatics | 8.57% (140 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: hamza ain | 6.59% (743 ref phonemes) |  | info | 0 |  |
| Listener | Gold PER: short vowels | 4.23% (5340 ref phonemes) |  | info | 0 |  |
| Listener | Gold raw PER vs spliced reference | 7.5% |  | info | 0 |  |
| Listener | Gold PER: never in tlog train | 8.88% (62 clips) |  | info | 0 | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | — |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | — |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info | 0 | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean-int8's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-int8's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-int8's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail | 0 | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass | 0 |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info | 0 |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info | 0 | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.1% (303/388) | ≥ 95.0% | fail | 0 | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass | 0 |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info | 0 |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info | 0 | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean-int8 |
| End to end | Stitched sessions: false wrong/added words per 5 min (free) | — |  | pending |  | teacher evidence only |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 87.5% (56/64) | ≥ 90.0% | fail | +5.1 pp | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: wrong_word precision | 82.8% (24/29) | ≥ 90.0% | fail | +7.8 pp |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word precision | 97.0% (32/33) | ≥ 90.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.018 (4/225) |  | info | -0.009 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.433 | ≤ 1 | pass | -0.216 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass | 0 | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info | 0 |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info | 0 | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 6/36 |  | info | 0 |  |
| Checker | E0: misses lost to tracker | 28/36 |  | info | 0 |  |
| Checker | E0: misses lost to checker | 2/36 |  | info | 0 | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean-fp16 @ 2bdd2be (2026-10-07)

Delta is against the baseline row 2bdd2be (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 80.0% (28/35) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 43.8% (28/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 33.3% (2/6) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 6.9% (2/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 100.0% (26/26) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 74.3% (26/35) | ≥ 80.0% | fail |  |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.013 (3/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 0.324 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 28.7 ms |
| End to end | Word highlight latency p95 | 492 ms (p50 338 ms, 4706 words) | ≤ 600 | pass |  | onnxruntime-node, 20 ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10 |
| End to end | Suspected latency p95 | 1022 ms (p50 306 ms, 132 errors) | ≤ 1000 | fail |  | from the onset of the word after the error; PLAN: within about 1 s |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 4.79% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 10.56% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 9.92% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.14% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 5.45% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 8.57% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 6.33% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.21% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 7.4% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 8.78% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | — |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | — |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean-fp16's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-fp16's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-fp16's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 78.1% (303/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.4% (21/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean-fp16 |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 83.6% (56/67) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 77.4% (24/31) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: omitted_word precision | 97.0% (32/33) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.022 (5/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.541 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 6/36 |  | info |  |  |
| Checker | E0: misses lost to tracker | 28/36 |  | info |  |  |
| Checker | E0: misses lost to checker | 2/36 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean-lite-onnx @ 7fb8fff (2026-10-07)

Delta is against the baseline row 7fb8fff (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 63.4% (26/41) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 40.6% (26/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 10.0% (1/10) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 92.6% (25/27) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 71.4% (25/35) | ≥ 80.0% | fail |  |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.044 (10/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 1.081 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 27.7 ms |
| End to end | Word highlight latency p95 | 494 ms (p50 332 ms, 4682 words) | ≤ 600 | pass |  | onnxruntime-node, 20 ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10 |
| End to end | Suspected latency p95 | 1013 ms (p50 281 ms, 153 errors) | ≤ 1000 | fail |  | from the onset of the word after the error; PLAN: within about 1 s |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 5.75% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 12.37% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 11.02% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.98% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 7.79% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 19.29% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 8.34% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.76% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 8.7% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 10.05% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | — |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | — |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 70.3% (45/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean-lite-onnx's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-lite-onnx's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-lite-onnx's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 77.6% (301/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean-lite-onnx |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 70.0% (56/80) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 56.1% (23/41) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 79.3% (23/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 91.7% (33/36) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 94.3% (33/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.084 (19/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 2.054 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/38 |  | info |  |  |
| Checker | E0: misses lost to tracker | 30/38 |  | info |  |  |
| Checker | E0: misses lost to checker | 3/38 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## Latest: spliced-clean-lite-int8 @ 7fb8fff (2026-10-07)

Delta is against the baseline row 7fb8fff (2026-10-07).

| Area | Metric | Value | Bar | Status | Δ baseline | Note |
|---|---|---:|---|---|---:|---|
| End to end | Memorization precision | 63.4% (26/41) | ≥ 90.0% | fail |  | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy |
| End to end | Memorization recall | 40.6% (26/64) | ≥ 80.0% | fail |  |  |
| End to end | wrong_word precision | 10.0% (1/10) | ≥ 90.0% | fail |  |  |
| End to end | wrong_word recall | 3.4% (1/29) | ≥ 80.0% | fail |  |  |
| End to end | omitted_word precision | 92.6% (25/27) | ≥ 90.0% | pass |  |  |
| End to end | omitted_word recall | 71.4% (25/35) | ≥ 80.0% | fail |  |  |
| End to end | added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/3) | ≥ 90.0% | fail |  |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.044 (10/225) |  | info |  |  |
| End to end | False warnings per 5 min (short clips) | 1.081 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass |  | perf.ts gate: device floor assumed 20× slower; slowest push 28.5 ms |
| End to end | Word highlight latency p95 | 493 ms (p50 330 ms, 4684 words) | ≤ 600 | pass |  | onnxruntime-node, 20 ms pushes, this machine; measureLatency (packages/sdk): from the decoded phoneme, not a labelled onset. Device runs: ticket 10 |
| End to end | Suspected latency p95 | 1010 ms (p50 281 ms, 153 errors) | ≤ 1000 | fail |  | from the onset of the word after the error; PLAN: within about 1 s |
| Listener | Gold PER (canon vs spliced, human-verified exact) | 5.74% (233 clips) | ≤ 3.7% | fail |  | bar: gold PER where fitted declared.precision falls 2 pts below the teacher (3.39%) |
| Listener | Gold PER: laptop | 12.27% (16 clips) |  | info |  |  |
| Listener | Gold PER: phone | 11.11% (26 clips) |  | info |  |  |
| Listener | Gold PER: tlog | 3.96% (191 clips) |  | info |  |  |
| Listener | Gold PER: pharyngeals | 7.79% (385 ref phonemes) |  | info |  |  |
| Listener | Gold PER: emphatics | 20.00% (140 ref phonemes) |  | info |  |  |
| Listener | Gold PER: hamza ain | 8.61% (743 ref phonemes) |  | info |  |  |
| Listener | Gold PER: short vowels | 4.72% (5340 ref phonemes) |  | info |  |  |
| Listener | Gold raw PER vs spliced reference | 8.6% |  | info |  |  |
| Listener | Gold PER: never in tlog train | 10.08% (62 clips) |  | info |  | help clips plus tlog test/val gold clips; fair for students trained before the gold exclusion |
| Listener | Fair PER on tlog clean-match test (canon vs spliced) | — |  | pending |  | not in the PER breakdown; run tools/per_breakdown.py on this run's decode |
| Listener | Raw PER vs reference (tlog clean-match test) | — |  | info |  | Quran-Lab zipformer 3.65% on phone audio: reference only, not a gate |
| Listener | Raw PER vs teacher | — |  | info |  |  |
| Listener | Planted memorization errors visible in the decode | 68.8% (44/64) |  | info |  | E0 heuristic: under half the planted word's consonants decode as written |
| Listener | E1 help clips: short vowel greedy recall | — |  | pending |  | E1/E1b not run on spliced-clean-lite-int8's posteriors |
| Listener | E1: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-lite-int8's posteriors |
| Listener | E1b: classes heard | — |  | pending |  | E1/E1b not run on spliced-clean-lite-int8's posteriors |
| Tracker | help clean: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | fail |  | fair gate number on user-like audio |
| Tracker | help clean: clips with a false jump | 0.0% (0/225) | ≤ 2.0% | pass |  |  |
| Tracker | help clean: clips with any lock outside the passage | 0.0% (0/225) |  | info |  |  |
| Tracker | help clean: median time to correct lock | 2.64 s |  | info |  | from first decoded phoneme |
| Tracker | help all: correct ayah within 5 s | 77.6% (301/388) | ≥ 95.0% | fail |  | error clips: a substitution forming another ayah counts as wrong |
| Tracker | help all: clips with a false jump | 0.0% (0/388) | ≤ 2.0% | pass |  |  |
| Tracker | help all: clips with any lock outside the passage | 5.7% (22/388) |  | info |  |  |
| Tracker | help all: median time to correct lock | 2.56 s |  | info |  | from first decoded phoneme |
| Tracker | QUD regression: correct ayah within 5 s | — |  | pending |  | no QUD evidence from spliced-clean-lite-int8 |
| Tracker | Ambiguous tlog clips (PLAN gate set) | — |  | unmeasurable |  | no passage truth: the sampled ambiguous clips are wrong-passage or no-recitation (ticket 06) |
| Tracker | Scripted-error recordings (PLAN gate set) | — |  | unmeasurable |  | no recordings yet (ticket 05) |
| Checker | Declared passage: Memorization precision | 70.0% (56/80) | ≥ 90.0% | fail |  | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: wrong_word precision | 56.1% (23/41) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: wrong_word recall | 79.3% (23/29) | ≥ 80.0% | fail |  |  |
| Checker | Declared passage: omitted_word precision | 91.7% (33/36) | ≥ 90.0% | pass |  |  |
| Checker | Declared passage: omitted_word recall | 94.3% (33/35) | ≥ 80.0% | pass |  |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail |  |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.084 (19/225) |  | info |  |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 2.054 | ≤ 1 | fail |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: Memorization precision | 100.0% (60/60) | ≥ 90.0% | pass |  | oracle lock + oracle evidence synthesised from labels (Decision 7); 0 clips with an unphonemizable planted word skipped |
| Checker | Oracle lock + evidence: Memorization recall | 93.8% (60/64) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word precision | 100.0% (25/25) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: wrong_word recall | 86.2% (25/29) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word precision | 100.0% (35/35) | ≥ 90.0% | pass |  |  |
| Checker | Oracle lock + evidence: omitted_word recall | 100.0% (35/35) | ≥ 80.0% | pass |  |  |
| Checker | Oracle lock + evidence: added_word precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: skipped_passage precision | — (0 raised) | ≥ 90.0% | unmeasurable |  |  |
| Checker | Oracle lock + evidence: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Oracle lock + evidence: False warnings per clean clip | 0.000 (0/225) |  | info |  |  |
| Checker | Oracle lock + evidence: False warnings per 5 min (short clips) | 0 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Oracle lock + evidence: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Oracle lock + evidence: positives raised in any state | 93.8% (60/64) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (confirmation delay on short clips) |
| Checker | E0: misses lost to evidence | 5/38 |  | info |  |  |
| Checker | E0: misses lost to tracker | 30/38 |  | info |  |  |
| Checker | E0: misses lost to checker | 3/38 |  | info |  | real upstream, free arm |
| Checker | Self-correction / long-session false warnings | — |  | unmeasurable |  | no recordings (ticket 05) |

## History

| Date | Commit | Listener | End to end: Memorization precision | End to end: Memorization recall | End to end: False warnings per 5 min (short clips) | Listener: Gold PER (canon vs spliced, human-verified exact) | Tracker: help clean: correct ayah within 5 s | Tracker: QUD regression: correct ayah within 5 s | Tracker: QUD regression: false jumps | Checker: Oracle lock: Memorization recall | Checker: Oracle lock + evidence: Memorization recall | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-10-07 | 3e91183 | teacher | 55.2% (32/58) | 50.0% (32/64) | 0.861 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 67.2% (43/64) | 59.4% (38/64) | Baseline: engine at 3e91183, offline Muaalem teacher evidence; first scoreboard row |
| 2026-10-07 | 3e91183 | e2-libri-full | — | — | — | — | — | — | — | — | — | Baseline student: e2-libri-full (E2 LibriSpeech init); PER only, help-clip evidence not decoded yet |
| 2026-10-07 | 8aca172 | teacher | 55.2% (32/58) | 50.0% (32/64) | 0.861 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Declared-passage mode in core (CheckerOptions.passage) + end-of-session finalisation in declared sessions; declared arm 62.6%/89.1%, free unchanged |
| 2026-10-07 | 004986d | teacher | 66.7% (32/48) | 50.0% (32/64) | 0.861 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Skip placement: prefer the equal placement on an ayah start when a word recurs across the skip; oracle false omitted_word 9 -> 0 |
| 2026-10-07 | 46a068e | teacher | 71.1% (32/45) | 50.0% (32/64) | 0.538 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | A word with no consonants (ghunna/madd only) needs a hard edit to be a wrong word |
| 2026-10-07 | 16d6396 | teacher | 76.2% (32/42) | 50.0% (32/64) | 0.215 | — | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Added word between words needs two consonants, as inside a word; declared 77.0%/89.1%, free 76.2%/50.0%, oracle 100%/92.2% |
| 2026-10-07 | bfc52cb | teacher | 76.2% (32/42) | 50.0% (32/64) | 0.215 | 3.39% (233 clips) | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 92.2% (59/64) | Gold PER (233 human-verified exact clips) is the primary listener PER, bar 3.65% calibrated on declared precision; teacher 3.39% gold PER |
| 2026-10-07 | bfc52cb | e5-top1-lite | 53.8% (28/52) | 43.8% (28/64) | 2.163 | 7.21% (233 clips) | 83.6% (188/225) | — | — | 84.4% (54/64) | 92.2% (59/64) | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e5-top1-lite; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e5-soft5-lite | 65.9% (29/44) | 45.3% (29/64) | 1.298 | 5.94% (233 clips) | 83.6% (188/225) | — | — | 84.4% (54/64) | 92.2% (59/64) | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e5-soft5-lite; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e2-scratch-full | 43.9% (29/66) | 45.3% (29/64) | 3.46 | 7.30% (233 clips) | 83.1% (187/225) | — | — | 85.9% (55/64) | 92.2% (59/64) | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e2-scratch-full; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e2-libri-full | 73.2% (30/41) | 46.9% (30/64) | 0.757 | 5.98% (233 clips) | 84.4% (190/225) | — | — | 82.8% (53/64) | 92.2% (59/64) | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e2-libri-full; calibration point for listener-target-v1 |
| 2026-10-07 | 72d1f04 | combo-libri-soft5-full | 83.8% (31/37) | 48.4% (31/64) | 0.216 | 4.82% (233 clips) | 84.4% (190/225) | — | — | 84.4% (54/64) | 92.2% (59/64) | Score the completed pretrained soft-target full student on existing held-out help evidence to measure mistake detection. |
| 2026-10-07 | 2d94ea0 | combo-libri-soft5-full | 83.8% (31/37) | 48.4% (31/64) | 0.216 | 4.82% (233 clips) | 84.4% (190/225) | — | — | 84.4% (54/64) | 92.2% (59/64) | LibriSpeech init + soft top-5, 12 epochs; trained on gold tlog clips, so gold PER is optimistic (use held-out slice) |
| 2026-10-07 | 2d94ea0 | combo-clean | 75.7% (28/37) | 43.8% (28/64) | 0.541 | 4.97% (233 clips) | 84.4% (190/225) | — | — | 79.7% (51/64) | 92.2% (59/64) | combo retrained with the 150 gold tlog clips excluded from train; same recipe |
| 2026-10-07 | 2d94ea0 | spliced-clean | 81.8% (27/33) | 42.2% (27/64) | 0.324 | 4.71% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 92.2% (59/64) | spliced-reference CTC target (reference between teacher anchors, runs collapsed) + soft top-5 at weight 2, gold excluded |
| 2026-10-07 | 457e2f0 | spliced-clean-onnx | 81.8% (27/33) | 42.2% (27/64) | 0.324 | — | 84.0% (189/225) | — | — | 87.5% (56/64) | 92.2% (59/64) | Ticket 09: spliced-clean as shipped: fp32 streaming ONNX on onnxruntime-node via the SDK listener (TS fbank, 320 ms windows); help evidence decoded from WAV, 381/388 decodes identical to PyTorch |
| 2026-10-07 | 2bdd2be | teacher | 76.7% (33/43) | 51.6% (33/64) | 0.215 | 3.39% (233 clips) | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 93.8% (60/64) | Cadence-independent acquisition (acquireMs grid, decodedTo) and checker anchors on end-of-session locks; scored at 320 ms chunks |
| 2026-10-07 | 2bdd2be | spliced-clean-onnx | 80.0% (28/35) | 43.8% (28/64) | 0.324 | 4.80% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | spliced-clean onnx export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 2bdd2be | spliced-clean-int8 | 77.8% (28/36) | 43.8% (28/64) | 0.433 | 4.82% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | spliced-clean int8 export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 2bdd2be | spliced-clean-fp16 | 80.0% (28/35) | 43.8% (28/64) | 0.324 | 4.79% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | spliced-clean fp16 export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 7fb8fff | spliced-clean-lite-onnx | 63.4% (26/41) | 40.6% (26/64) | 1.081 | 5.75% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | spliced-clean-lite (22M, spliced-clean recipe at size lite, 12 epochs from scratch, gold excluded) onnx export on onnxruntime-node, 20 ms pushes |
| 2026-10-07 | 7fb8fff | spliced-clean-lite-int8 | 63.4% (26/41) | 40.6% (26/64) | 1.081 | 5.74% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | spliced-clean-lite (22M, spliced-clean recipe at size lite, 12 epochs from scratch, gold excluded) int8 export on onnxruntime-node, 20 ms pushes |
| 2026-10-07 | 6343815 | teacher | 76.7% (33/43) | 51.6% (33/64) | 0.215 | 3.39% (233 clips) | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 93.8% (60/64) | Baseline for free-recitation latching: stitched-session free-checker false wrong/added words (ticket 13 sessions) |
| 2026-10-07 | db72fff | teacher | 76.7% (33/43) | 51.6% (33/64) | 0.215 | 3.39% (233 clips) | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 93.8% (60/64) | Free recitation latches onto the first lock as a declared passage (extends forward; re-anchors only on a 4+ word jump off it, or when the lock is lost) |
| 2026-10-09 | adbcafe | teacher | 84.6% (33/39) | 51.6% (33/64) | 0.215 | 3.39% (233 clips) | 84.9% (191/225) | 76.9% (30/39) | 8.8% (8/91) | 89.1% (57/64) | 93.8% (60/64) | Local-window warning confirmation: a wrong/omitted/added word is confirmed only if the decode beats the written words by 10 nats of CTC likelihood over the word ±1 (threshold frozen on train speakers). Declared 57/17/7 -> 57/11/7, free 33/10/31 -> 33/6/31, no new misses |
| 2026-10-09 | adbcafe | spliced-clean-int8 | 87.5% (28/32) | 43.8% (28/64) | 0.216 | 4.82% (233 clips) | 84.0% (189/225) | — | — | 87.5% (56/64) | 93.8% (60/64) | Local-window warning confirmation on the shipped int8 full student (SDK now passes listener posteriors to the checker only). Declared 56/12/8 -> 56/8/8, free 28/8/36 -> 28/4/36, no new misses |
