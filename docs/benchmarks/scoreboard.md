# Engine scoreboard

The release cockpit for the engine. One row per commit × listener, recomputed from evidence on disk (no GPU).

- **Release readiness** judges the candidate listener (`--candidate`, sticky) against the v1 gates in `GATES` (`packages/core/src/bench/scoreboard.ts`). It is ready when every gate passes and every requirement in `BLOCKED` has data. Gates are ranked by shortfall from the bar, so the top row is what most needs work.
- **Experiments** judges every run against its parent: the previous run of the same listener, or the candidate for a listener's first run. The verdict counts only gate moves larger than two clips; smaller ones are marked `~` and count as noise, since the eval set is 64 planted mistakes and 225 clean clips.
- **All metrics** lists every diagnostic for the candidate.

**Add a row.** From `packages/core`: `HELP_DIR=<repo>/.scratch/help QUD_DIR=<repo>/.scratch/bench/qud pnpm scoreboard --note '<what changed and why>' [--modules listener,tracker,checker]` (teacher). For a student, decode the help eval clips into a directory of `<clip>.evidence.json` and add `--listener <run-name> --evidence <dir>`. A run replaces the row with the same commit and listener, else appends. Commit `scoreboard.json` and `scoreboard.md` with the change. `pnpm scoreboard --render --candidate <listener>` switches the candidate without scoring.

**Dashboard.** `uv run tools/scoreboard_serve.py`, then http://127.0.0.1:8765/scoreboard.html.

**Sources.** Listener PER is gold PER (`tools/gold_set.py score`), barred by `listener-target-v1.json`. Help clips: 388 eval clips, 29 held-out speakers. Free memorization is gated `with lead-in` (`--leadin`, `tools/leadin_build.py`): 1–2 correctly recited preceding ayahs are stitched before each clip at the evidence level (same speaker's clean clip, else another help speaker's, else held-out tlog, else tlog train), errors and false positives are scored inside the original clip, and warnings on the lead-in are counted apart. For a student add `--leadin-evidence <dir>` with its decode of `.scratch/leadin/audio` (`pnpm --filter @tilawa/sdk decode`). Without a lead-in the free numbers are info; each arm is also split into `unambiguous` and `mutashabihat` (planted errors that form another ayah, `forms_other_ayah`). `Declared passage` is the memorization-session arm (`CheckerOptions.passage`); `oracle evidence` replaces the decode with one synthesised from the label, so its losses are the checker's own. E0 attributes each missed mistake to listener, tracker or checker.

## Release readiness: spliced-clean-int8 @ 88fad18 (2026-10-09)

**Not ready**: 6/12 gates pass, 5 fail, 1 not measured on this listener, 5 requirements not measurable yet. Worst shortfall first.

| Gate | Now | Bar | Shortfall | Δ previous run |
|---|---:|---|---:|---:|
| False warnings per 5 min (stitched sessions) | — |  | not measured |  |
| Listener gold PER | 4.82% (233 clips) | ≤ 3.7% | 32% short | 0 |
| Free recitation (after 1–2 recited ayahs): warnings correct | 71.9% (41/57) | ≥ 90.0% | 20% short |  |
| Free recitation (after 1–2 recited ayahs): mistakes caught | 64.1% (41/64) | ≥ 80.0% | 20% short |  |
| Tracker: correct ayah within 5 s | 84.0% (189/225) | ≥ 95.0% | 12% short | 0 |
| Memorization session: warnings correct | 82.4% (56/68) | ≥ 90.0% | 8% short | 0 |
| Memorization session: mistakes caught | 87.5% (56/64) | ≥ 80.0% | pass | 0 |
| False warnings per 5 min (short clips) | 0.433 | ≤ 1 | pass | 0 |
| False warnings on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |
| False warnings per 5 min (lead-in recitation) | 0.90 per 5 min (10 in 55.4 min, 346 lead-ins) | ≤ 1 | pass |  |
| Tracker: false jumps | 0.0% (0/225) | ≤ 2.0% | pass | 0 |
| Real-time factor | 0.002 | ≤ 0.05 | pass | 0 |

Not measurable yet (blocks release): Pronunciation: short vowels; Pronunciation: confusable pairs, shadda; Word highlight / warning latency on device; Self-correction and long-session false warnings; Scripted-error recordings.

**Biggest lever on recall: tracker**, which loses 28 of the 36 planted mistakes missed in free recitation (E0 attribution).

## Experiments

Newest first. Each run is judged against its parent: the previous run of the same listener, else the candidate. Only gates that moved are listed; `~` marks a move of two clips or fewer.

| Date | Commit | Listener | vs | Verdict | Gate moves | Note |
|---|---|---|---|---|---|---|
| 2026-10-09 | 88fad18 | spliced-clean-int8 | ded4480 | = unchanged |  | Lead-in eval on the shipped full tier (int8, onnxruntime-node): help clips from sdk-help, tlog lead-ins decoded from their audio with pnpm decode (20 ms pushes); free with lead-in is the tracked free metric |
| 2026-10-09 | 88fad18 | teacher | ded4480 | = unchanged |  | Lead-in eval: 1–2 correctly recited preceding ayahs stitched before each help clip (evidence level; same speaker, else other help speaker, else held-out tlog, else tlog train), errors scored inside the clip; free with lead-in is the tracked free metric, no-lead-in free numbers become info; lead-in false warnings counted apart |
| 2026-10-09 | ded4480 | spliced-clean-int8 | 2bdd2be | = unchanged |  | Mutashabihat split on the shipped full tier (int8, onnxruntime-node): free and declared memorization split by forms_other_ayah; free unambiguous is the tracked free metric, all FP stay there |
| 2026-10-09 | ded4480 | teacher | db72fff | = unchanged |  | Mutashabihat split: forms_other_ayah labels (recitation with the error fits another ayah as well); free and declared memorization split into unambiguous and mutashabihat; free unambiguous is the tracked free metric, all FP stay there |
| 2026-10-07 | db72fff | teacher | 6343815 | · noise | ▼~ False warnings per 5 min (stitched sessions) 2.04 (305 in 747 min) → 2.07 (309 in 747 min) | Free recitation latches onto the first lock as a declared passage (extends forward; re-anchors only on a 4+ word jump off it, or when the lock is lost) |
| 2026-10-07 | 6343815 | teacher | 2bdd2be | = unchanged |  | Baseline for free-recitation latching: stitched-session free-checker false wrong/added words (ticket 13 sessions) |
| 2026-10-07 | 7fb8fff | spliced-clean-lite-int8 | spliced-clean-int8 2bdd2be | ▼ worse | ▼ Memorization session: warnings correct 82.4% (56/68) → 70.0% (56/80)<br>▼ False warnings per 5 min (short clips) 0.433 → 1.081<br>▼ Listener gold PER 4.82% (233 clips) → 5.74% (233 clips) | spliced-clean-lite (22M, spliced-clean recipe at size lite, 12 epochs from scratch, gold excluded) int8 export on onnxruntime-node, 20 ms pushes |
| 2026-10-07 | 7fb8fff | spliced-clean-lite-onnx | spliced-clean-int8 2bdd2be | ▼ worse | ▼ Memorization session: warnings correct 82.4% (56/68) → 70.0% (56/80)<br>▼ False warnings per 5 min (short clips) 0.433 → 1.081<br>▼ Listener gold PER 4.82% (233 clips) → 5.75% (233 clips) | spliced-clean-lite (22M, spliced-clean recipe at size lite, 12 epochs from scratch, gold excluded) onnx export on onnxruntime-node, 20 ms pushes |
| 2026-10-07 | 2bdd2be | spliced-clean-fp16 | spliced-clean-int8 2bdd2be | ▲ better | ▲~ Memorization session: warnings correct 82.4% (56/68) → 83.6% (56/67)<br>▲ False warnings per 5 min (short clips) 0.433 → 0.324<br>▲~ Listener gold PER 4.82% (233 clips) → 4.79% (233 clips) | spliced-clean fp16 export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 2bdd2be | spliced-clean-int8 | first run |  |  | spliced-clean int8 export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 2bdd2be | spliced-clean-onnx | 457e2f0 | = unchanged |  | spliced-clean onnx export on onnxruntime-node, 20 ms pushes, 40 ms evidence cells |
| 2026-10-07 | 2bdd2be | teacher | bfc52cb | = unchanged |  | Cadence-independent acquisition (acquireMs grid, decodedTo) and checker anchors on end-of-session locks; scored at 320 ms chunks |
| 2026-10-07 | 457e2f0 | spliced-clean-onnx | first run |  |  | Ticket 09: spliced-clean as shipped: fp32 streaming ONNX on onnxruntime-node via the SDK listener (TS fbank, 320 ms windows); help evidence decoded from WAV, 381/388 decodes identical to PyTorch |
| 2026-10-07 | 2d94ea0 | spliced-clean | first run |  |  | spliced-reference CTC target (reference between teacher anchors, runs collapsed) + soft top-5 at weight 2, gold excluded |
| 2026-10-07 | 2d94ea0 | combo-clean | first run |  |  | combo retrained with the 150 gold tlog clips excluded from train; same recipe |
| 2026-10-07 | 2d94ea0 | combo-libri-soft5-full | 72d1f04 | = unchanged |  | LibriSpeech init + soft top-5, 12 epochs; trained on gold tlog clips, so gold PER is optimistic (use held-out slice) |
| 2026-10-07 | 72d1f04 | combo-libri-soft5-full | first run |  |  | Score the completed pretrained soft-target full student on existing held-out help evidence to measure mistake detection. |
| 2026-10-07 | bfc52cb | e2-libri-full | 3e91183 | = unchanged |  | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e2-libri-full; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e2-scratch-full | first run |  |  | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e2-scratch-full; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e5-soft5-lite | first run |  |  | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e5-soft5-lite; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | e5-top1-lite | first run |  |  | First help-clip decode (CPU fp32, chunk 16/left 128) and gold PER for e5-top1-lite; calibration point for listener-target-v1 |
| 2026-10-07 | bfc52cb | teacher | 16d6396 | = unchanged |  | Gold PER (233 human-verified exact clips) is the primary listener PER, bar 3.65% calibrated on declared precision; teacher 3.39% gold PER |
| 2026-10-07 | 16d6396 | teacher | 46a068e | ▲ better | ▲ Memorization session: warnings correct 74.0% (57/77) → 77.0% (57/74)<br>▲ False warnings per 5 min (short clips) 0.538 → 0.215 | Added word between words needs two consonants, as inside a word; declared 77.0%/89.1%, free 76.2%/50.0%, oracle 100%/92.2% |
| 2026-10-07 | 46a068e | teacher | 004986d | ▲ better | ▲ Memorization session: warnings correct 70.4% (57/81) → 74.0% (57/77)<br>▲ False warnings per 5 min (short clips) 0.861 → 0.538 | A word with no consonants (ghunna/madd only) needs a hard edit to be a wrong word |
| 2026-10-07 | 004986d | teacher | 8aca172 | ▲ better | ▲ Memorization session: warnings correct 62.6% (57/91) → 70.4% (57/81) | Skip placement: prefer the equal placement on an ayah start when a word recurs across the skip; oracle false omitted_word 9 -> 0 |
| 2026-10-07 | 8aca172 | teacher | 3e91183 | ▲ better | ▲ Memorization session: mistakes caught 67.2% (43/64) → 89.1% (57/64)<br>▲ Memorization session: warnings correct 61.4% (43/70) → 62.6% (57/91) | Declared-passage mode in core (CheckerOptions.passage) + end-of-session finalisation in declared sessions; declared arm 62.6%/89.1%, free unchanged |
| 2026-10-07 | 3e91183 | e2-libri-full | first run |  |  | Baseline student: e2-libri-full (E2 LibriSpeech init); PER only, help-clip evidence not decoded yet |
| 2026-10-07 | 3e91183 | teacher | first run |  |  | Baseline: engine at 3e91183, offline Muaalem teacher evidence; first scoreboard row |

## All metrics: spliced-clean-int8

| Area | Metric | Value | Bar | Status | Δ previous run | Note |
|---|---|---:|---|---|---:|---|
| End to end | Free, with lead-in: Memorization precision | 71.9% (41/57) | ≥ 90.0% | fail |  | free recitation with lead-in, the tracked free metric; leadin-v1: 346/388 clips get 1–2 preceding ayahs (first ayah of the surah 18, help-other-speaker 17, help-same-speaker 4, no clean recitation of the preceding ayah 24, tlog-heldout 255, tlog-train 70), 600 ms silence, stitched at the evidence level; errors and false positives scored inside the original clip |
| End to end | Free, with lead-in: Memorization recall | 64.1% (41/64) | ≥ 80.0% | fail |  |  |
| End to end | Free, with lead-in: False warnings per clean clip | 0.027 (6/225) |  | info |  |  |
| End to end | Free, with lead-in: False warnings per 5 min (short clips) | 0.649 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| End to end | Free, with lead-in: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| End to end | Free, with lead-in: unambiguous precision | 67.3% (33/49) | ≥ 90.0% | fail |  | excludes 26 planted errors whose recitation forms another ayah at least as well (forms_other_ayah); all false positives count here |
| End to end | Free, with lead-in: unambiguous recall | 86.8% (33/38) | ≥ 80.0% | pass |  |  |
| End to end | Free, with lead-in: mutashabihat precision | 100.0% (8/8) | ≥ 90.0% | pass |  | planted errors whose recitation forms another ayah at least as well (forms_other_ayah); no false positives attributed |
| End to end | Free, with lead-in: mutashabihat recall | 30.8% (8/26) | ≥ 80.0% | fail |  |  |
| End to end | Free, with lead-in: mutashabihat raised in any state | 69.2% (18/26) |  | info |  | the rest of the gap to recall is events still suspected/uncertain at clip end (free recitation has no end-of-session confirmation) |
| End to end | Free, with lead-in: false warnings in the lead-in | 0.90 per 5 min (10 in 55.4 min, 346 lead-ins) | ≤ 1 | pass |  | confirmed events of any memorization class on the lead-in's ayahs; all false |
| End to end | Memorization precision | 77.8% (28/36) |  | info | 0 | help eval split, held-out speakers: 388 clips, 29 speakers; free recitation, default policy; no lead-in, info only: the clip starts the recitation |
| End to end | Memorization recall | 43.8% (28/64) |  | info | 0 |  |
| End to end | wrong_word precision | 28.6% (2/7) |  | info | 0 |  |
| End to end | wrong_word recall | 6.9% (2/29) |  | info | 0 |  |
| End to end | omitted_word precision | 100.0% (26/26) |  | info | 0 |  |
| End to end | omitted_word recall | 74.3% (26/35) |  | info | 0 |  |
| End to end | added_word precision | 0.0% (0/1) |  | info | 0 |  |
| End to end | added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | skipped_passage precision | 0.0% (0/2) |  | info | 0 |  |
| End to end | skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| End to end | False warnings per clean clip | 0.018 (4/225) |  | info | 0 |  |
| End to end | False warnings per 5 min (short clips) | 0.433 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| End to end | False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| End to end | Free: unambiguous precision | 77.8% (28/36) |  | info | 0 | excludes 26 planted errors whose recitation forms another ayah at least as well (forms_other_ayah); all false positives count here |
| End to end | Free: unambiguous recall | 73.7% (28/38) |  | info | 0 |  |
| End to end | Free: mutashabihat precision | —% (0/0) |  | info |  | planted errors whose recitation forms another ayah at least as well (forms_other_ayah); no false positives attributed; needs declared passage or position hint |
| End to end | Free: mutashabihat recall | 0.0% (0/26) |  | info | 0 |  |
| End to end | Pronunciation: short_vowel | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 30 positives in eval split |
| End to end | Pronunciation: consonant_sad_sin | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 4 positives in eval split |
| End to end | Pronunciation: madd_natural | unsupported |  | unmeasurable |  | checker has no pronunciation classes; 1 positives in eval split |
| End to end | Pronunciation: other pairs, shadda | — |  | unmeasurable |  | no recordings with positives (ticket 05, phase 2) |
| End to end | Tracker real-time factor (this machine) | 0.002 | ≤ 0.05 | pass | 0 | perf.ts gate: device floor assumed 20× slower; slowest push 29.4 ms |
| End to end | Word highlight p95 / suspected / confirmed latency | — |  | unmeasurable |  | pass --latency <pnpm bench:help report> (packages/sdk); device runs: ticket 10 |
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
| Checker | Declared passage: Memorization precision | 82.4% (56/68) | ≥ 90.0% | fail | 0 | declared passage (CheckerOptions.passage): the prompted passage, real tracker and evidence |
| Checker | Declared passage: Memorization recall | 87.5% (56/64) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: wrong_word precision | 75.0% (24/32) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: wrong_word recall | 82.8% (24/29) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word precision | 97.0% (32/33) | ≥ 90.0% | pass | 0 |  |
| Checker | Declared passage: omitted_word recall | 91.4% (32/35) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: added_word precision | 0.0% (0/1) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: added_word recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: skipped_passage precision | 0.0% (0/2) | ≥ 90.0% | fail | 0 |  |
| Checker | Declared passage: skipped_passage recall | — (0 positives) |  | unmeasurable |  | no confirmed positives in the eval split |
| Checker | Declared passage: False warnings per clean clip | 0.027 (6/225) |  | info | 0 |  |
| Checker | Declared passage: False warnings per 5 min (short clips) | 0.649 | ≤ 1 | pass | 0 | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared passage: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass | 0 |  |
| Checker | Declared passage: unambiguous precision | 74.5% (35/47) | ≥ 90.0% | fail | 0 | excludes 26 planted errors whose recitation forms another ayah at least as well (forms_other_ayah); all false positives count here |
| Checker | Declared passage: unambiguous recall | 92.1% (35/38) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared passage: mutashabihat precision | 100.0% (21/21) | ≥ 90.0% | pass | 0 | planted errors whose recitation forms another ayah at least as well (forms_other_ayah); no false positives attributed |
| Checker | Declared passage: mutashabihat recall | 80.8% (21/26) | ≥ 80.0% | pass | 0 |  |
| Checker | Declared, with lead-in: Memorization precision | 74.3% (55/74) | ≥ 90.0% | fail |  | declared passage from the lead-in's first ayah; leadin-v1: 346/388 clips get 1–2 preceding ayahs (first ayah of the surah 18, help-other-speaker 17, help-same-speaker 4, no clean recitation of the preceding ayah 24, tlog-heldout 255, tlog-train 70), 600 ms silence, stitched at the evidence level; errors and false positives scored inside the original clip |
| Checker | Declared, with lead-in: Memorization recall | 85.9% (55/64) | ≥ 80.0% | pass |  |  |
| Checker | Declared, with lead-in: False warnings per clean clip | 0.031 (7/225) |  | info |  |  |
| Checker | Declared, with lead-in: False warnings per 5 min (short clips) | 0.757 | ≤ 1 | pass |  | short clips; the gating number needs phase-2 long sessions |
| Checker | Declared, with lead-in: False positives on repeats | 0 (23 repeat clips) | ≤ 0 | pass |  |  |
| Checker | Declared, with lead-in: unambiguous precision | 64.2% (34/53) | ≥ 90.0% | fail |  | excludes 26 planted errors whose recitation forms another ayah at least as well (forms_other_ayah); all false positives count here |
| Checker | Declared, with lead-in: unambiguous recall | 89.5% (34/38) | ≥ 80.0% | pass |  |  |
| Checker | Declared, with lead-in: mutashabihat precision | 100.0% (21/21) | ≥ 90.0% | pass |  | planted errors whose recitation forms another ayah at least as well (forms_other_ayah); no false positives attributed |
| Checker | Declared, with lead-in: mutashabihat recall | 80.8% (21/26) | ≥ 80.0% | pass |  |  |
| Checker | Declared, with lead-in: false warnings in the lead-in | 1.17 per 5 min (13 in 55.4 min) | ≤ 1 | fail |  |  |
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
