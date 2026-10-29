# 17 Two-model triage replaces the fixed 40-item listening check

Status: resolved
Type: task
Blocked by: 12

Spec: `../spec.md` decisions 16 and 17.

Two independent likelihood scorers check every case of a test set: Nemotron 3.5 Transcribe (LM pairwise) and NVIDIA's Arabic FastConformer (CTC pairwise). Every planted edit is scored as what was said against the original text at that spot, every repeat or restart with and without the repeat, and every clean stretch against a one-word change of its text.

- Both models say the label fits: auto-accepted. A seeded 10% of these go to the listening queue as a random spot check.
- Both models say it doesn't fit: queued for you to confirm. Nothing is flagged or rebuilt automatically.
- The models disagree, or either is unsure: queued.
- Seams can't be judged by the models. The builder's numeric seam checks cover them and are enforced in the build loop, and a seeded 10% of splices are queued for a seam listen.

The test set freezes when you have reviewed the queue with at most 1 bad verdict per 40 items. The old `v1-check` path stays.

## Acceptance

- `modal run modal/lab_triage.py --test-set lab-v1.1` writes `/lab/data/lab-v1.1/triage-v1.json` and `/lab/data/lab-v1.1/review/triage.json`. The queue passes `validate_queue`, has stable unique ids, and has plain titles that say why each item is there.
- The build loop enforces seam limits, refuses failing seams and runs triage after the automated checks pass.
- Results on lab-v1.1 are in `docs/benchmarks/lab-triage-v1.md`. Pure logic is tested in `tools/test_lab_triage.py`.

## Comments

Done on lab-v1.1 (`docs/benchmarks/lab-triage-v1.md`). There were 1576 cases: 1493 auto-accepted, 7 both-mismatch, 50 disagree and 26 unsure. The queue at `lab-v1.1/review/triage.json` has 257 items: 7 to confirm, 50 disagree, 26 unsure, 30 seam spot checks and 144 random spot checks, with maxBad 6. Eight items carry your v1-check verdicts. None of the 40 ok verdicts landed in both-mismatch. All 7 both-mismatch cases are linked added or wrong words that neither model hears, which suggests donor words mangled at link cuts.

The seam gate (`lab_sessions.SEAM_LIMITS`) is enforced in the build loop. On lab-v1.1, all 569 seams and all 8722 words pass, and the worst cut into a word is 15 ms. Clean windows use substitutions and insertions only, because CTC drops a heard word cheaply (settled on the smoke run, before the full run).

Open: the random spot check is 144 of the 257 items, 118 of them clean windows. No scoreboard row, because triage moves no lab metric.

lab-v1.2 (`docs/benchmarks/lab-triage-v1.md`): the spot check is now a fixed seeded sample of 15 auto-accepted edits and repeats plus 15 auto-accepted clean windows, and the seam spot check is 10% of the planted edits, at most 20 (spec decision 17). On lab-v1.1's own cases that gives 129 items instead of 257. On lab-v1.2 there were 1664 cases: 1585 auto-accepted, 10 both-mismatch (9 added words and a restart, all with seams at −7.5 dB or deeper), 53 disagree and 16 unsure. The queue at `lab-v1.2/review/triage.json` has 128 items with maxBad 3. Earlier verdicts carry over only on identical clip bytes and text; none of lab-v1.1's 47 matched.

lab-v1.3 (`docs/benchmarks/lab-triage-v1.md`): there were 1700 cases: 1624 auto-accepted, 6 both-mismatch (5 added words and a wrong word, all linked on both seams, cuts at −8.1 dB or deeper, donors from the host recording), 54 disagree and 16 unsure. The queue at `lab-v1.3/review/triage.json` has 117 items with maxBad 2. The both-mismatch items are also in `lab-v1.3/review/triage-confirm.json` (maxBad 0, diagnostic) to hear first. They share ids with the main queue, so verdicts count in both. Two lab-v1.2 verdicts carried over on identical clips.

API bake-off (`docs/benchmarks/lab-review-asr-v1.md`, "Commercial APIs"; `docs/benchmarks/lab-triage-v1.md`, "Nemotron + Soniox"). The second model was swapped for a commercial ASR's label-blind judge.
- On your 152 verdicts, Nemotron + Soniox auto-accepts 96/142 ok against 55 for Nemotron + FastConformer. It also auto-accepts 1/10 bad (lab-v1.2 p15-x1, Nemotron +38 nats), against 0/10. Speechmatics: 85/142 and 1/10. Soniox wins by the preregistered rule.
- Re-triaged lab-v1.3: 1559 auto-accepted, 0 both-mismatch, 19 disagree and 122 unsure. `lab-v1.3/review/triage-soniox.json` has 184 items (maxBad 4), against 117.
  - The growth is the preregistered margin: 56 of 60 skipped passages and 44 clean windows come out "too close to call". A post hoc attainable-gap margin would give about 112.
  - The 6 old confirm items (all ok) are now disagree. The Soniox confirm queue is empty.
- `triage.json` is unchanged and is still the queue to hear.
- No scoreboard row, because triage moves no lab metric.
- Open: re-preregister the judge margin on lab-v1.1 and lab-v1.2 only, then rerun lab-v1.3 from the cached transcripts. This needs no API calls.

Soniox margin v2 (`docs/benchmarks/lab-triage-v1.md`, "Soniox margin v2"). The `reach` margin was preregistered and committed (9f1dfe6) before any number, then calibrated on the lab-v1.1 and lab-v1.2 verdicts from cached transcripts. It needs half the gap a perfect transcript of the stretch the transcript covers would reach.
- It passes all three adoption checks. Safety: no newly accepted bad. Controls: every one still flagged. Gain: 101/137 ok auto-accepted, against 96. It is now the default (`review_api_core.JUDGE_RULE`).
- Only 5 items changed, all true skipped passages (unsure to match).
- On lab-v1.3's 117 verdicts: ok auto-accepted 75 → 79. Bad auto-accepted stays at 5 (e09-x3, p01-x2, p01-x4, p06-x1, p12-x8). Nemotron + FastConformer auto-accepted 4 of them.
- Both ASRs hear the planted word in all five, so no text margin catches them. The pause-to-pause rule of lab-v1.4 refuses all of them.

lab-v1.4 (`docs/benchmarks/lab-triage-v1.md`, "Results on lab-v1.4"). Nemotron + Soniox is the set's triage; the FastConformer queues are kept as `review/triage-fastconformer(-confirm).json`. The seam spot check has two pools: inserted audio 20% (at most 30) and the rest 10% (at most 20).
- 1591 cases: 1526 auto-accepted, 0 both-mismatch, 8 disagree and 57 unsure.
- `lab-v1.4/review/triage.json` has 126 items (8 disagree, 57 unsure, 32 seam with 14 inserted, 29 spot) with maxBad 3. It loads via `/api/queue`.
- `review/triage-confirm.json` is empty, so the API refuses it (422, "items must be a non-empty list").
- Soniox cost $0.36. No scoreboard row, because triage moves no lab metric.
