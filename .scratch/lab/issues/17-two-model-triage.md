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
