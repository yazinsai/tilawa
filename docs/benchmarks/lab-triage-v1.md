# Lab two-model triage v1 (lab-v1.1)

Ticket `.scratch/lab/issues/17-two-model-triage.md`, spec decisions 16 and 17. Two independent likelihood scorers score every case of the test set. A case they both accept is done. Everything else, plus seeded spot checks, goes to the listening queue.

- Run: `modal run modal/lab_triage.py --test-set lab-v1.1` (orchestrator in `modal/lab_triage.py`, logic in `tools/lab_triage_core.py`, tests in `tools/test_lab_triage.py`)
- Queue: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue=lab-v1.1/review/triage.json>
- Outputs: `/lab/data/lab-v1.1/triage-v1.json` (every case: both scores, decisions, outcome and spot flags), `review/triage.json` (the queue), `triage/` (per-spelling model records and the Nemotron transcripts). Summary: `lab-triage-v1-lab-v1.1.json`.
- Build loop: `modal run modal/lab_build_sessions.py --stage loop` renders, runs the seam gate, then the teacher decode and automated checks, refusing failures until nothing fails, then runs triage. `--stage gate` reports the seam gate alone.

## Rule (fixed before the run)

| | Nemotron 3.5 Transcribe | FastConformer |
| --- | --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` @ `9d8cd6b`, H100 | `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` @ `7f32349`, T4 |
| Reading | LM log-likelihood of the text, plain spellings | CTC log-likelihood, diacritised spellings |
| Margin | 2 nats (ticket 12) | 2 nats (ticket 12) |

Each model compares the labelled text (claimed) with one alternative over the same window. Its decision is match at 2 or more nats in favour of the label, mismatch at 2 or more nats against it, and unsure otherwise. A text with no CTC path counts as infinitely worse. Spellings come from each model's own label-blind transcript of the window, never from the label.

- **Planted edit**: the `review_splice` window (2.5 s either side, whole words), the same window and id as the v1 check. The alternative is the original text at that spot (`revert_text`).
- **Repeat or restart**: the `review_repeat` window (3 s either side). The alternative is the text without the repeat.
- **Clean**: each clean run (unplanted words with no seam inside, as `lab_checks` defines it) is cut at word boundaries into windows of about 8 s, each with at least 3 words. The alternative is one interior word substituted by a clearly different word of the session, or a foreign word inserted. The choice and position are seeded by the case id.

Studio copy only. Both match means auto-accepted. Both mismatch means **queued for you to confirm**, never flagged or rebuilt automatically. One of each means disagree, and anything else is unsure; both are queued. The spot check is a seeded 10% of the auto-accepted cases, and the seam spot check is a seeded 10% of the planted edits (splices, repeats and restarts). Each case appears once in the queue, with its reasons merged into the title (for example "Seam spot check · also a random spot check"). The bar is `maxBad = max(1, items // 40)`.

One deviation was settled on a two-session smoke run (p05, e05) before the full run: clean windows don't use deletions. A peaky CTC model can drop a clearly heard word for the cost of a few blank frames, which is less than one imperfect spelling costs. FastConformer preferred the shorter text on 3 of 19 clean deletions, by 13–21 nats, with the word present in its own greedy output. It did so on none of 32 substitutions and insertions. The margins did not change.

## Results

1576 cases: 360 planted edits, repeats and restarts, and 1216 clean windows (4–18 s, mean 8 s). They cover every said word of the 40 sessions.

| Class | Cases | Auto-accepted | Both mismatch | Disagree | Unsure |
| --- | ---: | ---: | ---: | ---: | ---: |
| wrong_word | 94 | 74 | 1 | 12 | 7 |
| added_word | 105 | 65 | 6 | 21 | 13 |
| omitted_word | 42 | 41 | 0 | 1 | 0 |
| skipped_passage | 69 | 69 | 0 | 0 | 0 |
| mutashabihat | 10 | 9 | 0 | 1 | 0 |
| repeat | 23 | 13 | 0 | 7 | 3 |
| restart | 17 | 14 | 0 | 1 | 2 |
| clean | 1216 | 1208 | 0 | 7 | 1 |
| **all** | **1576** | **1493** | **7** | **50** | **26** |

Per model, Nemotron gave 1534 match, 26 mismatch and 16 unsure, and FastConformer 1521, 44 and 11.

**Queue: 257 items, maxBad 6.**

| Why (first reason) | Items |
| --- | ---: |
| Both models think the label is wrong — please confirm | 7 |
| The two models disagree | 50 |
| A model couldn't tell — please check | 26 |
| Seam spot check | 30 (36 counting those also queued for another reason) |
| Random spot check | 144 (150; 118 of them clean windows) |

Eight queue items carry a verdict you already gave in the v1 check, on the same window and text, so the review page shows them as heard: p05-x6, p12-x7, p15-x8, e19-x12, e08-x1, p09-x8, e10-x9 and p10-x6.

All 7 both-mismatch cases are linked splices: six added words and one wrong word, all joined mid-phrase. In each, neither model's own transcript holds the planted word as labelled. For example, p19-x11 is heard as "عينا يشرب بها المقربون" with no ما, p01-x4 has no فِى, and e13-x8 is heard as junk around the join. They look like donor words that are inaudible or mangled at a link cut. This is a real candidate failure the numeric seam checks can't see.

Added words are the weakest class (40 of 105 not auto-accepted). Their alternative is the deletion, which is the same CTC length bias as above. Nemotron accepts most of the ones FastConformer rejects.

### Against your 40 v1-check verdicts (all ok)

34 auto-accepted, 4 disagree, 2 unsure, **0 both-mismatch (no false alarms)**. The six queued ones:

| Item | Class | Nemotron | FastConformer |
| --- | --- | --- | --- |
| p05-x6 | repeat | match +11.5 | mismatch −98.9 (calibration's known FastConformer miss) |
| p15-x8 | added_word | match +10.5 | mismatch −41.2 |
| e19-x12 | added_word | match +13.7 | mismatch −2.8 |
| p12-x7 | mutashabihat | mismatch −3.3 | match +30.0 |
| e08-x1 | repeat | unsure +1.0 | match +49.1 |
| p09-x8 | wrong_word | unsure +1.6 | unsure +0.2 |

The 14 clean windows of the v1 check (7 studio, 7 phone) were scored as extra cases and were not queued. They are included in the 40.

## Seam gate

The render's seam checks (`<id>.checks.json`) were computed but never enforced. `lab_sessions.SEAM_LIMITS` now gates them in the build loop. A failing seam or word refuses its splice's site and donor (`refusals_from_checks`), and the next plan rebuilds it before any decode. Each limit is set well outside lab-v1.1's worst value, which you accepted by ear:

| Check | Limit | lab-v1.1 worst | Pass |
| --- | --- | --- | ---: |
| Seam level against the source, either side | ≤ 1.0 dB, only where the source side is above −60 dBFS | 0.02 dB above −60 dBFS (1.4 dB at −86 dBFS digital silence in e05) | 569 / 569 |
| Click: peak step in the crossfade against the p99 step around it | ≤ 2.5 | 1.65 (p16-x9) | 569 / 569 |
| Seam depth against the session's speech level | ≤ −3 dB (the planner's `LINK_DB`) | −5.4 dB (p15-x1, linked) | 569 / 569 |
| Every unclipped said word found at its labelled time | lag 0, correlation ≥ 0.99 (was 0.999, reported only) | 0.9963 (p10 55:21:3, wrong-word donor at +4.3 dB, likely PCM16 saturation) | 8722 / 8722 |
| Word cut short by a cut | ≤ 25 ms (20 ms cut slack plus half the crossfade) | 15 ms | 135 / 135 clipped words |

The gate wrote `/lab/data/lab-v1.1/seams-v1.json` (report only, so lab-v1.1 is not re-planned).

## Cost

| | GPU | Time |
| --- | --- | --- |
| Nemotron transcribe (1487 new, 103 reused) | 4× H100 | 1923 s |
| Nemotron score | 4× H100 | 548 s |
| Nemotron model loads | | 4 × ~41 s |
| FastConformer score | 1× T4 | 128 s, plus an 8 s load |

The wall clock was 13 min 15 s. Total H100 compute was about 44 min busy, with containers held about 55 min (an upper bound). The cost was roughly $3.5 for the H100s and $0.15 for the T4, plus about $0.5 for the two smoke runs. These are estimates from Modal list prices.
