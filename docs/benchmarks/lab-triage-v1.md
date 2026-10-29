# Lab two-model triage v1 (lab-v1.1, lab-v1.2, lab-v1.3)

Ticket `.scratch/lab/issues/17-two-model-triage.md`, spec decisions 16 and 17. Two independent likelihood scorers score every case of the test set. A case they both accept is done. Everything else, plus seeded spot checks, goes to the listening queue.

- Run: `modal run modal/lab_triage.py --test-set lab-v1.3` (orchestrator in `modal/lab_triage.py`, logic in `tools/lab_triage_core.py`, tests in `tools/test_lab_triage.py`)
- Queues: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue=lab-v1.3/review/triage-confirm.json> (the both-mismatch items only, to hear first) and <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue=lab-v1.3/review/triage.json> (the full queue; earlier sets: `queue=lab-v1.2/review/triage.json`, `queue=lab-v1.1/review/triage.json`)
- Outputs: `/lab/data/<testSet>/triage-v1.json` (every case: both scores, decisions, outcome and spot flags), `review/triage.json` (the queue), `review/triage-confirm.json` (its both-mismatch items, maxBad 0, diagnostic only), `triage/` (per-spelling model records and the Nemotron transcripts). Summaries: `lab-triage-v1-lab-v1.1.json`, `lab-triage-v1-lab-v1.2.json`, `lab-triage-v1-lab-v1.3.json`.
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

Studio copy only. Both match means auto-accepted. Both mismatch means **queued for you to confirm**, never flagged or rebuilt automatically. One of each means disagree, and anything else is unsure; both are queued. From lab-v1.2 the spot check is a fixed seeded sample of the auto-accepted cases: 15 edits and repeats plus 15 clean windows (`SPOT_EDITS`, `SPOT_CLEAN`; `--spot-edits`, `--spot-clean`). The seam spot check is a seeded 10% of the planted edits (splices, repeats and restarts), at most 20 (`SEAM_SPOT`, `SEAM_CAP`). lab-v1.1 ran with a seeded 10% of the auto-accepted cases, which made 144 of its 257 items random spot checks, 118 of them clean windows both models had accepted; its 10% seam spot check had no cap. Verdicts on lab-v1.1 carry over to a lab-v1.2 item only when its clip bytes (the 16 kHz PCM the models hear) and text are identical (`--prior`). Each case appears once in the queue, with its reasons merged into the title (for example "Seam spot check · also a random spot check"). The bar is `maxBad = max(1, items // 40)`.

One deviation was settled on a two-session smoke run (p05, e05) before the full run: clean windows don't use deletions. A peaky CTC model can drop a clearly heard word for the cost of a few blank frames, which is less than one imperfect spelling costs. FastConformer preferred the shorter text on 3 of 19 clean deletions, by 13–21 nats, with the word present in its own greedy output. It did so on none of 32 substitutions and insertions. The margins did not change.

## Results on lab-v1.1

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

## Results on lab-v1.2

lab-v1.2 is lab-v1.1 rebuilt with the splice rules from your listens to its 7 both-mismatch cases (`lab-v1.2.md`). There were 1664 cases: 391 planted edits, repeats and restarts, and 1273 clean windows. 14 more clean windows from lab-v1.2's own v1 check were scored but not queued.

| Class | Cases | Auto-accepted | Both mismatch | Disagree | Unsure |
| --- | ---: | ---: | ---: | ---: | ---: |
| wrong_word | 45 | 43 | 0 | 1 | 1 |
| added_word | 170 | 117 | 9 | 33 | 11 |
| omitted_word | 30 | 30 | 0 | 0 | 0 |
| skipped_passage | 81 | 81 | 0 | 0 | 0 |
| mutashabihat | 25 | 24 | 0 | 1 | 0 |
| repeat | 23 | 15 | 0 | 6 | 2 |
| restart | 17 | 8 | 1 | 8 | 0 |
| clean | 1273 | 1267 | 0 | 4 | 2 |
| **all** | **1664** | **1585** | **10** | **53** | **16** |

Per model, Nemotron gave 1634 match, 22 mismatch and 8 unsure, and FastConformer 1601, 54 and 9.

**Queue: 128 items, maxBad 3** (lab-v1.1: 257 items).

| Why (first reason) | Items |
| --- | ---: |
| Both models think the label is wrong — please confirm | 10 |
| The two models disagree | 53 |
| A model couldn't tell — please check | 16 |
| Seam spot check | 19 (20 counting the one also queued for another reason) |
| Random spot check | 30 (15 edits and repeats, 15 clean windows) |

No verdict carried over. Of the 47 lab-v1.1 verdicts, none has a lab-v1.2 item with identical clip bytes and text, as expected after a rebuild.

The 10 both-mismatch cases are 9 added words and one restart (e04-x1). Every added word has at least one linked seam, but now at −7.5 dB or deeper, with donors of 0.32–1.72 s. That matches the kind you passed on lab-v1.1: four of its seven were fine added words, which suggests model bias against an inserted word.

Added words are still the weakest class: 53 of 170 are not auto-accepted, with FastConformer the stricter model (as on lab-v1.1).

On lab-v1.1's own cases, the new policy would have queued 129 instead of 257. That is its 83 model-queued items, plus 30 spot checks and 20 seam checks, 4 of which overlap.

## Results on lab-v1.3

lab-v1.3 is lab-v1.2 rebuilt with three more splice rules from your listens to lab-v1.2's queue (7 bad in 100, bar 3), plus a 30% cap on added words and skipped passages (`lab-v1.3.md`). There were 1700 cases: 375 planted edits, repeats and restarts, and 1325 clean windows. 14 more clean windows from lab-v1.3's own v1 check were scored but not queued. Same models, margins and spot policy as lab-v1.2.

| Class | Cases | Auto-accepted | Both mismatch | Disagree | Unsure |
| --- | ---: | ---: | ---: | ---: | ---: |
| wrong_word | 98 | 82 | 1 | 10 | 5 |
| added_word | 94 | 60 | 5 | 25 | 4 |
| omitted_word | 62 | 58 | 0 | 1 | 3 |
| skipped_passage | 60 | 60 | 0 | 0 | 0 |
| mutashabihat | 21 | 20 | 0 | 1 | 0 |
| repeat | 29 | 17 | 0 | 10 | 2 |
| restart | 11 | 7 | 0 | 4 | 0 |
| clean | 1325 | 1320 | 0 | 3 | 2 |
| **all** | **1700** | **1624** | **6** | **54** | **16** |

Per model, Nemotron gave 1674 match, 19 mismatch and 7 unsure, and FastConformer 1644, 47 and 9.

**Queue: 117 items, maxBad 2** (lab-v1.2: 128 items, maxBad 3).

| Why (first reason) | Items |
| --- | ---: |
| Both models think the label is wrong — please confirm | 6 |
| The two models disagree | 54 |
| A model couldn't tell — please check | 16 |
| Seam spot check | 12 (20 counting those also queued for another reason) |
| Random spot check | 29 (30; 15 edits and repeats, 15 clean windows) |

**Confirm queue: 6 items, maxBad 0** (`review/triage-confirm.json`). It holds the both-mismatch items of the main queue with the same ids. Verdicts are stored per test set and keyed by id, so the 6 you hear there count as heard in the main queue too. Hear it first: it is diagnostic, and the main queue's bar still decides.

Three of your 100 lab-v1.2 verdicts carried over on identical clip bytes and text: e15-x10 (lab-v1.2 e15-x12) and e19-x12, both queued and shown as heard, and p17-clean-0-0, not queued.

The 6 both-mismatch cases are 5 added words and one wrong word. Every one is linked on both seams, with cuts between −8.1 and −27.3 dB, and donors of 0.28–1.36 s from the host recording:

| Item | Said | Donor | Seam cuts (dB, out then in) |
| --- | --- | --- | --- |
| e03-x1 | رَبِّكَ added | 12:50:11, 0.76 s | −9.4 → −11.5, −15.1 → −9.4 |
| e09-x7 | كُنتُ added | 78:40:14, 1.36 s | −17.2 → −11.4, −27.3 → −17.2 |
| e10-x11 | بِكُلِّ added | 44:55:3, 0.64 s | −8.9 → −13.6, −16.1 → −8.9 |
| e13-x9 | قَبْلَ added | 50:39:8, 0.52 s | −12.2 → −14.9, −8.1 → −12.2 |
| e16-x9 | فِى added | 26:174:2, 0.28 s | −11.3 → −14.4, −12.5 → −11.3 |
| p11-x4 | كَانُوا۟ for ذَٰلِكَ | 26:6:6, 0.60 s | −9.1 → −8.7, −8.5 → −9.2 |

Added words are still the weakest class: 34 of 94 are not auto-accepted (lab-v1.2: 53 of 170). FastConformer is again the stricter model.

### Cost (lab-v1.3)

Nemotron transcribed 1714 clips (none reused) in 1712 s and scored them in 466 s, on 4 H100s with 133 s of model loads. FastConformer took 145 s plus a 7 s load on one T4. The wall clock was 13 min. That is about 39 min of busy H100 time, with 4 × 13 min held as an upper bound: roughly $2.5–3.5 at Modal list prices, plus $0.03 for the T4.

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

lab-v1.2: Nemotron transcribed 1678 clips (none reused) in 2149 s and scored them in 732 s, on 4 H100s with 164 s of model loads. FastConformer took 156 s plus an 8.5 s load on one T4. The wall clock was 15 min. That is about 51 min of busy H100 time, with 4 × 15 min held as an upper bound: roughly $3.5–4 at Modal list prices, plus $0.03 for the T4.

lab-v1.1:

| | GPU | Time |
| --- | --- | --- |
| Nemotron transcribe (1487 new, 103 reused) | 4× H100 | 1923 s |
| Nemotron score | 4× H100 | 548 s |
| Nemotron model loads | | 4 × ~41 s |
| FastConformer score | 1× T4 | 128 s, plus an 8 s load |

The wall clock was 13 min 15 s. Total H100 compute was about 44 min busy, with containers held about 55 min (an upper bound). The cost was roughly $3.5 for the H100s and $0.15 for the T4, plus about $0.5 for the two smoke runs. These are estimates from Modal list prices.

## Nemotron + Soniox on lab-v1.3 (2026-10-29)

Soniox `stt-async-v5` won the commercial API bake-off (`lab-review-asr-v1.md`, "Commercial APIs"). Here it replaces FastConformer as the second model.
- Its reading is the preregistered label-blind judge: the transcript's rasm distance to the claimed text against the alternative.
- Nemotron's decisions are reused as they are. The 1700 cases, windows and alternatives are asserted identical to `triage-v1.json`.
- The rule is unchanged: 15 + 15 random spot checks, and seam checks of 10% capped at 20, seed 17.
- `modal run modal/review_api.py::triage_main --vendor soniox` writes:
  - `/lab/data/lab-v1.3/triage-soniox-v1.json`;
  - `review/triage-soniox.json`;
  - `review/triage-soniox-confirm.json`;
  - `docs/benchmarks/lab-triage-soniox-lab-v1.3.json`.
- `triage.json` is untouched.

| Outcome | Nemotron + FastConformer | Nemotron + Soniox |
| --- | ---: | ---: |
| Auto-accepted | 1624 | 1559 |
| Both mismatch | 6 | **0** |
| Disagree | 54 | 19 |
| Unsure | 16 | 122 |

**Queue: 184 items, maxBad 4** (was 117, maxBad 2). 7 of your verdicts carry over.

| Why (first reason) | Old | New |
| --- | ---: | ---: |
| Both models think the label is wrong | 6 | 0 |
| The two models disagree | 54 | 19 |
| A model couldn't tell | 16 | 122 |
| Seam spot check | 12 | 14 |
| Random spot check | 29 | 29 |

**Confirm queue:** `review/triage-soniox-confirm.json` is empty, because Soniox never says mismatch on lab-v1.3 (1585 match, 115 unsure).
- Your 6 old confirm items, all of which you heard as ok, are still queued, now as disagree.
- On each, Nemotron says mismatch (−2.2 to −21.9 nats) and Soniox hears the label, for example e09-x7 "لا يذوقون. كنت فيها بردًا ولا".
- No false alarm reaches the confirm queue. On all 19 of Nemotron's mismatches, Soniox says match.

**Why the queue grew.** 113 of Soniox's 115 unsures are "too close to call" under the preregistered margin:
- 56 skipped passages: 56 of 60 are unsure, because the margin assumes the whole skipped passage separates the texts (see the bake-off doc);
- 44 clean windows: one-word substitutions that differ by only a few rasm letters;
- 11 wrong words, 5 mutashabihat and 4 omitted words.

**Overlap with the old queue:**
- 55 items are in both queues, 62 only in the old one and 129 only in the new one.
- Of the old pairing's 76 model-queued items, the new pairing auto-accepts 45, and calls 19 disagree and 12 unsure.
- Seam spot ids are the same, because they are seeded on the same planted list. Random spot items differ, because the auto-accepted set differs.

**Post hoc (tuned on this data, not used for the queue):** with the attainable-gap margin, lab-v1.3 would have 1631 auto-accepted, 19 disagree and 50 unsure. That is about 112 items with spot checks: roughly the old queue's size.

**Read.**
- On your verdicts, Soniox pairs better than FastConformer: 96 of 142 ok auto-accepted against 55. But it lets through one bad splice (p15-x1) that FastConformer's unsure held.
- On lab-v1.3, the preregistered queue is bigger, not smaller. The verdict set is enriched for items the old pairing queued, so it overstates the gain.
- The disagree bucket is now exactly "Nemotron hears a seam problem, Soniox hears the right words". That is a cleaner signal for bad splices than before.
- Hear `triage.json` (117), not this queue, unless the margin is fixed and re-preregistered.

**Cost:**
- Soniox transcribed 1714 clips (3.77 h) for $0.38 at list price, in 27 min wall time.
- 32 concurrent requests; chunk times step in about 70 s units, consistent with queueing or rate limits on Soniox's side.
- No GPU: Nemotron's scores are reused from `triage-v1.json`.
