# Listener target v1: gold PER and the calibrated bar

**Bar: a listener passes at gold PER ≤ 3.65%** (canon PER vs spliced reference on the gold set). That is where the fitted declared-arm memorization precision drops 2 points below the teacher's. The teacher is at 3.39%, and the best students are at 5.9–6.0%, so no student passes yet. Data: `gold-per-v1.json` (manifest), `gold-per-v1-scores.json` (per-listener scores), `listener-target-v1.json` (fit). Script: `tools/gold_set.py`.

## Gold set

| slice | source | clips | device | speakers |
|---|---|---|---|---|
| gold | ticket 06 bucket review, judged `exact` | 191 | tlog (crowd app audio; device unknown, mostly phones) | 48 clusters |
| gold | ticket 12 help clips, audit verdict 1 (label right) and kind `none` | 42 | 26 phone, 16 laptop | 26 |
| silver | help eval clips of kind `none` that were never audited | 211 | 149 phone, 62 laptop | 25 |

None of the clips is studio audio. tlog is crowd app recordings, and help is phone and laptop, so the split below is tlog vs phone vs laptop.

**Training overlap.** 171 of the 191 tlog gold clips are in tlog-v2 `train` (153 clean-match, 18 deviant). The other 20 are in val or test. All four students trained on them, with teacher targets. `gold-per-v1-exclude.txt` lists all 191, and `load_clips` in `modal/student_train.py` drops them from every train split, but not from val or test, so logged PER stays comparable. The ticket 08 combo run that is training now predates the exclusion. No help clip is in any student manifest. The `goldHeldOut` slice in the scores (62 clips: the 20 tlog val/test clips plus the 42 help clips) is uncontaminated, and it ranks the listeners the same way.

## Metric

The metric is canon PER vs the spliced reference, exactly as in `teacher-per-breakdown-v1.md`: `canon` (ں→ن, ۾→م, drop ڇ, collapse runs) on both sides. The reference is the label's phonemes between the teacher's first and last 3-phoneme anchors, with the teacher's own phonemes outside them. The reference is fixed per clip. Because the clips are human-verified exact, every interior edit is a listener error, not a recitation error. A phoneme-group PER counts canon edits that touch a group phoneme (on the reference side for sub/del, on the hypothesis side for ins/sub) per group phoneme in the reference. Raw PER vs the spliced reference is secondary.

## Gold PER

Students were decoded greedily, chunk 16 / left 128, fp32 on CPU (`modal/student_decode_cpu.py::gold`). The teacher column uses the teacher pass greedy (tlog pass1, help evidence).

| listener | gold canon | raw | tlog (191) | phone (26) | laptop (16) | pharyngeals ح ع | emphatics ص ض ط ظ | hamza/ain ء ع | short vowels | held-out (62) | silver (211) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher | **3.39%** | 5.23% | 1.77% | 9.12% | 7.61% | 5.45% | 11.43% | 5.11% | 3.20% | 7.27% | 3.01% |
| e5-soft5-lite | 5.94% | 7.88% | 4.31% | 10.73% | 12.18% | 9.61% | 20.71% | 8.88% | 4.98% | 9.95% | 4.48% |
| e2-libri-full | 5.98% | 7.70% | 4.17% | 11.63% | 12.18% | 10.91% | 15.71% | 9.15% | 4.91% | 10.62% | 4.19% |
| e5-top1-lite | 7.21% | 9.14% | 5.31% | 12.63% | 14.84% | 10.39% | 25.00% | 11.98% | 6.12% | 11.74% | 5.65% |
| e2-scratch-full | 7.30% | 9.12% | 5.57% | 12.58% | 13.61% | 11.95% | 28.57% | 12.25% | 5.81% | 11.51% | 5.99% |

Group sizes in the reference: 385 pharyngeals, 140 emphatics, 743 hamza/ain, 5,340 short vowels. Emphatics carry about ±3 points of noise per listener.

The teacher's tlog PER is low partly by construction: these clips were bucketed on teacher-vs-label PER, and the reference outside the anchors is the teacher's own output. The audited help clips are harder. They are learner audio from phones and laptops, and the audit sample over-weighted rare kinds. The teacher is at 7–9% on them and the students at 11–15%. Silver tracks gold closely, with the same order and slightly lower numbers.

## Calibration

These are end-to-end numbers on the 388 scored help eval clips (29 held-out speakers), from `pnpm scoreboard --listener <run> --evidence .scratch/gold/<run>`. Short vowel recall is the E1 greedy recall at the 30 planted sites (`tools/e1_report.py --help-clips` with `EVIDENCE_DIR`).

| listener | gold PER | declared recall | declared precision | free recall | free precision | short vowel recall |
|---|---|---|---|---|---|---|
| teacher | 3.39% | 89.1% (57/64) | 77.0% (57/74) | 50.0% | 76.2% | 63.3% (19/30) |
| e5-soft5-lite | 5.94% | 84.4% (54/64) | 62.8% (54/86) | 45.3% | 65.9% | 40.0% (12/30) |
| e2-libri-full | 5.98% | 82.8% (53/64) | 67.1% (53/79) | 46.9% | 73.2% | 56.7% (17/30) |
| e5-top1-lite | 7.21% | 84.4% (54/64) | 50.9% (54/106) | 43.8% | 53.8% | 60.0% (18/30) |
| e2-scratch-full | 7.30% | 85.9% (55/64) | 44.7% (55/123) | 45.3% | 43.9% | 60.0% (18/30) |

Least-squares fits against gold PER, over 5 points:

| metric | slope per PER point | r | PER at teacher − 2 pts |
|---|---|---|---|
| declared precision | −7.6 pts | −0.93 | **3.65%** |
| declared recall | −1.0 pts | −0.67 | 5.39% |
| short vowel recall | −0.8 pts | −0.14 | 5.84% (no signal; not used) |

Listener errors show up as false warnings, not as missed mistakes. Declared recall is flat at 83–86% for every student, because the checker still finds planted word errors through moderate phoneme noise. Precision falls about 7.6 points per point of gold PER, because each misheard phoneme in a clean passage can raise a wrong_word. So precision sets the bar. The bar uses only fits with r ≤ −0.5 and takes the tightest.

**Caveats.** These are speculative, so weigh them accordingly.
- There are 5 points, and the bar sits between the teacher and the nearest student, 2.5 PER points away. It is an interpolation along a line nobody has sampled between 3.4% and 5.9%.
- A 2-point precision change is about 1.5 false positives in 74 raised events. That is below the resolution of the help eval split. The bar is a direction, not a precise threshold.
- Short vowel recall at n = 30 moves 3.3 points per clip. e5-soft5-lite has the lowest PER and the worst short vowel recall, which looks like a real KD-softening effect (posterior-only), not noise. Gold PER won't catch it, so short vowel recall stays a separate listener metric.
- 171 tlog gold clips were in student training. On the held-out 62, the ranking is the same and the student−teacher gap is wider (2.7–4.5 points vs 2.5–3.9).

## Scoreboard

`listener.goldPer` is now the primary listener PER, gated at ≤ 3.65%. It has per-device and per-group breakdowns and gold raw PER as info. The clean-match fair PER from `teacher-per-breakdown-v1` stays as info. It replaces the old teacher floor + 1 pp bar (`STUDENT_MARGIN`).

Reproduce:

    G="uv run --python cpython-3.12-macos-aarch64-none tools/gold_set.py"
    $G build
    modal run modal/student_decode_cpu.py::gold --main-checkout <repo>
    $G score
    # per run: pnpm scoreboard --listener <run> --evidence <repo>/.scratch/gold/<run> --note '...'
    # per run: EVIDENCE_DIR=<repo>/.scratch/gold/<run> E1_OUT=<repo>/.scratch/gold/e1/<run>.json uv run ... tools/e1_report.py --help-clips
    $G calibrate

The Modal spend was one CPU pass, 16 containers × 8 cores × ~5 min, about $1.
