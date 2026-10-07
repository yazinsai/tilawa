# Combined student: mistake-detection evaluation v1

Listener `combo-libri-soft5-full`; checker/reference commit `72d1f04` / `ref-v2`. Scored 7 October 2026. Existing CPU fp32 student evidence, chunk 16 / left context 128, replayed in 1-second chunks. No retraining, decoding, or threshold tuning.

388 scored help evaluation clips from 29 held-out speakers; 413 total, 25 excluded by the import/audit. There are 64 confirmed memorization errors (29 wrong words, 35 omitted words). Letter-confusion substitutions count as pronunciation negatives. Repeats are non-errors. The same labels, checker code and default policies apply to all three listeners below; the checker and scoring implementation have not changed between baseline commit `bfc52cb` and this evaluation.

## Mistake detection

Declared passage gives the checker the intended passage. Free recitation follows the tracker without an intended passage.

| Listener | Arm | Hits | False warnings | Misses | Precision | Recall | Clean warnings / 225 clips |
|---|---|---:|---:|---:|---:|---:|---:|
| teacher | Declared passage | 57 | 17 | 7 | 77.0% | 89.1% | 6 |
| teacher | Free recitation | 32 | 10 | 32 | 76.2% | 50.0% | 2 |
| e2-libri-full | Declared passage | 53 | 26 | 11 | 67.1% | 82.8% | 18 |
| e2-libri-full | Free recitation | 30 | 11 | 34 | 73.2% | 46.9% | 7 |
| combo-libri-soft5-full | Declared passage | 54 | 12 | 10 | 81.8% | 84.4% | 5 |
| combo-libri-soft5-full | Free recitation | 31 | 6 | 33 | 83.8% | 48.4% | 2 |

The combined student improves declared precision to 81.8% but misses 10 errors, versus 7 for the teacher. It meets the 80% recall bar and fails the 90% precision bar. There are 5 false warnings over 46.24 minutes of clean student evidence (0.54 per five minutes), and none on 23 repeat clips. These short clips cannot establish the long-session warning gate. Small duration differences from the teacher reflect listener frame counts, not different evaluation clips.

Declared wrong words: 23 hits, 8 false warnings, 6 misses (74.2% precision, 79.3% recall). Omitted words: 31 hits, 1 false warning, 4 misses (96.9% precision, 88.6% recall). Added words and confirmed skips have no positive labels, so recall cannot be measured; they contribute 1 and 2 false warnings respectively.

The audit-verified subset has 59 clips from 21 speakers: 11 hits, 2 false warnings, no misses (84.6% precision, 100% recall), identical aggregate results to the teacher on that small subset.

## Listener accuracy and overlap

| Listener | Gold canon PER (233 clips) | Uncontaminated gold PER (62 clips) |
|---|---:|---:|
| teacher | 3.39% | 7.27% |
| e2-libri-full | 5.98% | 10.62% |
| e5-soft5-lite | 5.94% | 9.95% |
| combo-libri-soft5-full | 4.82% | 9.19% |

Gold uses canon PER against the existing fixed spliced reference; raw combined gold PER is 6.69%. All 233 gold and 211 silver evidence files were present. The earlier combined model was trained before gold exclusions: 171 tlog gold clips overlap its training. The 62-clip subset (20 tlog validation/test plus 42 audited help clips) excludes that overlap. No help clip was in the student training manifests. The full gold result is optimistic; the held-out subset still improves on previous students. The exploratory 3.65% full-gold bar and calibration are unchanged; the combined model does not meet it.

The combined training log reports 8.74% raw PER against reference and 4.29% against the teacher on 3,000 tlog clean-match test clips. These are phoneme metrics, distinct from mistake precision/recall.

Short-vowel greedy evidence preserves 13/30 planted errors (43.3%), versus teacher 19/30 (63.3%); posterior preference preserves 18/30. This remains a separate limitation despite better memorization precision. The pronunciation checker remains unsupported.

Clean tracker acquisition is 190/225 within five seconds (84.4%, below 95%); no clip has a false jump. Short individual clips do not test following real long sessions.

## Evidence and reproduction

- [Per-clip mistake results](acceptance-combo-libri-soft5-full-v1.json)
- [Short-vowel evidence results](acceptance-combo-libri-soft5-full-v1-e1.json)
- [Gold phoneme scores](gold-per-v1-scores.json)
- [Scoreboard](scoreboard.json)
- [Teacher acceptance](acceptance-v3.md)
- [Previous students and calibrated listener bar](listener-target-v1.md)

From `packages/core`, run `node --import tsx src/bench/scoreboard-cli.ts --listener combo-libri-soft5-full --evidence ../../.scratch/gold/combo-libri-soft5-full --modules listener --note "Score the completed pretrained soft-target full student on existing held-out help evidence to measure mistake detection."`.

Per-clip report: the existing acceptance CLI was run against `.scratch/gold/combo-eval-input`, which links the unchanged help labels/raw/audit files and the combined evidence at each label’s original wav path. Gold scores use `tools/gold_set.py`’s existing `score` and `decoded` functions on the frozen gold manifest, appending only this listener to the scores file. Short-vowel scoring uses `tools/e1_report.py --help-clips` with `EVIDENCE_DIR` set to the combined evidence and a listener-specific `E1_OUT`.

The separate `combo-clean` and `spliced-clean` training runs are not included in this report.
