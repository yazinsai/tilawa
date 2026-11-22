# 08: Student distillation

**What to build:** The `lite` and `full` streaming Zipformer phoneme-CTC students, distilled from the teacher, with the canonicalization ablations.

**Blocked by:** 01, 02, 06.

**Status:** done

- [x] E2: initialise from the multilingual sherpa-onnx streaming Zipformer versus training from scratch, on `clean-match` plus the other training sources, with augmentation.
- [ ] E3: spliced errors from QUD letter timestamps at 10–20% of each batch.
- [x] E5: soft top-5 posteriors versus top path only.
- [ ] E6: articulatory-attribute auxiliary head.
- [ ] Each ablation scored on PER (bar: 3.65% on phone audio) and on the ticket 07 scripted-error test.
- [x] Sizes informed by ticket 02.
- [ ] If E1 left classes `unsupported`, evaluate per-pair classifiers on tracker-aligned segments as a separate arm (speculative).

## Comments

**Agent, 2026-10-06. Cost estimate before the E2/E5 runs.** Measured on smoke runs (one H100, 16 CPU): `lite` (22.3M params) trains at ~1,500× real time and `full` (64.6M) at ~1,150×. At about $5.5/h (H100 + CPU + 128 GB RAM), each run on the ~560 h `clean-match` train split costs:

| run | size | epochs | est. GPU h | est. cost |
|---|---|---|---|---|
| e5-top1-lite | lite | 10 | ~3.7 | ~$20 |
| e5-soft5-lite | lite | 10 | ~3.7 | ~$20 |
| e2-scratch-full | full | 8 | ~4 | ~$22 |
| e2-libri-full | full | 8 | ~4 | ~$22 |

Data prep (743 shards to 16 kHz FLAC, plus noise/RIR bank) and four smoke runs cost about $10. Every run is under the $50 line, so they're launched.

**Agent, 2026-10-07. E2 and E5 done; see `docs/benchmarks/student-v1.md`.** PER vs reference on tlog-v2 clean-match test (3,000 clips; the teacher scores 8.5%):

| run | PER |
|---|---|
| e5-top1-lite | 11.8% |
| e5-soft5-lite | 10.4% |
| e2-scratch-full | 12.1% |
| e2-libri-full | 9.9% |

Soft top-5 targets help, and so does pretrained init. E2 initialised from LibriSpeech-M because the multilingual PengChengStarling model is ~330M parameters. Spend was about $75: four runs ≈ 11 GPU-h ≈ $61, plus prep, smoke runs and one diverged init run.

The scripted-error column is still empty. `transcribe` (L4) writes student evidence in the teacher row format for ticket 07's scorer. `tools/e1_report.py --e1b --in <dir> --model <run> --out <stem>` scores a student on the E1b spliced set. E3 and E6 have not started. Proposed E3 design: splice letter pairs and shadda within the same tlog train clip, using teacher alignment, with only the CTC loss on spliced clips.

**Agent, 2026-10-07. Ticket 08 student: `spliced-clean`; see `docs/benchmarks/student-v2.md`.** Both runs were retrained with the gold PER clips excluded (225,401 train clips) and scored on gold PER, PER on clean-match, E1b and the help eval split:

| run | gold PER | gold, never in train (62) | fair PER clean-match | declared P / R | E1b canonicalised |
|---|---|---|---|---|---|
| teacher | 3.39% | 7.27% | 2.44% | 77.0% / 89.1% | 34.5% |
| combo-clean (teacher-greedy target) | 4.97% | 9.45% | 4.14% | 75.0% / 79.7% | 34.0% |
| spliced-clean (spliced-reference target, KD 2) | 4.71% | 8.80% | 3.87% | 83.6% / 87.5% | 41.7% |

Neither run passes the 3.7% gold bar. Spend was about $40: two H100 runs ≈ 7 GPU-h, plus CPU decodes and ~3 min lost to two launches that got cancelled.

Proposal: defer E3 and E6 past v1. The teacher-target students canonicalise no more than the teacher (34.0–34.2% vs 34.5%), and v1's checker has no pronunciation classes, so neither experiment moves a v1 bar. One caveat for the human: the chosen spliced-target student does canonicalise more (41.7%), which is the thing E3 is meant to fix. If post-v1 pronunciation work builds on this student, E3 comes first. Status stays claimed until a human confirms.

- **Closed, 2026-10-07 (maintainer):** `spliced-clean` ships as the v1 student (`docs/benchmarks/student-v2.md`). E3 and E6 are deferred until after v1, because v1 is memorization-only (ticket 05 deferred). If pronunciation work builds on this student, run E3 first, because its canonicalised share is 41.7% against the teacher's 34.5%. The target-vs-KD-weight confound is left unresolved.
