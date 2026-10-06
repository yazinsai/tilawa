# 08: Student distillation

**What to build:** The `lite` and `full` streaming Zipformer phoneme-CTC students, distilled from the teacher, with the canonicalization ablations.

**Blocked by:** 01, 02, 06.

**Status:** claimed

- [ ] E2: initialise from the multilingual sherpa-onnx streaming Zipformer versus training from scratch, on `clean-match` plus the other training sources, with augmentation.
- [ ] E3: spliced errors from QUD letter timestamps at 10–20% of each batch.
- [ ] E5: soft top-5 posteriors versus top path only.
- [ ] E6: articulatory-attribute auxiliary head.
- [ ] Each ablation scored on PER (bar: 3.65% on phone audio) and on the ticket 07 scripted-error test.
- [ ] Sizes informed by ticket 02.
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
