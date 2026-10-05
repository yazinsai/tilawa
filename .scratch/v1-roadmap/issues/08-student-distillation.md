# 08: Student distillation

**What to build:** The `lite` and `full` streaming Zipformer phoneme-CTC students, distilled from the teacher, with the canonicalization ablations.

**Blocked by:** 01, 02, 06.

**Status:** needs-triage

- [ ] E2: initialise from the multilingual sherpa-onnx streaming Zipformer versus training from scratch, on `clean-match` plus the other training sources, with augmentation.
- [ ] E3: spliced errors from QUD letter timestamps at 10–20% of each batch.
- [ ] E5: soft top-5 posteriors versus top path only.
- [ ] E6: articulatory-attribute auxiliary head.
- [ ] Each ablation scored on PER (bar: 3.65% on phone audio) and on the ticket 07 scripted-error test.
- [ ] Sizes informed by ticket 02.
- [ ] If E1 left classes `unsupported`, evaluate per-pair classifiers on tracker-aligned segments as a separate arm (speculative).
