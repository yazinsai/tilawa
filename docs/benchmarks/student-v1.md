# Student distillation v1 (E2, E5)

Ticket 08. Streaming Zipformer2 phoneme CTC, 25 Hz output (same frame rate as the teacher), distilled from `obadx/muaalem-model-v3_2` on the tlog-v2 `clean-match` train split (535 h after dropping clips over 30 s), with noise, RIR, gain and SpecAugment. Code: `modal/student_prep.py`, `modal/student_train.py`, `tools/student.py`.

Eval: tlog-v2 `clean-match` test split, held out by speaker cluster, 3,000 clips. Greedy decode with a streaming chunk mask (chunk 16 frames, left context 128). The teacher's own PER against the reference on the same clips is **8.5%**.

| run | exp | size | params | init | objective | PER vs ref | PER vs teacher | scripted-error |
|---|---|---|---|---|---|---|---|---|
| e5-top1-lite | E5 | lite | 22.3M | scratch | top path (CTC on teacher greedy) | 11.8% | 7.7% | — |
| e5-soft5-lite | E5 | lite | 22.3M | scratch | top path + soft top-5 (weight 1) | 10.4% | 6.2% | — |
| e2-scratch-full | E2 | full | 64.6M | scratch | top path | 12.1% | 7.9% | — |
| e2-libri-full | E2 | full | 64.6M | LibriSpeech streaming Zipformer-M (lr 0.015) | top path | 9.9% | 5.5% | — |

The scripted-error column is empty for now. `transcribe` in `modal/student_train.py` writes student evidence in the teacher-pass row format, so ticket 07's acceptance scorer can read it.

## Readout

- **E5:** soft top-5 targets beat top path only by 1.4 points (11.8% → 10.4%) at the same size, data and steps.
- **E2:** initialising from a pretrained model helps `full` by 2.2 points (12.1% → 9.9%). That is the best run, 1.4 points above the teacher's 8.5%. The scratch `full` run did no better than scratch `lite` on this schedule: 8 epochs is too few for 65M parameters trained from scratch.
- The bar: Quran-Lab's 3.65% PER on phone audio is on a different test set (its tlog_holdout is excluded from our data), so it can't be compared directly. Against our own reference labels even the teacher scores 8.5%, because those labels carry noise (bucket cutoff PER ≤ 0.12). PER vs teacher is the cleaner measure of distillation quality: 5.5% for the best run.

## Deviations

- E2's init arm uses English LibriSpeech streaming Zipformer-M (`Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`), not the multilingual sherpa-onnx model. The only multilingual one with Arabic (PengChengStarling) has about 330M parameters, which fits neither tier.
- The init arm ran at base LR 0.015, against 0.045 for scratch. At 0.045 it diverged to NaN at step ~440.
- Training used tlog only. QUD, everyayah and the other sources still need their own teacher passes.
