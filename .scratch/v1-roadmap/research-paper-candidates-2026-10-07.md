# Paper candidates for Tilawa — 7 October 2026

Primary-source research. No training or evaluation was run. These are hypotheses for Tilawa; paper scores use different datasets and definitions and cannot be compared directly with Tilawa's passage-checker precision. Latest local baseline and pending combined-student run are established separately by the coordinating researcher.

## 1. Stage synthetic mistakes first, then adapt on real mistakes

[Yang et al., 23 June 2026](https://arxiv.org/html/2606.24086v1) train on about 79 hours native Arabic plus 80 hours synthetic mistakes, then adapt on about 2 hours real learner recordings. QuranMB.v2 F1: stage 1 alone 46.29%, naive mixing 43.05%, real-only 66.81%, staged single model 68.25%, six-model ensemble 72.01%. The last ensemble uses a small sequence prior built from its own predictions, not expected verse text. XLS-R-300M is too large to count as a phone solution; causality of the added convolution head does not make its encoder streaming.

**Tilawa experiment:** revise E3 into a controlled comparison: same student and fixed compute, clean-only versus 5,000 vetted audio-splice mistakes versus those mistakes followed by a short adaptation on 200–300 separately labelled learner clips. Keep evaluation speakers and passages separate. Record substitutions, omissions, vowels and shadda independently. Proposed gate: at least 5 percentage points precision gain, recall loss no more than 2 points; confirm on a larger set before shipping. Medium effort. Synthetic seam artifacts and small real-data overfitting are risks. This is additional work beyond clean soft5 and expected-middle targets.

Resources: [Iqra_TTS](https://huggingface.co/datasets/IqraEval/Iqra_TTS), 55,369 rows/5.94 GB, empty README; [real learner Extra](https://huggingface.co/datasets/IqraEval/Iqra_Extra_IS26). Audit labels, speaker identities and use permissions before importing.

## 2. Check that expected text is not erasing real mistakes

[Geng et al., 24 April 2026](https://arxiv.org/html/2604.22133v1) preserve dense acoustic evidence and use error information during training rather than expected-text prompts during inference. L2-ARCTIC: 2,429 training, 268 validation, 900 test utterances from six test speakers. Adding canonical prompts to their LLaMA variant drops recall from 59.08% to 28.83%, although PER improves from 15.85% to 13.55%. Their Arabic result is precision 73.25%, recall 70.20%, F1 71.70%, PER 3.72%; training uses synthetic speech then real-error adaptation. English full-system PER is 46.52% despite strong detection F1, so it is not a drop-in listener replacement.

**Tilawa experiment:** make a 120-item challenge pack: 30 correct controls, 30 substitutions, 30 omissions, 30 vowel/shadda mistakes, balanced across speakers. Compare clean soft5 and expected-middle targets; flag every case where the latter restores expected text over audible errors. Proposed gate: no more than 2 extra missed mistakes per category for any precision improvement. Low effort for the audit, high effort for replacing CTC with the paper's dense loss. This directly tests the risk of the already-running expected-middle experiment. [Code](https://github.com/Secondtonumb/IF-MDD).

## 3. Let streaming teacher targets move slightly in time

[Delayed-KD, 28 May 2025](https://arxiv.org/html/2505.22069v1) handles teacher/student emission timing differences by selecting the best target match within a short delay window. AISHELL-1: 178 training hours, 7,176 test utterances. At 40 ms chunks, direct distillation streaming character error is 7.02%; an 80 ms matching window gives 6.11%. The often-quoted 5.42% is utterance rescoring, not immediate streaming output. Median first-token delay rises from 100 to 140 ms. It also tests 10,000-hour WenetSpeech, approximately 33,100 test utterances.

**Tilawa experiment:** first inspect 200 held-out clips for systematic teacher/student phoneme peak offsets. Only if present, compare identical soft5 students with 0, 40 and 80 ms target buffers using a 10,000-clip clean subset. Proposed gate: at least 5% relative phoneme-error reduction without lower real-mistake recall; report first-warning P50/P90 latency and low-end-phone processing time. Medium effort, no added inference model size. It corrects timing, not wrong teacher labels. Unlike current soft5 training, it explicitly tests alignment mismatch; skip if the current student is not causally restricted.

## 4. E6 has support, but benefits are smaller than headline results

[Multi-View Multi-Task, 2 June 2023; revised 7 August 2023](https://arxiv.org/html/2306.01845v2) adds tasks for manner, place and vowel position. L2-ARCTIC uses 12 training, 6 validation and 6 test speakers. In the simple single-encoder comparison, auxiliary tasks improve precision 54.94%→58.30%, recall 53.90%→54.05%, PER 15.91%→14.60%. With the multilingual encoder, F1 barely changes, 57.82%→57.89%. The best 60.31% F1 combines two encoders and sequential auxiliary tasks; the whole gain cannot be attributed to the auxiliary head.

**Tilawa experiment:** add one training-only head for a carefully checked Arabic feature mapping (emphasis, voicing, vowel length, gemination), rather than two deployed encoders. Compare one fixed clean subset with and without the head. Proposed gate: 3 points checker precision improvement at unchanged recall, plus better confusion rates for the targeted contrasts. Medium effort. Features inherited from noisy teacher labels can reproduce its mistakes; begin with verified feature labels. This supports E6 as a bounded ablation, not as a demonstrated fix.

## 5. Update the Quran teacher/data reference to its August 2026 revision

[Abdelfattah et al. v2, 27 August 2026](https://arxiv.org/html/2509.00094v2) reports 848 hours, 286,537 utterances from 22 expert reciters; 11 output levels include phonemes and articulatory properties. Its 0.21% average PER on expert test speech is not learner error detection. The learner benchmark has 159 recordings of one verse, 120 female and 39 male reciters, 106 with errors and 53 correct; Tajweed F1 is 75.8%. The abstract's 1.94% learner average PER combines levels; the table's phoneme-only PER is 5.78%. All-expert training is male; single verse and single annotator limit generalization.

**Tilawa experiment:** use [qdat_bench](https://huggingface.co/datasets/obadx/qdat_bench) as a secondary out-of-domain test after checking overlap and mapping labels. Do not train on that test. Report by rule, gender, and correct versus incorrect recitation. Low/medium effort. This audits transfer beyond tlog rather than promising a new listener. The author has newer [streaming RNN](https://huggingface.co/obadx/muaalem-streaming-rnn-v0) and [mini](https://huggingface.co/obadx/muaalem-model-v3-mini) repositories, but their cards are autogenerated, with no usable training, size, latency or evaluation claims. Treat as inspection leads only.

## 6. Retrieval provides an independent acoustic cross-check, with insertion trouble

[Tu et al., 25 November 2025](https://arxiv.org/html/2511.20107v1) compares incoming acoustic frames with labelled examples instead of training a new phoneme recognizer. English L2-ARCTIC uses 12 training speakers and six test speakers; a pool from 500 labelled clips gives precision 71.78%, recall 67.56%, F1 69.60%, false-rejection 4.43%. However PER is 104.08%, largely because of insertions excluded from much of its detection scoring. With 1,800 clips, precision rises to 77.49% while recall falls to 62.83%. It uses a large English HuBERT model and phoneme time labels.

**Tilawa experiment:** use cached teacher embeddings from 200 manually aligned Arabic clips to examine 50 false warnings and 50 real errors. Test whether nearest examples can distinguish a decoding glitch from actual acoustic replacement. Proposed gate: independent evidence supports at least 20% of false-warning removals without removing any verified mistakes in the pilot. Medium effort; alignment and large embeddings make this an offline diagnostic first. Lower priority than staged adaptation or timing checks; do not use its F1 to justify a listener replacement.

## Practical order

1. Finish the current model's held-out checker scoring, then run the 120-item expected-text audit. This prevents optimizing precision by quietly erasing errors.
2. Revise E3 into staged synthetic-to-real adaptation, with matched-compute controls and a genuinely separate evaluation set.
3. Inspect timing on 200 clips before spending a run on Delayed-KD. If timing is already aligned, move to the small E6 head ablation.

All proposed success thresholds and sample sizes are planning choices, not measured Tilawa gains. Run the project scoreboard whenever an experiment changes listener/checker/evaluation behavior.
