# Tilawa listener research — 7 October 2026

Research only; no training, labels, code, or benchmark metrics changed. Recommendations are hypotheses to test, not promised improvements.

## Current evidence checked

Read AGENTS.md, PLAN.md, roadmap spec, acceptance-v3, student-v1, listener-target-v1, and ticket08 checkout `/Users/rock/ai/projects/tilawa-t08`.

- Four published students use approximately 535 hours of tlog audio. On 3,000 clean-match test clips, reference phoneme error ranges 9.9–12.1%; teacher 8.5%. These reference labels contain noise.
- On 388 scored help clips from 29 held-out speakers, declared teacher checker has 57 hits, 17 false warnings, 7 misses: precision 77.0%, recall 89.1%. About 11 of 17 false warnings were attributed to decoder noise. Students retain 82.8–85.9% recall but precision drops to 44.7–67.1%.
- Soft targets help recognition but the lite soft model detects only 12/30 planted vowel errors, versus teacher 19/30. A listener must pass an error-preservation test separately from clean recognition.
- Gold has 233 clips, but 171 overlap older student training; the uncontaminated comparison has 62 clips. Teacher held-out error 7.27%, best earlier student 9.95%. Reference uses teacher at clip edges, so even gold is not fully independent as-said transcription.
- Combined pretrained+soft5 outputs exist locally: 627 evidence files. End-to-end scores were not published in checked benchmark reports, and no combo E1 report appeared locally. Treat the coordinator's 8.74% reference error result as provisional context, not a new independently verified checker result. Clean combo and reference-middle training are already underway according to the coordinator; do not duplicate them.
- Tracker clean-start acquisition in acceptance-v3 is 191/225 (84.9%) within 5 seconds. The coordinator's newer approximately 86% is plausible but was not found in this published report; target remains 95%.

Sources: [acceptance-v3](../../../docs/benchmarks/acceptance-v3.md), [listener-target-v1](../../../docs/benchmarks/listener-target-v1.md), [student-v1](../../../docs/benchmarks/student-v1.md). Roadmap spec's opening status is stale.

## Three priority experiments

1. **Check whether a warning is supported by the audio, using existing outputs.** Compare the intended short word sequence with the actual alternative, allowing several possible timing alignments. Test teacher and best combo separately. Tune on training/development speakers only, then freeze thresholds for the 388-clip evaluation. Initial success: remove at least 5 of the teacher's 17 false positives while keeping at least 55/64 true errors (recall 85.9%; precision at 55 hits/12 FP is 82.1%). This is a useful intermediate win, below the 90% shipping gate. Aim for at most 6 FP with at least 55 hits to exceed 90%. Do not simply suppress first words or force the intended passage into listener output.
2. **Teach actual mistakes in stages.** After current clean combo results, compare one additional fine-tuning arm mixing 10% synthetic substitutions/omissions with an arm using the same synthetic phase followed by a short real-learner adaptation phase. Use 2,000–5,000 synthetic training examples, plus 100–300 reviewed real training-speaker clips if available. Keep the existing help eval speakers entirely out. Success: fewer false warnings, no more than two lost word-error hits, and at least teacher-level 19/30 vowel detection on the frozen test. Those 30 sites cannot establish a shipping-quality pronunciation claim.
3. **Align teacher and streaming student timing before strengthening soft targets.** Pilot 0/40/80/120 ms target alignment allowances on 20–50 training hours with identical steps and data, then take one winner to full data. Success: at least 0.5 percentage-point held-out phoneme-error reduction and at least five fewer checker false positives, without losing more than two word-error hits or one vowel hit. Measure actual total device delay; a target-alignment buffer is not automatically added audio lookahead.

These sizes and success criteria are proposed Tilawa choices, not paper findings. Start with experiment 1; it needs no new full training run. Experiments 2 and 3 address different failures and should be tested separately.

## Evidence and resources

Implementation ranking: staged learner adaptation (entry 3), warning confirmation (entry 1), timing-aware distillation (entry 5), expected-text audit (entry 4, a required safety check for the running arm), training-only attributes (entry 6), external Arabic second opinion (entry 2), updated teacher benchmark (entry 7). Entries are grouped below for readable provenance.

### 1. Segmentation-free pronunciation scoring: cheap warning confirmation

[Segmentation-Free Goodness of Pronunciation](https://arxiv.org/abs/2507.16838), first submitted July 2025, revised 5 February 2026; [implementation](https://github.com/frank613/CTC-based-GOP).

**Measured:** evaluates English data: CMU Kids has 5,180 sentences from 89 children (9.1 hours); speechocean762 has 5,000 utterances from 250 learners. Best feature-based score correlation is 0.648 versus 0.605 for the compared baseline. This correlation measures agreement with pronunciation scores, not Tilawa precision. Context of 8–14 phones performed similarly to full context.

**Idea:** score intended versus alternative sounds across possible timings, rather than judging one decoded path. **Tilawa test:** priority experiment 1, using small word windows and explicit substitution/deletion alternatives. **Effort:** roughly 1–3 engineering days for an offline prototype; phone implementation needs profiling. **Risk:** truncated top-5 outputs may omit relevant alternatives; retrieve full posterior windows for uncertain sites. Local context still requires some following audio. A canonical reference can guide checking without changing the observed listener transcript. This extends checker posterior use; it does not duplicate clean-reference training.

### 2. An independent Arabic error-aware model: diagnostic second opinion (resource priority, below staged training for implementation)

[MostafaMaroof/wav2vec2-arabic-phoneme-asr](https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr), current v2 card checked today; card does not establish its exact release date.

**Reported by model author:** trained on roughly 74,000 utterances; 14.42% phoneme error on an 800-example merged-pool holdout. Its revised vocabulary preserves emphatics, hamza, and doubled consonants, unlike v1. The holdout is not stated to be speaker-disjoint, so the number is not comparable to Tilawa.

**Tilawa test:** decode 100 reviewed disagreements: 40 false-warning sites, 30 genuine errors, 30 clean controls, all from development/training speakers. First manually verify vocabulary mapping. Ask whether it sees mistakes that Muaalem smooths away, and whether agreement can identify reliable pseudo-labels. **Success:** improve reviewed pseudo-label correctness without reducing error coverage; report counts by vowel/pair/shadda. **Effort:** about a day for a diagnostic pass. **Risk:** shared synthetic training biases, incompatible Quran conventions, and no validated low-end streaming performance. Use offline only initially. Apache-2.0 model card does not establish every source dataset's permissions.

### 3. Stage synthetic mistakes first, then adapt on real mistakes

[Yang et al., 23 June 2026](https://arxiv.org/html/2606.24086v1) train on about 79 hours native Arabic plus 80 hours synthetic mistakes, then adapt on about 2 hours real learner recordings. QuranMB.v2 F1: stage 1 alone 46.29%, naive mixing 43.05%, real-only 66.81%, staged single model 68.25%, six-model ensemble 72.01%. The last ensemble uses a small sequence prior built from its own predictions, not expected verse text. XLS-R-300M is too large to count as a phone solution; causality of the added convolution head does not make its encoder streaming.

**Tilawa experiment:** revise E3 into a controlled comparison: same student and fixed compute, clean-only versus 5,000 vetted audio-splice mistakes versus those mistakes followed by a short adaptation on 200–300 separately labelled learner clips. Keep evaluation speakers and passages separate. Record substitutions, omissions, vowels and shadda independently. Proposed gate: at least 5 percentage points precision gain, recall loss no more than 2 points; confirm on a larger set before shipping. Medium effort. Synthetic seam artifacts and small real-data overfitting are risks. This is additional work beyond clean soft5 and expected-middle targets.

Resources: [Iqra_TTS](https://huggingface.co/datasets/IqraEval/Iqra_TTS), 55,369 rows/5.94 GB, empty README; [real learner Extra](https://huggingface.co/datasets/IqraEval/Iqra_Extra_IS26). Audit labels, speaker identities and use permissions before importing.

### 4. Check that expected text is not erasing real mistakes

[Geng et al., 24 April 2026](https://arxiv.org/html/2604.22133v1) preserve dense acoustic evidence and use error information during training rather than expected-text prompts during inference. L2-ARCTIC: 2,429 training, 268 validation, 900 test utterances from six test speakers. Adding canonical prompts to their LLaMA variant drops recall from 59.08% to 28.83%, although PER improves from 15.85% to 13.55%. Their Arabic result is precision 73.25%, recall 70.20%, F1 71.70%, PER 3.72%; training uses synthetic speech then real-error adaptation. English full-system PER is 46.52% despite strong detection F1, so it is not a drop-in listener replacement.

**Tilawa experiment:** make a 120-item challenge pack: 30 correct controls, 30 substitutions, 30 omissions, 30 vowel/shadda mistakes, balanced across speakers. Compare clean soft5 and expected-middle targets; flag every case where the latter restores expected text over audible errors. Proposed gate: no more than 2 extra missed mistakes per category for any precision improvement. Low effort for the audit, high effort for replacing CTC with the paper's dense loss. This directly tests the risk of the already-running expected-middle experiment. [Code](https://github.com/Secondtonumb/IF-MDD).

### 5. Let streaming teacher targets move slightly in time

[Delayed-KD, 28 May 2025](https://arxiv.org/html/2505.22069v1) handles teacher/student emission timing differences by selecting the best target match within a short delay window. AISHELL-1: 178 training hours, 7,176 test utterances. At 40 ms chunks, direct distillation streaming character error is 7.02%; an 80 ms matching window gives 6.11%. The often-quoted 5.42% is utterance rescoring, not immediate streaming output. Median first-token delay rises from 100 to 140 ms. It also tests 10,000-hour WenetSpeech, approximately 33,100 test utterances.

**Tilawa experiment:** first inspect 200 held-out clips for systematic teacher/student phoneme peak offsets. Only if present, compare identical soft5 students with 0, 40 and 80 ms target buffers using a 10,000-clip clean subset. Proposed gate: at least 5% relative phoneme-error reduction without lower real-mistake recall; report first-warning P50/P90 latency and low-end-phone processing time. Medium effort, no added inference model size. It corrects timing, not wrong teacher labels. Unlike current soft5 training, it explicitly tests alignment mismatch; skip if the current student is not causally restricted.

### 6. E6 has support, but benefits are smaller than headline results

[Multi-View Multi-Task, 2 June 2023; revised 7 August 2023](https://arxiv.org/html/2306.01845v2) adds tasks for manner, place and vowel position. L2-ARCTIC uses 12 training, 6 validation and 6 test speakers. In the simple single-encoder comparison, auxiliary tasks improve precision 54.94%→58.30%, recall 53.90%→54.05%, PER 15.91%→14.60%. With the multilingual encoder, F1 barely changes, 57.82%→57.89%. The best 60.31% F1 combines two encoders and sequential auxiliary tasks; the whole gain cannot be attributed to the auxiliary head.

**Tilawa experiment:** add one training-only head for a carefully checked Arabic feature mapping (emphasis, voicing, vowel length, gemination), rather than two deployed encoders. Compare one fixed clean subset with and without the head. Proposed gate: 3 points checker precision improvement at unchanged recall, plus better confusion rates for the targeted contrasts. Medium effort. Features inherited from noisy teacher labels can reproduce its mistakes; begin with verified feature labels. This supports E6 as a bounded ablation, not as a demonstrated fix.

### 7. Update the Quran teacher/data reference to its August 2026 revision

[Abdelfattah et al. v2, 27 August 2026](https://arxiv.org/html/2509.00094v2) reports 848 hours, 286,537 utterances from 22 expert reciters; 11 output levels include phonemes and articulatory properties. Its 0.21% average PER on expert test speech is not learner error detection. The learner benchmark has 159 recordings of one verse, 120 female and 39 male reciters, 106 with errors and 53 correct; Tajweed F1 is 75.8%. The abstract's 1.94% learner average PER combines levels; the table's phoneme-only PER is 5.78%. All-expert training is male; single verse and single annotator limit generalization.

**Tilawa experiment:** use [qdat_bench](https://huggingface.co/datasets/obadx/qdat_bench) as a secondary out-of-domain test after checking overlap and mapping labels. Do not train on that test. Report by rule, gender, and correct versus incorrect recitation. Low/medium effort. This audits transfer beyond tlog rather than promising a new listener. The author has newer [streaming RNN](https://huggingface.co/obadx/muaalem-streaming-rnn-v0) and [mini](https://huggingface.co/obadx/muaalem-model-v3-mini) repositories, but their cards are autogenerated, with no usable training, size, latency or evaluation claims. Treat as inspection leads only.


## Evaluation before accepting any result

Report hits/false positives/misses, not rounded percentages alone. Show clean phone audio separately from laptop and tlog. Retain repeats, clip edges, missing words and substitutions as separate slices. Use the 62 uncontaminated gold clips for comparison with older students; preserve the same fixed reference for paired comparisons. New models excluding all gold IDs may use the full gold set, but comparison with old models still needs the common held-out slice.

Expand independent evidence: target 50 positive sites for each supported pronunciation class, including missing shadda and each consonant pair; seek 5–15 held-out volunteers as PLAN.md already specifies. Collect at least 30 minutes of clean long-session audio across several speakers for an initial false-warning-rate check. This is a pilot, not enough to confidently certify the shipping gate. Avoid using the current eval repeatedly to choose thresholds.

Speaker clustering is approximate; cross-source professional reciters need deduplication. Review 100 imported examples before mixing a new corpus into training, stratified by sound class, real/synthetic origin, and intended/as-said mismatch. Split speakers and source recording identities before augmentation. A splice's donor and recipient must remain in the same split.

For phone candidates, report actual model download bytes, peak memory, real-time factor, p95 highlight delay and p95 confirmation delay on the Android floor. A causal output head on a large bidirectional encoder does not make the entire model streaming. Preserve separate clean recognition and real-mistake gates after quantization. Once a metric-moving experiment is implemented, add the required scoreboard row; this research note itself needs none.
