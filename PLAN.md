# Tilawa — Quran recitation engine plan

Agreed 4 October 2026. Supersedes the original working plan where they differ.

## Product

- **Deliverable:** SDK for Node, React Native, and web. Apps build UI on top.
- **Business model:** freemium. No paid feature may depend on Quran-Lab weights (No-Profit License).
- **v1 users:** adult non-Arab hifz revisers. Children are v2 with their own data.
- **Tradition:** Hafs ʿan ʿĀṣim; conventions follow `quran-transcript` `MoshafAttributes`.
- **Priority:** product first; publication optional.

### v1 feedback scope

| Area | v1 | Gated / later |
|---|---|---|
| Memorization | Wrong, omitted, added words; unresolved skips | — |
| Pronunciation | Short-vowel substitutions; confusable pairs ح/ه ع/ء ص/س ض/د ط/ت ق/ك ذ/ز ث/س; missing shadda | Natural madd (2 vs 0), gated on timing reliability |
| Tajweed | — | ghunnah, qalqalah, tafkheem |

Each pronunciation class ships independently once it meets its precision bar. Classes that don't meet it are reported by the SDK as `unsupported`.

## SDK architecture

- **Core:** TypeScript, used by all three targets. It contains the tracker, checker, and reference index. Port the hot loop to Rust/WASM only if profiling shows it's needed.
- **Listener runtime:** onnxruntime-web (WebGPU, falling back to WASM SIMD), onnxruntime-node, and onnxruntime-react-native.
- **Model tiers, downloaded on first use and versioned separately from the SDK:**
  - `lite`, about 25 MB, for web and low-end devices.
  - `full`, about 100 MB, for flagship phones.
  - The total budget is 200 MB.
- **API layers:**
  1. Core typed event streams. `position` events carry candidates and confidence. Checker events carry `suspected | confirmed | resolved | uncertain` plus the error class and location. The event schema is versioned from day one.
  2. A session layer: `startMemorization({ surah, fromAyah, toAyah })` and `startFreeRecitation()`, with mode and policy applied.
- **Device floor:** 2020 mid-range Android (Snapdragon 7xx, 4 GB), iPhone XR/11, and the last two versions of Chrome and Safari. The `lite` tier must run in real time on the Android floor with WASM SIMD, without WebGPU.

## Listener

- A streaming Zipformer with a phoneme CTC head, distilled from **`obadx/muaalem-model-v3_2`** (MIT, 660M w2v-BERT, multi-level CTC).
- Phoneme inventory and phonemizer: **`quran-transcript`** (MIT).
- The bar to beat is `Quran-Lab/zipformer_p-arabic-v3`: 65M params, 3.65% PER on phone audio, 72 MB int8. It is a reference only, never a dependency.
- E2 compares two arms: initialising from the multilingual sherpa-onnx streaming Zipformer versus training from scratch.
- **Canonicalization countermeasures.** The bias is already measured: Quran-Lab's model emitted ikhfaa on 81 of 94 plain-noon clips. Each countermeasure below is a separate ablation evaluated on the scripted-error test:
  - E3: SpeechBlender-style spliced errors built from QUD letter timestamps, making up 10–20% of each batch.
  - E5: distilling soft top-5 posteriors rather than only the teacher's top path.
  - E6: an articulatory-attribute auxiliary head (emphatic, pharyngeal, gemination).
- Grade madd from alignment timing, not from symbol run-length.

## tlog processing (`modal/tlog_pass.py`)

tlog filename labels can't be trusted, so treat the corpus as unlabelled audio.

1. Run the teacher over a 10% stratified shard sample, then the full corpus. Store the greedy phonemes, per-phoneme confidence, and top-5 fp16 posteriors for every frame.
2. Track the clip against the whole Quran, using the filename ayah only as a prior.
3. Relabel a clip only when the tracker is confident which passage it is; otherwise exclude it. Ambiguous clips go to tracker evaluation only.
4. Sort clips into buckets:
   - `clean-match`: PER ≤ ~3% against the matched passage, with high teacher confidence. Used for distillation.
   - `deviant`: confident location but local mismatches. Mined for natural errors; needs a second model or human review before it counts as ground truth.
   - `ambiguous`
   - `reject`: kept as non-recitation negatives.
5. Tune the bucket cutoffs on about 300 hand-checked clips stratified by bucket.
6. Speaker splits use metadata if it exists, otherwise ECAPA/WavLM-SV embedding clusters.

## Data

Untagged-license data: training accepted at our risk. NC-tagged and Quran-Lab data: excluded. No author outreach.

| Role | Sources |
|---|---|
| Distillation / training audio | `tarteel-ai/tlog` (CC-BY), `QUD-Technologies/quranic-universal-ayahs` (CC-BY, Hafs filter), `zaibihassan/Quranic-Recitation-Data` (Apache), everyayah / EA-DI / EA-UD, `obadx/muaalem-annotated-v3` (MIT), `obadx/qdat` (MIT), Iqra data where useful |
| Augmentation | OpenSLR-28 `EaseZh/rirs_noises` (Apache), MIT IR survey (CC-BY), MUSAN noise subset only, `obadx/freesound-commercial-50k-noise-only` |
| Tracker eval | QUD alignment benchmark (CC-BY), `nurlingo/quran-recitation-detect-split`, ambiguous tlog clips |
| Pronunciation eval | Scripted-error recordings, `obadx/qdat_bench`, `azizbekphd/ikhlas_recitations`, reviewed RetaSy clips |
| Excluded | Tadabur, FaresElmenshawi, quranic-orgz, all Quran-Lab / Muno459 assets, WHAM, RIR-Mega |

**Leakage rules:**

- Split by reciter, never by clip. Professional reciters recur across everyayah, its mirrors, QUD, and zaibihassan.
- Remove Quran-Lab benchmark IDs (including `tlog_holdout`) before training.
- Dedupe `tamm5y5m5/quran_with_time` against tlog if it is used.

**Attribution:** the SDK docs must credit tlog and QUD (CC-BY).

## Ground truth (no labelling budget)

- **Scripted-error recordings:** 5–15 volunteers reading generated scripts with planted errors, so labels are known by construction. Mixed gender, mostly non-Arab, several phones. Held out by speaker.
- **RetaSy clips:** flagged by the pipeline, then reviewed word by word.
- **Mined tlog `deviant` clips:** become natural errors once reviewed.
- **After launch:** an opt-in "report wrong feedback" button.

## Tracker

- Stage 1: a QUD-style deterministic aligner on decoded phonemes. Steps: VAD, then n-gram anchor voting, then substring Levenshtein DP with word-boundary constraints.
- Stage 2: CTC posterior scoring reranks the shortlist only.
- Keep separate expected and observed bookmarks.
- Search globally on acquisition and favor continuity once locked. Re-broaden when confidence drops.

## Acceptance bars (v1)

| Area | Bar |
|---|---|
| Memorization events | Precision ≥ 90%, recall ≥ 80% on the scripted-error test; ≤ 1 false warning per 5 min of correct recitation |
| Each pronunciation class | Precision ≥ 80% at whatever recall it achieves; otherwise `unsupported` |
| Tracker | Correct ayah within 5 s in ≥ 95% of starts; ≤ 2% false jumps |
| Latency | Word highlight p95 ≤ 600 ms on the device floor; `suspected` within ~1 s; `confirmed` within 2–3 s or at the next word onset (the confirm delay is a policy setting) |

## Budget

- $20k in Modal credits for compute.
- No cash for labelling.
- Rough costs: one teacher pass over tlog is about $150–400; one student training run is under $500.

## Experiments (kept from the original plan)

| ID | Purpose |
|---|---|
| E0 | Baseline diagnosis: which component loses mistakes |
| E1 | Transcript-only versus posterior-aware checking. Also measures the teacher's own canonicalization bias |
| E2 | Student initialisation arms |
| E3 | Spliced/human error adaptation |
| E4 | Optional Iqra_TTS synthesis |
| E5 | Soft-posterior / objective variants |
| E6 | Attribute head |
| E7 | Quantize and export; device measurements |
