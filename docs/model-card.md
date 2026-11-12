# Model card: `spliced-clean@1-int8` and `spliced-clean-lite@1-int8`

The listener models behind `@tilawa/core` 1.x. They turn recitation audio into a stream of phonemes. The SDK's tracker and checker then match those phonemes against the Quran reference to find the reader's place and flag memorization mistakes.

## At a glance

| | `full` | `lite` |
|---|---|---|
| Id | `spliced-clean@1-int8` | `spliced-clean-lite@1-int8` |
| File | `models/spliced-clean/1-int8/model.onnx` | `models/spliced-clean-lite/1-int8/model.onnx` |
| Size | 67,759,744 bytes | 24,650,054 bytes |
| sha256 | `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b` | `4974de78d3c35d2542e7a24d99b747e1160d6e31e366fb68492b74a789890f1b` |
| Quantized from (fp32 sha256) | `a8552d517fa1393b91c9fe00145fafe0c80166e2aa408aeda08b2685930c71a8` | `3af366b9ae5c2587293748b2a09ea664c531206112829bfcd88ca84edcb544ab` |
| Default for | browsers and Node | low-memory devices, mobile browsers without WebGPU, React Native |
| Licence | CC BY 4.0 | CC BY 4.0 |

The hashes and manifests are in `packages/core/src/runtime/models-v1.json`, which the SDK bundles and checks every download against.

## Architecture

- **Streaming Zipformer2 with a CTC head** (`zipformer2-streaming-ctc`), following the k2-fsa icefall recipe and ONNX export.
- **Input:** 16 kHz mono audio, as 80-bin log-mel filterbank features (25 ms window, 10 ms shift).
- **Output:** phoneme posteriors every **40 ms**, over 42 phoneme tokens plus the CTC blank. The inventory is `muaalem-v3_2-phonemes`, from quran-transcript.
- **Streaming:** windows of 45 feature frames, advancing 32 at a time, with a left context of 128 frames. The encoder carries its state between windows, so latency doesn't grow with the length of the recitation.
- **Quantization:** dynamic int8 on the MatMul weights. It runs on onnxruntime-web (WebGPU or WASM), onnxruntime-node and onnxruntime-react-native.
- The `lite` model uses the same recipe at a smaller size.

The models don't output text, and they aren't a general Arabic speech recogniser. Finding the place and judging mistakes is done by the SDK, not the model.

## Training

- **Distillation.** Both models are distilled from the muaalem model (`obadx/muaalem-model-v3_2`, MIT), and use its phoneme inventory.
- **Initialisation.** The `full` model is initialised from icefall's LibriSpeech streaming Zipformer (Apache-2.0). The `lite` model is trained from scratch.
- **Audio.** The sources approved for training in the research plan: recitation from tlog (`tarteel-ai/tlog`, Tarteel, CC-BY) and Quranic Universal Ayahs (`QUD-Technologies/quranic-universal-ayahs`, QUD Technologies, CC-BY, Hafs only), with other openly licensed Quran recitation sets, including `zaibihassan/Quranic-Recitation-Data` (Apache-2.0), EveryAyah-derived sets, `obadx/muaalem-annotated-v3` (MIT) and `obadx/qdat` (MIT).
- **Augmentation.** Room impulse responses and noise from the same plan: OpenSLR-28 (Apache-2.0), the MIT IR survey (CC-BY), the MUSAN noise subset, and `obadx/freesound-commercial-50k-noise-only`.
- **Excluded.** All Quran-Lab / Muno459 models and data (NPL-1.2) and data tagged non-commercial. These models are not derived from the 0.x Quran-Lab model.
- **Licensing of the training data.** Some sources carry no licence tag. They were used for training at the project's own risk.
- **Leakage controls.** Splits are by reciter, never by clip. Quran-Lab benchmark clips (including `tlog_holdout`) were removed before training. Evaluation uses held-out reciters and held-out help-page speakers (see Evaluation).
- **Reference text.** The Quran reference (`ref-v2`) is the Hafs riwayah, Uthmani text from Tanzil, phonemized with quran-transcript 0.6.4 (MIT).

### Credits

tlog (Tarteel, CC-BY) and Quranic Universal Ayahs / the QUD alignment benchmark (QUD Technologies, CC-BY) for training and evaluation audio. muaalem and quran-transcript (`obadx`, MIT) for the teacher model and phonemization. k2-fsa icefall (Apache-2.0) for the architecture, recipe and export. Tanzil for the Quran text. Full notices: [NOTICE.md](../NOTICE.md).

## Licence

The model weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit "Tilawa (tilawa.dev)" and the sources above when you redistribute them.

The SDK code is MIT.

## Intended use

- Following a reader's recitation of the Quran, word by word, on their own device.
- Flagging **memorization mistakes** as they happen: `wrong_word`, `omitted_word`, `added_word` and `skipped_passage`.
- Memorization (hifz) practice and revision apps, recitation followers, and research.

Use the warnings as prompts for the reader, not as verdicts. Each one says "this sounded different", with a confidence; the reader, or a teacher, decides.

## Out of scope

- **Pronunciation and tajweed.** The models don't judge vowels, letter articulation, shadda, madd or any tajweed rule. Those error classes are listed in `capabilities.unsupported` and never emitted.
- **Other qira'at.** The reference is Hafs only.
- **Grading or certifying** a reader, or any decision about a person, without a human teacher.
- **Identifying speakers**, and any use outside Quran recitation.

## Limitations

- **Hafs only.** Recitation in another riwayah will be followed poorly and produce false warnings.
- **Memorization classes only.** A pronunciation mistake that doesn't change the word may go unflagged, or show up as `wrong_word`.
- **Adult reciters.** Training and evaluation audio is mostly adult recitation. **Children's voices have not been evaluated.**
- **Recording conditions.** Noise, echo, distance from the microphone and very fast recitation (hadr) reduce accuracy. Fast recitation is not part of the evaluation.
- **Repeated and near-identical ayahs.** Passages that recur across the Quran (for example in Ar-Rahman or Ash-Shu'ara) can be ambiguous in free recitation until the reader reaches a word that tells the copies apart. Declaring the passage (`startMemorization`) helps.
- **False warnings and missed mistakes both happen.** The rates are below.
- **`lite` is less accurate than `full`.** It is meant for devices that can't run `full`.

## Evaluation

Release readiness of engine `73c6a92f` with `spliced-clean@1-int8`, 11 November 2026. **Preview quality: 7 of Tilawa's 13 release bars pass, and the 6 below their bar are marked.** The bars have not been lowered.

| Measure (`full`) | Result | Release bar |
|---|---|---|
| Listener phoneme error rate (gold set, 233 clips) | 4.82% | ≤ 3.7% (not met) |
| Long recitations, simulated phone audio, held-out reciters (427 min): mistakes caught | 91.2% (485 of 532) | — |
| Same: warnings that were real mistakes | 88.8% | — |
| Same: false warnings per 5 min | 0.71 | — |
| Same: locked within 5 s / on the right place / wrong jumps | 98.7% / 99.2% / 38 in 427 min | — |
| Learner clips (help page), free recitation with 1–2 ayahs before and after: mistakes caught | 87.5% (56 of 64) | ≥ 80% |
| Same: warnings that were real mistakes | 81.2% | ≥ 90% (not met) |
| Declared passage: mistakes caught / warnings correct | 87.5% / 83.6% | ≥ 80% / ≥ 90% (not met) |
| Correct ayah within 5 s, short learner clips | 84.0% | ≥ 95% (not met) |
| Wrong jumps, short learner clips | 0 of 225 | ≤ 2% |
| False warnings per 5 min: short clips / repeats / recitation before / after the clip / long stitched sessions | 0.76 / 0 / 0.54 / 1.69 / 1.35 | ≤ 1 each (last two not met) |
| Real users' clean recitation | 0.36 false warnings per 5 min (126 min) | — |
| Speed, desktop CPU (Apple M4, onnxruntime-node) | about 1/18 of real time per session | — |

`lite` (`spliced-clean-lite@1-int8`): gold phoneme error rate 5.74%; about 1/27 of real time per session on the same desktop. Its mistake and warning rates were not re-measured for this release; earlier measurements showed similar recall and about twice the false warnings of `full`.

Not measured yet: latency and memory on phones and in browsers, long sessions with self-corrections, and scripted-error recordings from volunteers. Added words and skipped passages are measured only on the lab set (no learner examples), where planted additions sit at pauses.

## Versioning

A model is `<id>@<version>` plus its sha256. Files at a model's URL never change: a new model gets a new version and a new path. The SDK's default model changes only in an SDK minor release, with a changelog line. See the [version policy](api.md#version-policy).

## Reporting problems

If the model gets something wrong, readers can report it from apps that use [reporting](reporting.md), or recite at [tilawa.dev/help](https://tilawa.dev/help/). Code issues: [github.com/yazinsai/tilawa/issues](https://github.com/yazinsai/tilawa/issues).
