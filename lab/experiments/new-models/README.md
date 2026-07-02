# New Models Benchmark for Offline Quran Verse Identification

## Purpose

Evaluate newer/alternative ASR models for offline Quran verse identification, complementing the existing benchmark which tests mlx-whisper models and HF models with LoRA.

## Models Tested

### Priority 1 -- Most likely to work well
| Model | Status | Notes |
|-------|--------|-------|
| `openai/whisper-large-v3-turbo` (HF) | Tested | 3/7 correct, good Arabic transcription |
| `tarteel-ai/whisper-base-ar-quran` (HF) | Tested | 3/7 correct, Quran-specific fine-tuning |
| `distil-whisper/distil-large-v3` | Tested | 0/7 -- **English only**, does not support Arabic |

### Priority 2 -- Newer models
| Model | Status | Notes |
|-------|--------|-------|
| `UsefulSensors/moonshine-tiny-ar` | Tested | 3/7 correct, smallest model, fastest inference |
| `facebook/seamless-m4t-v2-large` | Tested | 0/7 -- outputs empty strings for Arabic ASR |
| `facebook/mms-1b-all` (Arabic adapter) | Tested | 3/7 correct, largest model, slowest |

### Priority 3 -- Quran-specific
| Model | Status | Notes |
|-------|--------|-------|
| `Nuwaisir/Quran_speech_recognizer` | Tested | 0/7 -- outputs Buckwalter transliteration, not Arabic |
| `HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr` | Tested | 2/7 correct, garbled character-level output |

## Key Findings

### Top performers (3/7 accuracy on test set):

1. **Moonshine Tiny Arabic** -- Best efficiency. 103 MB, 1.26s avg inference, 27M params. Produces accurate Arabic transcriptions for clear reference audio. Struggles with user recordings and very short surahs.

2. **Whisper Large-v3-Turbo (HF)** -- Best general quality. 3 GB, 2.04s avg inference. Excellent Arabic transcription quality. Same accuracy but much larger.

3. **Tarteel Whisper Base (HF)** -- Best specialization. 277 MB, 2.08s avg inference. Fine-tuned specifically for Quran recitation. Very similar output to Whisper-turbo. Good balance of size and accuracy.

4. **MMS-1B-All (Arabic)** -- Most accurate on long verses. 3.7 GB, 4.01s avg inference. Got the highest match score on Ayat al-Kursi (0.945). But slow and huge.

### Models that did NOT work:

- **Distil-Whisper Large-v3**: English-only model. Cannot transcribe Arabic at all. Outputs English translations/gibberish.
- **SeamlessM4T-v2 Large**: 5.7 GB model that outputs empty strings. The S2T submodel doesn't produce usable Arabic text ASR output through the standard API.
- **Nuwaisir Quran Recognizer**: Outputs Buckwalter transliteration (ASCII encoding of Arabic) instead of actual Arabic characters. Would need a Buckwalter-to-Arabic converter to be useful.

### Challenging test cases (no model got right):

- **An-Nisa:51 (user recording)**: All models matched to Aal-i-Imraan 3:23 instead. The recording's verse is similar to 3:23 in wording.
- **Al-Ikhlas:2-3 (multi-ayah)**: Subtle transcription errors like "يند" instead of "يلد" caused wrong matches.
- **Al-Ikhlas:1 (ref)**: Very short verse "قل هو الله أحد" -- correctly transcribed by most but matched to wrong verse (too short for unique identification).
- **Ya-Sin:1 (ref)**: Single letter "يس" is too short for any model to match correctly.

## Recommendations

### For offline Quran identification:

1. **Best overall**: Use **Moonshine Tiny Arabic** as the primary model. At 103 MB and 1.26s inference, it offers the best size/speed/accuracy tradeoff. It produces clean Arabic output with diacritics.

2. **For maximum accuracy**: Consider **Whisper Large-v3-Turbo** or **Tarteel Whisper Base** as a secondary/validation model when Moonshine's confidence score is low.

3. **For edge deployment**: Moonshine Tiny Arabic is the clear winner at 27M params (vs 809M for Whisper-turbo).

### What needs improvement:

- Short verse identification needs a different approach (embedding similarity rather than text matching)
- User recordings need noise-robust models or preprocessing
- Consider combining multiple models for ensemble voting on difficult cases

## How to Run

```bash
# From project root venv
cd experiments/new-models-benchmark/
python benchmark.py          # Runs all models (can take 10+ minutes)
python results_table.py      # Generates results.md
```

## Files

- `benchmark.py` -- Main benchmark script
- `results_table.py` -- Results table generator
- `benchmark_results.json` -- Raw results data
- `results.md` -- Formatted markdown comparison table
