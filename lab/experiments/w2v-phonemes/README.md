# w2v-phonemes

Phoneme-based Quran verse identification using wav2vec2 CTC models. Decodes audio into IPA phoneme sequences and matches against pre-phonemized Quran reference text.

## Models

| Variant | Model | Params | Size (int8) |
|---------|-------|--------|-------------|
| **base-int8** | `hetchyy/r15_95m_onnx_int8` | 95M (wav2vec2-base) | 116 MB |
| **large-int8** | `hetchyy/r7_onnx_int8` | 1B (wav2vec2-xls-r-1b) | 970 MB |

Both are ONNX int8 quantized. The models are fine-tuned on phoneme-labelled Quranic recitations.
## Data files

Two pre-computed data files are required:

| File | Size | Description |
|------|------|-------------|
| `data/phoneme_cache.pkl` | 7.6 MB | Per-verse phoneme reference for all 6,236 verses |
| `data/phoneme_ngram_index_5.pkl` | 6.0 MB | Pre-computed 5-gram index over all Quran phonemes for candidate retrieval |

Total additional storage: ~14 MB on top of model weights.

## Strategy

1. **Full CTC decode**: Run the entire audio through wav2vec2, greedy-decode the CTC output into an IPA phoneme sequence.
2. **Pass 1 — Levenshtein**: Score the decoded phonemes against all 6,236 verse phoneme strings using Levenshtein ratio. Take the top 5 surahs.
3. **Pass 1b — N-gram anchoring**: Extract 5-grams from the ASR phonemes and vote across a pre-built index to find additional candidate surahs (weighted by n-gram rarity). Adds up to 5 more surahs to the candidate pool.
4. **Pass 2 — Multi-verse spans**: For all candidate surahs (union of Levenshtein + n-gram), score consecutive verse spans (up to 7 verses).
5. **Best score wins** across single-verse and span candidates.

## Results

Latency numbers are not directly comparable to other experiments due to differences in local hardware.

### Overall (v1 corpus — 53 samples)

| Variant | SeqAcc | Recall | Precision | Size |
|---------|--------|--------|-----------|------|
| **large-int8** | **100%** | **100%** | **100%** | 970 MB |
| **base-int8** | 89% | 89% | 89% | 116 MB |

### Overall (v2 corpus — 43 samples)

| Variant | SeqAcc | Recall | Precision | Size |
|---------|--------|--------|-----------|------|
| **large-int8** | **95%** | **95%** | **95%** | 970 MB |
| **base-int8** | 88% | 88% | 88% | 116 MB |

### Per-category breakdown

#### large-int8

| Category | v1 | v2 |
|----------|----|----|
| Short | 17/17 (100%) | 23/25 (92%) |
| Medium | 18/18 (100%) | 4/4 (100%) |
| Long | 9/9 (100%) | 10/10 (100%) |
| Multi | 9/9 (100%) | 4/4 (100%) |

#### base-int8

| Category | v1 | v2 |
|----------|----|----|
| Short | 14/17 (82%) | 20/25 (80%) |
| Medium | 15/18 (83%) | 4/4 (100%) |
| Long | 9/9 (100%) | 10/10 (100%) |
| Multi | 9/9 (100%) | 4/4 (100%) |

Both models achieve 100% on long and multi-verse samples across both corpuses. All failures are in short/medium RetaSy crowd-sourced clips.

## Running

```bash
# Both int8 variants
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes

# Single variant
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/base-int8
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/large-int8

# Against v2 corpus
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes --corpus test_corpus_v2
```

## Next steps

1. **Fine-tune hyperparameters and verse detection strategy** — tune thresholds (e.g. `MATCH_THRESHOLD`, `TOP_SURAHS`, `MAX_SPAN`), experiment with alternative scoring (e.g. weighted Levenshtein, custom phoneme substitution costs), and improve short-verse detection where accuracy is weakest.
2. **Integrate with the web app and streaming** — connect the phoneme matching pipeline to the live web app, support real-time streaming audio input with incremental CTC decoding and progressive verse matching.
3. **(Optional) Retrain models with different architectures or additional data** — explore alternative architectures or augment training data with more diverse recitation styles and crowd-sourced recording.

## Benchmark runner changes

Two changes were made to `benchmark/runner.py` to support this experiment:

- **`--corpus` flag**: Selects which corpus directory to use (default: `test_corpus`). Usage: `--corpus test_corpus_v2`.
- **`also_accept` in manifest**: Samples can specify alternative accepted verse sequences via an `also_accept` field, for cases where two verses are identical (e.g., 3:23/4:44). The runner picks the alternative if it produces a higher sequence accuracy score.
