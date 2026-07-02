# Embedding Search Experiment

Audio fingerprinting approach to Quran verse identification using HuBERT embeddings and FAISS nearest-neighbor search.

## Approach

Instead of transcribing speech to text and matching against verse text (the ASR approach), this experiment treats verse identification as an **audio similarity** problem:

1. **Offline indexing**: Extract a fixed-length embedding from each of the 6,236 Quran verses using a pre-trained speech representation model (HuBERT)
2. **Build a FAISS index**: Store all 6,236 embeddings in a FAISS index for fast nearest-neighbor search
3. **Query**: Extract an embedding from the user's recitation audio and find the closest verse in the index

```
User Audio → [HuBERT] → 768-dim embedding → [FAISS search] → Top-K verse matches
                                                                   ↓
Reference Audio Corpus → [HuBERT] → embeddings → FAISS Index    Results
(6,236 verses, Alafasy)                                         (surah, ayah, score)
```

### Why This Approach

- **No language model needed**: Works purely on acoustic similarity, no Arabic text processing
- **Fast search**: FAISS search over 6K vectors takes <1ms
- **Simple pipeline**: Two steps (embed + search) vs. transcribe + normalize + text-match
- **Offline-friendly**: Index is only 18MB, model runs on-device

### Components

| File | Purpose |
|------|---------|
| `download_verses.py` | Downloads verse-by-verse audio from EveryAyah.com |
| `build_index.py` | Extracts HuBERT embeddings from audio corpus, builds FAISS index |
| `search.py` | Queries the FAISS index with test audio |
| `benchmark.py` | Benchmarks accuracy and speed across test cases |

## How to Reproduce

### Prerequisites

```bash
pip install torch transformers faiss-cpu librosa tqdm numpy
```

### 1. Download reference audio

```bash
python download_verses.py --reciter alafasy
# Downloads 6,236 MP3 files (~1.5 GB) to audio_corpus/alafasy/
```

### 2. Build the FAISS index

```bash
python build_index.py --model hubert
# Processes 6,236 files in ~26 minutes on Apple Silicon (CPU)
# Output: indices/hubert_alafasy.index (18 MB)
```

The build script supports checkpointing -- if interrupted, it resumes from the last checkpoint.

### 3. Run search on a single file

```bash
python search.py path/to/audio.m4a --model hubert --top-k 5
```

### 4. Run the full benchmark

```bash
python benchmark.py --model hubert
# Tests 7 audio files (5 reference, 2 user recordings)
# Outputs: benchmark_results.json, results.md
```

## Benchmark Results

**Model**: `facebook/hubert-base-ls960` (768-dim embeddings)
**Index**: 6,236 vectors (all Quran verses, Alafasy recitation)
**Index size on disk**: 18 MB (.index) + 18 MB (.npy) + 477 KB (.json metadata)

### Accuracy Summary

| Category | Count | Top-1 | Top-3 | Top-5 | Top-10 | Surah@5 |
|----------|-------|-------|-------|-------|--------|---------|
| **All** | 7 | 71.4% | 71.4% | 71.4% | 71.4% | 85.7% |
| Reference (same reciter) | 5 | **100%** | **100%** | **100%** | **100%** | 100% |
| User (different speaker) | 2 | **0%** | **0%** | **0%** | **0%** | 50% |

### Speed

| Metric | Value |
|--------|-------|
| Avg embedding extraction | 376 ms |
| Avg FAISS search | 1.2 ms |
| Avg total latency | 377 ms |
| Model load time | 9.1 s |
| Index load time | included in model load |

### Per-File Detail

#### Reference Audio (same reciter = Alafasy)

| Test Case | Top-1 | Score | Time |
|-----------|-------|-------|------|
| Al-Fatiha:1 | **1:1** (exact match, score=1.000) | 97 ms |
| Al-Fatiha:2 | **1:2** (exact match, score=1.000) | 215 ms |
| Ayat al-Kursi (2:255) | **2:255** (score=0.974) | 674 ms |
| Al-Ikhlas:1 | **112:1** (exact match, score=1.000) | 168 ms |
| Ya-Sin:1 | **36:1** (exact match, score=1.000) | 117 ms |

Reference audio achieves **perfect score=1.000** for shorter verses (identical audio goes in, identical embedding comes out). Even Ayat al-Kursi (the longest test verse at ~52s, truncated to 15s) scores 0.974.

#### User Recordings (different speaker)

| Test Case | Expected | Top-1 Result | Score | Correct Surah in Top-5? |
|-----------|----------|-------------|-------|------------------------|
| An-Nisa:51 | 4:51 | Al-Qalam 68:39 | 0.861 | No |
| Al-Ikhlas:2-3 | 112:2 | An-Najm 53:62 | 0.721 | Yes (112:4 at #2) |

User recordings **completely fail** at verse-level matching. The embeddings encode speaker characteristics (voice timbre, recording quality, microphone) more than linguistic content. The Ikhlas recording at least gets the correct surah in the top-5, suggesting some content signal exists but is overwhelmed by speaker variation.

## Strengths

1. **Perfect on same-reciter audio**: 100% top-1 accuracy when the query audio comes from the same reciter as the index
2. **Extremely fast search**: FAISS lookup takes <1ms over 6,236 vectors
3. **Simple architecture**: No language model, no text normalization, no Arabic NLP pipeline
4. **Small index**: 18 MB on disk for the entire Quran
5. **Deterministic**: Same input always produces the same result

## Limitations

1. **Speaker-dependent**: HuBERT embeddings capture speaker identity more than linguistic content, making cross-speaker matching nearly impossible
2. **No partial matching**: The mean-pooled embedding represents the entire audio clip; a user reciting half a verse produces a very different embedding than the full verse
3. **Duration sensitivity**: Embedding changes significantly with audio length, so different-length recitations of the same verse produce different embeddings
4. **Single-verse only**: Cannot identify which verse within a multi-verse recitation

## Comparison with ASR-Based Approach

| Dimension | Embedding Search | Streaming ASR |
|-----------|-----------------|---------------|
| **Same-reciter accuracy** | **100% top-1** | 29-43% (mode-dependent) |
| **Cross-speaker accuracy** | **0%** | 29-43% (same as same-reciter) |
| **Search latency** | **<1ms** | N/A (text matching) |
| **Total latency** | ~377ms | ~1.2s (batch) |
| **Partial verse support** | No | **Yes** (streaming modes) |
| **Speaker independence** | No | **Yes** |
| **Multi-verse support** | No | **Yes** (text accumulates) |
| **Model size** | ~360 MB (HuBERT base) | ~150 MB (whisper-base) |
| **Index size** | 18 MB | 0 (text matching only) |

### Verdict

The embedding search approach is **not viable** as a general-purpose Quran verse identifier because it fails completely on cross-speaker audio. HuBERT embeddings (pre-trained on English LibriSpeech) encode too much speaker-specific information and too little phonetic/linguistic content to generalize across speakers.

However, it could work as a **supplementary signal** in a hybrid system:
- Use ASR as the primary identification method
- Use embedding search as a fast pre-filter when the reciter matches a known voice
- For a personal app where only one user recites, fine-tuning on the user's voice could dramatically improve results

### Potential Improvements (Not Tested)

1. **Arabic speech models**: Use a model pre-trained on Arabic (e.g., wav2vec2-large-xlsr-53) instead of English HuBERT
2. **Speaker-invariant embeddings**: Fine-tune with contrastive learning to map different speakers reciting the same verse to similar embeddings
3. **Segment matching**: Instead of mean-pooling the entire clip, use DTW on frame-level embeddings for time-alignment-based matching
4. **Multi-reciter index**: Index multiple reciters to improve coverage of different speaking styles
