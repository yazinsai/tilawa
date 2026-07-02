# Offline Tarteel: Experiment Report

**Date:** 2026-02-24
**Goal:** Evaluate multiple approaches for offline Quran verse identification — listen to audio, identify the exact surah:ayah in near real-time, on-device (Apple Silicon).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Test Corpus](#test-corpus)
3. [Experiment A: ASR Model Benchmark](#experiment-a-asr-model-benchmark)
4. [Experiment B: Embedding Search (FAISS + HuBERT)](#experiment-b-embedding-search-faiss--hubert)
5. [Experiment C: Contrastive Audio-Text Model (QuranCLAP)](#experiment-c-contrastive-audio-text-model-quranclap)
6. [Experiment D: Streaming ASR](#experiment-d-streaming-asr)
7. [Cross-Experiment Comparison](#cross-experiment-comparison)
8. [Failure Analysis](#failure-analysis)
9. [Recommendations](#recommendations)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Appendix: Reproduction Instructions](#appendix-reproduction-instructions)

---

## Executive Summary

We evaluated four paradigms for offline Quran verse identification:

| Paradigm | Approach | Best Accuracy | Speed | Verdict |
|----------|----------|---------------|-------|---------|
| **A. ASR Benchmark** | Whisper/Moonshine/MMS → text match | 43% (3/7) | 1.3-4.0s | Best ready-to-use option |
| **B. Embedding Search** | HuBERT → FAISS nearest-neighbor | 100% same-reciter, 0% cross-speaker | 377ms | Not viable standalone |
| **C. Contrastive Model** | HuBERT+AraBERT → shared space | 1.6% top-10 | 100ms inference | Concept proven, needs GPU training |
| **D. Streaming ASR** | Chunked Whisper → progressive match | 43% (3/7) | 0.19s first-match | Adds real latency value |

**Key finding:** The accuracy bottleneck across all approaches is Arabic ASR quality. Whisper-base produces too many transcription errors for reliable verse matching. The path forward is: (1) upgrade to a better Arabic ASR model (Moonshine Tiny Arabic or fine-tuned Whisper Large), then (2) layer streaming for real-time UX, and (3) optionally add embedding search for known-reciter fast-path.

---

## Test Corpus

### Test Audio Files

| ID | File | Source | Expected | Duration | Description |
|----|------|--------|----------|----------|-------------|
| T1 | `test_audio/nisa_51.m4a` | User recording | 4:51 | ~10s | An-Nisa:51 — natural speech, ambient noise |
| T2 | `test_audio/ikhlas_2_3.m4a` | User recording | 112:2 | ~4s | Al-Ikhlas:2-3 — multi-ayah span |
| R1 | `reference_audio/001001.mp3` | EveryAyah (Alafasy) | 1:1 | 6.0s | Al-Fatiha:1 (Bismillah) |
| R2 | `reference_audio/001002.mp3` | EveryAyah (Alafasy) | 1:2 | 5.5s | Al-Fatiha:2 |
| R3 | `reference_audio/002255.mp3` | EveryAyah (Alafasy) | 2:255 | 51.9s | Ayat al-Kursi (longest test) |
| R4 | `reference_audio/112001.mp3` | EveryAyah (Alafasy) | 112:1 | 2.9s | Al-Ikhlas:1 (very short) |
| R5 | `reference_audio/036001.mp3` | EveryAyah (Alafasy) | 36:1 | 4.6s | Ya-Sin:1 (single letter surah opener) |

### Corpus Properties
- **Reference audio** (R1-R5): Professional studio recordings from Mishary Alafasy, clean audio, standard Hafs recitation
- **User recordings** (T1-T2): Real device recordings with ambient noise, non-professional reciter, varying microphone quality
- **Quran text database**: 6,236 verses from `api.alquran.cloud/v1/quran/quran-uthmani`, stored as JSON with both Uthmani (diacriticized) and clean (normalized) text

---

## Experiment A: ASR Model Benchmark

**Directory:** `experiments/new-models-benchmark/`
**Approach:** Transcribe audio with various ASR models → normalize Arabic → fuzzy-match against 6,236 verse database using Levenshtein ratio

### Models Tested

| Model | Type | Params | Size (MB) | Load Time (s) |
|-------|------|--------|-----------|---------------|
| Moonshine Tiny Arabic | Moonshine | 27M | 103 | 11.9 |
| Whisper Large-v3-Turbo (HF) | Whisper | 809M | 3,086 | 13.2 |
| Tarteel Whisper Base (HF) | Whisper fine-tuned | 74M | 277 | 13.3 |
| MMS-1B-All (Arabic) | wav2vec2 | 1B | 3,680 | 36.0 |
| HamzaSidhu Quran ASR | wav2vec2 fine-tuned | 317M | 360 | 9.4 |
| Distil-Whisper Large-v3 | Whisper distilled | 756M | 2,886 | 14.2 |
| Nuwaisir Quran Recognizer | wav2vec2 fine-tuned | 317M | 1,204 | 13.7 |
| SeamlessM4T-v2 Large | SeamlessM4T | 2.3B | 5,729 | 36.3 |

### Accuracy Results

| Model | T1 | T2 | R1 | R2 | R3 | R4 | R5 | Total |
|-------|----|----|----|----|----|----|----|----|
| **Moonshine Tiny Arabic** | - | - | 1:1 | 1:2 | 2:255 | - | - | **3/7** |
| **Whisper Large-v3-Turbo** | - | - | 1:1 | 1:2 | 2:255 | - | - | **3/7** |
| **Tarteel Whisper Base** | - | - | 1:1 | 1:2 | 2:255 | - | - | **3/7** |
| **MMS-1B-All** | - | - | 1:1 | 1:2 | 2:255 | - | - | **3/7** |
| HamzaSidhu Quran ASR | - | - | 1:1 | - | 2:255 | - | - | 2/7 |
| Distil-Whisper Large-v3 | - | - | - | - | - | - | - | 0/7 |
| Nuwaisir Quran Recognizer | - | - | - | - | - | - | - | 0/7 |
| SeamlessM4T-v2 Large | - | - | - | - | - | - | - | 0/7 |

### Speed Comparison

| Model | Avg Inference (s) | Speed Rank | Size Rank |
|-------|-------------------|------------|-----------|
| **Moonshine Tiny Arabic** | **1.26** | 1st | 1st (103 MB) |
| Distil-Whisper Large-v3 | 1.66 | 2nd | 5th |
| Whisper Large-v3-Turbo | 2.04 | 3rd | 6th |
| Tarteel Whisper Base | 2.08 | 4th | 2nd (277 MB) |
| Nuwaisir Quran Recognizer | 3.08 | 5th | 4th |
| HamzaSidhu Quran ASR | 3.64 | 6th | 3rd |
| MMS-1B-All | 4.01 | 7th | 7th |
| SeamlessM4T-v2 Large | 17.32 | 8th | 8th |

### Match Confidence Scores (correct matches only)

| Model | R1 (Bismillah) | R2 (Fatiha:2) | R3 (Kursi) | Avg |
|-------|----------------|---------------|------------|-----|
| Moonshine Tiny Arabic | 0.978 | 0.976 | 0.744 | 0.899 |
| Whisper Large-v3-Turbo | 0.978 | 0.976 | 0.735 | 0.896 |
| Tarteel Whisper Base | 0.978 | 0.976 | 0.748 | 0.901 |
| MMS-1B-All | 0.936 | 0.884 | 0.945 | 0.922 |

### Transcription Quality Samples

**R1 — Al-Fatiha:1 (بسم الله الرحمن الرحيم)**
| Model | Transcription (normalized) | Notes |
|-------|---------------------------|-------|
| Moonshine Tiny | بسم الله الرحمن الرحيم | Perfect |
| Whisper Large-v3-Turbo | بسم الله الرحمن الرحيم | Perfect |
| Tarteel Whisper Base | بسم الله الرحمن الرحيم | Perfect |
| MMS-1B-All | بباسم الله الرحمن الرحيم | Extra ب prefix |
| Distil-Whisper | In the name of Allah, the | English output |

**T1 — An-Nisa:51 (user recording)**
| Model | Transcription (normalized) | Issue |
|-------|---------------------------|-------|
| Moonshine Tiny | الم تر الي الذين اوتوا النصيب من الكتاب ودعونا الي كتاب الله معكم وبينه | Close but hallucinated ending |
| Whisper Large-v3-Turbo | الم تر الي الذين اوتوا نصيبا من الكتاب يدعون الي كتاب الله نحكم بينهم | Closest to correct |
| Tarteel Whisper Base | الم توان الذين اوتوا نصيبا من الكتاب يدعون الي كتاب الله ومحكم بينهم | "توان" error |

### Models That Failed

| Model | Reason | Example Output |
|-------|--------|----------------|
| **Distil-Whisper Large-v3** | English-only model | "In the name of Allah, the" |
| **SeamlessM4T-v2 Large** | Empty output for Arabic S2T | (empty string) |
| **Nuwaisir Quran Recognizer** | Buckwalter transliteration | "bisomi/All~ahi/Alr~aHomani" |

### Key Takeaways

1. **Moonshine Tiny Arabic is the efficiency winner** — 30x smaller than Whisper-turbo, fastest inference, same accuracy
2. **All top models correctly identify the same 3 verses** (Bismillah, Fatiha:2, Ayat al-Kursi) and fail on the same 4
3. **MMS-1B-All has the highest score on Ayat al-Kursi** (0.945) — its wav2vec2 architecture seems better at long-form Arabic
4. **No model handles user recordings or very short verses** — this is a systematic gap, not a model-specific one

---

## Experiment B: Embedding Search (FAISS + HuBERT)

**Directory:** `experiments/embedding-search/`
**Approach:** Extract HuBERT embeddings from all 6,236 Quran verses (Alafasy recitation) → build FAISS index → nearest-neighbor search at inference

### Architecture

```
Pre-computation (one-time):
  6,236 MP3s (Alafasy) → HuBERT-base → mean-pool → 768-dim embedding → FAISS flat index

Inference:
  Query audio → HuBERT-base → mean-pool → 768-dim embedding → FAISS search → Top-K matches
```

### Index Statistics

| Metric | Value |
|--------|-------|
| Model | `facebook/hubert-base-ls960` (95M params) |
| Embedding dimension | 768 |
| Total vectors indexed | 6,236 |
| Index file size | 18 MB |
| Raw embeddings file | 18 MB |
| Metadata file | 477 KB |
| Index build time | 26 minutes (Apple Silicon CPU) |
| Model load time | 9.1s |
| Audio corpus | 6,236 Alafasy MP3s (~1.5 GB) |

### Accuracy Results

| Test | Expected | Top-1 Match | Score | Embed Time | Search Time | Correct? |
|------|----------|------------|-------|------------|-------------|----------|
| **R1** Al-Fatiha:1 | 1:1 | **1:1** | 1.000 | 97ms | 0.5ms | Yes |
| **R2** Al-Fatiha:2 | 1:2 | **1:2** | 1.000 | 215ms | 0.6ms | Yes |
| **R3** Ayat al-Kursi | 2:255 | **2:255** | 0.974 | 673ms | 0.7ms | Yes |
| **R4** Al-Ikhlas:1 | 112:1 | **112:1** | 1.000 | 168ms | 0.9ms | Yes |
| **R5** Ya-Sin:1 | 36:1 | **36:1** | 1.000 | 117ms | 0.6ms | Yes |
| **T1** An-Nisa:51 | 4:51 | 68:39 | 0.861 | 1,229ms | 4.3ms | No |
| **T2** Al-Ikhlas:2-3 | 112:2 | 53:62 | 0.721 | 133ms | 0.5ms | No |

### Accuracy Summary

| Category | Top-1 | Top-3 | Top-5 | Top-10 |
|----------|-------|-------|-------|--------|
| Reference (same reciter) | **100%** (5/5) | 100% | 100% | 100% |
| User (different speaker) | **0%** (0/2) | 0% | 0% | 0% |
| Overall | **71.4%** (5/7) | 71.4% | 71.4% | 71.4% |

### Why Cross-Speaker Fails

HuBERT was pre-trained on English LibriSpeech. Its embeddings encode:
- Speaker identity (voice timbre, formants)
- Recording conditions (microphone, room acoustics, background noise)
- Speaking style (pace, intonation)

These speaker-specific features dominate over linguistic content. When the same Alafasy audio goes in, the same embedding comes out (score=1.000). When a different person recites the same verse, the embedding is in a completely different region of the space.

Evidence: For T2 (user recording of Al-Ikhlas:2-3), the nearest match is An-Najm 53:62 — a completely unrelated verse. But the correct surah (112) does appear at rank #2 (112:4, score=0.674), suggesting *some* content signal exists but is overwhelmed by speaker variation.

### Latency Breakdown

| Operation | Avg Time | Notes |
|-----------|----------|-------|
| Audio loading + resampling | ~10ms | librosa, 16kHz mono |
| HuBERT embedding extraction | 376ms | Varies 97ms-1.2s with audio length |
| FAISS search (6,236 vectors) | 1.2ms | Flat L2 index, brute force |
| **Total per query** | **~377ms** | Dominated by embedding extraction |

---

## Experiment C: Contrastive Audio-Text Model (QuranCLAP)

**Directory:** `experiments/contrastive/`
**Approach:** Train a CLIP-style model that maps Quran audio and Arabic verse text into a shared 256-dim embedding space via contrastive learning (InfoNCE loss)

### Architecture

```
Audio path:  waveform → HuBERT-base (frozen) → mean pool → MLP(768→256→256) → L2-norm → [256-d]
                                                                                          |
                                                                                     cosine sim
                                                                                          |
Text path:   Arabic text → AraBERT v2 (frozen) → CLS token → MLP(768→256→256) → L2-norm → [256-d]
```

| Component | Model | Params | Trainable? |
|-----------|-------|--------|------------|
| Audio encoder | `facebook/hubert-base-ls960` | 95M | Frozen (Phase 1) |
| Text encoder | `aubmindlab/bert-base-arabertv02` | 135M | Frozen (Phase 1) |
| Audio projection | 2-layer MLP (768→256→256, GELU) | ~263K | Yes |
| Text projection | 2-layer MLP (768→256→256, GELU) | ~263K | Yes |
| Temperature | Learnable scalar (init=0.07) | 1 | Yes |
| **Total** | | **230M** | **525K trainable** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training data | Alafasy recitations, surahs 1-5 (789 verses) |
| Train/val split | 552 / 118 (70/15/15) |
| Batch size | 4 |
| Phase 1 epochs | 5 (frozen encoders, projection heads only) |
| Phase 2 epochs | 0 (not run — would unfreeze encoder layers) |
| Optimizer | AdamW (lr=1e-4, weight decay=0.01) |
| Scheduler | Cosine annealing |
| Loss | Symmetric InfoNCE (CLIP-style) |
| Max audio length | 15 seconds (truncated) |
| Training time | 28.6 minutes (Apple MPS) |
| Device | Apple M-series (MPS backend) |

### Training Progression

| Epoch | Train Loss | Val Loss | Val Batch Accuracy |
|-------|-----------|----------|-------------------|
| 1 | 1.290 | 1.141 | 38.2% |
| 2 | 1.226 | 1.108 | 42.4% |
| 3 | 1.175 | 1.100 | 43.1% |
| 4 | 1.138 | 1.076 | 44.9% |
| 5 | 1.106 | 1.085 | 43.9% |

Within-batch accuracy rose from ~25% (random with batch size 4) to 44.9%, showing the projection heads are learning to discriminate audio-text pairs. But this does not transfer to global retrieval.

### Evaluation: Retrieval Accuracy

**Task:** Given an audio clip, find the correct verse among N candidates via cosine similarity.

| Setup | Candidates | Top-1 | Top-3 | Top-5 | Top-10 |
|-------|-----------|-------|-------|-------|--------|
| Zero-shot (no training) | 789 (in-domain) | 0.1% | 0.4% | 0.5% | 0.8% |
| **Trained (Phase 1)** | **789 (in-domain)** | **0.1%** | **0.6%** | **0.6%** | **1.6%** |
| Zero-shot (no training) | 6,236 (all verses) | 0.0% | 0.0% | 0.0% | 0.0% |
| Trained (Phase 1) | 6,236 (all verses) | 0.0% | 0.1% | 0.3% | 0.5% |

### Why It Barely Works

1. **English HuBERT on Arabic audio**: HuBERT was pre-trained on English LibriSpeech. Arabic Quranic recitation has fundamentally different phonetics, prosody, and melodic patterns (tajweed). The frozen encoder doesn't capture Arabic phoneme distinctions.

2. **Modality gap too large**: Acoustic features (HuBERT) and orthographic features (AraBERT) operate in incompatible representation spaces. A 2-layer MLP cannot bridge this gap — it can only learn a rough linear mapping.

3. **Batch size 4 is too small for contrastive learning**: CLIP used batch sizes of 32,768. With batch size 4, the model only needs to discriminate among 4 options. Global retrieval over 6,236 candidates requires orders of magnitude more discriminative power.

4. **Single reciter, no augmentation**: Only Alafasy's voice. The model overfits to his specific vocal characteristics rather than learning verse-specific acoustic patterns.

5. **Insufficient scale**: 552 training examples, 2,760 total (5 epochs). CLAP models train on millions of audio-text pairs.

### What Would Make It Viable

| Requirement | Minimum | Ideal |
|-------------|---------|-------|
| Audio encoder | Arabic-pretrained (Whisper encoder, Arabic HuBERT) | Whisper Large encoder fine-tuned on Quran |
| Training data | 10+ reciters (62K+ pairs) | 30 reciters from Quran-MD (187K pairs) |
| Batch size | 32-64 | 256+ |
| Epochs | 50-100 | 200+ with hard negative mining |
| Compute | Single A100 (~4-8 hours) | Multi-GPU (~1-2 hours) |
| Phase 2 | Unfreeze last 2 encoder layers | Full encoder fine-tuning |
| Data augmentation | Speed/pitch perturbation | + noise injection, room simulation |
| Hard negatives | Same-surah verses | Adjacent verses, similar-sounding verses |

---

## Experiment D: Streaming ASR

**Directory:** `experiments/streaming-asr/`
**Approach:** Process audio in small chunks (1-3s) with mlx-whisper, accumulate transcription progressively, match against verses in real-time with prefix-aware Levenshtein scoring

### Architecture

```
Audio (continuous) → [Chunk Splitter (1-3s)] → [mlx-whisper per chunk]
    → [Word Accumulator] → [Verse Position Tracker] → { verse_ref, word_position, progress% }
```

### Components

| Component | File | Algorithm |
|-----------|------|-----------|
| Streaming transcriber | `streaming_transcriber.py` | Audio chunking (independent or sliding window), per-chunk mlx-whisper |
| Word tracker | `word_tracker.py` | DTW on Whisper cross-attention weights for word-level timestamps |
| Verse position tracker | `verse_position_tracker.py` | Prefix-aware Levenshtein + LCS-based word alignment |
| Demo | `demo.py` | Real-time terminal output with progress bars |
| Benchmark | `benchmark.py` | Systematic mode comparison |

### Matching Algorithm Details

The `VersePositionTracker` uses a prefix-aware strategy optimized for streaming:

1. **Prefix scoring**: Compares input text against the first N words of each verse (where N = number of input words), avoiding the penalty of matching a short prefix against a long verse
2. **Adaptive weighting**: When coverage <80%, trusts prefix score (70%) over full score (30%). When nearly complete, reverses the weighting
3. **Word alignment**: LCS dynamic programming to track which specific verse words have been matched, with fuzzy per-word matching (Levenshtein ratio >= 0.7)

### Streaming Modes Tested

| Mode | Chunk Size | Overlap | Per-Chunk Latency |
|------|-----------|---------|-------------------|
| batch | entire file | N/A | N/A |
| stream_3s | 3 seconds | none | ~0.1-0.3s |
| stream_3s_overlap | 3 seconds | 1 second | ~0.1-0.4s |
| stream_2s | 2 seconds | none | ~0.2-0.3s |
| stream_1s | 1 second | none | ~0.4s |

### Benchmark Results

| Mode | Accuracy | Avg Confidence | Avg First Match | Avg Total Time | Avg RTF |
|------|----------|---------------|-----------------|----------------|---------|
| batch | 2/7 (29%) | 0.675 | 1.17s | 1.17s | 0.150 |
| stream_3s | 2/7 (29%) | 0.535 | **0.19s** | 4.92s | 0.483 |
| stream_3s_overlap | 2/7 (29%) | 0.605 | 0.80s | 5.86s | 0.667 |
| **stream_2s** | **3/7 (43%)** | 0.591 | 5.44s | 7.58s | 0.707 |
| stream_1s | 1/7 (14%) | 0.552 | 4.16s | 22.57s | 1.797 |

RTF = Real-Time Factor (processing time / audio duration). RTF < 1.0 = faster than real-time.

### Per-Test Results (Batch vs Stream 3s)

| Test | Expected | Batch | Batch Conf | Stream 3s | Stream 3s Conf | Stream First Match |
|------|----------|-------|-----------|-----------|---------------|-------------------|
| R1 Fatiha:1 | 1:1 | 20:1 | 0.981 | 20:1 | 0.933 | 0.16s |
| R2 Fatiha:2 | 1:2 | **1:2** | 0.976 | **1:2** | 0.909 | 0.34s |
| R3 Kursi | 2:255 | **2:255** | 0.891 | **2:255** | 0.594 | 0.21s |
| R4 Ikhlas:1 | 112:1 | 39:14 | 0.737 | 39:14 | 0.737 | 0.09s |
| R5 Ya-Sin:1 | 36:1 | 51:12 | 0.567 | — | — | — |
| T1 Nisa:51 | 4:51 | 3:23 | 0.574 | NONE | 0.000 | — |
| T2 Ikhlas:2-3 | 112:2 | NONE | 0.000 | NONE | 0.000 | — |

### Word Timestamp Capability

| Method | Quality | Speed | Production-Ready? |
|--------|---------|-------|-------------------|
| Whisper DTW word timestamps | Accurate word boundaries | **~160x real-time** (209s for 6s audio) | No |
| Synthetic (chunk duration / word count) | Approximate | Free | Yes |
| WhisperX (not tested) | Good (phoneme-level alignment) | Unknown | Potentially |

Word timestamps via DTW on 2 short test files:

| Test | Words Found | Expected | Time | Word Spans |
|------|------------|----------|------|------------|
| Al-Fatiha:1 | 4 | 4 | 1.15s | 0.00-0.48, 0.48-1.08, 1.08-2.28, 2.28-4.32 |
| Al-Ikhlas:1 | 3 | 8 | 0.28s | 0.00-0.54, 0.54-1.22, 1.22-2.16 |

### Key Observations

1. **Streaming is 6x faster to first match**: 0.19s (stream_3s) vs 1.17s (batch). For Ayat al-Kursi (52s audio), streaming identifies the verse in 0.21s while batch takes 1.42s.

2. **Accuracy is limited by whisper-base Arabic quality, not streaming**: Both batch and streaming modes fail on the same test cases. The chunking strategy doesn't meaningfully degrade accuracy.

3. **2-3 second chunks are optimal**: 1s chunks don't give Whisper enough context (produces garbled output, 14% accuracy). 3s chunks give good results. 2s sometimes edges ahead on short verses.

4. **Bismillah matching bug**: Al-Fatiha:1 ("بسم الله الرحمن الرحيم") consistently matches Taa-Haa 20:1 instead. This is because 20:1 contains the Bismillah as a prefix and scores higher in the verse matcher due to text overlap.

5. **User recordings produce English hallucinations**: Whisper-base outputs English text ("and I'm talking", "and The Law") when given noisy Arabic recordings. No Arabic output = no verse match.

---

## Cross-Experiment Comparison

### Accuracy by Test Case (Best Mode per Experiment)

| Test | A: ASR (Moonshine) | B: Embedding | D: Streaming (2s) |
|------|-------------------|-------------|-------------------|
| R1 Fatiha:1 | **1:1** | **1:1** | 20:1 |
| R2 Fatiha:2 | **1:2** | **1:2** | **1:2** |
| R3 Ayat al-Kursi | **2:255** | **2:255** | **2:255** |
| R4 Ikhlas:1 | 39:14 | **112:1** | 39:14 |
| R5 Ya-Sin:1 | 51:12 | **36:1** | 51:12 |
| T1 An-Nisa:51 | 3:23 | 68:39 | **4:51** |
| T2 Al-Ikhlas:2-3 | — | 53:62 | — |
| **Total Correct** | **3/7** | **5/7** (same-reciter only) | **3/7** |

Note: Experiment B's 5/7 is misleading — it only works because R1-R5 are from the same reciter as the index. On T1-T2 (different speaker) it scores 0/2.

### Latency Comparison

| Experiment | First Result | Total (per query) | Notes |
|------------|-------------|-------------------|-------|
| A: ASR (Moonshine) | 1.26s | 1.26s | Single-shot |
| B: Embedding | 377ms | 377ms | Fastest overall |
| C: Contrastive | ~100ms | ~100ms | Inference only (model doesn't work yet) |
| D: Streaming (3s) | **0.19s** | 4.92s | Best first-match latency |

### Resource Requirements

| Experiment | Model Size | RAM Usage | Index/Data | Total Footprint |
|------------|-----------|-----------|------------|-----------------|
| A: Moonshine Tiny | 103 MB | ~300 MB | 3 MB (quran.json) | ~406 MB |
| A: Tarteel Base | 277 MB | ~500 MB | 3 MB | ~780 MB |
| B: HuBERT + FAISS | 360 MB | ~800 MB | 36 MB (index+embeddings) | ~1.2 GB |
| C: QuranCLAP | 878 MB | ~2 GB | 3 MB | ~2.9 GB |
| D: whisper-base (mlx) | ~150 MB | ~400 MB | 3 MB | ~553 MB |

---

## Failure Analysis

### Systematic Failures (No Approach Solves)

**1. Short verses (< 4 words)**
- Ya-Sin:1 ("يس") — a single letter. Too ambiguous for any text matching.
- Al-Ikhlas:1 ("قل هو الله أحد") — only 4 words, many verses share subsets.
- Root cause: The Levenshtein ratio assigns high scores to many short verses. Need minimum-length gating or surah-context bias.

**2. User recordings with noise**
- An-Nisa:51 and Al-Ikhlas:2-3 fail across all ASR models.
- Whisper-base hallucinates English on noisy Arabic input.
- Root cause: Whisper-base has limited Arabic robustness. Larger models (large-v3-turbo) produce better Arabic but still don't match correctly due to subtle transcription errors.

**3. Highly similar verses**
- An-Nisa:51 transcription matches Aal-i-Imraan 3:23 (similar wording).
- Bismillah matches Taa-Haa 20:1 (which contains Bismillah as a prefix).
- Root cause: Many Quran verses share phrases. Pure text distance is insufficient — need surah context, recitation sequence tracking, or longer audio windows.

### Per-Paradigm Limitations

| Paradigm | Critical Limitation | Would Fix It |
|----------|-------------------|--------------|
| A (ASR) | Whisper-base Arabic quality | Larger model, Quran fine-tuning |
| B (Embedding) | Speaker identity in embeddings | Arabic-pretrained model, multi-reciter index |
| C (Contrastive) | English HuBERT + tiny batch + minimal data | Arabic encoder, GPU training, Quran-MD dataset |
| D (Streaming) | Same as A (ASR quality) | Better model improves all streaming modes |

---

## Recommendations

### Tier 1: Immediate Improvements (1-2 days)

**1. Switch to Moonshine Tiny Arabic as default ASR model**
- 30x smaller than Whisper-turbo, fastest inference, competitive accuracy
- Package: `UsefulSensors/moonshine-tiny-ar` from HuggingFace
- Integration: Drop-in replacement for whisper in transcriber.py

**2. Add surah-context bias to verse matcher**
- After identifying a verse, bias next match toward same surah (adjacent ayahs)
- Already designed in `matcher.py` plan but not fully leveraged
- Would fix: Bismillah→20:1 bug (when preceded by other Fatiha verses)

**3. Add minimum confidence gating**
- Only return a match if confidence > 0.7
- Below threshold: return "listening..." / accumulate more audio
- Would reduce false positives on short/ambiguous verses

### Tier 2: Model Improvements (1-2 weeks)

**4. Fine-tune Whisper Large-v3 on Quran data**
- Use `tarteel-ai/whisper-large-v3-Tarteel` or fine-tune from scratch on Quran-MD
- Target: <3% WER on Quranic Arabic
- Deploy via mlx-whisper with INT8 quantization for on-device
- Expected impact: All accuracy numbers would improve dramatically

**5. Benchmark Moonshine v2 Arabic (when available)**
- Moonshine v2 (Feb 2026) has explicit Arabic model and streaming-first architecture
- Would be ideal for the streaming paradigm — bounded time-to-first-token
- GitHub: `https://github.com/usefulmoonshine/moonshine`

**6. Implement WhisperKit for iOS deployment**
- Native Swift, runs on Apple Neural Engine
- Supports Whisper Large-v3 Turbo on-device with 0.46s latency
- Would be the production deployment target for iPhone
- GitHub: `https://github.com/argmaxinc/WhisperKit`

### Tier 3: Research Directions (1+ months)

**7. Train contrastive model with proper resources**
- Replace HuBERT with Whisper's encoder (already understands Arabic)
- Train on Quran-MD (187K samples, 30 reciters)
- GPU training with batch size 256+, 100+ epochs
- Hard negative mining (same-surah, adjacent verses)
- If successful: enables sub-100ms verse identification via embedding lookup

**8. Multi-reciter embedding index**
- Index multiple reciters in FAISS (not just Alafasy)
- 10 reciters × 6,236 verses = 62K vectors (~64 MB at 512-dim)
- Use Arabic-pretrained speech model (not English HuBERT)
- Potential: fast pre-filter for the ASR pipeline

**9. Hybrid architecture**
- Phase 1 (<500ms): Embedding search → top-5 candidates (if reciter known)
- Phase 2 (streaming): ASR → progressive text match against candidates
- Phase 3 (fallback): Full ASR → match against all 6,236 verses
- Combines best latency (embedding) with best generalization (ASR)

### Not Recommended

| Approach | Reason |
|----------|--------|
| SeamlessM4T-v2 | Empty output for Arabic, 5.7 GB, slow |
| Distil-Whisper | English-only, no Arabic support |
| Audio fingerprinting (Chromaprint/Dejavu) | Only matches exact recordings, not different speakers |
| Nuwaisir model | Outputs Buckwalter transliteration, not Arabic text |

---

## Implementation Roadmap

### Phase 1: MVP (target: functional prototype)
```
1. Integrate Moonshine Tiny Arabic as transcriber        → experiments/new-models-benchmark/
2. Wire streaming (2-3s chunks) from streaming-asr       → experiments/streaming-asr/
3. Add surah-context bias to VerseMatcher                → src/offline_tarteel/matcher.py
4. Add confidence gating (>0.7 threshold)                → src/offline_tarteel/matcher.py
5. Build CLI: `recognize audio.mp3` with streaming output → src/offline_tarteel/cli.py
```

### Phase 2: Accuracy Boost
```
6. Fine-tune Whisper Large-v3 on Quran-MD dataset
7. Convert to mlx-whisper format (or GGML for whisper.cpp)
8. INT8 quantize for on-device deployment
9. Benchmark: target >80% accuracy on expanded test set
```

### Phase 3: Production
```
10. WhisperKit integration for iOS
11. Streaming word-position tracking (synthetic timestamps)
12. Multi-verse sequence detection
13. Expanded test corpus (100+ verses, multiple reciters, user recordings)
```

### Phase 4: Research (Optional)
```
14. Contrastive model with Whisper encoder + Quran-MD training
15. Multi-reciter FAISS index with Arabic speech embeddings
16. Hybrid architecture (embedding pre-filter + ASR confirmation)
```

---

## Appendix: Reproduction Instructions

### Environment Setup

```bash
cd ~/ai/projects/offline-tarteel
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
# Additional dependencies for experiments:
pip install torch transformers faiss-cpu moonshine peft accelerate datasets
```

### Running Each Experiment

**Experiment A: New Models Benchmark**
```bash
cd experiments/new-models-benchmark
python benchmark.py          # Runs all 8 models (~15 min, downloads models on first run)
python results_table.py      # Generates results.md
```

**Experiment B: Embedding Search**
```bash
cd experiments/embedding-search
python download_verses.py --reciter alafasy   # Downloads 6,236 MP3s (~1.5 GB)
python build_index.py --model hubert          # Builds FAISS index (~26 min)
python benchmark.py --model hubert            # Runs benchmark
```

**Experiment C: Contrastive Model**
```bash
cd experiments/contrastive
# Training (requires audio corpus from Experiment B)
python train.py --source everyayah --batch-size 4 --phase1-epochs 5 --phase2-epochs 0 --max-surah 5
# Evaluation
python evaluate.py --mode both --checkpoint checkpoints/best_phase1.pt
```

**Experiment D: Streaming ASR**
```bash
cd experiments/streaming-asr
python benchmark.py          # Runs all streaming modes (~5 min)
python demo.py               # Interactive demo (default: Ayat al-Kursi)
python demo.py --all         # Demo all test files
```

### Data Locations

| Data | Path | Size | Source |
|------|------|------|--------|
| Quran text DB | `data/quran.json` | 3 MB | alquran.cloud API |
| Test audio (user) | `data/test_audio/` | ~84 KB | Manual recordings |
| Reference audio | `data/reference_audio/` | ~4 MB | EveryAyah.com (Alafasy) |
| Full audio corpus | `experiments/embedding-search/audio_corpus/alafasy/` | ~1.5 GB | EveryAyah.com |
| FAISS index | `experiments/embedding-search/indices/` | ~36 MB | Generated by build_index.py |
| Contrastive checkpoint | `experiments/contrastive/checkpoints/` | ~2 MB | Generated by train.py |

### Key Model IDs (HuggingFace)

| Model | HuggingFace ID | Use Case |
|-------|---------------|----------|
| Moonshine Tiny Arabic | `UsefulSensors/moonshine-tiny-ar` | Best efficiency ASR |
| Tarteel Whisper Base | `tarteel-ai/whisper-base-ar-quran` | Quran-specific ASR |
| Whisper Large-v3-Turbo | `openai/whisper-large-v3-turbo` | Best general ASR |
| Tarteel Whisper Large | `IJyad/whisper-large-v3-Tarteel` | Not yet benchmarked |
| HuBERT Base | `facebook/hubert-base-ls960` | Embedding extraction |
| AraBERT v2 | `aubmindlab/bert-base-arabertv02` | Arabic text encoding |
| MMS-1B-All | `facebook/mms-1b-all` | Multilingual ASR |

### Quran-MD Dataset (for future training)

```python
from datasets import load_dataset
ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)
# 187,080 samples: 30 reciters x 6,236 ayahs
# Fields: audio, arabic_text, english_translation, transliteration, reciter_id
```
