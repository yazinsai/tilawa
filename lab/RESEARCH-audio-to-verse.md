# Deep Research: Cutting-Edge Approaches for Offline Audio-to-Text Verse Identification

**Date:** 2026-02-24
**Scope:** Beyond standard Whisper fine-tuning -- what's possible TODAY for offline Quran verse identification on Apple Silicon

---

## 1. Audio Embedding Models for Direct Audio-to-Verse Matching

### The Core Idea
Instead of transcribing audio to text and then matching text to verses, map audio directly into an embedding space where similar recitations cluster together. This eliminates the ASR bottleneck entirely.

### Models Available Today

| Model | Type | Embedding Dim | Key Strength | On-Device Feasibility |
|-------|------|--------------|--------------|----------------------|
| **CLAP** (LAION) | Audio-Text contrastive | 512 | Maps audio AND text to same space | Medium (330M params) |
| **tinyCLAP** | Distilled CLAP | 512 | Only 6% of CLAP params, <5% perf drop | Excellent |
| **WavLink** | Whisper + global token | Sub-100 dim possible | SOTA retrieval, 8x smaller embeddings via Matryoshka | Excellent |
| **HuBERT** | Self-supervised speech | 256/768/1024 | Strong semantic structure, cosine similarity works well | Good (Base: 95M) |
| **WavLM** | Self-supervised speech | 768/1024 | Robust to noise, overlapping speech | Good |
| **BEATs** | Audio pre-training | 768 | Iterative audio pre-training, strong on audio events | Good |
| **AudioMAE** | Masked autoencoder | Variable | Self-supervised, captures fine-grained audio patterns | Good |

### Practical Assessment for Verse Matching

**WavLink (January 2026)** is the most exciting development. It augments Whisper's encoder with a single learnable "global token" that compresses a 30-second clip from 1500 frame features down to a single embedding. With Matryoshka-style training, embeddings as small as sub-100 dimensions retain competitive retrieval performance. This means:
- Pre-compute one embedding per verse per reciter = very small database
- At inference: encode user audio -> nearest neighbor search -> return matching verse
- Leverages Whisper's existing multilingual capabilities (already good at Arabic)

**HuBERT** embeddings have strong semantic structure where nearby representations correspond to similar linguistic content. This means cosine similarity search over pre-computed verse embeddings is viable. The Base model at 95M params runs comfortably on Apple Silicon.

**tinyCLAP** uses only 6% of original CLAP parameters with minimal quality loss. Uses PhiNet (an edge-optimized backbone). Ideal for on-device deployment. However, CLAP was trained primarily on environmental sounds, not speech/recitation -- would need fine-tuning.

### Verdict
**WavLink + FAISS on-device is the most promising novel approach.** Pre-compute embeddings for all 6,236 verses across N reciters, store in a FAISS index, and do nearest-neighbor lookup at inference. This completely sidesteps ASR.

---

## 2. Whisper Alternatives with Arabic Support

### Models Ranked by Practicality for Offline Arabic ASR on Apple Silicon

#### Tier 1: Ready to Deploy Today

**Moonshine v2** (Feb 2026)
- Streaming ergodic encoder architecture with sliding-window self-attention
- **Arabic model officially released** under permissive open-source license
- Bounded time-to-first-token independent of utterance length
- Runs on-device: Python, iOS, Android, macOS, Linux, Raspberry Pi
- Matches models 6x their size while running significantly faster
- Local attention rivals global attention accuracy at fraction of compute
- Source: https://github.com/moonshine-ai/moonshine

**Parakeet TDT (NVIDIA) via CoreML**
- 0.6B params, RTFx > 2,000 (processes audio dramatically faster than Whisper)
- **CoreML conversion available** (FluidInference/parakeet-tdt-0.6b-v3-coreml)
- On M4 Pro: 110x RTF = 1 minute of audio in ~0.5 seconds
- On iPhone 16 Pro: Neural Engine is 4.3x faster than GPU for this model
- **Caveat: English-only currently.** No Arabic support.
- Source: https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml

**WhisperKit (Argmax)**
- Native Swift implementation optimized for Apple Neural Engine (ANE)
- Runs Whisper Large v3 Turbo (1B params) on-device
- Block-diagonal attention + KV caching for streaming
- 0.46s mean hypothesis latency, 2.2% WER
- **Full Arabic support** (inherits from Whisper)
- Streaming support with partial audio processing
- Source: https://github.com/argmaxinc/WhisperKit

#### Tier 2: Viable with Some Work

**Meta MMS (Massively Multilingual Speech)**
- 1B params, covers 1,107 languages including Arabic
- Based on wav2vec 2.0 architecture
- Trained partly on religious texts (including Quran-adjacent data)
- Can run offline via HuggingFace Transformers
- Requires uroman preprocessing for Arabic text
- Source: https://huggingface.co/facebook/mms-1b-all

**SeamlessM4T v2 (Meta)**
- Supports Arabic (MSA + Moroccan + Egyptian dialects)
- Speech-to-text, translation, and more in one model
- SeamlessStreaming variant supports near-real-time ASR
- Open-source, offline-capable
- Heavier than alternatives (~2.3B params)
- Source: https://huggingface.co/facebook/seamless-m4t-v2-large

**NVIDIA Canary 1B**
- Tops Hugging Face Open ASR Leaderboard (5.63% WER)
- Strong multilingual including Arabic
- Flash variant for fast inference
- Would need conversion for Apple Silicon (NeMo framework)
- Source: https://developer.nvidia.com/blog/nvidia-speech-ai-models-deliver-industry-leading-accuracy-and-performance/

#### Tier 3: Worth Watching

**Google USM** - 2B params, 300+ languages, but only available via private API. Not open-source.

**GLM-ASR-Nano** - Streaming and batch modes, but focused on Chinese/English.

### Verdict
**WhisperKit is the best ready-to-deploy option for Arabic on Apple Silicon today.** Moonshine v2 with its new Arabic model is the most exciting alternative for edge/streaming scenarios. Parakeet is blazing fast but English-only.

---

## 3. Audio Fingerprinting at Scale

### Traditional Approaches

| System | Algorithm | Min Query | Key Feature | License |
|--------|-----------|-----------|-------------|---------|
| **Panako** | Spectral peaks | 5 seconds | Handles pitch shift, time stretch | AGPL |
| **Dejavu** | Constellation (Shazam-like) | 5 seconds | Python, simple API | MIT |
| **Chromaprint** | Chroma features | ~10 seconds | Very compact fingerprints, AcoustID | LGPL |

### Neural Audio Fingerprinting (State of the Art)

Recent advances (2025-2026):
- **Contrastive learning** with triplet loss and hard positive/negative sampling outperforms older objectives
- **Realistic degradation chains** (room impulse responses, mic characteristics, reverb) critical for real-world robustness
- **PeakNetFP** (June 2025): Peak-based neural fingerprinting for improved efficiency
- **Music Foundation Model fingerprints** (2025): Using pre-trained music models as feature extractors for fingerprinting
- **Synthetic fingerprint generation** via Rectified Flow diffusion models for benchmarking at population scale
- **HRR (Holographic Reduced Representations)** for aggregating fingerprints while preserving position information

### Feasibility for 6,236 Verses x N Reciters

**The math works out beautifully:**
- 6,236 verses x 10 reciters = 62,360 reference embeddings
- At 512-dim float32: 62,360 x 512 x 4 bytes = ~128 MB
- At 128-dim float32: ~32 MB
- With WavLink sub-100-dim: ~25 MB
- With quantization (int8): ~6-12 MB

This fits trivially in memory on any Apple Silicon device. FAISS with HNSW index can search millions of vectors in <1ms. For 62K vectors, brute-force cosine similarity would work fine.

### Existing Quran Fingerprinting Work
- **"Sout Albayan"** - An existing iPhone app that uses audio fingerprinting to identify reciter, ayah, and surah
- Research combining pre-trained embedding models + speech-to-text + Elasticsearch for verse identification
- 2025 paper achieving 97.12% reciter recognition rate using VGG features on Quran chapter 36

### Vector Search on Apple Silicon
- **Faiss-mlx**: Metal-accelerated FAISS port for Apple Silicon via MLX framework
- Standard FAISS also compiles natively on macOS with ARM support
- HNSW handles up to tens of millions of vectors efficiently
- For 62K vectors, even flat index with brute-force search is sub-millisecond

### Verdict
**Pre-computing embeddings for all verses across multiple reciters and doing nearest-neighbor search is absolutely feasible and likely the fastest approach.** The database would be tiny (6-128 MB depending on embedding dim and quantization). Combined with WavLink or HuBERT embeddings, this could provide sub-100ms verse identification.

---

## 4. Distilled/Quantized Models for On-Device Arabic Speech

### Whisper Compression Techniques

| Technique | Compression | WER Impact | Source |
|-----------|-------------|------------|--------|
| **INT8 quantization** | ~2x size reduction | No WER deterioration | arxiv:2503.09905 |
| **INT4 quantization** | ~4x (69% size reduction) | Slight improvement in some cases | arxiv:2503.09905 |
| **DQ-Whisper** (distill + quantize) | 5.18-10.48x | Negligible WER loss | arxiv:2305.10788 |
| **Distil-Whisper Large v3** | 6x faster, 50% smaller | Within 1% WER | HuggingFace |
| **WhisperKit** (ANE optimized) | Full Large v3 on-device | 2.2% WER | Argmax |

### Critical Warning for Arabic
Quantization **amplifies bias against low-resource languages**. Arabic (especially dialectal) can be disproportionately affected. Key findings:
- All models achieve best results on MSA (Modern Standard Arabic)
- Significant WER decline on dialects (Egyptian, Khaliji)
- Blindly applying INT4 to small models for Arabic could increase errors substantially
- **Recommendation: Use larger models (Large v3) with moderate quantization (INT8) rather than tiny models with aggressive quantization for Arabic**

### Quran-Specific Fine-Tuned Models (Available on HuggingFace)

| Model | Base | WER | Notes |
|-------|------|-----|-------|
| **tarteel-ai/whisper-base-ar-quran** | Whisper Base | 5.75% | Purpose-built for Quranic Arabic |
| **KheemP/whisper-base-quran-lora** | Whisper Base (LoRA) | 5.98% | Diacritic-sensitive, 2024 |
| **IJyad/whisper-large-v3-Tarteel** | Whisper Large v3 | Not published | Fine-tuned on Tarteel Everyayah dataset |
| **tarteel-ai/whisper-tiny-ar-quran** | Whisper Tiny | Higher | Smallest, fastest option |

### Optimal On-Device Strategy
1. **Best quality**: whisper-large-v3-Tarteel + INT8 quantization via WhisperKit on ANE
2. **Best speed**: whisper-base-ar-quran + INT4 quantization via mlx-whisper
3. **Best balance**: Distil-Whisper Large v3 fine-tuned on Quran data + INT8

### MLX-Whisper on Apple Silicon
- 30-40% faster than other implementations
- Supports 4bit and 8bit quantization
- Distil-Large-v3 available as MLX model
- Lightning Whisper MLX: extremely fast implementation specifically for Apple Silicon

### Verdict
**The tarteel-ai models with INT8 quantization via WhisperKit/MLX is the most practical path for on-device Quran ASR today.** The Large v3 Tarteel fine-tune running on ANE via WhisperKit would give the best quality. For faster inference, the base model with INT4 via mlx-whisper.

---

## 5. Streaming ASR for Real-Time Verse Identification

### Available Streaming Approaches

| Model | Architecture | Latency | Chunk Size | Arabic Support |
|-------|-------------|---------|------------|----------------|
| **Moonshine v2** | Ergodic streaming encoder | Bounded TTFT | Variable | Yes |
| **WhisperKit** | Modified Whisper + ANE | 0.46s | Streaming | Yes (via Whisper) |
| **NVIDIA Nemotron Streaming** | FastConformer + RNN-T | <300ms | 80-160ms | English only |
| **SeamlessStreaming** | SeamlessM4T v2 + streaming | Near real-time | Configurable | Yes |
| **Kyutai** | Streaming decoder | 1-2.5s to first output | Streaming | Limited |

### How Streaming Works
- Audio is sliced into 100-200ms chunks
- Sliding-window causal attention with limited right-chunk lookahead
- Cached encoder states avoid recomputation
- New audio chunks are incrementally processed while reusing cached context
- End-to-end latency typically under 300ms

### Moonshine v2 Streaming Details (Most Relevant)
- Ergodic streaming with sliding-window self-attention
- Caches input encoding and part of decoder state between chunks
- Skips recomputation for previously seen audio
- TTFT is bounded regardless of utterance length
- Arabic model specifically trained and released

### WhisperKit Streaming Details
- Modified Whisper architecture for native streaming
- Audio Encoder natively supports streaming inference
- Text Decoder yields accurate output even on partial audio
- Runs on Apple Neural Engine for maximum efficiency
- 0.46s mean hypothesis latency

### Practical Application for Verse Tracking
For "tracking position within a verse as it's being recited":
1. Stream audio chunks every 100-200ms
2. Run incremental ASR on each chunk
3. Compare partial transcription against known verse text
4. Update UI with current word position in real-time
5. Use word-level timestamps (see section 7) for precise alignment

### Verdict
**Moonshine v2 (Arabic) for streaming ASR + WhisperKit for iOS deployment.** Both support incremental processing with cached context. Moonshine v2's explicit Arabic model and streaming-first design make it ideal for live verse tracking.

---

## 6. Contrastive Learning for Audio-to-Verse Lookup

### The Architecture

```
                    Shared Embedding Space

   Audio Encoder -----> [embedding] <----> [embedding] <----- Text Encoder
   (recitation)              |                    |           (verse text)
                             |                    |
                        cosine similarity
                             |
                      verse identification
```

### Available Frameworks

**CLAP (Contrastive Language-Audio Pretraining)**
- Audio analog of CLIP
- Trains audio + text encoders to maximize similarity for aligned pairs
- At inference: compute audio embedding, find nearest text embedding
- GLAP (2025): Extends to 145 languages via auto-translation
- Source: https://github.com/LAION-AI/CLAP

**WavLink (January 2026)**
- Builds on Whisper encoder with learnable global token
- Surpasses all CLAP variants in retrieval tasks
- Matryoshka supervision: embeddings from 768d down to sub-100d
- Two-stage training recipe across three model sizes
- Competitive with much larger audio-LLMs (Qwen2-Audio, Falcon3-Audio)
- Source: https://arxiv.org/abs/2601.15118

**tinyCLAP**
- 6% of CLAP parameters, <5% performance drop
- PhiNet backbone optimized for edge
- Unimodal distillation (doesn't need captions at distillation time)
- Source: https://arxiv.org/abs/2311.14517

### Custom Training for Quran Verse Matching

You could train a contrastive model specifically for Quran:

1. **Positive pairs**: (audio of verse X by any reciter, text of verse X)
2. **Negative pairs**: (audio of verse X, text of verse Y)
3. **Architecture**: Whisper encoder (audio) + Arabic text encoder (text)
4. **Loss**: InfoNCE / NT-Xent contrastive loss
5. **Data**: Everyayah dataset (multiple reciters, all verses)

**Advantages over pure ASR:**
- No need for perfect transcription
- Handles recitation style variations inherently
- Direct verse lookup without text matching step
- Robust to tajweed variations (elongation, nasalization, etc.)

**Key insight from 2025 research**: "Promptable embeddings" allow a single model to generate task-specific representations. Instead of a fixed vector, embeddings are conditional on both content AND an instruction. This could enable queries like "find the verse being recited" vs "identify the reciter" from the same model.

### Verdict
**Training a custom contrastive model on Quran data is the most powerful novel approach.** Use WavLink's architecture (Whisper encoder + global token) as the audio side, and an Arabic text encoder for the text side. Pre-compute text embeddings for all 6,236 verses. At inference, encode audio and find nearest verse embedding. This approach handles reciter variation, tajweed differences, and partial recitations naturally.

---

## 7. Whisper Word-Level Timestamps

### Available Tools

| Tool | Approach | Precision | Arabic Support | Speed |
|------|----------|-----------|----------------|-------|
| **WhisperX** | Forced alignment via wav2vec2 | High (phoneme-level) | Yes (via HF models) | Fast |
| **CrisperWhisper** | DTW on cross-attention scores | Higher than WhisperX | Needs verification | Medium |
| **whisper-timestamped** | DTW on cross-attention weights | Good | Yes | Medium |
| **stable-ts** | Forced alignment | Good | Yes | Fast |
| **Whisper Internal Aligner** (2025) | Attention head filtering + teacher forcing | Best (outperforms WhisperX) | Needs testing | Fast |

### WhisperX Details
- Uses wav2vec 2.0 phoneme model for forced alignment
- Transfers timing from CTC-based model onto Whisper transcripts
- Supports Arabic via HuggingFace alignment models
- Pipeline: Whisper transcription -> phoneme alignment -> word timestamps
- Source: https://github.com/m-bain/whisperX

### CrisperWhisper (2024)
- Fine-tunes Whisper for verbatim transcription
- Dynamic time warping on decoder cross-attention scores
- More robust in noisy conditions than WhisperX
- Better at timestamping pauses between words
- Does NOT depend on external wav2vec2 model
- Source: https://arxiv.org/html/2408.16589v1

### Whisper's Internal Word Aligner (September 2025)
- Discovery: Whisper already contains attention heads that capture accurate word alignments
- Unsupervised approach: filter attention heads while teacher-forcing with characters
- Outperforms WhisperX on TIMIT and LibriSpeech benchmarks
- No external alignment model needed
- Source: https://arxiv.org/html/2509.09987v1

### Application to Real-Time Verse Position Tracking

```
User reciting verse 2:255 (Ayat al-Kursi):
"Allahu la ilaha illa huwa alhayyu alqayyumu..."

Timeline:
[0.0s] "Allahu" -> word 1/50 highlighted
[0.3s] "la" -> word 2/50 highlighted
[0.5s] "ilaha" -> word 3/50 highlighted
[0.8s] "illa" -> word 4/50 highlighted
[1.0s] "huwa" -> word 5/50 highlighted
...
```

**Implementation approach:**
1. Run streaming ASR (Moonshine v2 or WhisperKit)
2. Get word-level timestamps from each chunk
3. Match recognized words against verse text using fuzzy matching
4. Highlight current word position in the Quran text display
5. Handle tajweed elongations (madd) that stretch single words

### Verdict
**Whisper's internal aligner (2025 discovery) is the cleanest approach** -- no external models needed. For production, WhisperX with Arabic wav2vec2 alignment models is proven and reliable. CrisperWhisper offers better noise robustness but Arabic support needs verification.

---

## Recommended Architecture: Hybrid Approach

Based on all research, here is the optimal architecture for offline Quran verse identification:

### Phase 1: Quick Verse Identification (< 500ms)
```
User Audio (3-5 sec)
  -> HuBERT/WavLink Embedding
  -> FAISS Nearest Neighbor Search (pre-computed verse embeddings)
  -> Top-5 candidate verses
```

### Phase 2: Confirmation + Position Tracking (streaming)
```
User Audio (streaming, 100ms chunks)
  -> Moonshine v2 Arabic / WhisperKit streaming ASR
  -> Word-level timestamps (internal aligner)
  -> Fuzzy match against candidate verse texts
  -> Real-time word position highlighting
```

### Phase 3: Precision Refinement
```
If confidence < threshold:
  -> Run full Whisper (tarteel-ai/whisper-large-v3-Tarteel)
  -> Exact text match against Quran corpus
  -> Return definitive verse ID
```

### Pre-computation (one-time, offline)
```
For each verse (6,236) x each reciter (N):
  -> Extract HuBERT/WavLink embedding
  -> Store in FAISS index (~6-128 MB total)
  -> Store verse text with diacritics for matching
```

### Hardware Requirements (Apple Silicon)
- **iPhone**: WhisperKit on ANE, FAISS flat index in RAM
- **Mac (M-series)**: mlx-whisper or Moonshine v2, Faiss-mlx with Metal acceleration
- **Storage**: ~100-500 MB for models + embeddings
- **RAM**: ~500 MB - 2 GB during inference
- **Latency target**: < 500ms for verse ID, real-time for word tracking

---

## Key Sources

1. [WavLink: Compact Audio-Text Embeddings](https://arxiv.org/abs/2601.15118) - January 2026
2. [Moonshine v2 Streaming ASR](https://arxiv.org/abs/2602.12241) - February 2026
3. [WhisperKit On-Device ASR](https://github.com/argmaxinc/WhisperKit) - Argmax
4. [Parakeet TDT CoreML](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) - FluidInference
5. [Whisper Internal Word Aligner](https://arxiv.org/html/2509.09987v1) - September 2025
6. [tinyCLAP](https://arxiv.org/abs/2311.14517) - Edge-optimized CLAP
7. [DQ-Whisper Compression](https://arxiv.org/abs/2305.10788) - Distillation + Quantization
8. [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) - Quran-specific Whisper
9. [Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx) - Metal-accelerated FAISS for Apple Silicon
10. [Neural Audio Fingerprinting](https://ieeexplore.ieee.org/document/9414337/) - Contrastive learning for audio retrieval
11. [Quran Recitation Recognition E2E](https://arxiv.org/abs/2305.07034) - End-to-end deep learning
12. [Best Open Source STT 2026 Benchmarks](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
13. [CLAP - LAION](https://github.com/LAION-AI/CLAP) - Contrastive Language-Audio Pretraining
14. [WhisperX Word Timestamps](https://github.com/m-bain/whisperX)
15. [CrisperWhisper](https://arxiv.org/html/2408.16589v1) - Accurate word timestamps
16. [Panako Fingerprinting](https://github.com/JorenSix/Panako)
17. [MLX Whisper Quantized](https://github.com/mustafaaljadery/lightning-whisper-mlx)
18. [Meta MMS](https://huggingface.co/facebook/mms-1b-all) - 1,107 language ASR
19. [SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) - Multilingual speech
20. [Open Universal Arabic ASR Leaderboard](https://www.isca-archive.org/interspeech_2025/wang25_interspeech.pdf) - Interspeech 2025
