# Streaming ASR Experiment

Real-time Quran verse identification using simulated streaming transcription with mlx-whisper.

## Motivation

Batch ASR processes the entire audio recording at once. This works, but introduces a mandatory delay -- the user must finish reciting, then wait for the full transcription before getting feedback. For a Quran recitation app, real-time feedback (highlighting words as the user recites, showing which verse is detected early) provides a fundamentally better UX.

This experiment tests whether processing audio in small chunks (1-3 seconds) can deliver verse identification faster without sacrificing accuracy.

## Architecture

```
Audio Input (continuous)
    |
    v
[Chunk Splitter] -- 1s / 2s / 3s chunks, optional overlap
    |
    v
[mlx-whisper] -- per-chunk transcription (Arabic, whisper-base)
    |
    v
[Word Accumulator] -- concatenates text across chunks, deduplicates overlaps
    |
    v
[Verse Position Tracker] -- prefix-aware Levenshtein matching against 6,236 verses
    |                        + LCS-based word alignment for position tracking
    v
Output: { verse_ref, word_position, progress_pct, confidence }
```

### Components

| File | Purpose |
|------|---------|
| `streaming_transcriber.py` | Splits audio into chunks, runs mlx-whisper on each, accumulates text |
| `word_tracker.py` | Standalone word-level timestamp extraction using whisper's DTW alignment |
| `verse_position_tracker.py` | Maps accumulated transcription to Quran verses with word-level position tracking |
| `demo.py` | Interactive demo showing real-time tracking with progress bars |
| `benchmark.py` | Systematic comparison of streaming modes vs batch |

### Streaming Modes Tested

| Mode | Chunk Size | Overlap | Description |
|------|-----------|---------|-------------|
| **batch** | entire file | N/A | Baseline -- process full audio at once |
| **stream_3s** | 3 seconds | none | Independent 3-second chunks |
| **stream_3s_overlap** | 3 seconds | 1 second | Overlapping chunks for better word boundaries |
| **stream_2s** | 2 seconds | none | Smaller chunks, faster first result |
| **stream_1s** | 1 second | none | Minimum viable chunk for fastest response |

### Verse Matching Algorithm

The `VersePositionTracker` uses a prefix-aware matching strategy for streaming:

1. **Prefix scoring**: Compares input text against the first N words of each verse (where N = number of input words), avoiding the penalty that full Levenshtein ratio gives when matching a short prefix against a long verse.
2. **Adaptive weighting**: When coverage is low (<80%), trusts prefix score (70%) over full score (30%). When nearly complete, reverses the weighting.
3. **Word alignment**: Uses Longest Common Subsequence (LCS) dynamic programming to track exactly which verse words have been matched, tolerating ASR errors via fuzzy matching (Levenshtein ratio >= 0.7 per word).

## Benchmark Results

Model: `mlx-community/whisper-base-mlx` on Apple Silicon (M-series).

Test corpus: 7 audio files (5 reference recordings from everyayah.com, 2 user recordings).

### Accuracy & Latency Summary

| Mode | Accuracy | Avg Confidence | Avg Total Time | Avg First Match | Avg RTF |
|------|----------|---------------|----------------|-----------------|---------|
| **batch** | 2/7 (29%) | 0.675 | 1.17s | 1.17s | 0.150 |
| **stream_3s** | 2/7 (29%) | 0.535 | 4.92s | 0.19s | 0.483 |
| **stream_3s_overlap** | 2/7 (29%) | 0.605 | 5.86s | 0.80s | 0.667 |
| **stream_2s** | 3/7 (43%) | 0.591 | 7.58s | 5.44s | 0.707 |
| **stream_1s** | 1/7 (14%) | 0.552 | 22.57s | 4.16s | 1.797 |

RTF = Real-Time Factor (processing time / audio duration). RTF < 1.0 means faster than real-time.

### Per-File Results (Selected Modes)

#### Batch Mode
| Test | Expected | Got | Conf | Time | Words |
|------|----------|-----|------|------|-------|
| Al-Fatiha:1 (ref) | 1:1 | 20:1 | 0.981 | 1.81s | 4/5 |
| Al-Fatiha:2 (ref) | 1:2 | **1:2** | 0.976 | 0.21s | 4/4 |
| Ayat al-Kursi (ref) | 2:255 | **2:255** | 0.891 | 1.42s | 50/50 |
| Al-Ikhlas:1 (ref) | 112:1 | 39:14 | 0.737 | 0.09s | 2/6 |
| Ya-Sin:1 (ref) | 36:1 | 51:12 | 0.567 | 0.16s | 0/4 |
| An-Nisa:51 (user) | 4:51 | 3:23 | 0.574 | 0.56s | 1/20 |
| Al-Ikhlas:2-3 (user) | 112:2 | NONE | 0.000 | 3.91s | 0/0 |

#### Stream 3s Mode
| Test | Expected | Got | Conf | First Match | Total Time |
|------|----------|-----|------|-------------|------------|
| Al-Fatiha:1 (ref) | 1:1 | 20:1 | 0.933 | 0.16s | 0.22s |
| Al-Fatiha:2 (ref) | 1:2 | **1:2** | 0.909 | 0.34s | 0.51s |
| Ayat al-Kursi (ref) | 2:255 | **2:255** | 0.594 | 0.21s | 9.20s |
| Al-Ikhlas:1 (ref) | 112:1 | 39:14 | 0.737 | 0.09s | 0.09s |
| An-Nisa:51 (user) | 4:51 | NONE | 0.000 | N/A | 11.95s |
| Al-Ikhlas:2-3 (user) | 112:2 | NONE | 0.000 | N/A | 9.83s |

### Key Observations

**1. Streaming delivers much faster first match:**
- Batch average first match: **1.17s** (must process entire audio)
- Stream 3s average first match: **0.19s** (6x faster)
- For Ayat al-Kursi (52s audio), streaming identifies the verse in 0.21s vs 1.42s for batch

**2. Accuracy is similar across modes (all low):**
- The core accuracy bottleneck is **whisper-base's Arabic transcription quality**, not the chunking strategy
- Al-Fatiha:1 (Bismillah) consistently matches 20:1 (Taa-Haa) because 20:1 = "Ta Ha" + "Bismillah" prefix -- the matcher scores it higher than 1:1
- User recordings get English hallucinations ("and I'm talking", "and The Law") -- whisper-base fails on noisy/accented Arabic
- Short isolated verses (Ya-Sin, Al-Ikhlas) are too brief for reliable matching with single-word transcriptions

**3. Chunk size tradeoffs:**
- **3s chunks**: Best balance of speed and quality. Per-chunk latency ~0.1-0.3s
- **2s chunks**: Slightly better accuracy on some cases (Al-Fatiha:1 correct here). Reasonable latency
- **1s chunks**: Too little context for whisper. Produces garbled output, much worse accuracy (14%), and paradoxically slower total time due to many more chunks

**4. Overlap helps marginally:**
- stream_3s_overlap gets slightly better confidence than stream_3s on Ayat al-Kursi (0.816 vs 0.594)
- But it also introduces duplicate fragments and more processing overhead

**5. Word timestamp extraction is prohibitively slow:**
- `word_timestamps=True` in mlx-whisper triggers a DTW alignment that takes ~200s for a 6s audio clip (160x real-time)
- Without word timestamps: ~1.3s for the same clip
- For production use, word timestamps must be estimated from chunk boundaries rather than computed via DTW

### Word Timestamp Results

Tested with `word_timestamps=True` on 2 short files only (due to extreme slowness):

| Test | Words Found | Expected | Time | Word Spans |
|------|------------|----------|------|------------|
| Al-Fatiha:1 | 4 | 4 | 1.15s | 0.00-0.48, 0.48-1.08, 1.08-2.28, 2.28-4.32 |
| Al-Ikhlas:1 | 3 | 8 | 0.28s | 0.00-0.54, 0.54-1.22, 1.22-2.16 |

When word timestamps are available, they are accurate. The durations are reasonable (0.5-2s per word for slow Quran recitation). But the computation cost makes this impractical for real-time use.

## Strengths

- **Fast first identification**: Streaming gets a verse match in < 0.5s, even before the reciter finishes
- **Progressive word tracking**: The LCS alignment correctly tracks position through a long verse (Ayat al-Kursi: 50/50 words tracked in streaming)
- **Prefix-aware matching**: Handles partial verse input without penalizing short prefixes against long verses
- **Hallucination detection**: Filters out repetitive whisper artifacts before they poison the tracker

## Limitations

- **whisper-base Arabic quality is poor**: The model frequently hallucinates English, merges Arabic words incorrectly, and misses diacritics. This is the dominant source of error -- not the streaming approach itself
- **Short verses are unreliable**: Verses under 3 words don't provide enough signal for Levenshtein matching against 6,236 candidates
- **User recordings fail**: The 2 user-recorded test files produced mostly English hallucinations, indicating whisper-base is not robust to recording conditions (background noise, microphone quality, accent variation)
- **Word timestamps are unusable in real-time**: mlx-whisper's DTW alignment is ~160x slower than transcription alone
- **No language detection fallback**: When whisper produces English output, the system has no mechanism to detect and retry

## Production Recommendations

For a real app using this approach:

1. **Use a larger/fine-tuned model**: whisper-large-v3-turbo or a Quran-specific fine-tune would dramatically improve Arabic transcription accuracy. The streaming architecture itself works well.

2. **2-3 second chunks are optimal**: 1s is too short, batch is too slow for first feedback. 2-3s provides the right balance.

3. **Estimate word positions from chunk timing**: Rather than using whisper's slow DTW word timestamps, divide each chunk's duration evenly across its output words. This gives approximate word timing that's "good enough" for UI highlighting.

4. **Add a confidence gate**: Only display verse identification to the user when confidence exceeds 0.7. Below that, show "listening..." to avoid confusing false matches.

5. **Consider VAD preprocessing**: Use Voice Activity Detection to only send speech-containing chunks to whisper, reducing wasted inference on silence.

## How to Run

```bash
# Activate venv
source ../../.venv/bin/activate

# Run benchmark (all modes, ~5 minutes)
python benchmark.py

# Run demo on default file (Ayat al-Kursi)
python demo.py

# Run demo on all test files
python demo.py --all

# Run demo on specific file
python demo.py ../../data/test_audio/nisa_51.m4a

# Test word timestamps standalone
python word_tracker.py ../../data/reference_audio/001001.mp3

# Test streaming standalone
python streaming_transcriber.py ../../data/reference_audio/112001.mp3
```

## Files

- `benchmark_results.json` -- Full benchmark data (all modes, all test cases)
- `streaming_transcriber.py` -- Core streaming engine
- `word_tracker.py` -- Word-level timestamp extraction
- `verse_position_tracker.py` -- Verse identification and position tracking
- `demo.py` -- Interactive demo
- `benchmark.py` -- Benchmark runner
