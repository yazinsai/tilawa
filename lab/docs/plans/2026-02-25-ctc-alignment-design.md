# CTC Forced Alignment for Quran Verse Recognition

**Date:** 2026-02-25

## Core Idea

Replace "transcribe then search" with "encode then align." A CTC encoder produces frame-level character scores from audio. We score candidate verses directly against those frames. No decoder, no hallucination.

## Architecture

```
Audio (2-3s chunk)
  → wav2vec2 CTC encoder (fine-tuned on Quran)
  → frame-level character logits  [T × V matrix]
  → Two parallel paths:

    Path A (pruning):
      argmax greedy decode → rough Arabic text → Levenshtein top-50 candidates

    Path B (scoring):
      CTC forward algorithm on each candidate → alignment score

  → Best scoring candidate → verse ID + alignment path → word position
```

## Components

1. **CTC Model** — wav2vec2-xls-r-300m fine-tuned with CTC on Quran audio (Modal A10G)
2. **CTC Scorer** — CTC forward algorithm: P(text | audio frames)
3. **Candidate Pruner** — greedy decode + Levenshtein top-50
4. **Sequential Tracker** — after first match, narrow to next ~10 verses

## Training

- Base: `facebook/wav2vec2-xls-r-300m` (300M, multilingual)
- Dataset: `Buraaq/quran-md-ayahs` (187K samples, 30 reciters)
- CTC loss, Arabic character vocab (~40 chars)
- Modal A10G, 10-20 epochs, batch 16-32

## Experiment Interface

Follows repo convention: `experiments/ctc-alignment/run.py` with `predict()` and `model_size()`.
