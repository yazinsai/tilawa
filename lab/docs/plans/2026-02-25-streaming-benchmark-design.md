# Streaming Verse Detection — Design

## Problem

The current benchmark feeds whole audio clips to models and expects a single `(surah, ayah, ayah_end)` prediction. The real product is continuous streaming — users recite and we identify verses on the fly. Multi-ayah samples fail because nothing segments the audio or tracks verse transitions.

## Architecture

```
Audio → [Chunker] → [ASR Backend] → [Verse Tracker] → [(surah, ayah)] emissions
                       ↑ pluggable
                 (ctc / whisper-lora / tarteel-whisper-base / mlx-whisper)
```

Three layers:

1. **Chunker** (from existing `StreamingTranscriber`): Splits audio into overlapping chunks, feeds to ASR, produces accumulated transcript over time.

2. **ASR Backend** (pluggable): Any experiment. Each exposes `transcribe(audio_path) -> str`. Existing `predict()` functions already transcribe internally — we extract that step.

3. **Verse Tracker** (new, replaces `MultiVerseTracker` stub): Takes accumulated transcript, continuously matches against QuranDB. Detects verse completion, emits `(surah, ayah)`, resets accumulation for next verse.

## Verse Transition Detection

The tracker maintains:
- `current_best_match`: verse with highest score against accumulated text
- `peak_score`: highest score seen for current match

A verse is emitted when:
- **Score drop**: match score drops significantly below peak (transcript moved past this verse)
- **Match shift**: best-matching verse changes (entered new territory)
- **End of audio**: accumulated text emitted as final detection

After emission, trim accumulated text (remove matched portion) and continue.

## Continuation Bias

Many verses share identical text (e.g. "فبأي آلاء ربكما تكذبان" 31x in Ar-Rahman, bismillah in 113 surahs). Context resolves ambiguity:

- If already emitted verse N from surah S, strongly prefer S:N+1 as next match
- Scoring bonus: `effective_score = levenshtein_score + CONTINUATION_BONUS` when candidate is the next sequential verse
- Tunable bonus (start with 0.15)
- No prior context: fall back to best global match

## Pluggable Backend Interface

Each experiment adds:

```python
def transcribe(audio_path: str) -> str:
    """Return raw Arabic transcript without verse matching."""
```

## Benchmark Scoring

All 55 samples go through the streaming pipeline. Expected output is an ordered sequence of `(surah, ayah)`.

- Single-ayah `2:255` → expected: `[(2, 255)]`
- Multi-ayah `103:1-3` → expected: `[(103, 1), (103, 2), (103, 3)]`

Metrics:
- **Verse recall**: fraction of expected verses detected (in order)
- **Precision**: fraction of detected verses that are correct
- **Sequence accuracy**: 1.0 only if full ordered sequence matches exactly
- **Latency**: time to first correct detection
- Per-experiment, per-category breakdowns

## File Changes

| File | Change |
|---|---|
| `shared/streaming.py` | New — pluggable streaming pipeline (chunker + verse tracker) |
| `shared/quran_db.py` | Add `get_next_verse(surah, ayah)` helper |
| `experiments/*/run.py` | Add `transcribe()` function to each |
| `benchmark/runner.py` | Rewrite eval to use streaming pipeline, score ordered sequences |
| `benchmark/test_corpus/manifest.json` | Expand expected format to explicit verse sequences |
