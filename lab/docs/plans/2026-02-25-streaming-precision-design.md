# Streaming Precision Recovery Design

## Problem

Chunked streaming helps multi-ayah detection but tanks single-ayah precision. Short chunks produce garbage transcripts, the tracker matches garbage against 6,236 verses and always finds something, and there's no mechanism to reject bad input or hold suspect emissions.

| Metric | Full transcript | Streaming (5s) |
|---|---|---|
| Single-ayah SeqAcc | 74% | 50% |
| Single-ayah Precision | 77% | 60% |
| Multi-ayah SeqAcc | 0% | 22% |
| Multi-ayah Recall | 28% | 66% |

## Architecture

Four layers of defense, each independent and testable:

```
Audio chunk
  → [Layer 1: Confidence Gate]     reject garbage chunks before they enter the tracker
  → [Layer 2: Streaming Thresholds] higher bar for what counts as a "match"
  → [Layer 3: Tracker Matching]     unchanged core logic
  → [Layer 4: Emission Buffer]      hold suspect emissions, confirm or retract
  → Confirmed verse emission
```

## Layer 1: Chunk Confidence Gating

**Where:** `shared/streaming.py` (pipeline) + `experiments/streaming-asr/streaming_transcriber.py`

Use Whisper's own confidence signal to reject bad chunks before they reach the tracker. Whisper segments include `avg_logprob` (average log-probability of decoded tokens). Clean Arabic produces values around -0.3 to -0.5. Garbled output from short/noisy chunks drops to -1.0 or worse.

**Changes:**
- `transcribe_fn` interface: `(path) -> str | dict`. When returning a dict: `{"text": str, "avg_logprob": float}`. Backward compatible.
- `StreamingTranscriber` exposes `avg_logprob` by averaging across segments.
- `StreamingPipeline.run_on_audio_chunked` checks logprob and skips chunks below threshold.
- Skip chunks producing < 2 words.

**Constants:**
```python
MIN_CHUNK_LOG_PROB = -1.0
MIN_CHUNK_WORDS = 2
```

## Layer 2: Streaming-Aware Thresholds

**Where:** `shared/verse_tracker.py`

The tracker uses `MIN_EMIT_SCORE = 0.3` for both modes. 0.3 is way too permissive for noisy chunked input.

**Changes:**
- `VerseTracker.__init__` gets `streaming_mode: bool = False`.
- When `streaming_mode=True`:
  - `MIN_EMIT_SCORE` becomes 0.5 (up from 0.3).
  - Minimum accumulation gate: `_find_best_match` returns `None` if accumulated words < 3.
- Non-streaming mode unchanged.

**Constants:**
```python
STREAMING_MIN_EMIT_SCORE = 0.5
MIN_WORDS_FOR_MATCH = 3
```

## Layer 3: Core Tracker Logic

No changes. The existing `_evaluate`, `_score_verse`, `_emit`, `_try_split_and_emit` logic stays as-is.

## Layer 4: Buffered Confirmation

**Where:** `shared/streaming.py` (pipeline level, wrapping tracker output)

Instead of immediately reporting every emission, hold low-confidence ones as "tentative" and confirm or retract based on what happens next.

**Key asymmetry:** Real verses are temporally stable — the next chunk produces coherent text. Spurious matches from garbage are random — the next chunk matches something unrelated or gets gated out.

**Emission rules:**

| Score | Action |
|---|---|
| >= 0.7 | Immediate emit. Also confirms any pending tentative. |
| 0.5 - 0.7 | Hold as tentative. If already a tentative, confirm the old one. |

**Tentative resolution:**

| Event | Action |
|---|---|
| Next valid chunk arrives (passes gate) | Confirm |
| 3 consecutive gated/skipped chunks | Retract |
| `finalize()` (end of audio) | Confirm if score >= 0.5, else retract |

**Constants:**
```python
HIGH_CONFIDENCE_THRESHOLD = 0.7
MAX_HOLD_CHUNKS = 3
```

**Implementation (pipeline level):**
```python
tentative = None
tentative_age = 0
confirmed = []

for each chunk:
    text, logprob = transcribe_with_confidence(chunk)

    if logprob < MIN_CHUNK_LOG_PROB or word_count < 2:
        tentative_age += 1
        if tentative and tentative_age >= MAX_HOLD_CHUNKS:
            tentative = None
        continue

    emissions = tracker.process_delta(text)

    if tentative:
        confirmed.append(tentative)
        tentative = None
        tentative_age = 0

    for e in emissions:
        if e["score"] >= HIGH_CONFIDENCE_THRESHOLD:
            confirmed.append(e)
        else:
            if tentative:
                confirmed.append(tentative)
            tentative = e
            tentative_age = 0

if tentative and tentative["score"] >= STREAMING_MIN_EMIT_SCORE:
    confirmed.append(tentative)
confirmed.extend(tracker.finalize())
```

## Files Changed

| File | Change |
|---|---|
| `shared/verse_tracker.py` | Add `streaming_mode` param, conditional thresholds, min-word gate |
| `shared/streaming.py` | Confidence gating + emission buffer in `run_on_audio_chunked` |
| `experiments/streaming-asr/streaming_transcriber.py` | Expose `avg_logprob` from Whisper result |
| `benchmark/runner.py` | Pass `streaming_mode=True` to tracker when using chunked pipeline |

## What Doesn't Change

- `_score_verse`, `_emit`, `_try_split_and_emit`, `_evaluate`
- `process_text` path (full-transcript mode)
- `run_on_full_transcript`
- Test corpus, manifest, scoring logic
- All non-streaming experiments

## Expected Impact

| Metric | Current Streaming | Projected |
|---|---|---|
| Single-ayah Precision | 60% | ~75% |
| Single-ayah SeqAcc | 50% | ~68% |
| Multi-ayah Recall | 66% | ~60-66% |
| Multi-ayah SeqAcc | 22% | ~25-30% |

## Tuning Strategy

All thresholds are constants at the top of their files. After implementation:
1. Run benchmark with current thresholds
2. If precision still low: tighten `MIN_CHUNK_LOG_PROB` (e.g., -0.8) or raise `STREAMING_MIN_EMIT_SCORE`
3. If recall drops too much: loosen `MIN_CHUNK_LOG_PROB` (e.g., -1.2) or lower `MIN_WORDS_FOR_MATCH` to 2

## Future Additions (not in scope)

- Context conditioning: pass previous transcript as Whisper's `initial_prompt`
- Adaptive chunking: split at silence boundaries instead of fixed intervals
- Match margin filter: reject matches where best and 2nd-best scores are too close
