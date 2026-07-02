# Streaming Precision Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover single-ayah precision in streaming mode without sacrificing multi-ayah recall, using confidence gating, streaming thresholds, and buffered confirmation.

**Architecture:** Four independent layers: (1) reject bad chunks via Whisper's avg_logprob, (2) raise match thresholds in streaming mode, (3) unchanged core tracker, (4) hold low-confidence emissions for 1-chunk confirmation. All layers are additive and testable independently.

**Tech Stack:** Python, mlx-whisper, Levenshtein, pytest

---

### Task 1: Streaming-Aware VerseTracker Thresholds

**Files:**
- Modify: `shared/verse_tracker.py`
- Modify: `tests/test_verse_tracker.py`

**Step 1: Write failing tests for streaming_mode**

Add to `tests/test_verse_tracker.py`:

```python
def test_streaming_mode_rejects_low_score():
    """In streaming mode, MIN_EMIT_SCORE is 0.5, so a weak match shouldn't emit."""
    tracker = VerseTracker(db, streaming_mode=True)
    # A short garbled fragment that might match something at 0.3-0.4
    garbage = "يا ايها"
    emissions = tracker.process_delta(garbage)
    emissions += tracker.finalize()
    # Should emit nothing — score too low for streaming threshold
    assert emissions == []


def test_streaming_mode_min_words_gate():
    """In streaming mode, fewer than 3 accumulated words should not match."""
    tracker = VerseTracker(db, streaming_mode=True)
    # Just 2 words — should be gated even if they partially match a verse
    emissions = tracker.process_delta("بسم الله")
    assert emissions == []
    # Now add more words to cross the threshold
    v = db.get_verse(1, 1)
    emissions = tracker.process_delta(v["text_clean"])
    emissions += tracker.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1


def test_non_streaming_mode_unchanged():
    """Non-streaming mode should still work with the old MIN_EMIT_SCORE of 0.3."""
    tracker = VerseTracker(db)  # default: streaming_mode=False
    v = db.get_verse(1, 1)
    emissions = tracker.process_text(v["text_clean"])
    emissions += tracker.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1 and emissions[0]["ayah"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_verse_tracker.py::test_streaming_mode_rejects_low_score tests/test_verse_tracker.py::test_streaming_mode_min_words_gate -v`

Expected: FAIL — `VerseTracker.__init__()` doesn't accept `streaming_mode`

**Step 3: Implement streaming_mode in VerseTracker**

In `shared/verse_tracker.py`:

1. Add constants after existing ones:
```python
STREAMING_MIN_EMIT_SCORE = 0.5
MIN_WORDS_FOR_MATCH = 3
```

2. Update `__init__` signature:
```python
def __init__(
    self,
    db: QuranDB = None,
    last_emission: tuple[int, int] | None = None,
    streaming_mode: bool = False,
):
    self.db = db or QuranDB()
    self._streaming_mode = streaming_mode
    self._min_emit_score = STREAMING_MIN_EMIT_SCORE if streaming_mode else MIN_EMIT_SCORE
    # ... rest unchanged
```

3. Replace all uses of `MIN_EMIT_SCORE` with `self._min_emit_score`:
   - `_find_best_match` line 81: `if best and best["score"] >= self._min_emit_score:`
   - `_evaluate` line 164: `if self._current_match and self._current_match["score"] >= self._min_emit_score:`
   - `finalize` line 225: `if self._current_match and self._current_match["score"] >= self._min_emit_score:`

4. Add min-word gate in `_find_best_match`:
```python
def _find_best_match(self, text: str) -> dict | None:
    if not text.strip():
        return None
    # Streaming mode: require minimum word count
    if self._streaming_mode and len(text.split()) < MIN_WORDS_FOR_MATCH:
        return None
    # ... rest unchanged
```

**Step 4: Run all verse tracker tests**

Run: `.venv/bin/python -m pytest tests/test_verse_tracker.py -v`

Expected: ALL PASS (new tests + existing tests)

**Step 5: Commit**

```bash
git add shared/verse_tracker.py tests/test_verse_tracker.py
git commit -m "feat: add streaming_mode to VerseTracker with higher thresholds"
```

---

### Task 2: Expose avg_logprob from StreamingTranscriber

**Files:**
- Modify: `experiments/streaming-asr/streaming_transcriber.py`

**Step 1: Update `_transcribe_chunk` to return avg_logprob**

In `experiments/streaming-asr/streaming_transcriber.py`, modify `_transcribe_chunk`:

```python
def _transcribe_chunk(self, audio_chunk: np.ndarray) -> dict:
    """Transcribe a single audio chunk, optionally with word timestamps."""
    result = self._mlx_whisper.transcribe(
        audio_chunk,
        path_or_hf_repo=self.model_id,
        language="ar",
        word_timestamps=self.word_timestamps,
        condition_on_previous_text=False,
        hallucination_silence_threshold=1.0,
        no_speech_threshold=0.5,
    )
    return result
```

No change to `_transcribe_chunk` itself — it already returns the full result dict.

**Step 2: Add `_extract_avg_logprob` helper method**

```python
@staticmethod
def _extract_avg_logprob(result: dict) -> float:
    """Extract average log probability across all segments."""
    segments = result.get("segments", [])
    if not segments:
        return -float("inf")
    logprobs = [s.get("avg_logprob", -float("inf")) for s in segments]
    return sum(logprobs) / len(logprobs)
```

**Step 3: Update `stream()` to yield avg_logprob**

In the `stream()` method, after `result = self._transcribe_chunk(chunk)` (line 140), extract and yield the logprob:

```python
avg_logprob = self._extract_avg_logprob(result)
```

Add `"avg_logprob": avg_logprob` to the yielded dict (alongside `chunk_text`, `accumulated_text`, etc.).

**Step 4: Update `transcribe_batch()` to return avg_logprob**

In `transcribe_batch()`, add:
```python
avg_logprob = self._extract_avg_logprob(result)
```
And include `"avg_logprob": avg_logprob` in the returned dict.

**Step 5: Commit**

```bash
git add experiments/streaming-asr/streaming_transcriber.py
git commit -m "feat: expose avg_logprob from StreamingTranscriber"
```

---

### Task 3: Confidence Gating + Buffered Confirmation in Pipeline

**Files:**
- Modify: `shared/streaming.py`
- Modify: `tests/test_streaming_pipeline.py`

**Step 1: Write failing tests for confidence gating**

Add to `tests/test_streaming_pipeline.py`:

```python
def test_chunked_gates_low_confidence():
    """Chunks with low avg_logprob should be skipped."""
    pipeline = StreamingPipeline(db=db)
    v = db.get_verse(112, 1)  # "قل هو الله احد"

    call_count = 0

    def mock_transcribe(path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First chunk: garbage with low confidence
            return {"text": "يا المسلمون الكرام", "avg_logprob": -2.0}
        else:
            # Second chunk: real verse with good confidence
            return {"text": v["text_clean"], "avg_logprob": -0.3}

    # Create a short fake audio (2 chunks worth)
    import numpy as np
    import tempfile, os, soundfile as sf
    audio = np.zeros(int(16000 * 6))  # 6 seconds = 2 chunks at 3s
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 16000)
    tmp.close()

    try:
        emissions = pipeline.run_on_audio_chunked(tmp.name, mock_transcribe, chunk_seconds=3.0)
    finally:
        os.unlink(tmp.name)

    # Should only have the real verse, not the garbage
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 112 and emissions[0]["ayah"] == 1


def test_chunked_backward_compat_string_transcribe():
    """transcribe_fn returning plain str should still work (no gating)."""
    pipeline = StreamingPipeline(db=db)
    v = db.get_verse(112, 1)

    def mock_transcribe(path):
        return v["text_clean"]  # returns str, not dict

    import numpy as np
    import tempfile, os, soundfile as sf
    audio = np.zeros(int(16000 * 3))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 16000)
    tmp.close()

    try:
        emissions = pipeline.run_on_audio_chunked(tmp.name, mock_transcribe, chunk_seconds=3.0)
    finally:
        os.unlink(tmp.name)

    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 112


def test_buffered_confirmation_high_confidence_immediate():
    """High confidence emissions (score >= 0.7) should emit immediately."""
    pipeline = StreamingPipeline(db=db)
    v = db.get_verse(112, 1)

    def mock_transcribe(path):
        return {"text": v["text_clean"], "avg_logprob": -0.3}

    import numpy as np
    import tempfile, os, soundfile as sf
    audio = np.zeros(int(16000 * 3))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 16000)
    tmp.close()

    try:
        emissions = pipeline.run_on_audio_chunked(tmp.name, mock_transcribe, chunk_seconds=3.0)
    finally:
        os.unlink(tmp.name)

    # Perfect verse text → high score → immediate emission
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 112
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py::test_chunked_gates_low_confidence tests/test_streaming_pipeline.py::test_chunked_backward_compat_string_transcribe -v`

Expected: FAIL — pipeline doesn't handle dict return from transcribe_fn

**Step 3: Implement confidence gating + buffered confirmation**

In `shared/streaming.py`, add constants at the top:

```python
MIN_CHUNK_LOG_PROB = -1.0
MIN_CHUNK_WORDS = 2
HIGH_CONFIDENCE_THRESHOLD = 0.7
MAX_HOLD_CHUNKS = 3
```

Replace the `run_on_audio_chunked` method with:

```python
def run_on_audio_chunked(
    self,
    audio_path: str,
    transcribe_fn,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.0,
) -> list[dict]:
    audio = load_audio(audio_path)
    chunk_size = int(chunk_seconds * SAMPLE_RATE)
    overlap_size = int(overlap_seconds * SAMPLE_RATE)
    step_size = max(chunk_size - overlap_size, 1)

    tracker = VerseTracker(self.db, streaming_mode=True)
    confirmed = []
    tentative = None
    tentative_age = 0

    pos = 0
    while pos < len(audio):
        chunk_end = min(pos + chunk_size, len(audio))
        chunk = audio[pos:chunk_end]

        if len(chunk) < MIN_CHUNK_SAMPLES:
            break

        if len(chunk) < SAMPLE_RATE:
            chunk = np.pad(chunk, (0, SAMPLE_RATE - len(chunk)))

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, chunk, SAMPLE_RATE)
            tmp.close()
            raw = transcribe_fn(tmp.name)
        except Exception:
            raw = ""
        finally:
            os.unlink(tmp.name)

        # Handle both str and dict returns from transcribe_fn
        if isinstance(raw, dict):
            chunk_text = raw.get("text", "").strip()
            avg_logprob = raw.get("avg_logprob", 0.0)
        else:
            chunk_text = str(raw).strip() if raw else ""
            avg_logprob = 0.0  # no gating for plain str

        # Layer 1: Confidence gating
        chunk_words = len(chunk_text.split()) if chunk_text else 0
        is_gated = False
        if isinstance(raw, dict):  # only gate when confidence info available
            if avg_logprob < MIN_CHUNK_LOG_PROB or chunk_words < MIN_CHUNK_WORDS:
                is_gated = True

        if is_gated or not chunk_text:
            # Track tentative age
            if tentative is not None:
                tentative_age += 1
                if tentative_age >= MAX_HOLD_CHUNKS:
                    tentative = None  # retract
                    tentative_age = 0
            pos += step_size
            continue

        # Feed to tracker
        emissions = tracker.process_delta(chunk_text)

        # Layer 4: Buffered confirmation
        # Valid chunk arrived → confirm any pending tentative
        if tentative is not None:
            confirmed.append(tentative)
            tentative = None
            tentative_age = 0

        for e in emissions:
            if e["score"] >= HIGH_CONFIDENCE_THRESHOLD:
                confirmed.append(e)
            else:
                if tentative is not None:
                    confirmed.append(tentative)
                tentative = e
                tentative_age = 0

        pos += step_size

    # Finalize: confirm tentative if score is decent
    from shared.verse_tracker import STREAMING_MIN_EMIT_SCORE
    if tentative is not None and tentative["score"] >= STREAMING_MIN_EMIT_SCORE:
        confirmed.append(tentative)

    confirmed.extend(tracker.finalize())
    return confirmed
```

**Step 4: Run all streaming pipeline tests**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add shared/streaming.py tests/test_streaming_pipeline.py
git commit -m "feat: add confidence gating and buffered confirmation to streaming pipeline"
```

---

### Task 4: Wire Up Benchmark Runner

**Files:**
- Modify: `benchmark/runner.py`

**Step 1: Verify existing benchmark tests still pass**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`

Expected: ALL PASS (scoring logic is unchanged)

**Step 2: Update `run_experiment` to use streaming_mode**

No changes needed — the pipeline's `run_on_audio_chunked` now internally creates a `VerseTracker(streaming_mode=True)`. The runner calls `pipeline.run_on_audio_chunked()` which handles it.

Verify by reading the code: the runner at line 182 calls `pipeline.run_on_audio_chunked(audio_path, transcribe_fn, chunk_seconds=chunk_seconds)`. The pipeline method now creates `VerseTracker(self.db, streaming_mode=True)` for chunked mode. The full transcript path (`run_on_full_transcript`) still creates `VerseTracker(self.db)` with `streaming_mode=False`. Correct.

**Step 3: Update streaming-asr experiment's transcribe to return dict**

In `experiments/streaming-asr/run.py`, check if `transcribe()` returns plain str. If so, update it to return `{"text": str, "avg_logprob": float}` when called by the benchmark runner.

Check the file first — the `transcribe()` function is what the benchmark runner calls. It should now return a dict with confidence info.

**Step 4: Commit**

```bash
git add experiments/streaming-asr/run.py
git commit -m "feat: streaming-asr transcribe returns confidence info for gating"
```

---

### Task 5: Integration Test — Run Benchmark

**Step 1: Run streaming benchmark on tarteel-whisper-base**

```bash
.venv/bin/python -m benchmark.runner --experiment tarteel-whisper-base --mode streaming --chunk 5
```

Record: Recall, Precision, SeqAcc.

**Step 2: Compare with previous results**

Check `benchmark/results/latest.json` for the pre-change streaming numbers. The key metrics to compare:
- Single-ayah precision: was 60%, target ~75%
- Multi-ayah recall: was 66%, should stay ~60-66%

**Step 3: Tune thresholds if needed**

If precision is still low:
- Tighten `MIN_CHUNK_LOG_PROB` from -1.0 to -0.8 in `shared/streaming.py`
- Raise `STREAMING_MIN_EMIT_SCORE` from 0.5 to 0.55 in `shared/verse_tracker.py`

If recall drops too much:
- Loosen `MIN_CHUNK_LOG_PROB` to -1.2
- Lower `MIN_WORDS_FOR_MATCH` from 3 to 2

**Step 4: Run full benchmark after tuning**

```bash
.venv/bin/python -m benchmark.runner --experiment tarteel-whisper-base --mode streaming --chunk 5
.venv/bin/python -m benchmark.runner --experiment tarteel-whisper-base --mode full
```

Compare full vs streaming to see the gap.

**Step 5: Commit final threshold values**

```bash
git add shared/verse_tracker.py shared/streaming.py
git commit -m "chore: tune streaming thresholds based on benchmark results"
```

---

### Task 6: Run Full Test Suite

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: ALL PASS

**Step 2: Final commit if any cleanup needed**

```bash
git add <changed files>
git commit -m "chore: cleanup after streaming precision improvements"
```
