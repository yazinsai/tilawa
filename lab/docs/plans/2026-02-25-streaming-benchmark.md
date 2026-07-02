# Streaming Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-prediction benchmark with a streaming pipeline that processes audio in chunks, detects verse transitions, and scores an ordered sequence of verse emissions — using any ASR backend.

**Architecture:** Audio is chunked and transcribed progressively. A `VerseTracker` matches accumulated text against QuranDB, emits `(surah, ayah)` when a verse completes (score drop, match shift, or end-of-audio), then resets for the next verse. A continuation bias favors the next sequential verse to resolve duplicate-text ambiguity. The benchmark scores the emitted sequence against the expected ordered sequence using recall, precision, and exact-sequence-match metrics.

**Tech Stack:** Python 3.13, Levenshtein, existing shared/ utilities (QuranDB, normalizer, audio), existing ASR backends (CTC wav2vec2, Whisper LoRA, tarteel-whisper-base, mlx-whisper)

---

### Task 1: Add `get_next_verse()` to QuranDB

**Files:**
- Modify: `shared/quran_db.py:28-30`
- Test: `tests/test_quran_db.py` (create)

**Step 1: Write the failing test**

Create `tests/test_quran_db.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quran_db import QuranDB

db = QuranDB()

def test_get_next_verse_normal():
    nxt = db.get_next_verse(1, 1)
    assert nxt is not None
    assert nxt["surah"] == 1 and nxt["ayah"] == 2

def test_get_next_verse_end_of_surah():
    # Last ayah of Al-Fatiha (1:7) -> first of Al-Baqarah (2:1)
    nxt = db.get_next_verse(1, 7)
    assert nxt is not None
    assert nxt["surah"] == 2 and nxt["ayah"] == 1

def test_get_next_verse_last_verse():
    # Last verse in Quran (114:6)
    nxt = db.get_next_verse(114, 6)
    assert nxt is None

def test_get_next_verse_invalid():
    nxt = db.get_next_verse(999, 1)
    assert nxt is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_quran_db.py -v`
Expected: FAIL — `AttributeError: 'QuranDB' object has no attribute 'get_next_verse'`

**Step 3: Write the implementation**

Add to `shared/quran_db.py` after `get_surah()` (line 32):

```python
def get_next_verse(self, surah: int, ayah: int) -> dict | None:
    """Return the next verse after surah:ayah, or None if last verse."""
    verses = self._by_surah.get(surah, [])
    for i, v in enumerate(verses):
        if v["ayah"] == ayah:
            if i + 1 < len(verses):
                return verses[i + 1]
            # End of surah — try first verse of next surah
            next_surah = self._by_surah.get(surah + 1, [])
            return next_surah[0] if next_surah else None
    return None
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_quran_db.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add shared/quran_db.py tests/test_quran_db.py
git commit -m "feat: add get_next_verse() to QuranDB"
```

---

### Task 2: Add `transcribe()` to each experiment

Each experiment's `run.py` needs a `transcribe(audio_path: str) -> str` function that returns the raw Arabic transcript without verse matching.

**Files:**
- Modify: `experiments/ctc-alignment/run.py`
- Modify: `experiments/whisper-lora/run.py`
- Modify: `experiments/tarteel-whisper-base/run.py`
- Modify: `experiments/streaming-asr/run.py`
- Test: `tests/test_transcribe.py` (create)

**Step 1: Write the failing test**

Create `tests/test_transcribe.py`:

```python
"""Verify each experiment exports a transcribe() function."""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENTS = {
    "ctc-alignment": ROOT / "experiments" / "ctc-alignment" / "run.py",
    "whisper-lora": ROOT / "experiments" / "whisper-lora" / "run.py",
    "tarteel-whisper-base": ROOT / "experiments" / "tarteel-whisper-base" / "run.py",
    "streaming-asr": ROOT / "experiments" / "streaming-asr" / "run.py",
}

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_all_experiments_have_transcribe():
    for name, path in EXPERIMENTS.items():
        if not path.exists():
            continue
        mod = _load(name, path)
        assert hasattr(mod, "transcribe"), f"{name} missing transcribe()"
        # Verify signature: takes audio_path, returns str
        import inspect
        sig = inspect.signature(mod.transcribe)
        params = list(sig.parameters.keys())
        assert "audio_path" in params, f"{name}.transcribe() missing audio_path param"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -v`
Expected: FAIL — `AssertionError: ctc-alignment missing transcribe()`

**Step 3: Add transcribe() to each experiment**

**ctc-alignment/run.py** — add after `_ensure_loaded()` (line 58), before `predict()`:

```python
def transcribe(audio_path: str) -> str:
    """Return raw greedy-decoded transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(pred_ids.cpu().numpy())
    return normalize_arabic(text)
```

**whisper-lora/run.py** — add after `_ensure_loaded()` (line 32), before `predict()`:

```python
def transcribe(audio_path: str) -> str:
    """Return raw Whisper transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = _model.generate(
            inputs["input_features"],
            max_new_tokens=225,
            repetition_penalty=1.2,
            language="ar",
            task="transcribe",
        )
    return _processor.batch_decode(ids, skip_special_tokens=True)[0]
```

**tarteel-whisper-base/run.py** — add after `_ensure_loaded()` (line 31), before `predict()`:

```python
def transcribe(audio_path: str) -> str:
    """Return raw Whisper transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = _model.generate(inputs["input_features"], max_new_tokens=225)
    text = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    return _SPECIAL_PREFIX.sub("", text).strip()
```

**streaming-asr/run.py** — add after `_ensure_loaded()` (line 20), before `predict()`:

```python
def transcribe(audio_path: str) -> str:
    """Return raw mlx-whisper transcript."""
    _ensure_loaded()
    result = _transcriber.transcribe_batch(audio_path)
    return result.get("text", "")
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add experiments/ctc-alignment/run.py experiments/whisper-lora/run.py experiments/tarteel-whisper-base/run.py experiments/streaming-asr/run.py tests/test_transcribe.py
git commit -m "feat: add transcribe() to all experiment backends"
```

---

### Task 3: Build the VerseTracker

The core streaming logic: takes accumulated text, detects verse completions, emits `(surah, ayah)` in order.

**Files:**
- Create: `shared/verse_tracker.py`
- Test: `tests/test_verse_tracker.py` (create)

**Step 1: Write the failing tests**

Create `tests/test_verse_tracker.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.verse_tracker import VerseTracker
from shared.quran_db import QuranDB

db = QuranDB()

def test_single_verse_detection():
    """Feed the full text of 1:1, expect one emission."""
    tracker = VerseTracker(db)
    v = db.get_verse(1, 1)
    emissions = tracker.process_text(v["text_clean"])
    # Finalize to flush
    emissions += tracker.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1
    assert emissions[0]["ayah"] == 1

def test_two_consecutive_verses():
    """Feed text of 112:1 + 112:2, expect two emissions in order."""
    tracker = VerseTracker(db)
    v1 = db.get_verse(112, 1)
    v2 = db.get_verse(112, 2)
    combined = v1["text_clean"] + " " + v2["text_clean"]
    emissions = tracker.process_text(combined)
    emissions += tracker.finalize()
    assert len(emissions) >= 2
    assert emissions[0]["surah"] == 112 and emissions[0]["ayah"] == 1
    assert emissions[1]["surah"] == 112 and emissions[1]["ayah"] == 2

def test_continuation_bias():
    """After detecting 55:12, the same text should match 55:13 not 55:16."""
    tracker = VerseTracker(db)
    # Feed 55:12 first
    v12 = db.get_verse(55, 12)
    tracker.process_text(v12["text_clean"])
    tracker.finalize()  # emit 55:12
    # Now reset and feed the repeated refrain
    tracker2 = VerseTracker(db, last_emission=(55, 12))
    v13 = db.get_verse(55, 13)  # same text as many other ayahs in Ar-Rahman
    emissions = tracker2.process_text(v13["text_clean"])
    emissions += tracker2.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 55
    assert emissions[0]["ayah"] == 13

def test_finalize_emits_partial():
    """Even incomplete text should emit on finalize if there's a reasonable match."""
    tracker = VerseTracker(db)
    # Feed partial text of Ayat al-Kursi
    partial = "الله لا اله الا هو الحي القيوم"
    emissions = tracker.process_text(partial)
    emissions += tracker.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 2
    assert emissions[0]["ayah"] == 255

def test_empty_input():
    tracker = VerseTracker(db)
    emissions = tracker.process_text("")
    emissions += tracker.finalize()
    assert emissions == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_verse_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.verse_tracker'`

**Step 3: Write the implementation**

Create `shared/verse_tracker.py`:

```python
"""
VerseTracker — Detect verse boundaries in streaming Quran transcription.

As accumulated text grows, matches against QuranDB and emits (surah, ayah)
when a verse is "complete" (match shifts to a different verse or score drops).
Continuation bias resolves duplicate-text ambiguity by preferring the next
sequential verse after the last emission.
"""

from Levenshtein import ratio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

CONTINUATION_BONUS = 0.15
SCORE_DROP_THRESHOLD = 0.15  # emit when score drops this much below peak
MIN_EMIT_SCORE = 0.3  # minimum score to consider a match real


class VerseTracker:
    """Track and emit verse detections from streaming text."""

    def __init__(
        self,
        db: QuranDB = None,
        last_emission: tuple[int, int] | None = None,
    ):
        self.db = db or QuranDB()
        self._accumulated = ""
        self._current_match: dict | None = None  # {surah, ayah, score}
        self._peak_score: float = 0.0
        self._emissions: list[dict] = []
        self._last_emitted: tuple[int, int] | None = last_emission

    def _score_verse(self, text: str, verse: dict) -> float:
        """Score a verse against accumulated text, with continuation bias."""
        base = ratio(text, verse["text_clean"])
        # Apply continuation bias
        if self._last_emitted:
            next_v = self.db.get_next_verse(*self._last_emitted)
            if next_v and next_v["surah"] == verse["surah"] and next_v["ayah"] == verse["ayah"]:
                base += CONTINUATION_BONUS
        return base

    def _find_best_match(self, text: str) -> dict | None:
        """Find the best matching single verse for the given text."""
        if not text.strip():
            return None

        best = None
        best_score = 0.0

        # Prefix-aware: compare against verse prefixes when text is short
        text_words = text.split()
        n_words = len(text_words)

        for v in self.db.verses:
            verse_clean = v["text_clean"]
            verse_words = verse_clean.split()

            # Prefix scoring for partial verses
            prefix_len = min(n_words, len(verse_words))
            prefix = " ".join(verse_words[:prefix_len])
            prefix_score = ratio(text, prefix)
            full_score = ratio(text, verse_clean)

            coverage = n_words / max(len(verse_words), 1)
            if coverage > 0.8:
                raw = 0.3 * prefix_score + 0.7 * full_score
            else:
                raw = 0.7 * prefix_score + 0.3 * full_score

            # Apply continuation bias
            if self._last_emitted:
                next_v = self.db.get_next_verse(*self._last_emitted)
                if next_v and next_v["surah"] == v["surah"] and next_v["ayah"] == v["ayah"]:
                    raw += CONTINUATION_BONUS

            if raw > best_score:
                best_score = raw
                best = {
                    "surah": v["surah"],
                    "ayah": v["ayah"],
                    "text_clean": verse_clean,
                    "score": best_score,
                }

        if best and best["score"] >= MIN_EMIT_SCORE:
            return best
        return None

    def _emit(self, match: dict) -> dict:
        """Emit a verse detection and reset state."""
        emission = {"surah": match["surah"], "ayah": match["ayah"], "score": match["score"]}
        self._emissions.append(emission)
        self._last_emitted = (match["surah"], match["ayah"])

        # Trim accumulated text: remove the portion that matched
        matched_text = match["text_clean"]
        norm_acc = self._accumulated
        # Try to remove the matched verse text from the start of accumulated
        matched_words = matched_text.split()
        acc_words = norm_acc.split()
        # Find how many leading accumulated words belong to the emitted verse
        overlap = min(len(matched_words), len(acc_words))
        # Simple: remove roughly the number of words in the matched verse
        remaining_words = acc_words[overlap:]
        self._accumulated = " ".join(remaining_words)

        self._current_match = None
        self._peak_score = 0.0

        return emission

    def process_text(self, text: str) -> list[dict]:
        """Process new text, return any verse emissions.

        Args:
            text: The full accumulated transcript so far (not a delta).

        Returns:
            List of emitted verses [{"surah": int, "ayah": int, "score": float}]
        """
        normalized = normalize_arabic(text)
        if not normalized.strip():
            return []

        self._accumulated = normalized
        emissions = []

        match = self._find_best_match(self._accumulated)
        if not match:
            return []

        same_verse = (
            self._current_match
            and self._current_match["surah"] == match["surah"]
            and self._current_match["ayah"] == match["ayah"]
        )

        if same_verse:
            # Same verse — update peak score
            if match["score"] > self._peak_score:
                self._peak_score = match["score"]
            elif self._peak_score - match["score"] > SCORE_DROP_THRESHOLD:
                # Score dropped — verse is likely complete, emit it
                emissions.append(self._emit(self._current_match))
                # Try to match remainder
                if self._accumulated.strip():
                    next_match = self._find_best_match(self._accumulated)
                    if next_match:
                        self._current_match = next_match
                        self._peak_score = next_match["score"]
                    else:
                        self._current_match = None
                        self._peak_score = 0.0
            else:
                self._current_match = match
        else:
            # Different verse detected
            if self._current_match and self._current_match["score"] >= MIN_EMIT_SCORE:
                # Emit the previous verse
                emissions.append(self._emit(self._current_match))
            # Start tracking the new verse
            self._current_match = match
            self._peak_score = match["score"]

        if not self._current_match:
            self._current_match = match
            self._peak_score = match["score"]

        return emissions

    def finalize(self) -> list[dict]:
        """Flush any remaining match as a final emission."""
        if self._current_match and self._current_match["score"] >= MIN_EMIT_SCORE:
            return [self._emit(self._current_match)]
        return []
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_verse_tracker.py -v`
Expected: PASS (all 5 tests). The continuation bias test is the trickiest — may need tuning of `CONTINUATION_BONUS` if it fails.

**Step 5: Commit**

```bash
git add shared/verse_tracker.py tests/test_verse_tracker.py
git commit -m "feat: add VerseTracker with continuation bias for streaming detection"
```

---

### Task 4: Build the streaming pipeline

Connects chunker + ASR backend + VerseTracker into a single `stream_predict()` function.

**Files:**
- Create: `shared/streaming.py`
- Test: `tests/test_streaming_pipeline.py` (create)

**Step 1: Write the failing test**

Create `tests/test_streaming_pipeline.py`:

```python
"""Test the streaming pipeline with a mock transcriber."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB

db = QuranDB()

class MockTranscriber:
    """Simulates chunked transcription by returning preset text."""
    def __init__(self, chunks: list[str]):
        self._chunks = chunks
        self._idx = 0

    def transcribe_chunk(self, audio_chunk) -> str:
        if self._idx < len(self._chunks):
            text = self._chunks[self._idx]
            self._idx += 1
            return text
        return ""

def test_pipeline_single_verse():
    v = db.get_verse(1, 1)
    # Simulate one chunk returning the full verse
    pipeline = StreamingPipeline(db=db)
    emissions = pipeline.run_on_text([v["text_clean"]])
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1 and emissions[0]["ayah"] == 1

def test_pipeline_multi_verse_sequence():
    v1 = db.get_verse(103, 1)
    v2 = db.get_verse(103, 2)
    v3 = db.get_verse(103, 3)
    # Simulate 3 chunks, one per verse
    pipeline = StreamingPipeline(db=db)
    emissions = pipeline.run_on_text([
        v1["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"] + " " + v3["text_clean"],
    ])
    assert len(emissions) >= 3
    assert emissions[0] == {"surah": 103, "ayah": 1, "score": emissions[0]["score"]}
    assert emissions[1]["surah"] == 103 and emissions[1]["ayah"] == 2
    assert emissions[2]["surah"] == 103 and emissions[2]["ayah"] == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.streaming'`

**Step 3: Write the implementation**

Create `shared/streaming.py`:

```python
"""
Streaming pipeline — connects chunked audio transcription to verse detection.

Can be used with any ASR backend that exposes a transcribe() function.
Also provides run_on_text() for testing with pre-transcribed text chunks.
"""

from shared.quran_db import QuranDB
from shared.verse_tracker import VerseTracker


class StreamingPipeline:
    """Streaming verse detection pipeline."""

    def __init__(self, db: QuranDB = None):
        self.db = db or QuranDB()

    def run_on_text(self, text_chunks: list[str]) -> list[dict]:
        """Run verse detection on a sequence of accumulated text snapshots.

        Args:
            text_chunks: List of accumulated transcripts (each chunk is the
                         full text so far, not a delta). This matches how
                         StreamingTranscriber.stream() yields accumulated_text.

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        tracker = VerseTracker(self.db)
        all_emissions = []

        for text in text_chunks:
            emissions = tracker.process_text(text)
            all_emissions.extend(emissions)

        # Finalize — flush any remaining match
        all_emissions.extend(tracker.finalize())
        return all_emissions

    def run_on_audio(self, audio_path: str, transcribe_fn, chunk_seconds: float = 3.0) -> list[dict]:
        """Run full streaming pipeline on an audio file.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str that transcribes audio.
                           For streaming, we use the StreamingTranscriber to chunk
                           and transcribe progressively.
            chunk_seconds: Chunk size for streaming

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        # Import here to avoid hard dependency on mlx_whisper
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "streaming-asr"))
        from streaming_transcriber import StreamingTranscriber

        transcriber = StreamingTranscriber(chunk_seconds=chunk_seconds)
        tracker = VerseTracker(self.db)
        all_emissions = []

        for event in transcriber.stream(audio_path):
            accumulated = event.get("accumulated_text", "")
            if accumulated:
                emissions = tracker.process_text(accumulated)
                all_emissions.extend(emissions)

        all_emissions.extend(tracker.finalize())
        return all_emissions

    def run_on_full_transcript(self, audio_path: str, transcribe_fn) -> list[dict]:
        """Run verse detection on a full transcript (non-streaming).

        Transcribes the whole file at once, then feeds to VerseTracker.
        Useful as a baseline and for backends that don't support chunking.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        transcript = transcribe_fn(audio_path)
        tracker = VerseTracker(self.db)
        emissions = tracker.process_text(transcript)
        emissions += tracker.finalize()
        return emissions
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add shared/streaming.py tests/test_streaming_pipeline.py
git commit -m "feat: add StreamingPipeline connecting chunker to verse tracker"
```

---

### Task 5: Update manifest format for ordered sequences

The manifest needs explicit expected verse sequences for scoring.

**Files:**
- Modify: `benchmark/test_corpus/manifest.json`
- Test: manual verification

**Step 1: Write a script to update the manifest**

Create `scripts/update_manifest_sequences.py`:

```python
"""Add expected_verses field to manifest entries."""
import json
from pathlib import Path

MANIFEST = Path(__file__).parent.parent / "benchmark" / "test_corpus" / "manifest.json"

with open(MANIFEST) as f:
    manifest = json.load(f)

for sample in manifest["samples"]:
    surah = sample["surah"]
    ayah_start = sample["ayah"]
    ayah_end = sample.get("ayah_end")

    if ayah_end:
        sample["expected_verses"] = [
            {"surah": surah, "ayah": a}
            for a in range(ayah_start, ayah_end + 1)
        ]
    else:
        sample["expected_verses"] = [{"surah": surah, "ayah": ayah_start}]

with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Updated {len(manifest['samples'])} samples with expected_verses")
```

**Step 2: Run the script**

Run: `.venv/bin/python scripts/update_manifest_sequences.py`
Expected: "Updated 55 samples with expected_verses"

**Step 3: Verify a multi-ayah entry looks right**

Check that `multi_103_001_003` has:
```json
"expected_verses": [
  {"surah": 103, "ayah": 1},
  {"surah": 103, "ayah": 2},
  {"surah": 103, "ayah": 3}
]
```

**Step 4: Commit**

```bash
git add benchmark/test_corpus/manifest.json scripts/update_manifest_sequences.py
git commit -m "feat: add expected_verses sequences to benchmark manifest"
```

---

### Task 6: Rewrite the benchmark runner

Replace the single-prediction evaluation with streaming sequence scoring.

**Files:**
- Modify: `benchmark/runner.py`
- Test: `tests/test_scoring.py` (create)

**Step 1: Write the failing test for scoring logic**

Create `tests/test_scoring.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import score_sequence

def test_perfect_match():
    expected = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    predicted = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["sequence_accuracy"] == 1.0

def test_partial_match():
    expected = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    predicted = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 3}]  # missed ayah 2
    s = score_sequence(expected, predicted)
    assert s["recall"] == 2 / 3  # 2 of 3 found in order
    assert s["precision"] == 1.0  # both predictions correct
    assert s["sequence_accuracy"] == 0.0  # not exact match

def test_wrong_verse():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = [{"surah": 2, "ayah": 1}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 0.0
    assert s["precision"] == 0.0
    assert s["sequence_accuracy"] == 0.0

def test_single_verse_correct():
    expected = [{"surah": 2, "ayah": 255}]
    predicted = [{"surah": 2, "ayah": 255}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["sequence_accuracy"] == 1.0

def test_empty_predicted():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = []
    s = score_sequence(expected, predicted)
    assert s["recall"] == 0.0
    assert s["precision"] == 0.0
    assert s["sequence_accuracy"] == 0.0

def test_extra_predictions():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = [{"surah": 1, "ayah": 1}, {"surah": 1, "ayah": 2}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5  # 1 of 2 predictions correct
    assert s["sequence_accuracy"] == 0.0  # not exact
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_sequence' from 'benchmark.runner'`

**Step 3: Rewrite benchmark/runner.py**

Rewrite `benchmark/runner.py` to:
1. Export `score_sequence()` for testing
2. Use `StreamingPipeline.run_on_full_transcript()` for each experiment
3. Score with recall/precision/sequence_accuracy
4. Keep backward-compatible output format with new metrics added

```python
"""
Benchmark runner for all experiments — streaming sequence evaluation.

Usage:
    python -m benchmark.runner                           # all experiments
    python -m benchmark.runner --experiment whisper-lora  # one experiment
    python -m benchmark.runner --category short           # filter by category
"""

import sys
import json
import time
import importlib.util
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",
    "embedding-search": EXPERIMENTS_DIR / "embedding-search" / "run.py",
    "contrastive": EXPERIMENTS_DIR / "contrastive" / "run.py",
    "streaming-asr": EXPERIMENTS_DIR / "streaming-asr" / "run.py",
    "ctc-alignment": EXPERIMENTS_DIR / "ctc-alignment" / "run.py",
    "tarteel-whisper-base": EXPERIMENTS_DIR / "tarteel-whisper-base" / "run.py",
}

NEW_MODELS_PATH = EXPERIMENTS_DIR / "new-models" / "run.py"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_manifest() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return data["samples"]


def score_sequence(expected: list[dict], predicted: list[dict]) -> dict:
    """Score a predicted verse sequence against expected.

    Uses ordered subsequence matching: a predicted verse counts as a
    recall hit if it matches an expected verse and appears in the
    correct relative order.

    Args:
        expected: [{"surah": int, "ayah": int}, ...]
        predicted: [{"surah": int, "ayah": int, ...}, ...]

    Returns:
        {"recall": float, "precision": float, "sequence_accuracy": float}
    """
    if not expected:
        return {"recall": 1.0, "precision": 1.0, "sequence_accuracy": 1.0}

    if not predicted:
        return {"recall": 0.0, "precision": 0.0, "sequence_accuracy": 0.0}

    # Find longest ordered subsequence of correct predictions
    pred_tuples = [(p["surah"], p["ayah"]) for p in predicted]
    exp_tuples = [(e["surah"], e["ayah"]) for e in expected]

    # Greedy ordered match: walk through expected, find each in predicted (in order)
    matched = 0
    pred_idx = 0
    matched_pred_indices = set()
    for exp in exp_tuples:
        for j in range(pred_idx, len(pred_tuples)):
            if pred_tuples[j] == exp:
                matched += 1
                matched_pred_indices.add(j)
                pred_idx = j + 1
                break

    recall = matched / len(exp_tuples)
    precision = len(matched_pred_indices) / len(pred_tuples)
    seq_acc = 1.0 if pred_tuples == exp_tuples else 0.0

    return {"recall": recall, "precision": precision, "sequence_accuracy": seq_acc}


def discover_experiments(filter_name: str | None = None) -> list[dict]:
    experiments = []

    for name, run_path in EXPERIMENT_REGISTRY.items():
        if filter_name and filter_name != name:
            continue
        if not run_path.exists():
            print(f"Warning: {name} run.py not found at {run_path}")
            continue
        experiments.append({"name": name, "run_path": run_path, "model_name": None})

    if not filter_name or filter_name.startswith("new-models"):
        try:
            mod = _load_module("new_models_run", NEW_MODELS_PATH)
            for model_name in mod.list_models():
                entry_name = f"new-models/{model_name}"
                if filter_name and filter_name != entry_name and filter_name != "new-models":
                    continue
                experiments.append({
                    "name": entry_name,
                    "run_path": NEW_MODELS_PATH,
                    "model_name": model_name,
                })
        except Exception as e:
            print(f"Warning: could not load new-models: {e}")

    return experiments


def run_experiment(exp: dict, samples: list[dict], pipeline: StreamingPipeline) -> dict:
    """Run one experiment against all samples using streaming evaluation."""
    mod = _load_module(exp["name"].replace("/", "_").replace("-", "_"), exp["run_path"])

    if not hasattr(mod, "transcribe"):
        print(f"  Skipping {exp['name']} — no transcribe() function")
        return None

    transcribe_fn = mod.transcribe
    if exp["model_name"]:
        # For multi-model experiments, wrap transcribe with model_name
        base_fn = transcribe_fn
        transcribe_fn = lambda path, _mn=exp["model_name"]: base_fn(path, model_name=_mn)

    # Warmup
    warmup_sample = samples[0]
    audio_path = str(CORPUS_DIR / warmup_sample["file"])
    try:
        transcribe_fn(audio_path)
    except Exception as e:
        print(f"  Warmup failed for {exp['name']}: {e}")

    # Get model size
    try:
        if exp["model_name"]:
            size = mod.model_size(model_name=exp["model_name"])
        else:
            size = mod.model_size()
    except Exception:
        size = 0

    total_recall = 0.0
    total_precision = 0.0
    total_seq_acc = 0.0
    latencies = []
    per_sample = []

    for sample in samples:
        audio_path = str(CORPUS_DIR / sample["file"])
        expected = sample.get("expected_verses", [{"surah": sample["surah"], "ayah": sample["ayah"]}])

        try:
            start = time.perf_counter()
            emissions = pipeline.run_on_full_transcript(audio_path, transcribe_fn)
            elapsed = time.perf_counter() - start
        except Exception as e:
            emissions = []
            elapsed = 0.0

        scores = score_sequence(expected, emissions)
        total_recall += scores["recall"]
        total_precision += scores["precision"]
        total_seq_acc += scores["sequence_accuracy"]
        latencies.append(elapsed)

        per_sample.append({
            "id": sample["id"],
            "expected": expected,
            "predicted": emissions,
            "recall": scores["recall"],
            "precision": scores["precision"],
            "sequence_accuracy": scores["sequence_accuracy"],
            "latency": elapsed,
        })

    n = len(samples)
    avg_latency = sum(latencies) / n if n else 0

    return {
        "name": exp["name"],
        "recall": total_recall / n if n else 0,
        "precision": total_precision / n if n else 0,
        "sequence_accuracy": total_seq_acc / n if n else 0,
        "total": n,
        "avg_latency": avg_latency,
        "model_size": size,
        "per_sample": per_sample,
    }


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def print_table(results: list[dict]):
    print()
    print(f"{'Experiment':<30} {'Recall':>8} {'Precision':>10} {'SeqAcc':>8} {'Latency':>10} {'Size':>10}")
    print("-" * 78)
    for r in results:
        rec = f"{r['recall']:.0%}"
        prec = f"{r['precision']:.0%}"
        seq = f"{r['sequence_accuracy']:.0%}"
        lat = f"{r['avg_latency']:.2f}s"
        size = format_size(r['model_size'])
        print(f"{r['name']:<30} {rec:>8} {prec:>10} {seq:>8} {lat:>10} {size:>10}")
    print()


def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")

    latest_path = RESULTS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            latest = {r["name"]: r for r in json.load(f)}
    else:
        latest = {}

    for r in results:
        summary = {
            "name": r["name"],
            "recall": r["recall"],
            "precision": r["precision"],
            "sequence_accuracy": r["sequence_accuracy"],
            "total": r["total"],
            "avg_latency": r["avg_latency"],
            "model_size": r["model_size"],
            "timestamp": timestamp,
        }
        prev = latest.get(r["name"])
        if prev is None or r["sequence_accuracy"] > prev.get("sequence_accuracy", 0) or (
            r["sequence_accuracy"] == prev.get("sequence_accuracy", 0) and r["avg_latency"] < prev.get("avg_latency", float("inf"))
        ):
            latest[r["name"]] = summary

    with open(latest_path, "w") as f:
        json.dump(sorted(latest.values(), key=lambda x: x["name"]), f, indent=2, default=str)
    print(f"Updated {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all experiments (streaming)")
    parser.add_argument("--experiment", type=str, help="Run only this experiment")
    parser.add_argument("--category", type=str, help="Filter samples by category")
    args = parser.parse_args()

    samples = load_manifest()
    if args.category:
        samples = [s for s in samples if s["category"] == args.category]
        print(f"Filtered to {len(samples)} samples in category '{args.category}'")

    experiments = discover_experiments(args.experiment)
    if not experiments:
        print(f"No experiments found matching '{args.experiment}'")
        return

    db = QuranDB()
    pipeline = StreamingPipeline(db=db)

    print(f"Running {len(experiments)} experiment(s) on {len(samples)} sample(s)...")

    results = []
    for exp in experiments:
        print(f"\n>>> {exp['name']}")
        result = run_experiment(exp, samples, pipeline)
        if result is None:
            continue
        results.append(result)
        print(f"    Recall: {result['recall']:.0%}  Precision: {result['precision']:.0%}  SeqAcc: {result['sequence_accuracy']:.0%}")

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
```

**Step 4: Run scoring tests**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add benchmark/runner.py tests/test_scoring.py
git commit -m "feat: rewrite benchmark runner with streaming sequence scoring"
```

---

### Task 7: Run the full benchmark and verify

**Step 1: Run the benchmark on one experiment to verify**

Run: `.venv/bin/python -m benchmark.runner --experiment ctc-alignment`

Verify:
- Output shows recall/precision/sequence_accuracy columns
- Results saved to `benchmark/results/`
- Single-ayah samples still score correctly
- Multi-ayah samples now produce sequence-based scores

**Step 2: Run all three working experiments individually**

```bash
.venv/bin/python -m benchmark.runner --experiment ctc-alignment
.venv/bin/python -m benchmark.runner --experiment whisper-lora
.venv/bin/python -m benchmark.runner --experiment tarteel-whisper-base
```

**Step 3: Review results and fix issues**

Check per-sample results for multi-ayah samples. The VerseTracker may need tuning:
- `CONTINUATION_BONUS` (default 0.15) — increase if duplicate verses misidentified
- `SCORE_DROP_THRESHOLD` (default 0.15) — decrease if verses emitted too early
- `MIN_EMIT_SCORE` (default 0.3) — increase if garbage verses emitted

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: tune verse tracker parameters based on benchmark results"
```

---

### Task 8: Update CLAUDE.md and clean up

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md experiment table**

Replace the current experiment table with updated metrics (recall/precision/seqacc instead of accuracy).

**Step 2: Update benchmark section**

Update the "Running Benchmarks" section to mention the new streaming-based scoring.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with streaming benchmark metrics"
```
