"""
Simulated Streaming ASR for Quran Recitation.

Processes audio in small chunks to simulate real-time streaming,
accumulating text as chunks arrive. Supports two modes:

1. Independent chunks: Each chunk transcribed independently
2. Sliding window: Chunks include overlap with previous, deduplicated
"""

import sys
from pathlib import Path
from typing import Generator

import numpy as np

# Reuse parent project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio

# Default models
MLX_MODEL = "mlx-community/whisper-base-mlx"
SAMPLE_RATE = 16000


class StreamingTranscriber:
    """Simulated streaming ASR that yields partial transcriptions chunk by chunk."""

    def __init__(
        self,
        model_id: str = MLX_MODEL,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 0.0,
        min_chunk_seconds: float = 0.5,
        word_timestamps: bool = False,
    ):
        self.model_id = model_id
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.min_chunk_seconds = min_chunk_seconds
        self.word_timestamps = word_timestamps
        self._model_loaded = False

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if not self._model_loaded:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
            # Warm up by loading model (first call triggers download)
            self._model_loaded = True

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

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Detect hallucinated/repetitive output from Whisper."""
        if not text.strip():
            return False
        words = text.strip().split()
        if len(words) < 3:
            return False
        # Check for excessive repetition (same word repeated 3+ times)
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
        # Check for very long output from a short chunk (likely hallucination)
        if len(words) > 30:
            return True
        return False

    @staticmethod
    def _is_silent_chunk(audio_chunk: np.ndarray, threshold: float = 0.005) -> bool:
        """Check if a chunk is mostly silence based on RMS energy."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < threshold

    @staticmethod
    def _extract_avg_logprob(result: dict) -> float:
        """Extract average log probability across all segments."""
        segments = result.get("segments", [])
        if not segments:
            return -float("inf")
        logprobs = [s.get("avg_logprob", -float("inf")) for s in segments]
        return sum(logprobs) / len(logprobs)

    def stream(self, audio_path: str) -> Generator[dict, None, None]:
        """
        Generator that yields partial transcriptions as audio is processed chunk by chunk.

        Yields dicts with:
            chunk_idx: int - which chunk this is
            chunk_text: str - text from this chunk only
            accumulated_text: str - all text so far
            timestamp_s: float - start time of this chunk in the audio
            words: list[dict] - word-level info [{"word": str, "start": float, "end": float}]
            processing_time_s: float - time to process this chunk
        """
        import time

        self._ensure_model()
        audio = load_audio(audio_path)
        total_duration = len(audio) / SAMPLE_RATE

        chunk_size = int(self.chunk_seconds * SAMPLE_RATE)
        overlap_size = int(self.overlap_seconds * SAMPLE_RATE)
        step_size = chunk_size - overlap_size
        min_chunk_size = int(self.min_chunk_seconds * SAMPLE_RATE)

        accumulated_text = ""
        all_words = []
        chunk_idx = 0

        pos = 0
        while pos < len(audio):
            # Extract chunk (with overlap from previous)
            chunk_start = max(0, pos - overlap_size) if chunk_idx > 0 else 0
            chunk_end = min(pos + chunk_size, len(audio))
            chunk = audio[chunk_start:chunk_end]

            # Skip very short final chunks
            if len(chunk) < min_chunk_size:
                break

            # Pad short chunks to at least 1 second for Whisper
            if len(chunk) < SAMPLE_RATE:
                chunk = np.pad(chunk, (0, SAMPLE_RATE - len(chunk)))

            # Skip silent chunks
            if self._is_silent_chunk(chunk):
                pos += step_size
                chunk_idx += 1
                continue

            # Transcribe
            t0 = time.time()
            result = self._transcribe_chunk(chunk)
            processing_time = time.time() - t0
            avg_logprob = self._extract_avg_logprob(result)

            chunk_text = result.get("text", "").strip()

            # Skip hallucinated output
            if self._is_hallucination(chunk_text):
                pos += step_size
                chunk_idx += 1
                continue

            # Extract word-level timestamps, adjusted to global time
            chunk_start_time = chunk_start / SAMPLE_RATE
            chunk_end_time = chunk_end / SAMPLE_RATE
            words = []
            if self.word_timestamps:
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        word_text = w["word"].strip()
                        if not word_text:
                            continue
                        words.append({
                            "word": word_text,
                            "start": round(w["start"] + chunk_start_time, 3),
                            "end": round(w["end"] + chunk_start_time, 3),
                        })
            else:
                # Without word timestamps, create synthetic entries from text
                text_words = chunk_text.split()
                if text_words:
                    word_dur = (chunk_end_time - chunk_start_time) / len(text_words)
                    for i, w in enumerate(text_words):
                        words.append({
                            "word": w.strip(),
                            "start": round(chunk_start_time + i * word_dur, 3),
                            "end": round(chunk_start_time + (i + 1) * word_dur, 3),
                        })

            # Deduplicate overlapping words
            if self.overlap_seconds > 0 and all_words:
                words = self._deduplicate_words(all_words, words, chunk_start_time)

            all_words.extend(words)

            # Build accumulated text from all words
            if words:
                new_text = " ".join(w["word"] for w in words)
                if accumulated_text:
                    accumulated_text += " " + new_text
                else:
                    accumulated_text = new_text

            yield {
                "chunk_idx": chunk_idx,
                "chunk_text": chunk_text,
                "accumulated_text": accumulated_text.strip(),
                "timestamp_s": round(pos / SAMPLE_RATE, 3),
                "duration_s": round(total_duration, 3),
                "words": words,
                "all_words": list(all_words),
                "processing_time_s": round(processing_time, 3),
                "avg_logprob": avg_logprob,
            }

            pos += step_size
            chunk_idx += 1

    def _deduplicate_words(
        self,
        existing_words: list[dict],
        new_words: list[dict],
        overlap_start_time: float,
    ) -> list[dict]:
        """Remove words from new_words that overlap with existing words."""
        if not existing_words or not new_words:
            return new_words

        # Find the latest end time in existing words
        last_existing_end = max(w["end"] for w in existing_words)

        # Only keep new words that start after the last existing word ends
        # (with a small tolerance for timing imprecision)
        tolerance = 0.15  # 150ms tolerance
        deduplicated = []
        for w in new_words:
            if w["start"] >= last_existing_end - tolerance:
                deduplicated.append(w)

        return deduplicated

    def transcribe_batch(self, audio_path: str) -> dict:
        """Transcribe entire audio at once (baseline for comparison)."""
        import time

        self._ensure_model()
        audio = load_audio(audio_path)

        t0 = time.time()
        result = self._transcribe_chunk(audio)
        processing_time = time.time() - t0
        avg_logprob = self._extract_avg_logprob(result)

        text = result.get("text", "").strip()

        words = []
        if self.word_timestamps:
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    words.append({
                        "word": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })
        else:
            # Synthetic word entries from text
            text_words = text.split()
            duration = len(audio) / SAMPLE_RATE
            if text_words:
                word_dur = duration / len(text_words)
                for i, w in enumerate(text_words):
                    words.append({
                        "word": w.strip(),
                        "start": round(i * word_dur, 3),
                        "end": round((i + 1) * word_dur, 3),
                    })

        return {
            "text": text,
            "words": words,
            "processing_time_s": round(processing_time, 3),
            "duration_s": round(len(audio) / SAMPLE_RATE, 3),
            "avg_logprob": avg_logprob,
        }


def create_streaming_modes() -> dict[str, StreamingTranscriber]:
    """Create all streaming mode variants for benchmarking."""
    return {
        "batch": StreamingTranscriber(chunk_seconds=9999),  # effectively batch
        "stream_3s": StreamingTranscriber(chunk_seconds=3.0, overlap_seconds=0.0),
        "stream_3s_overlap": StreamingTranscriber(chunk_seconds=3.0, overlap_seconds=1.0),
        "stream_1s": StreamingTranscriber(chunk_seconds=1.0, overlap_seconds=0.0),
        "stream_2s": StreamingTranscriber(chunk_seconds=2.0, overlap_seconds=0.0),
    }


if __name__ == "__main__":
    import sys as _sys

    audio_file = _sys.argv[1] if len(_sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "reference_audio" / "112001.mp3"
    )

    print(f"Streaming: {audio_file}")
    print(f"Chunk: 3s, No overlap")
    print("-" * 60)

    transcriber = StreamingTranscriber(chunk_seconds=3.0)
    for event in transcriber.stream(audio_file):
        print(
            f"[{event['timestamp_s']:.1f}s] "
            f"chunk={event['chunk_idx']} "
            f"({event['processing_time_s']:.2f}s) "
            f"| {event['chunk_text']}"
        )
        if event["words"]:
            for w in event["words"]:
                print(f"    {w['start']:.2f}-{w['end']:.2f}: {w['word']}")

    print(f"\nFinal: {event['accumulated_text']}")
