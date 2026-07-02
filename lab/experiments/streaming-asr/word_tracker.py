"""
Word-level timestamp extraction from Whisper.

Provides a clean interface for extracting word-level timestamps
from audio using mlx-whisper's built-in word_timestamps feature.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# Reuse parent project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio

MLX_MODEL = "mlx-community/whisper-base-mlx"
SAMPLE_RATE = 16000


@dataclass
class TimestampedWord:
    """A single word with its start and end timestamps."""
    word: str
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self):
        return f"[{self.start:.2f}-{self.end:.2f}] {self.word}"


class WordTimestampExtractor:
    """Extract word-level timestamps from audio using mlx-whisper."""

    def __init__(self, model_id: str = MLX_MODEL):
        self.model_id = model_id
        self._mlx_whisper = None

    def _ensure_model(self):
        if self._mlx_whisper is None:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper

    def extract(self, audio_input) -> list[TimestampedWord]:
        """
        Extract word-level timestamps from audio.

        Args:
            audio_input: Path string or numpy array of audio data (16kHz float32)

        Returns:
            List of TimestampedWord objects
        """
        self._ensure_model()

        if isinstance(audio_input, (str, Path)):
            audio = load_audio(str(audio_input))
        else:
            audio = audio_input

        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_id,
            language="ar",
            word_timestamps=True,
            condition_on_previous_text=True,
            hallucination_silence_threshold=1.0,
        )

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word_text = w["word"].strip()
                if not word_text:
                    continue
                words.append(TimestampedWord(
                    word=word_text,
                    start=round(w["start"], 3),
                    end=round(w["end"], 3),
                ))

        # Filter out hallucinated words (repeated sequences at the end)
        words = self._filter_hallucinations(words)

        return words

    def extract_from_chunk(
        self,
        audio_chunk: np.ndarray,
        time_offset: float = 0.0,
    ) -> list[TimestampedWord]:
        """
        Extract words from a chunk, offsetting timestamps to global time.

        Args:
            audio_chunk: numpy array of audio (16kHz float32)
            time_offset: seconds to add to each timestamp

        Returns:
            List of TimestampedWord objects with globally adjusted timestamps
        """
        self._ensure_model()

        # Pad short chunks
        if len(audio_chunk) < SAMPLE_RATE:
            audio_chunk = np.pad(audio_chunk, (0, SAMPLE_RATE - len(audio_chunk)))

        result = self._mlx_whisper.transcribe(
            audio_chunk,
            path_or_hf_repo=self.model_id,
            language="ar",
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
        )

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word_text = w["word"].strip()
                if not word_text:
                    continue
                words.append(TimestampedWord(
                    word=word_text,
                    start=round(w["start"] + time_offset, 3),
                    end=round(w["end"] + time_offset, 3),
                ))

        words = self._filter_hallucinations(words)
        return words

    @staticmethod
    def _filter_hallucinations(words: list[TimestampedWord]) -> list[TimestampedWord]:
        """Remove hallucinated repetitions from the word list."""
        if len(words) < 4:
            return words

        # Find where repetition starts
        clean = []
        repeat_count = 0
        for i, w in enumerate(words):
            if i > 0 and w.word == words[i - 1].word:
                repeat_count += 1
                if repeat_count >= 2:
                    # 3+ consecutive same word = hallucination, stop here
                    break
            else:
                repeat_count = 0
            clean.append(w)

        return clean


if __name__ == "__main__":
    import sys as _sys

    audio_file = _sys.argv[1] if len(_sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "reference_audio" / "001001.mp3"
    )

    print(f"Extracting word timestamps: {audio_file}")
    print("-" * 60)

    extractor = WordTimestampExtractor()
    words = extractor.extract(audio_file)

    total_duration = 0
    for w in words:
        print(w)
        total_duration = max(total_duration, w.end)

    print(f"\nTotal words: {len(words)}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Full text: {' '.join(w.word for w in words)}")
