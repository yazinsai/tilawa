"""
Streaming pipeline — connects chunked audio transcription to verse detection.

Can be used with any ASR backend that exposes a transcribe() function.
Supports three modes:
  - run_on_text(): test with pre-transcribed text chunks
  - run_on_full_transcript(): transcribe whole file, then detect verses
  - run_on_audio_chunked(): split audio into chunks, transcribe each,
    feed growing accumulated text to VerseTracker progressively
"""

import os
import tempfile

import numpy as np
import soundfile as sf

from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.verse_tracker import VerseTracker

SAMPLE_RATE = 16000
MIN_CHUNK_SAMPLES = 8000  # 0.5s — skip chunks shorter than this
MIN_CHUNK_LOG_PROB = -1.0
MIN_CHUNK_WORDS = 2
HIGH_CONFIDENCE_THRESHOLD = 0.7
MAX_HOLD_CHUNKS = 3


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

        all_emissions.extend(tracker.finalize())
        return all_emissions

    def run_on_full_transcript(self, audio_path: str, transcribe_fn) -> list[dict]:
        """Run verse detection on a full transcript (non-streaming).

        Transcribes the whole file at once, then iteratively calls
        match_verse to peel off verses from the transcript front-to-back.
        Works for multi-verse recordings because trigram-indexed match_verse
        is length-agnostic — trigram overlap doesn't penalize length mismatch.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        transcript = transcribe_fn(audio_path)

        from shared.normalizer import normalize_arabic
        remaining = normalize_arabic(transcript)
        if not remaining.strip():
            return []

        emissions = []
        hint = None
        min_score = 0.3
        for _ in range(20):  # safety bound
            if not remaining.strip():
                break
            result = self.db.match_verse(remaining, max_span=8, hint=hint)
            if not result or result.get("score", 0) < min_score:
                break
            min_score = 0.7  # after first match, require higher confidence
            surah = result["surah"]
            ayah_start = result["ayah"]
            ayah_end = result.get("ayah_end") or ayah_start
            for ayah in range(ayah_start, ayah_end + 1):
                emissions.append({"surah": surah, "ayah": ayah, "score": result["score"]})
            matched_words = result["text_clean"].split()
            rem_words = remaining.split()
            overlap = min(len(matched_words), len(rem_words))
            remaining = " ".join(rem_words[overlap:])
            hint = (surah, ayah_end)
        return emissions

    def run_on_audio_chunked(
        self,
        audio_path: str,
        transcribe_fn,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 0.0,
    ) -> list[dict]:
        """Run streaming verse detection with chunked audio.

        Splits audio into chunks, transcribes each independently using
        the provided backend, accumulates text, and feeds the growing
        transcript to VerseTracker after each chunk.

        Supports confidence gating: if transcribe_fn returns a dict with
        "text" and "avg_logprob" keys, chunks with low confidence are skipped.
        For backward compatibility, transcribe_fn may also return a plain str.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str | dict
            chunk_seconds: Duration of each audio chunk
            overlap_seconds: Overlap between consecutive chunks

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
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
            # Valid chunk arrived -- confirm any pending tentative
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
