"""Test the streaming pipeline with text-based input."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB

db = QuranDB()

def test_pipeline_single_verse():
    v = db.get_verse(1, 1)
    pipeline = StreamingPipeline(db=db)
    emissions = pipeline.run_on_text([v["text_clean"]])
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1 and emissions[0]["ayah"] == 1

def test_pipeline_multi_verse_sequence():
    v1 = db.get_verse(103, 1)
    v2 = db.get_verse(103, 2)
    v3 = db.get_verse(103, 3)
    pipeline = StreamingPipeline(db=db)
    # Simulate accumulated text growing over time
    emissions = pipeline.run_on_text([
        v1["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"] + " " + v3["text_clean"],
    ])
    assert len(emissions) >= 3
    assert emissions[0]["surah"] == 103 and emissions[0]["ayah"] == 1
    assert emissions[1]["surah"] == 103 and emissions[1]["ayah"] == 2
    assert emissions[2]["surah"] == 103 and emissions[2]["ayah"] == 3

def test_pipeline_run_on_full_transcript():
    """Test run_on_full_transcript with a mock transcribe function."""
    v = db.get_verse(112, 1)
    pipeline = StreamingPipeline(db=db)
    # Mock transcribe that returns the verse text
    mock_transcribe = lambda path: v["text_clean"]
    emissions = pipeline.run_on_full_transcript("dummy_path.wav", mock_transcribe)
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 112 and emissions[0]["ayah"] == 1


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
