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
    """After detecting 55:12, the repeated refrain should match 55:13."""
    tracker = VerseTracker(db, last_emission=(55, 12))
    v13 = db.get_verse(55, 13)
    emissions = tracker.process_text(v13["text_clean"])
    emissions += tracker.finalize()
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 55
    assert emissions[0]["ayah"] == 13

def test_finalize_emits_partial():
    """Partial text should emit on finalize if reasonable match."""
    tracker = VerseTracker(db)
    # Use enough of Ayat al-Kursi to distinguish it from 3:2 (which is the
    # same opening but only 7 words long).
    partial = "الله لا اله الا هو الحي القيوم لا تاخذه سنه ولا نوم"
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


def test_streaming_mode_rejects_single_word():
    """In streaming mode, a single word should be gated by MIN_WORDS_FOR_MATCH."""
    tracker = VerseTracker(db, streaming_mode=True)
    # A single word — should be gated by the minimum word count
    emissions = tracker.process_delta("يا")
    emissions += tracker.finalize()
    assert emissions == []


def test_streaming_mode_min_words_gate():
    """In streaming mode, fewer than 2 accumulated words should not match."""
    tracker = VerseTracker(db, streaming_mode=True)
    # Just 1 word — should be gated even if it partially matches a verse
    emissions = tracker.process_delta("بسم")
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
