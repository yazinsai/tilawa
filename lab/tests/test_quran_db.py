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
    nxt = db.get_next_verse(1, 7)
    assert nxt is not None
    assert nxt["surah"] == 2 and nxt["ayah"] == 1

def test_get_next_verse_last_verse():
    nxt = db.get_next_verse(114, 6)
    assert nxt is None

def test_get_next_verse_invalid():
    nxt = db.get_next_verse(999, 1)
    assert nxt is None
