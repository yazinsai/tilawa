"""
Verse Position Tracker — Track position within a Quran verse in real-time.

As words arrive from streaming ASR, this module:
1. Identifies which verse is being recited
2. Tracks how far into the verse the recitation has reached
3. Uses sequence alignment (DP) to handle imperfect transcriptions
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

# Reuse parent project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic


@dataclass
class VerseMatch:
    """Represents a matched verse with position tracking."""
    surah: int
    ayah: int
    surah_name_en: str
    verse_text_uthmani: str
    verse_text_clean: str
    verse_words: list[str]
    confidence: float
    word_position: int  # how many verse words matched so far
    total_words: int
    matched_word_indices: list[int] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        return (self.word_position / self.total_words * 100) if self.total_words > 0 else 0

    @property
    def reference(self) -> str:
        return f"{self.surah_name_en} {self.surah}:{self.ayah}"

    def __repr__(self):
        return (
            f"VerseMatch({self.reference}, "
            f"position={self.word_position}/{self.total_words}, "
            f"{self.progress_pct:.0f}%, "
            f"conf={self.confidence:.3f})"
        )


class VersePositionTracker:
    """Track position within a verse as words arrive from streaming ASR."""

    def __init__(self, quran_db: QuranDB = None, match_threshold: float = 0.25):
        self.db = quran_db or QuranDB()
        self.match_threshold = match_threshold
        self._accumulated_words: list[str] = []
        self._current_match: VerseMatch | None = None
        self._match_history: list[VerseMatch] = []

    def reset(self):
        """Reset tracker for a new recitation."""
        self._accumulated_words = []
        self._current_match = None
        self._match_history = []

    @property
    def current_match(self) -> VerseMatch | None:
        return self._current_match

    @property
    def accumulated_text(self) -> str:
        return " ".join(self._accumulated_words)

    def update(self, new_words: list[str]) -> VerseMatch | None:
        """
        Update with new recognized words and return current verse match.

        Args:
            new_words: List of newly recognized Arabic words

        Returns:
            VerseMatch if a verse is identified, None otherwise
        """
        # Clean and accumulate
        for w in new_words:
            cleaned = normalize_arabic(w.strip())
            if cleaned:
                self._accumulated_words.append(cleaned)

        if not self._accumulated_words:
            return None

        text = " ".join(self._accumulated_words)

        # Use prefix-aware matching for streaming
        match = self._prefix_aware_match(text)
        if not match:
            # Fall back to standard match
            match = self.db.match_verse(text, threshold=self.match_threshold)
        if not match:
            return None

        verse_words = match["text_clean"].split()

        # Compute word-level alignment
        position, matched_indices = self._align_position(
            self._accumulated_words, verse_words
        )

        self._current_match = VerseMatch(
            surah=match["surah"],
            ayah=match["ayah"],
            surah_name_en=match["surah_name_en"],
            verse_text_uthmani=match["text_uthmani"],
            verse_text_clean=match["text_clean"],
            verse_words=verse_words,
            confidence=match["score"],
            word_position=position,
            total_words=len(verse_words),
            matched_word_indices=matched_indices,
        )

        self._match_history.append(self._current_match)
        return self._current_match

    def update_from_text(self, accumulated_text: str) -> VerseMatch | None:
        """
        Update from a full accumulated text string (alternative to word-by-word).

        Args:
            accumulated_text: Full text accumulated so far

        Returns:
            VerseMatch if identified, None otherwise
        """
        cleaned = normalize_arabic(accumulated_text)
        self._accumulated_words = cleaned.split()
        return self.update([])  # trigger match with already-accumulated words

    def _prefix_aware_match(self, text: str, top_k: int = 5) -> dict | None:
        """
        Match that considers the text might be a prefix of a longer verse.

        Standard Levenshtein ratio penalizes matching a short text against
        a long verse. This method uses prefix-based scoring: it compares
        the input text against the first N words of each verse (where N
        is the number of input words), favoring verses where the prefix
        matches well.
        """
        from Levenshtein import ratio

        text_clean = normalize_arabic(text)
        if not text_clean.strip():
            return None

        input_words = text_clean.split()
        n_words = len(input_words)

        candidates = []
        for v in self.db.verses:
            verse_clean = v["text_clean"]
            verse_words = verse_clean.split()

            # Compute prefix score: compare input against the same number
            # of words from the start of the verse
            prefix_len = min(n_words, len(verse_words))
            verse_prefix = " ".join(verse_words[:prefix_len])
            prefix_score = ratio(text_clean, verse_prefix)

            # Also compute full text score for disambiguation
            full_score = ratio(text_clean, verse_clean)

            # Combined score: prefix match matters most during streaming,
            # but full score helps when we have most of the verse
            coverage = n_words / max(len(verse_words), 1)
            if coverage > 0.8:
                # Nearly complete verse: trust full score more
                combined = 0.3 * prefix_score + 0.7 * full_score
            else:
                # Partial verse: trust prefix score more
                combined = 0.7 * prefix_score + 0.3 * full_score

            candidates.append({
                **v,
                "prefix_score": prefix_score,
                "full_score": full_score,
                "score": combined,
                "coverage": coverage,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if candidates and candidates[0]["score"] >= self.match_threshold:
            return candidates[0]
        return None

    def _align_position(
        self,
        recognized_words: list[str],
        verse_words: list[str],
    ) -> tuple[int, list[int]]:
        """
        Find how far into the verse the recognized words reach.

        Uses a modified Longest Common Subsequence (LCS) approach that
        prefers sequential matches from the beginning of the verse,
        tolerating ASR errors and insertions.

        Returns:
            (position, matched_indices) where position is the furthest
            verse word index matched, and matched_indices is the list
            of verse word indices that matched.
        """
        if not recognized_words or not verse_words:
            return 0, []

        n = len(recognized_words)
        m = len(verse_words)

        # DP for longest common subsequence with position tracking
        # dp[i][j] = length of LCS of recognized[:i] and verse[:j]
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if self._words_match(recognized_words[i - 1], verse_words[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Backtrack to find which verse words matched
        matched_indices = []
        i, j = n, m
        while i > 0 and j > 0:
            if self._words_match(recognized_words[i - 1], verse_words[j - 1]):
                matched_indices.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        matched_indices.reverse()

        # Position = furthest verse word matched + 1 (1-indexed count)
        if matched_indices:
            position = matched_indices[-1] + 1
        else:
            position = 0

        return position, matched_indices

    @staticmethod
    def _words_match(w1: str, w2: str, threshold: float = 0.7) -> bool:
        """
        Check if two Arabic words match, with tolerance for ASR errors.

        Uses Levenshtein ratio for fuzzy matching.
        """
        # Exact match (fast path)
        if w1 == w2:
            return True

        # Very short words — require exact match
        if len(w1) <= 2 or len(w2) <= 2:
            return w1 == w2

        # Fuzzy match using Levenshtein ratio
        from Levenshtein import ratio
        return ratio(w1, w2) >= threshold

    def get_highlighted_text(self) -> str | None:
        """
        Return the verse text with position markers for display.

        Words that have been matched are wrapped in brackets.
        """
        if not self._current_match:
            return None

        m = self._current_match
        parts = []
        matched_set = set(m.matched_word_indices)

        for i, word in enumerate(m.verse_words):
            if i in matched_set:
                parts.append(f"[{word}]")
            else:
                parts.append(word)

        return " ".join(parts)


class MultiVerseTracker:
    """
    Track position across multiple verses (for multi-ayah recitations).

    When the confidence for the current verse drops below a threshold
    while new words keep coming, it tries to match the new words against
    the next verse(s) in sequence.
    """

    def __init__(self, quran_db: QuranDB = None):
        self.db = quran_db or QuranDB()
        self.trackers: list[VersePositionTracker] = []
        self._current_tracker_idx = 0

    def reset(self):
        self.trackers = []
        self._current_tracker_idx = 0

    def update(self, new_words: list[str]) -> list[VerseMatch]:
        """Update with new words and return all verse matches."""
        if not self.trackers:
            tracker = VersePositionTracker(self.db)
            self.trackers.append(tracker)

        current = self.trackers[self._current_tracker_idx]
        match = current.update(new_words)

        matches = []
        for t in self.trackers:
            if t.current_match:
                matches.append(t.current_match)

        return matches


if __name__ == "__main__":
    # Demo: simulate progressive word arrival
    print("Verse Position Tracker Demo")
    print("=" * 60)

    db = QuranDB()
    tracker = VersePositionTracker(db)

    # Simulate words arriving from Al-Fatiha:1
    simulated_words = [
        ["بسم"],
        ["الله"],
        ["الرحمن"],
        ["الرحيم"],
    ]

    for word_batch in simulated_words:
        match = tracker.update(word_batch)
        if match:
            print(f"  Words so far: {tracker.accumulated_text}")
            print(f"  -> {match}")
            print(f"  Highlighted: {tracker.get_highlighted_text()}")
        else:
            print(f"  Words so far: {tracker.accumulated_text} -> No match yet")
        print()

    print("\n" + "=" * 60)
    print("Simulating Ayat al-Kursi (partial)...")
    tracker.reset()

    kursi_words = "الله لا اله الا هو الحي القيوم لا تاخذه سنه".split()
    for i in range(0, len(kursi_words), 2):
        batch = kursi_words[i:i+2]
        match = tracker.update(batch)
        if match:
            print(f"  [{tracker.accumulated_text}]")
            print(f"  -> {match.reference}: {match.word_position}/{match.total_words} ({match.progress_pct:.0f}%)")
        print()
