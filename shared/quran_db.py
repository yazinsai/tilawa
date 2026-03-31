import array
import json
import math
from collections import defaultdict
from pathlib import Path
from Levenshtein import ratio
from shared.normalizer import normalize_arabic


def partial_ratio(short: str, long: str) -> float:
    """Levenshtein ratio of *short* against its best-matching window in *long*.

    Useful for detecting when a short transcription is a fragment of a longer
    verse that was already emitted.
    """
    if not short or not long:
        return 0.0
    if len(short) > len(long):
        short, long = long, short
    window = len(short)
    best = 0.0
    for i in range(max(1, len(long) - window + 1)):
        r = ratio(short, long[i:i + window])
        if r > best:
            best = r
            if best == 1.0:
                break
    return best

# Resolve to project root / data / quran.json
DATA_PATH = Path(__file__).parent.parent / "data" / "quran.json"


_BSM_CLEAN = normalize_arabic("بسم الله الرحمن الرحيم")
_FRAGMENT_MIN_WORDS = 4
_FRAGMENT_PARTIAL_WEIGHT = 0.75


class QuranDB:
    def __init__(self, path: Path = DATA_PATH):
        with open(path) as f:
            self.verses = json.load(f)
        self._by_ref = {}
        self._by_surah = {}
        # Strip BOM from text_clean (quran.json verse 1:1 has a stray BOM)
        for v in self.verses:
            v["text_clean"] = v["text_clean"].lstrip('\ufeff')
        for v in self.verses:
            self._by_ref[(v["surah"], v["ayah"])] = v
            self._by_surah.setdefault(v["surah"], []).append(v)
            # Pre-compute bismillah-stripped text for verse 1 of each surah
            # (Al-Fatiha 1:1 IS the bismillah, At-Tawbah 9 has none)
            if (
                v["ayah"] == 1
                and v["surah"] not in (1, 9)
                and v["text_clean"].startswith(_BSM_CLEAN)
            ):
                stripped = v["text_clean"][len(_BSM_CLEAN):].strip()
                v["text_clean_no_bsm"] = stripped if stripped else None
            else:
                v["text_clean_no_bsm"] = None

        # Build reverse-lookup and trigram index for fast candidate retrieval
        self._ref_to_idx: dict[tuple, int] = {}
        for i, v in enumerate(self.verses):
            self._ref_to_idx[(v["surah"], v["ayah"])] = i
        self._build_trigram_index()

    @property
    def total_verses(self):
        return len(self.verses)

    @property
    def surah_count(self):
        return len(self._by_surah)

    def get_verse(self, surah: int, ayah: int):
        return self._by_ref.get((surah, ayah))

    def get_surah(self, surah: int):
        return self._by_surah.get(surah, [])

    def get_next_verse(self, surah: int, ayah: int) -> dict | None:
        """Return the next verse after surah:ayah, or None if last verse."""
        verses = self._by_surah.get(surah, [])
        for i, v in enumerate(verses):
            if v["ayah"] == ayah:
                if i + 1 < len(verses):
                    return verses[i + 1]
                next_surah = self._by_surah.get(surah + 1, [])
                return next_surah[0] if next_surah else None
        return None

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        text = normalize_arabic(text)
        scored = []
        for v in self.verses:
            score = ratio(text, v["text_clean"])
            scored.append({**v, "score": score, "text": v["text_uthmani"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _continuation_bonuses(
        self, hint: tuple[int, int] | None
    ) -> dict[tuple[int, int], float]:
        """Build a map of (surah, ayah) → score bonus for expected next verses."""
        if not hint:
            return {}
        h_surah, h_ayah = hint
        bonuses: dict[tuple[int, int], float] = {}
        nv = self._by_ref.get((h_surah, h_ayah + 1))
        if nv:
            bonuses[(h_surah, h_ayah + 1)] = 0.22
            if self._by_ref.get((h_surah, h_ayah + 2)):
                bonuses[(h_surah, h_ayah + 2)] = 0.12
            if self._by_ref.get((h_surah, h_ayah + 3)):
                bonuses[(h_surah, h_ayah + 3)] = 0.06
        else:
            # Last ayah in surah — bonus carries to first ayah(s) of next surah
            next_verses = self._by_surah.get(h_surah + 1, [])
            for i, nv in enumerate(next_verses[:3]):
                bonus = [0.22, 0.12, 0.06][i]
                bonuses[(nv["surah"], nv["ayah"])] = bonus
        return bonuses

    @staticmethod
    def _char_trigrams(text: str) -> set[str]:
        """Extract character trigrams from text."""
        return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else set()

    def _build_trigram_index(self):
        """Build char-trigram inverted index over all verses for fast retrieval."""
        posting: dict[str, set[int]] = defaultdict(set)
        N = len(self.verses)
        for idx, v in enumerate(self.verses):
            trigrams = self._char_trigrams(v["text_clean"])
            if v.get("text_clean_no_bsm"):
                trigrams |= self._char_trigrams(v["text_clean_no_bsm"])
            for tri in trigrams:
                posting[tri].add(idx)
        # Convert to sorted array.array('H') for compactness (~3MB)
        self._trigram_index: dict[str, array.array] = {}
        self._idf: dict[str, float] = {}
        for tri, indices in posting.items():
            self._trigram_index[tri] = array.array('H', sorted(indices))
            self._idf[tri] = math.log(N / len(indices))

    def _trigram_candidates(self, text: str, top_k: int = 50) -> list[int]:
        """Return top_k verse indices by IDF-weighted trigram overlap with text."""
        trigrams = self._char_trigrams(text)
        if not trigrams:
            return []
        scores: dict[int, float] = {}
        for tri in trigrams:
            if tri not in self._trigram_index:
                continue
            w = self._idf[tri]
            for idx in self._trigram_index[tri]:
                scores[idx] = scores.get(idx, 0.0) + w
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _suffix_prefix_score(text: str, verse_text: str) -> float:
        """Best Levenshtein ratio from matching suffixes of *text* against
        equal-length prefixes of *verse_text*.

        After a window reset the transcription often starts with residual
        words from the *previous* verse followed by the start of the *next*
        verse. This method finds the best alignment by sliding the split
        point through the transcription.
        """
        words_t = text.split()
        words_v = verse_text.split()
        if len(words_t) < 2 or len(words_v) < 2:
            return 0.0
        best = 0.0
        max_trim = min(len(words_t) // 2, 4)
        for trim in range(1, max_trim + 1):
            suffix = " ".join(words_t[trim:])
            n = len(words_t) - trim
            prefix = " ".join(words_v[:min(n, len(words_v))])
            best = max(best, ratio(suffix, prefix))
        return best

    @staticmethod
    def _fragment_score(text: str, verse_text: str, full_ratio: float) -> float:
        """Score a partial transcript against a longer verse.

        Streaming windows often contain only a fragment of a long ayah. A
        full-string Levenshtein ratio suppresses those matches too heavily, so
        blend in partial matching when the query is long enough to be specific.
        Short candidate verses are penalized to avoid spuriously matching a
        tiny verse that happens to occur inside a longer transcript.
        """
        query_words = text.split()
        verse_words = verse_text.split()
        if len(query_words) < _FRAGMENT_MIN_WORDS or len(verse_words) < 2:
            return full_ratio

        fragment = partial_ratio(text, verse_text)
        if fragment <= full_ratio:
            return full_ratio

        shorter_penalty = min(1.0, len(verse_words) / max(len(query_words), 1))
        blended = (
            (1.0 - _FRAGMENT_PARTIAL_WEIGHT) * full_ratio
            + _FRAGMENT_PARTIAL_WEIGHT * fragment * shorter_penalty
        )
        return max(full_ratio, blended)

    def match_verse(
        self,
        text: str,
        threshold: float = 0.3,
        max_span: int = 3,
        hint: tuple[int, int] | None = None,
        return_top_k: int = 0,
    ) -> dict | None:
        """Find the best matching verse or consecutive verse span.

        Two-pass: first find top single-verse candidates (fast), then try
        multi-ayah spans only around those candidates.

        If *hint* is provided as (surah, ayah) of the last matched verse,
        the expected next verses receive a score bonus so sequential
        recitation is favoured over re-inferring from scratch.

        If *return_top_k* > 0, the returned dict includes a ``"runners_up"``
        list with the next-best candidates (each with raw_score and bonus).
        """
        text = normalize_arabic(text)
        if not text.strip():
            return None

        bonuses = self._continuation_bonuses(hint)

        # Pass 1: trigram retrieval → Levenshtein rerank on small candidate set
        candidate_idxs = set(self._trigram_candidates(text, top_k=50))

        # Always inject continuation bonus candidates
        for ref in bonuses:
            idx = self._ref_to_idx.get(ref)
            if idx is not None:
                candidate_idxs.add(idx)

        # Fallback: if trigram retrieval returns too few, full scan
        if len(candidate_idxs) < 20:
            candidate_idxs = set(range(len(self.verses)))

        scored = []
        for idx in candidate_idxs:
            v = self.verses[idx]
            raw = ratio(text, v["text_clean"])
            raw = self._fragment_score(text, v["text_clean"], raw)
            if v["text_clean_no_bsm"]:
                stripped_raw = ratio(text, v["text_clean_no_bsm"])
                stripped_raw = self._fragment_score(
                    text,
                    v["text_clean_no_bsm"],
                    stripped_raw,
                )
                raw = max(raw, stripped_raw)
            bonus = bonuses.get((v["surah"], v["ayah"]), 0.0)
            if bonus > 0:
                sp = self._suffix_prefix_score(text, v["text_clean"])
                raw = max(raw, sp)
            scored.append((v, raw, bonus, min(raw + bonus, 1.0)))
        scored.sort(key=lambda x: x[3], reverse=True)

        if not scored:
            return None

        best_v, best_raw, best_bonus, best_score = scored[0]
        best = {**best_v, "score": best_score, "raw_score": best_raw, "bonus": best_bonus}

        # Collect single-verse runners-up before span pass
        top_singles = [
            {
                "surah": v["surah"],
                "ayah": v["ayah"],
                "raw_score": round(raw, 3),
                "bonus": round(bon, 3),
                "score": round(total, 3),
                "text_clean": v["text_clean"][:60],
            }
            for v, raw, bon, total in scored[:max(return_top_k, 5)]
        ]

        # Pass 2: try multi-ayah spans around all candidate surahs
        # (candidate set is already small from trigram retrieval)
        seen_surahs = set()
        for v, _raw, _bon, _total in scored:
            s = v["surah"]
            if s in seen_surahs:
                continue
            seen_surahs.add(s)
            verses = self._by_surah[s]
            for i, sv in enumerate(verses):
                for span in range(2, max_span + 1):
                    if i + span > len(verses):
                        break
                    chunk = verses[i:i + span]
                    # Use no-bismillah text for the first verse in a span
                    first_text = (
                        chunk[0]["text_clean_no_bsm"] or chunk[0]["text_clean"]
                    )
                    combined = " ".join(
                        [first_text] + [c["text_clean"] for c in chunk[1:]]
                    )
                    raw = ratio(text, combined)
                    bonus = bonuses.get((chunk[0]["surah"], chunk[0]["ayah"]), 0.0)
                    score = min(raw + bonus, 1.0)
                    if score > best_score:
                        best_score = score
                        best = {
                            "surah": s,
                            "ayah": chunk[0]["ayah"],
                            "ayah_end": chunk[-1]["ayah"],
                            "text": " ".join(c["text_uthmani"] for c in chunk),
                            "text_clean": combined,
                            "score": score,
                            "raw_score": raw,
                            "bonus": bonus,
                        }

        if best_score >= threshold:
            if return_top_k > 0:
                best["runners_up"] = top_singles[:return_top_k]
            return best
        return None
