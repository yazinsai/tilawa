import json
from pathlib import Path
from Levenshtein import ratio
from offline_tarteel.normalizer import normalize_arabic

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "quran.json"


class QuranDB:
    def __init__(self, path: Path = DATA_PATH):
        with open(path) as f:
            self.verses = json.load(f)
        self._by_ref = {}
        self._by_surah = {}
        for v in self.verses:
            self._by_ref[(v["surah"], v["ayah"])] = v
            self._by_surah.setdefault(v["surah"], []).append(v)

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

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        text = normalize_arabic(text)
        scored = []
        for v in self.verses:
            score = ratio(text, v["text_clean"])
            scored.append({**v, "score": score, "text": v["text_uthmani"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def match_verse(self, text: str, threshold: float = 0.3, max_span: int = 3) -> dict | None:
        """Find the best matching verse or consecutive verse span.

        Two-pass: first find top single-verse candidates (fast), then try
        multi-ayah spans only around those candidates.
        """
        text = normalize_arabic(text)
        if not text.strip():
            return None

        # Pass 1: score all single verses
        scored = []
        for v in self.verses:
            score = ratio(text, v["text_clean"])
            scored.append((v, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        best = {**scored[0][0], "score": scored[0][1]}
        best_score = scored[0][1]

        # Pass 2: try multi-ayah spans around top 20 candidates
        seen_surahs = set()
        for v, _ in scored[:20]:
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
                    combined = " ".join(c["text_clean"] for c in chunk)
                    score = ratio(text, combined)
                    if score > best_score:
                        best_score = score
                        best = {
                            "surah": s,
                            "ayah": chunk[0]["ayah"],
                            "ayah_end": chunk[-1]["ayah"],
                            "text": " ".join(c["text_uthmani"] for c in chunk),
                            "text_clean": combined,
                            "score": score,
                        }

        if best_score >= threshold:
            return best
        return None
