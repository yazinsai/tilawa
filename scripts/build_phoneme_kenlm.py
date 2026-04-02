"""Build phoneme-level KenLM assets for CTC beam search decoding.

Outputs under data/kenlm_phoneme/:
  - quran_phoneme_corpus.txt   (one sentence of phoneme-words per verse)
  - quran_phoneme_unigrams.txt (sorted unique phoneme-words)
  - phoneme_ngrams.json        (compact bigram/trigram table for browser use)

Usage:
  .venv/bin/python scripts/build_phoneme_kenlm.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QURAN_PHONEMES = PROJECT_ROOT / "web" / "frontend" / "public" / "quran_phonemes.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kenlm_phoneme"


def load_phoneme_sentences() -> list[list[str]]:
    """Load quran_phonemes.json and return list of phoneme-word sentences."""
    data = json.loads(QURAN_PHONEMES.read_text(encoding="utf-8"))
    sentences = []
    for verse in data:
        words = verse.get("phoneme_words", [])
        if words:
            sentences.append(words)
    return sentences


def build_corpus_and_unigrams(sentences: list[list[str]]) -> tuple[Path, Path, set[str]]:
    """Write corpus and unigrams files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = OUTPUT_DIR / "quran_phoneme_corpus.txt"
    unigrams_path = OUTPUT_DIR / "quran_phoneme_unigrams.txt"

    vocab: set[str] = set()
    with corpus_path.open("w", encoding="utf-8") as f:
        for words in sentences:
            f.write(" ".join(words) + "\n")
            vocab.update(words)

    sorted_vocab = sorted(vocab)
    with unigrams_path.open("w", encoding="utf-8") as f:
        for word in sorted_vocab:
            f.write(word + "\n")

    print(f"Corpus: {len(sentences)} sentences → {corpus_path}")
    print(f"Vocabulary: {len(sorted_vocab)} unique phoneme-words → {unigrams_path}")
    return corpus_path, unigrams_path, vocab


def build_ngram_json(sentences: list[list[str]]) -> Path:
    """Build compact n-gram JSON for browser-side beam search scoring."""
    unigram_counts: Counter[str] = Counter()
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for words in sentences:
        for i, w in enumerate(words):
            unigram_counts[w] += 1
            if i > 0:
                bigram_counts[words[i - 1]][w] += 1
            if i > 1:
                key = f"{words[i - 2]} {words[i - 1]}"
                trigram_counts[key][w] += 1

    # Compact format: only keep counts >= 1
    ngrams = {
        "unigrams": dict(unigram_counts),
        "bigrams": {k: dict(v) for k, v in bigram_counts.items()},
        "trigrams": {k: dict(v) for k, v in trigram_counts.items()},
        "total_words": sum(unigram_counts.values()),
        "vocab_size": len(unigram_counts),
    }

    out_path = OUTPUT_DIR / "phoneme_ngrams.json"
    out_path.write_text(json.dumps(ngrams, ensure_ascii=False), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"N-grams: {len(unigram_counts)} unigrams, "
          f"{sum(len(v) for v in bigram_counts.values())} bigrams, "
          f"{sum(len(v) for v in trigram_counts.values())} trigrams → {out_path} ({size_kb:.0f} KB)")
    return out_path


def main():
    sentences = load_phoneme_sentences()
    corpus_path, unigrams_path, vocab = build_corpus_and_unigrams(sentences)
    ngrams_path = build_ngram_json(sentences)

    # Try to build KenLM if lmplz is available
    import shutil
    lmplz = shutil.which("lmplz")
    if lmplz:
        import subprocess
        order = 5
        arpa_path = OUTPUT_DIR / f"quran_phoneme_{order}gram.arpa"
        print(f"\nBuilding {order}-gram model with lmplz...")
        subprocess.run([
            lmplz, "-o", str(order), "--discount_fallback",
            "--text", str(corpus_path), "--arpa", str(arpa_path),
        ], check=True)
        print(f"ARPA model: {arpa_path}")

        build_binary = shutil.which("build_binary")
        if build_binary:
            binary_path = OUTPUT_DIR / f"quran_phoneme_{order}gram.binary"
            subprocess.run([build_binary, str(arpa_path), str(binary_path)], check=True)
            print(f"Binary model: {binary_path}")
    else:
        print("\nlmplz not found — skipping ARPA/binary model build.")
        print("Install KenLM to build: pip install https://github.com/kpu/kenlm/archive/master.zip")


if __name__ == "__main__":
    main()
