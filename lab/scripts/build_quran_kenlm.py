"""Build Quran KenLM assets for shallow-fusion decoding.

Outputs under data/kenlm/:
  - quran_corpus.txt        (LM training corpus)
  - quran_unigrams.txt      (word list for pyctcdecode)
  - quran_{order}gram.arpa  (if lmplz is available)
  - quran_{order}gram.binary (if build_binary is available)

Usage:
  .venv/bin/python scripts/build_quran_kenlm.py --order 5
  .venv/bin/python scripts/build_quran_kenlm.py --order 5 \
    --model_path data/kenlm/quran_5gram.binary \
    --unigrams_path data/kenlm/quran_unigrams.txt

If kenlm binaries are missing, this still generates corpus + unigrams so
model build can be completed later.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QURAN_JSON = PROJECT_ROOT / "data" / "quran.json"
OUT_DIR = PROJECT_ROOT / "data" / "kenlm"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def load_quran_lines() -> list[str]:
    with open(QURAN_JSON, "r", encoding="utf-8") as f:
        verses = json.load(f)
    lines = []
    for v in verses:
        t = normalize_whitespace(v.get("text_clean", ""))
        if t:
            lines.append(t)
    return lines


def write_corpus_and_unigrams(lines: list[str], unigrams_path: Path | None = None) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "quran_corpus.txt"
    if unigrams_path is None:
        unigrams_path = OUT_DIR / "quran_unigrams.txt"
    unigrams_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    vocab = sorted({w for line in lines for w in line.split() if w})
    with open(unigrams_path, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")

    return corpus_path, unigrams_path


def resolve_model_paths(order: int, model_path: str | None) -> tuple[Path, Path]:
    if not model_path:
        arpa_path = OUT_DIR / f"quran_{order}gram.arpa"
        bin_path = OUT_DIR / f"quran_{order}gram.binary"
        return arpa_path, bin_path

    p = Path(model_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".arpa":
        return p, p.with_suffix(".binary")
    if p.suffix in {".binary", ".bin"}:
        return p.with_suffix(".arpa"), p

    # If extension is omitted or unknown, treat as prefix.
    return p.with_suffix(".arpa"), p.with_suffix(".binary")


def find_tool(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path

    # Try common local build locations.
    candidates = [
        PROJECT_ROOT / "data" / "kenlm" / "build" / "bin" / name,
        PROJECT_ROOT / "data" / "kenlm" / "bin" / name,
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_models(corpus_path: Path, order: int, model_path: str | None = None) -> tuple[Path | None, Path | None]:
    lmplz = find_tool("lmplz")
    build_binary = find_tool("build_binary")

    arpa_path, bin_path = resolve_model_paths(order=order, model_path=model_path)

    if not lmplz:
        print("Warning: `lmplz` not found. Skipping ARPA build.")
        return None, None

    run_cmd([
        lmplz,
        "-o",
        str(order),
        "--discount_fallback",
        "--text",
        str(corpus_path),
        "--arpa",
        str(arpa_path),
    ])

    if build_binary:
        run_cmd([build_binary, str(arpa_path), str(bin_path)])
        return arpa_path, bin_path

    print("Warning: `build_binary` not found. ARPA created, binary skipped.")
    return arpa_path, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Target KenLM model path (.binary/.bin or .arpa). If omitted, uses data/kenlm defaults.",
    )
    parser.add_argument(
        "--unigrams_path",
        type=str,
        default=None,
        help="Output path for unigrams file. If omitted, uses data/kenlm/quran_unigrams.txt.",
    )
    args = parser.parse_args()

    lines = load_quran_lines()
    if not lines:
        raise RuntimeError(f"No lines loaded from {QURAN_JSON}")

    unigrams_override = Path(args.unigrams_path) if args.unigrams_path else None
    corpus_path, unigrams_path = write_corpus_and_unigrams(lines, unigrams_path=unigrams_override)
    print(f"Wrote corpus:   {corpus_path}")
    print(f"Wrote unigrams: {unigrams_path}")

    arpa_path, bin_path = build_models(corpus_path, args.order, model_path=args.model_path)
    if arpa_path:
        print(f"Built ARPA:     {arpa_path}")
    if bin_path:
        print(f"Built binary:   {bin_path}")


if __name__ == "__main__":
    main()
