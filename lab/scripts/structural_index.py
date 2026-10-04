"""Write the SDK's structural-rules asset from the Quran text (no audio).

``packages/core/src/recitation/structural-index.json``:

- ``pairs``: the similar-verse look-alike index (``similar_verse.build_index``, same order),
  compacted to ``"s:a" -> [[ms, ma, tag, i1, i2, j1, j2, tag, ...], ...]`` with tag
  0 = replace, 1 = delete, 2 = insert. Word indices only; no Quran text.
- ``ident``: ayahs the ayah-order rule never flags (``ayah_order.ident_guarded``).

    ../.venv/bin/python scripts/structural_index.py --out ../packages/core/src/recitation/structural-index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ayah_order import ident_guarded  # noqa: E402
from similar_verse import DEFAULT_CORPUS, Corpus, build_index  # noqa: E402

TAGS = {"replace": 0, "delete": 1, "insert": 2}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    corpus = Corpus(args.corpus)
    idx = build_index(corpus)
    pairs = {}
    for key, looks in idx.items():
        pairs[key] = [[*c["m"], *[v for r in c["regions"] for v in (TAGS[r[0]], *r[1:])]] for c in looks]
    ident = [f"{s}:{a}" for (s, a) in sorted(corpus.ph) if ident_guarded(corpus, s, a)]
    out = {"v": 1, "pairs": pairs, "ident": ident}
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(pairs)} ayahs, {sum(len(v) for v in pairs.values())} pairs, {len(ident)} guarded, "
          f"{args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
