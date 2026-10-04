"""Listen to one TLOG slip candidate locally.

Fetches ``audio/tlog/<id>.flac`` from the ``zipformer-ctc-training`` volume
into ``/tmp/tlog_slips/audio`` and prints the ayah's reference words with the
candidate word marked. The waveform is not copied anywhere else.

  TILAWA_DATA_ROOT=/workspace/lab/data python scripts/review_slips.py <clip-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from locate_slips import _fetch  # noqa: E402

AUDIO_DIR = Path("/tmp/tlog_slips/audio")
CANDIDATES = Path("/tmp/tlog_slips/tlog_candidates.jsonl")


def _display_words(surah: int, ayah: int) -> list[str]:
    from shared.phoneme_labels import resolve_quran_json

    data = json.loads(resolve_quran_json().read_text(encoding="utf-8"))
    for surah_row in data["surahs"]:
        if surah_row["n"] != surah:
            continue
        for ayah_row in surah_row["ayahs"]:
            if ayah_row["n"] == ayah:
                return [w[2] if len(w) > 2 else w[1] for w in ayah_row["w"]]
    raise ValueError(f"no ayah {surah}:{ayah}")


def _load_rows(path: Path, clip_id: str) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") == clip_id:
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_id")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--surah", type=int, default=0)
    parser.add_argument("--ayah", type=int, default=0)
    args = parser.parse_args(argv)

    rows = _load_rows(args.candidates, args.clip_id)
    surah = args.surah or (rows[0]["surah"] if rows else 0)
    ayah = args.ayah or (rows[0]["ayah"] if rows else 0)
    if not surah or not ayah:
        # id shape tlog_<n>_<surah>_<ayah>
        parts = args.clip_id.split("_")
        if len(parts) >= 4 and not surah:
            surah, ayah = int(parts[-2]), int(parts[-1])
    if not surah or not ayah:
        raise SystemExit("pass --surah and --ayah, or a candidates jsonl that contains this id")

    import modal

    dest = args.audio_dir / f"{args.clip_id}.flac"
    _fetch(modal.Volume.from_name("zipformer-ctc-training"), args.clip_id, dest)
    words = _display_words(surah, ayah)
    # Honour a phrase end when the evidence recorded one.
    spans = []
    for row in rows:
        end = row["word_index"] + 1
        for side in (row.get("evidence") or {}).values():
            if isinstance(side, dict) and side.get("word_end"):
                end = max(end, int(side["word_end"]))
        spans.append((row["word_index"], end, row))

    print(f"id {args.clip_id}  {surah}:{ayah}")
    print(f"audio {dest}")
    if not rows:
        print("no candidate row in the local jsonl; printing the reference only")
    for index, word in enumerate(words):
        hit = [row for start, end, row in spans if start <= index < end]
        mark = ">>" if hit else "  "
        extra = ""
        if hit:
            row = hit[0]
            span = row.get("span_s") or ["?", "?"]
            extra = f"   {row['kind']} {row['extent']}  {span[0]}-{span[1]}s  c={row['confidence']}"
        print(f"{mark} {index:3d}  {word}{extra}")


if __name__ == "__main__":
    main()
