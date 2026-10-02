"""Pick the clean calibration pool. Prints counts only.

TLOG clean train is clip-disjoint from TLOG clean dev (``clean_ids`` already
has dev removed). Help test speakers are not eligible. The word target is
reference words; tracking yield is lower, so the target sits above 50k.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB))

from shared.phoneme_labels import PhonemeCorpus  # noqa: E402

# 65k TLOG reference words + EveryAyah dev (~5k) + help clean dev (~1k).
# At a 75% tracking yield that is still >= 50k tracked words.
TLOG_WORD_TARGET = 65000


def _words(corpus: PhonemeCorpus, surah: int, ayah: int) -> int:
    try:
        return len(corpus.word_phonemes(int(surah), int(ayah)))
    except (ValueError, KeyError):
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=LAB / "data" / "zipformer" / "quran.json")
    ap.add_argument("--help-manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--help-audio", type=Path, default=Path("/tmp/vol/help"))
    ap.add_argument("--ea-manifest", type=Path, default=Path("/tmp/meta/ea/manifest.json"))
    ap.add_argument("--ea-audio", type=Path, default=Path("/tmp/vol/ea"))
    ap.add_argument("--tlog-cuts", type=Path, default=Path("/tmp/meta/tlog/tlog_cuts.jsonl.gz"))
    ap.add_argument("--clean-ids", type=Path, default=Path("/tmp/meta/tlog/clean_ids.txt"))
    ap.add_argument("--dev-ids", type=Path, default=Path("/tmp/meta/tlog/dev_ids.txt"))
    ap.add_argument("--tlog-audio", type=Path, default=Path("/tmp/vol/tlog"))
    ap.add_argument("--tlog-words", type=int, default=TLOG_WORD_TARGET)
    args = ap.parse_args()
    corpus = PhonemeCorpus(args.corpus)
    help_rows = json.loads(args.help_manifest.read_text(encoding="utf-8"))["samples"]
    test_speakers = {r["speaker"] for r in help_rows if r.get("split") == "test"}
    dev_speakers = {r["speaker"] for r in help_rows if r.get("split") == "dev"}
    if test_speakers & dev_speakers:
        raise SystemExit("help speaker crosses the split")

    ea_rows = json.loads(args.ea_manifest.read_text(encoding="utf-8"))["samples"]
    ea_speakers = {r["reciter"] for r in ea_rows}
    if ea_speakers & test_speakers:
        raise SystemExit("an EveryAyah dev reciter is a help test speaker")

    clean = {ln.strip() for ln in args.clean_ids.read_text(encoding="utf-8").splitlines() if ln.strip()}
    dev = {ln.strip() for ln in args.dev_ids.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if clean & dev:
        raise SystemExit("TLOG clean train ids overlap clean dev")

    train = []
    with gzip.open(args.tlog_cuts, "rt", encoding="utf-8") as handle:
        for line in handle:
            cut = json.loads(line)
            if cut["id"] not in clean:
                continue
            custom = cut["supervisions"][0]["custom"]
            train.append((
                cut["id"], int(custom["surah"]), int(custom["ayah"]), float(cut["duration"]),
                _words(corpus, custom["surah"], custom["ayah"]),
            ))
    train.sort(key=lambda row: row[0])
    picked = []
    words = 0
    for row in train:
        if words >= args.tlog_words:
            break
        picked.append(row)
        words += row[4]

    args.out.mkdir(parents=True, exist_ok=True)
    ea_path = args.out / "ea.jsonl"
    tlog_path = args.out / "tlog-train.jsonl"
    with ea_path.open("w", encoding="utf-8") as handle:
        for row in ea_rows:
            handle.write(json.dumps({
                "id": row["id"],
                "audio": str(args.ea_audio / row["file"]),
                "split": "dev",
                "speaker": row["reciter"],
                "expected_verses": row.get("expected_verses") or [],
            }, ensure_ascii=False) + "\n")
    with tlog_path.open("w", encoding="utf-8") as handle:
        for cid, surah, ayah, dur, _nw in picked:
            handle.write(json.dumps({
                "id": cid,
                "audio": str(args.tlog_audio / f"{cid}.flac"),
                "split": "dev",
                # One group per clip: the manifest speaker is the literal "tlog".
                "speaker": f"clip:{cid}",
                "duration_s": round(dur, 3),
                "expected_verses": [{"surah": surah, "ayah": ayah}],
            }, ensure_ascii=False) + "\n")
    (args.out / "tlog-train.ids").write_text("".join(cid + "\n" for cid, *_ in picked), encoding="utf-8")
    ea_words = 0
    for row in ea_rows:
        for verse in row.get("expected_verses") or []:
            ea_words += _words(corpus, verse["surah"], verse["ayah"])
    help_words = 0
    help_min = 0.0
    for row in help_rows:
        if row.get("use") != "clean" or row.get("split") != "dev":
            continue
        help_min += float(row.get("duration_s") or 0)
        for verse in row.get("expected_verses") or []:
            help_words += _words(corpus, verse["surah"], verse["ayah"])
    print(
        f"tlog-train clips={len(picked)} ref_words={words} hours={sum(r[3] for r in picked)/3600:.2f} "
        f"ea clips={len(ea_rows)} ref_words={ea_words} reciters={len(ea_speakers)} "
        f"help-clean-dev ref_words={help_words} min={help_min/60:.2f} "
        f"pool_ref_words={words + ea_words + help_words}"
    )


if __name__ == "__main__":
    main()
