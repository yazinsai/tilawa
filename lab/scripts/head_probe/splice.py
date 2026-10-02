"""Dev-only substitution and vowel splices. Audio stays under /tmp.

A vowel splice replaces a word with another word from the same speaker whose
consonant skeleton matches and whose phonemes differ. A substitution splice
uses a different skeleton. The slot length is preserved (crossfaded) so the
clip clock does not move. Test speakers and guard sets are not read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.audio import load_audio  # noqa: E402
from shared.phoneme_labels import PhonemeCorpus  # noqa: E402
from common import fit_splice, skeleton, units_from_track  # noqa: E402

SR = 16000
HOP = 0.04


def _index_audio(help_manifest: Path, help_audio: Path, ea_clips: Path) -> dict[str, str]:
    out = {}
    for row in json.loads(help_manifest.read_text(encoding="utf-8"))["samples"]:
        if row.get("use") == "clean" and row.get("split") == "dev":
            out[row["id"]] = str(help_audio / row["file"])
    for line in ea_clips.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row["audio"]
    return out


def _collect(rec_dir: Path, audio_of: dict[str, str], corpus: PhonemeCorpus) -> list[dict]:
    rows = []
    phon_cache: dict[tuple[int, int], list[str]] = {}
    for set_name in ("help-clean", "ea"):
        path = rec_dir / f"{set_name}.jsonl"
        if not path.is_file():
            continue
        seen: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                seen[rec["id"]] = rec
        for rec in seen.values():
            if rec.get("error") or rec.get("split") not in (None, "dev"):
                continue
            if rec["id"] not in audio_of:
                continue
            for unit in units_from_track(rec.get("trackPost")):
                if unit["skipped"] or unit["surah"] < 0:
                    continue
                key = (unit["surah"], unit["ayah"])
                if key not in phon_cache:
                    try:
                        phon_cache[key] = corpus.word_phonemes(*key)
                    except (ValueError, KeyError):
                        phon_cache[key] = []
                phs = phon_cache[key]
                if not 0 <= unit["word"] < len(phs):
                    continue
                dur_s = (unit["b"] - unit["a"]) * HOP
                if not 0.08 <= dur_s <= 1.2:
                    continue
                ph = phs[unit["word"]]
                sk = skeleton(ph)
                if len(sk) < 2:
                    continue
                rows.append({
                    "cid": rec["id"], "speaker": str(rec.get("speaker") or ""),
                    "surah": unit["surah"], "ayah": unit["ayah"], "word": unit["word"],
                    "a": unit["a"], "b": unit["b"], "ph": ph, "sk": sk,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--help-manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--help-audio", type=Path, default=Path("/tmp/vol/help"))
    ap.add_argument("--ea-clips", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=LAB / "data" / "zipformer" / "quran.json")
    ap.add_argument("--n-sub", type=int, default=1000)
    ap.add_argument("--n-vowel", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    corpus = PhonemeCorpus(args.corpus)
    audio_of = _index_audio(args.help_manifest, args.help_audio, args.ea_clips)
    rows = _collect(args.rec, audio_of, corpus)
    by_speaker: dict[str, list[int]] = defaultdict(list)
    by_vowel: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_speaker[row["speaker"]].append(i)
        by_vowel[(row["speaker"], row["sk"])].append(i)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    manifest = args.out_dir / "synth.jsonl"

    def choose(kind: str, n: int, used_host: set[tuple]) -> int:
        wrote = 0
        order = rng.permutation(len(rows))
        cache: dict[str, np.ndarray] = {}
        with manifest.open("a", encoding="utf-8") as handle:
            for i in order:
                if wrote >= n:
                    break
                host = rows[int(i)]
                host_key = (host["cid"], host["surah"], host["ayah"], host["word"])
                if host_key in used_host:
                    continue
                if kind == "vowel":
                    pool = [j for j in by_vowel[(host["speaker"], host["sk"])]
                            if rows[j]["ph"] != host["ph"] and rows[j]["cid"] != host["cid"]]
                else:
                    pool = [j for j in by_speaker[host["speaker"]]
                            if rows[j]["sk"] != host["sk"] and rows[j]["cid"] != host["cid"]]
                if not pool:
                    continue
                # Prefer a donor whose duration is close to the slot.
                donor = None
                for _ in range(8):
                    cand = rows[int(pool[int(rng.integers(0, len(pool)))])]
                    ratio = ((cand["b"] - cand["a"]) / max(1, host["b"] - host["a"]))
                    if 0.6 <= ratio <= 1.6:
                        donor = cand
                        break
                if donor is None:
                    continue
                if host["cid"] not in cache:
                    cache[host["cid"]] = load_audio(audio_of[host["cid"]])
                if donor["cid"] not in cache:
                    cache[donor["cid"]] = load_audio(audio_of[donor["cid"]])
                host_audio = cache[host["cid"]]
                a0 = int(round(host["a"] * HOP * SR))
                a1 = int(round(host["b"] * HOP * SR))
                a0 = max(0, min(a0, len(host_audio) - 1))
                a1 = max(a0 + 1, min(a1, len(host_audio)))
                if a1 - a0 < int(0.08 * SR) or a0 > len(host_audio) - int(0.05 * SR):
                    continue
                d0 = int(round(donor["a"] * HOP * SR))
                d1 = int(round(donor["b"] * HOP * SR))
                d0 = max(0, min(d0, len(cache[donor["cid"]]) - 1))
                d1 = max(d0 + 1, min(d1, len(cache[donor["cid"]])))
                spliced = fit_splice(host_audio, a0, a1, cache[donor["cid"]], d0, d1)
                if spliced is None:
                    continue
                # Drop the cached host so a later splice of another word re-reads clean audio.
                cid = f"s{args.seed:02d}{kind[0]}{wrote:04d}"
                wav = wav_dir / f"{cid}.wav"
                sf.write(str(wav), spliced, SR)
                handle.write(json.dumps({
                    "id": cid,
                    "audio": str(wav),
                    "split": "dev",
                    "speaker": host["speaker"],
                    "surah": host["surah"],
                    "ayah": host["ayah"],
                    "word": host["word"],
                    "label_kind": "vowel" if kind == "vowel" else "substitution",
                    "span_s": [round(a0 / SR, 3), round(a1 / SR, 3)],
                    "expected_verses": [{"surah": host["surah"], "ayah": host["ayah"]}],
                }, ensure_ascii=False) + "\n")
                used_host.add(host_key)
                wrote += 1
                if len(cache) > 32:
                    cache.clear()
        return wrote

    manifest.write_text("", encoding="utf-8")
    used: set[tuple] = set()
    n_v = choose("vowel", args.n_vowel, used)
    n_s = choose("sub", args.n_sub, used)
    print(
        f"pool_words={len(rows)} speakers={len(by_speaker)} "
        f"vowel_skeletons={sum(1 for k, v in by_vowel.items() if len(v) >= 2)} "
        f"wrote_vowel={n_v} wrote_sub={n_s}"
    )


if __name__ == "__main__":
    main()
