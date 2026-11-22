# /// script
# requires-python = ">=3.12"
# dependencies = ["quran-transcript==0.6.4", "rapidfuzz==3.14.1", "pyarrow", "numpy"]
# ///
"""E1b spliced-error probe: synthetic letter-pair and missing-shadda positives cut from clean recitations.

Source audio is everyayah per-ayah murattal (juz 29–30). Letter boundaries come from teacher CTC forced
alignment of each ayah's canonical phonemes on a clean teacher pass. Every positive has a control: the
same host and seam with no error (letters: the same consonant and vowel from another word; shadda: a
smaller cut that leaves the doubled consonant doubled). Needs ffmpeg.

    uv run --python cpython-3.12-macos-aarch64-none tools/e1b_build.py clean
    modal run modal/e1b_pass.py --stage clean
    uv run --python cpython-3.12-macos-aarch64-none tools/e1b_build.py splice
    modal run modal/e1b_pass.py --stage spliced
    uv run --python cpython-3.12-macos-aarch64-none tools/e1_report.py --e1b
"""

import json
import random
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import PROFILE  # noqa: E402
from e1_report import NOT_CONSONANT, PAIRS, VOWELS, log_probs, runs, viterbi  # noqa: E402
from splice import SR, boundaries, replace, shorten  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".scratch/e1b"
SRC = WORK / "src"
PASS = ROOT / ".scratch/bench/e1"
RECITERS = ["Husary_128kbps", "Minshawy_Murattal_128kbps"]
SURAHS = range(67, 115)
DATASET = "e1b-spliced"
# Positives per pair direction, and per shadda class.
PER_DIRECTION = 150
SHADDA = 300
MAX_DONORS_PER_HOST = 3
# A donor's segment must be within this duration ratio of the host's.
DUR_RATIO = (0.6, 1.6)
# Clean ayahs whose greedy decode is further than this from the canonical don't align well enough to cut.
MAX_CLEAN_PER = 0.15
# A missing shadda is only a real reduction if the single-consonant target is well under the geminate.
SHADDA_MAX_KEEP = 0.7
CONTROL_KEEP = 0.85
# Clips are trimmed at pauses (blank runs this long) at least CONTEXT_S from the splice.
PAUSE_FRAMES = 4
CONTEXT_S = 1.5
LISTEN = 20
SEED = 11


def ayahs() -> list[tuple[int, int]]:
    out = []
    for s in SURAHS:
        a = 1
        while True:
            try:
                Aya(s, a).get()
            except Exception:
                break
            out.append((s, a))
            a += 1
    return out


def letter_sites(canon: str) -> list[int]:
    """Indices of a pair consonant followed by a short vowel, not part of a doubled consonant."""
    letters = {c for p in PAIRS for c in p}
    return [
        i
        for i in range(1, len(canon) - 2)
        if canon[i] in letters and canon[i + 1] in VOWELS and canon[i - 1] != canon[i] and canon[i + 2] != canon[i]
    ]


def geminate_sites(canon: str) -> list[tuple[int, int]]:
    """Doubled consonants (exactly two) away from the clip edges."""
    return [(lo, hi) for lo, hi in runs(canon) if hi - lo == 2 and lo > 0 and hi < len(canon)]


def mp3(reciter: str, ayah: str) -> Path:
    s, a = ayah.split(":")
    return SRC / reciter / f"{int(s):03d}{int(a):03d}.mp3"


def load(path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"], capture_output=True, check=True
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def flac(x: np.ndarray) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", "-sample_fmt", "s16", "-f", "flac", "pipe:1"],
        input=np.clip(x, -1, 1).astype(np.float32).tobytes(),
        capture_output=True,
        check=True,
    ).stdout


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def clean() -> None:
    moshaf = MoshafAttributes(**PROFILE)
    rows = []
    for s, a in ayahs():
        canon = quran_phonetizer(Aya(s, a).get().uthmani, moshaf, remove_spaces=True, sura_idx=s).phonemes
        sites = sorted(set(letter_sites(canon)) | {lo for lo, _ in geminate_sites(canon)})
        for r in RECITERS:
            path = mp3(r, f"{s}:{a}")
            if not path.exists():
                continue
            rows.append({"dataset": DATASET, "id": f"{r}/{s}:{a}/clean", "ayah": f"{s}:{a}", "reciter": r, "kind": "clean", "cls": "",
                         "canonical": canon, "label": canon, "sites": sites, "ayah_site": -1, "donor": "", "audio": path.read_bytes()})
    WORK.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), WORK / "clean-in.parquet")
    print(f"{len(rows)} clean ayahs -> {WORK / 'clean-in.parquet'}")


class Ayah:
    def __init__(self, row: dict):
        self.row = row
        self.canon = row["canonical"]
        self.wave = load(mp3(row["reciter"], row["ayah"]))
        _, self.spans = viterbi(log_probs(row), self.canon)
        if self.spans is not None:
            self.segs = boundaries(self.spans, row["n_frames"], len(self.wave))
            self.per_frame = len(self.wave) / row["n_frames"]

    def window(self, i0: int, i1: int) -> tuple[int, int]:
        """Token range around tokens [i0, i1), cut at pauses at least CONTEXT_S away."""
        ctx = CONTEXT_S * SR
        gaps = [k for k in range(1, len(self.spans)) if self.spans[k][0] - self.spans[k - 1][1] >= PAUSE_FRAMES]
        j0 = max((k for k in [0, *gaps] if k <= i0 and self.segs[i0][0] - self.segs[k][0] >= ctx), default=0)
        j1 = min((k for k in [*gaps, len(self.canon)] if k >= i1 and self.segs[k - 1][1] - self.segs[i1 - 1][1] >= ctx), default=len(self.canon))
        return j0, j1

    def clip(self, out: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, int, int]:
        """The spliced ayah trimmed to the window around the splice, and the window's token range."""
        j0, j1 = self.window(i0, i1)
        start = self.segs[j0][0] if j0 > 0 else 0
        end = (self.segs[j1 - 1][1] if j1 < len(self.canon) else len(self.wave)) + len(out) - len(self.wave)
        return out[start:end], j0, j1


def splice() -> None:
    rng = random.Random(SEED)
    clean_rows = pq.read_table(PASS / "e1b-clean.parquet").to_pylist()
    src: list[Ayah] = []
    for r in clean_rows:
        if Levenshtein.distance(r["canonical"], r["greedy"]) / len(r["canonical"]) > MAX_CLEAN_PER:
            continue
        a = Ayah(r)
        if a.spans is not None:
            src.append(a)
    print(f"{len(src)}/{len(clean_rows)} clean ayahs aligned")

    # Letter units: consonant plus vowel, keyed by reciter, consonant and vowel.
    units: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    single: dict[tuple[str, str], list[int]] = defaultdict(list)
    for k, a in enumerate(src):
        for i in letter_sites(a.canon):
            units[(a.row["reciter"], a.canon[i], a.canon[i + 1])].append((k, i))
        for i in range(1, len(a.canon) - 1):
            c = a.canon[i]
            if c not in NOT_CONSONANT and a.canon[i - 1] != c and a.canon[i + 1] != c:
                single[(a.row["reciter"], c)].append(a.segs[i][1] - a.segs[i][0])

    def span(k: int, i: int, n: int) -> tuple[int, int]:
        return src[k].segs[i][0], src[k].segs[i + n - 1][1]

    def pick_donor(k: int, i: int, pool: list[tuple[int, int]], used: set) -> tuple[int, int] | None:
        a, b = span(k, i, 2)
        ok = [(dk, di) for dk, di in pool if dk != k and (dk, di) not in used]
        fits = [u for u in ok if DUR_RATIO[0] <= (lambda s: (s[1] - s[0]) / (b - a))(span(*u, 2)) <= DUR_RATIO[1]]
        return rng.choice(fits) if fits else None

    out_rows, listen = [], defaultdict(list)

    def emit(k: int, i0: int, i1: int, cls: str, kind: str, said: str, out: np.ndarray, donor: str, pair_id: int) -> None:
        a = src[k]
        x, j0, j1 = a.clip(out, i0, i1)
        canon_clip = a.canon[j0:j1]
        said_clip = said[j0 : j1 + len(said) - len(a.canon)]
        rid = f"{a.row['reciter']}/{a.row['ayah']}/{cls}/{kind}/{pair_id}"
        out_rows.append({"dataset": DATASET, "id": rid, "ayah": a.row["ayah"], "reciter": a.row["reciter"], "kind": kind, "cls": cls,
                         "canonical": canon_clip, "label": said_clip, "sites": [i0 - j0], "ayah_site": i0, "donor": donor, "audio": flac(x)})
        if len(listen[(cls, kind)]) < LISTEN:
            listen[(cls, kind)].append((rid, x, canon_clip, said_clip))

    pid = 0
    for p, q in PAIRS:
        cls = f"{p}{q}"
        for c, partner in ((p, q), (q, p)):
            hosts = [(r, v, k, i) for (r, cc, v), us in units.items() if cc == c for k, i in us]
            rng.shuffle(hosts)
            made, used = 0, set()
            for _ in range(MAX_DONORS_PER_HOST):
                for r, v, k, i in hosts:
                    if made >= PER_DIRECTION:
                        break
                    donor = pick_donor(k, i, units.get((r, partner, v), []), used)
                    same = pick_donor(k, i, [u for u in units.get((r, c, v), []) if u != (k, i)], set())
                    if donor is None or same is None:
                        continue
                    used.add(donor)
                    a, b = span(k, i, 2)
                    w = src[k].wave
                    said = src[k].canon[:i] + partner + src[k].canon[i + 1 :]
                    dk, di = donor
                    emit(k, i, i + 2, cls, "spliced", said, replace(w, a, b, src[dk].wave, *span(dk, di, 2)), f"{src[dk].row['ayah']}@{di}", pid)
                    sk, si = same
                    emit(k, i, i + 2, cls, "control", src[k].canon, replace(w, a, b, src[sk].wave, *span(sk, si, 2)), f"{src[sk].row['ayah']}@{si}", pid)
                    made += 1
                    pid += 1
            print(f"{c}->{partner}: {made} positives from {len({(src[k].row['ayah'], i) for _, _, k, i in hosts})} candidate sites")

    gems = [(k, lo, hi) for k, a in enumerate(src) for lo, hi in geminate_sites(a.canon)]
    rng.shuffle(gems)
    all_single = [d for ds in single.values() for d in ds]
    made = 0
    for k, lo, hi in gems:
        if made >= SHADDA:
            break
        a = src[k]
        c = a.canon[lo]
        ds = single.get((a.row["reciter"], c), [])
        keep = int(np.median(ds if len(ds) >= 5 else all_single))
        s0, s1 = span(k, lo, 2)
        if keep >= SHADDA_MAX_KEEP * (s1 - s0):
            continue
        said = a.canon[:lo] + c + a.canon[hi:]
        emit(k, lo, hi, "shadda", "spliced", said, shorten(a.wave, s0, s1, keep), "", pid)
        emit(k, lo, hi, "shadda", "control", a.canon, shorten(a.wave, s0, s1, int(CONTROL_KEEP * (s1 - s0))), "", pid)
        made += 1
        pid += 1
    print(f"shadda: {made} positives")

    pq.write_table(pa.Table.from_pylist(out_rows), WORK / "spliced-in.parquet")
    print(f"{len(out_rows)} clips -> {WORK / 'spliced-in.parquet'}")
    write_listening_pack(listen)


def write_listening_pack(listen: dict) -> None:
    root = WORK / "listen"
    lines = ["# E1b listening pack", "", "Mark each splice pass if it sounds like the target error (controls: like a clean recitation).", "",
             "| file | canonical | said | pass |", "|---|---|---|---|"]
    for (cls, kind), items in sorted(listen.items()):
        for n, (rid, x, canon, said) in enumerate(items):
            name = f"{cls}/{kind}-{n:02d}.wav"
            write_wav(root / name, x)
            lines.append(f"| `{name}` ({rid}) | {canon} | {said} | |")
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(f"listening pack -> {root}")


if __name__ == "__main__":
    {"clean": clean, "splice": splice}[sys.argv[1]]()
