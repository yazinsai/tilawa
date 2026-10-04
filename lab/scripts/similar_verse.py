"""Similar-verse substitution / skip check for correction mode (lab prototype).

A rule over the Quran text plus the free decode; nothing is fit on labels.

1. ``index``: for every ayah, its look-alikes (other ayahs whose word sequence
   mostly matches), with the word regions where the two differ.
2. ``detect``: per take, align the free-decode phonemes to the passage, and to
   the passage with one region of one ayah swapped for its look-alike's
   wording (or with words skipped between a repeated phrase). When a variant
   fits the heard phonemes better than the canonical text by ``margin`` cost
   units, flag the first word of the region: ``possible_substitution`` for a
   swap, ``possible_omission`` for a dropped region.

The passage is the take's expected passage when the host knows it
(``setExpected``); otherwise the tracker's emitted verses, where a variant must
also beat every pure look-alike reading of that slot (a reciter who reads a
whole other ayah correctly is not flagged).

Private data only: inputs and outputs live under /tmp. Commit aggregates only.

    ../.venv/bin/python scripts/similar_verse.py index --out /tmp/sv/index.json
    ../.venv/bin/python scripts/similar_verse.py detect --run /tmp/tc/a0w-base --index /tmp/sv/index.json \\
        --out /tmp/sv/a0w --passage expected
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numba import njit
from rapidfuzz.distance import Levenshtein

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from shared.normalizer import normalize_arabic  # noqa: E402

DEFAULT_CORPUS = LAB / "data" / "zipformer" / "quran.json"

# Mirror of packages/core/src/recitation/phonemeCost.ts.
ALPHABET = "ءابتثجحخدذرزسشصضطظعغفقكلمنهويۥۦں۾ٲأإآؤئٱىَُِڇؙۣ۪ٞۜـ"
TABLE = 52
UNKNOWN = 51
_CANON = {"ۦ": "ي", "ۥ": "و", "ں": "ن", "۾": "م", "ٱ": "ا", "ى": "ي"}
_HAMZA = set("ءأإآاؤئٲ")
_SHORT = set("َُِ")
_OTHER_MARKS = set("ڇؙۣ۪ٞۜـ")
_MARKS = _SHORT | _OTHER_MARKS
_GROUPS = ["ذدضتط", "ظزذصسث", "جزش", "ةهت", "قكغ", "فبم"]
_PAIRS = [("ه", "ح"), ("غ", "خ"), ("ء", "ع"), ("ن", "م"), ("ن", "ل"), ("ظ", "ض")]


def _neighbors() -> set[frozenset]:
    s = set()
    for g in _GROUPS:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                s.add(frozenset((g[i], g[j])))
    for a, b in _PAIRS:
        s.add(frozenset((a, b)))
    return s


_NEIGH = _neighbors()


def char_cost(h: str, e: str) -> float:
    if h == e:
        return 0.0
    ch, ce = _CANON.get(h, h), _CANON.get(e, e)
    if ch == ce:
        return 0.0
    hm, em = h in _MARKS, e in _MARKS
    if hm or em:
        if hm and em:
            return 0.1 if (h in _SHORT and e in _SHORT) else 0.25
        return 1.0
    if ch in _HAMZA and ce in _HAMZA:
        return 0.1
    if frozenset((ch, ce)) in _NEIGH:
        return 0.25
    return 1.0


def cost_matrix() -> np.ndarray:
    m = np.ones((TABLE, TABLE), dtype=np.float32)
    for i, a in enumerate(ALPHABET):
        for j, b in enumerate(ALPHABET):
            m[i, j] = char_cost(a, b)
    return m


COST = cost_matrix()
_ID = {c: i for i, c in enumerate(ALPHABET)}


def encode(text: str) -> np.ndarray:
    return np.array([_ID.get(c, UNKNOWN) for c in text], dtype=np.int64)


@njit(cache=True)
def align_cost(q, r, cost, skip):
    """Reference global, query ends skippable at ``skip`` per char."""
    n, m = q.shape[0], r.shape[0]
    prev = np.empty(m + 1, dtype=np.float64)
    cur = np.empty(m + 1, dtype=np.float64)
    for j in range(m + 1):
        prev[j] = j
    best = prev[m] + n * skip
    for i in range(1, n + 1):
        cur[0] = i * skip
        qi = q[i - 1]
        for j in range(1, m + 1):
            c = prev[j - 1] + cost[qi, r[j - 1]]
            u = prev[j] + 1.0
            if u < c:
                c = u
            lf = cur[j - 1] + 1.0
            if lf < c:
                c = lf
            cur[j] = c
        tail = cur[m] + (n - i) * skip
        if tail < best:
            best = tail
        prev, cur = cur, prev
    return best


@njit(cache=True)
def align_path(q, r, cost, skip):
    """Same alignment; returns, per reference char, the last query index aligned at or before it (-1 if none)."""
    n, m = q.shape[0], r.shape[0]
    D = np.empty((n + 1, m + 1), dtype=np.float64)
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    for j in range(m + 1):
        D[0, j] = j
        T[0, j] = 2
    for i in range(1, n + 1):
        D[i, 0] = i * skip
        T[i, 0] = 3
        qi = q[i - 1]
        for j in range(1, m + 1):
            c = D[i - 1, j - 1] + cost[qi, r[j - 1]]
            t = 0
            u = D[i - 1, j] + 1.0
            if u < c:
                c = u
                t = 1
            lf = D[i, j - 1] + 1.0
            if lf < c:
                c = lf
                t = 2
            D[i, j] = c
            T[i, j] = t
    bi = 0
    bc = D[0, m] + n * skip
    for i in range(1, n + 1):
        v = D[i, m] + (n - i) * skip
        if v < bc:
            bc = v
            bi = i
    last = np.full(m, -1, dtype=np.int64)
    rc = np.zeros(m, dtype=np.float64)
    i, j = bi, m
    while j > 0:
        t = T[i, j]
        if i == 0:
            t = 2
        if t == 0:
            last[j - 1] = i - 1
            rc[j - 1] += cost[q[i - 1], r[j - 1]]
            i -= 1
            j -= 1
        elif t == 1:
            rc[j - 1] += 1.0
            i -= 1
        elif t == 2:
            last[j - 1] = i - 1 if i > 0 else -1
            rc[j - 1] += 1.0
            j -= 1
        else:
            break
    return last, rc


# ---------------------------------------------------------------- corpus + index


class Corpus:
    def __init__(self, path: Path = DEFAULT_CORPUS):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ph: dict[tuple[int, int], list[str]] = {}
        self.keys: dict[tuple[int, int], list[str]] = {}
        for s in raw["surahs"]:
            for a in s["ayahs"]:
                k = (int(s["n"]), int(a["n"]))
                self.ph[k] = [w[1] for w in a["w"]]
                self.keys[k] = [word_key(w[2]) for w in a["w"]]

    def has(self, k: tuple[int, int]) -> bool:
        return k in self.ph


def word_key(plain: str) -> str:
    return normalize_arabic(plain, strip_hamza=True)


def build_index(corpus: Corpus, min_shared: int = 3, min_frac: float = 0.5, top_k: int = 8,
                max_df: int = 400) -> dict[str, list[dict]]:
    vocab: dict[str, int] = {}
    seqs = {k: [vocab.setdefault(w, len(vocab)) for w in ws] for k, ws in corpus.keys.items()}
    bigrams: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for k, s in seqs.items():
        for bg in set(zip(s, s[1:])):
            bigrams[bg].append(k)
    out: dict[str, list[dict]] = {}
    for k, s in seqs.items():
        if len(s) < min_shared:
            continue
        counts: Counter = Counter()
        for bg in set(zip(s, s[1:])):
            post = bigrams[bg]
            if len(post) > max_df:
                continue
            for m in post:
                if m != k:
                    counts[m] += 1
        cands = []
        for m, shared in counts.items():
            t = seqs[m]
            ops = Levenshtein.opcodes(s, t)
            matched = sum(o.src_end - o.src_start for o in ops if o.tag == "equal")
            if matched < min_shared or matched / len(s) < min_frac:
                continue
            regions = [[o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end] for o in ops if o.tag != "equal"]
            if not regions:
                continue
            cands.append({"m": list(m), "sim": round(matched / max(len(s), len(t)), 3),
                          "frac": round(matched / len(s), 3), "regions": regions})
        cands.sort(key=lambda c: -c["sim"])
        if cands:
            out[f"{k[0]}:{k[1]}"] = cands[:top_k]
    return out


def variants_for(corpus: Corpus, e: tuple[int, int], look: dict) -> list[dict]:
    """E with one region swapped for look-alike M's wording (merged runs and single ops)."""
    m = tuple(look["m"])
    ew, mw = corpus.ph[e], corpus.ph[m]
    regs = [r for r in look["regions"] if r[0] != "insert"]
    runs: list[list] = []
    for r in look["regions"]:
        if runs and runs[-1][2] == r[1] and runs[-1][4] == r[3]:
            runs[-1] = ["run", runs[-1][1], r[2], runs[-1][3], r[4]]
        else:
            runs.append(list(r))
    seen = set()
    out = []
    for tag, i1, i2, j1, j2 in regs + [r for r in runs if r[0] == "run"]:
        if i2 <= i1:
            continue
        words = ew[:i1] + mw[j1:j2] + ew[i2:]
        key = (i1, i2, tuple(mw[j1:j2]))
        if key in seen:
            continue
        seen.add(key)
        kind = "possible_omission" if j2 <= j1 else "possible_substitution"
        out.append({"words": words, "word": i1, "end": i1 + (j2 - j1), "kind": kind, "from": list(m),
                    "family": "lookalike", "span": i2 - i1})
        if tag == "replace" and (i2 - i1) == (j2 - j1) > 1:
            for d in range(i2 - i1):
                key1 = (i1 + d, i1 + d + 1, (mw[j1 + d],))
                if key1 in seen:
                    continue
                seen.add(key1)
                out.append({"words": ew[: i1 + d] + [mw[j1 + d]] + ew[i1 + d + 1:], "word": i1 + d,
                            "end": i1 + d + 1, "kind": "possible_substitution", "from": list(m),
                            "family": "lookalike", "span": 1})
    return out


def skip_variants(corpus: Corpus, e: tuple[int, int], max_skip: int = 6) -> list[dict]:
    """Eye-skip inside E: after phrase A, the reader jumps past the next A (``A X A Y`` read ``A Y``)."""
    keys, ph = corpus.keys[e], corpus.ph[e]
    out = []
    n = len(keys)
    for i in range(1, n):
        for j in range(i + 1, min(n, i + max_skip) + 1):
            if keys[i - 1] == keys[j - 1] and j < n:
                out.append({"words": ph[:i] + ph[j:], "word": i, "end": i, "kind": "possible_omission",
                            "from": list(e), "family": "eyeskip", "span": j - i})
    return out


def drop_variants(corpus: Corpus, e: tuple[int, int], max_drop: int = 2) -> list[dict]:
    """Skip ahead: E with one or two consecutive interior words missing."""
    ph = corpus.ph[e]
    out = []
    for i in range(1, len(ph) - 1):
        for d in range(1, max_drop + 1):
            if i + d < len(ph):
                out.append({"words": ph[:i] + ph[i + d:], "word": i, "end": i, "kind": "possible_omission",
                            "from": list(e), "family": "drop", "span": d})
    return out


# ---------------------------------------------------------------- detection


def passage_of(row: dict, mode: str) -> list[tuple[int, int]]:
    if mode == "expected" and row.get("expected_verses"):
        vs = row["expected_verses"]
        surahs = {int(v["surah"]) for v in vs}
        if len(surahs) == 1:
            s = surahs.pop()
            ayahs = [int(v["ayah"]) for v in vs]
            return [(s, a) for a in range(min(ayahs), max(ayahs) + 1)]
    out: list[tuple[int, int]] = []
    for v in row.get("verses") or []:
        k = (int(v[0]), int(v[1]))
        if k not in out:
            out.append(k)
    return out


def heard_of(row: dict) -> tuple[np.ndarray, np.ndarray]:
    chars, times = [], []
    for sym, _frame, t in row.get("tokens") or []:
        for c in sym:
            chars.append(c)
            times.append(float(t))
    return encode("".join(chars)), np.array(times, dtype=np.float64)


def detect_take(corpus: Corpus, index: dict, row: dict, mode: str, skip: float = 0.5,
                families: tuple[str, ...] = ("lookalike", "eyeskip", "drop")) -> list[dict]:
    """Every candidate (variant) for the take with its margin; thresholds are applied later."""
    passage = [k for k in passage_of(row, mode) if corpus.has(k)]
    q, times = heard_of(row)
    if not passage or q.shape[0] < 8:
        return []
    known = mode == "expected" and bool(row.get("expected_verses"))
    slot_words = [list(corpus.ph[k]) for k in passage]

    def cost_of(slot: int, words: list[str] | None) -> float:
        ws = [w for t, sw in enumerate(slot_words) for w in (words if t == slot and words is not None else sw)]
        return float(align_cost(q, encode("".join(ws)), COST, skip))

    base = cost_of(-1, None)
    out = []
    for t, k in enumerate(passage):
        alts = [k] if known else [k] + [tuple(c["m"]) for c in index.get(f"{k[0]}:{k[1]}", [])]
        pure: dict[tuple[int, int], float] = {k: base}
        for a in alts:
            if a not in pure:
                pure[a] = cost_of(t, corpus.ph[a])
        for e in alts:
            looks = index.get(f"{e[0]}:{e[1]}", [])
            if not known:
                for c in looks:
                    mk = tuple(c["m"])
                    if mk not in pure:
                        pure[mk] = cost_of(t, corpus.ph[mk])
            vs: list[dict] = []
            if "lookalike" in families:
                for c in looks:
                    vs.extend(variants_for(corpus, e, c))
            if "eyeskip" in families:
                vs.extend(skip_variants(corpus, e))
            if "drop" in families and (known or e == k):
                vs.extend(drop_variants(corpus, e))
            ref_e = pure[e]
            floor = ref_e if known else min(pure.values())
            for v in vs:
                cv = cost_of(t, v["words"])
                out.append({"slot": t, "surah": e[0], "ayah": e[1], "word": v["word"], "kind": v["kind"],
                            "family": v["family"], "span": v["span"], "from": v["from"],
                            "margin": round(ref_e - cv, 3), "margin_all": round(floor - cv, 3),
                            "known": known, "_words": v["words"], "_end": v["end"]})
    return _with_times(corpus, passage, slot_words, q, times, out, skip, row)


def _with_times(corpus, passage, slot_words, q, times, cands, skip, row) -> list[dict]:
    """When the region plus one following word had been heard (seconds), and the fit of the variant's
    region (plus one word each side) and of its ayah, as alignment cost per reference char."""
    dur = float(row.get("duration_s") or 0.0) + 2.0
    for c in cands:
        c["at"] = c["loc"] = c["ayah_d"] = None
        if c["margin"] > 0 and c.get("margin_all", 0) > 0:
            words = [w for t, sw in enumerate(slot_words) for w in (c["_words"] if t == c["slot"] else sw)]
            starts = np.cumsum([0] + [len(w) for w in words])
            w0 = sum(len(sw) for sw in slot_words[: c["slot"]])
            wi = w0 + c["_end"]
            last, rc = align_path(q, encode("".join(words)), COST, skip)
            if wi < len(words):
                qi = int(last[int(starts[wi + 1]) - 1])
                c["at"] = round(float(times[qi]), 2) if qi >= 0 else round(dur, 2)
            else:
                c["at"] = round(dur, 2)
            lo = int(starts[max(w0, w0 + c["word"] - 1)])
            hi = int(starts[min(w0 + len(c["_words"]), wi + 1)])
            c["loc"] = round(float(rc[lo:hi].sum()) / max(1, hi - lo), 3)
            a0, a1 = int(starts[w0]), int(starts[w0 + len(c["_words"])])
            c["ayah_d"] = round(float(rc[a0:a1].sum()) / max(1, a1 - a0), 3)
        for k in ("_words", "_end"):
            c.pop(k, None)
    return cands


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("index")
    ip.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ip.add_argument("--out", type=Path, required=True)
    ip.add_argument("--min-shared", type=int, default=3)
    ip.add_argument("--min-frac", type=float, default=0.5)
    ip.add_argument("--top-k", type=int, default=8)
    dp = sub.add_parser("detect")
    dp.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    dp.add_argument("--index", type=Path, required=True)
    dp.add_argument("--run", type=Path, required=True, help="tracker_correction_eval output dir (rows with tokens)")
    dp.add_argument("--sets", default="help-acted,help-clean,tlog-dev,v1")
    dp.add_argument("--passage", choices=("expected", "tracker"), default="expected")
    dp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    corpus = Corpus(args.corpus)
    if args.cmd == "index":
        idx = build_index(corpus, args.min_shared, args.min_frac, args.top_k)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        n = sum(len(v) for v in idx.values())
        print(f"{len(idx)} ayahs with look-alikes, {n} pairs")
        return
    index = json.loads(args.index.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name in [s for s in args.sets.split(",") if s]:
        src = args.run / f"{name}.jsonl"
        if not src.is_file():
            continue
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for line in src.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("error"):
                    continue
                cands = detect_take(corpus, index, row, args.passage)
                fh.write(json.dumps({"id": row["id"], "split": row.get("split"), "cands": cands},
                                    ensure_ascii=False) + "\n")
        print(name, flush=True)


if __name__ == "__main__":
    main()
