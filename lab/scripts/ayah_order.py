"""Pause-segmented ayah-order check for skipped ayahs (lab prototype).

A rule over the Quran text plus the free decode; nothing is fit on labels.

Reciters pause (waqf) at or near ayah ends. Per take:

1. Cut the free-decode token stream at pauses: a gap of at least ``gap``
   encoder frames (blank run) between two emitted tokens.
2. Align each segment to every contiguous word span of a local window of
   ayahs (``[current-1 .. current+4]`` from the tracker's first verse, or the
   expected passage when the host knows it). Segments may also be
   isti'adha / basmala, or unexplained audio at ``garbage`` cost per char.
3. Chain the segments with a DP over the window, twice:
   - in order: each segment continues where the last one ended; restarts
     (going back) cost ``restart``; a forward gap may skip words at ``wcost``
     per char but may not contain a whole ayah;
   - free: a forward gap may also skip whole ayahs, at ``jump`` per jump.
4. When the free chain beats the in-order chain by at least ``margin`` cost
   units and its jump skips ayah X (X is not read later, the segments on both
   sides of the jump fit at ``fit`` cost per char or better, and no
   unexplained audio sits between them), flag ``possible_skipped_ayah`` on X.
   The flag time is when the segment after the jump has been decoded.

Private data only: inputs and outputs live under /tmp. Commit aggregates only.

    ../.venv/bin/python scripts/ayah_order.py detect --run /tmp/ao/run/a0w-nopass --out /tmp/ao/cand/a0w-nopass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similar_verse import COST, Corpus, encode  # noqa: E402

FRAME_S = 0.04
# Mirror of packages/core/src/recitation/search.ts.
ISTIADHA = "ءَعُۥۥذُبِللَااهِمِنَششَييطَاانِررَجِۦۦم"
BASMALA = "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم"
PREAMBLES = (ISTIADHA, BASMALA, ISTIADHA + BASMALA)

DEFAULTS = {
    "gap": 12,         # frames (0.48 s) of blank between tokens that cut a segment
    "min_chars": 3,    # shorter segments are merged into the previous one
    "garbage": 0.5,    # cost per char of unexplained audio
    "wcost": 0.5,      # cost per skipped char inside the in-order chain
    "restart": 2.0,    # cost of going back (restart / repeat)
    "jump": 0.5,       # cost of a forward jump over a whole ayah
    "back": 1,         # window starts this many ayahs before the anchor (no passage)
    "ahead": 4,        # and ends this many after it
}


# Frozen on dev (acted dev skip_ayah + help clean dev + v1; TLOG not used).
# ``no_ident``: never flag an ayah whose text equals a neighbour's (see ``ident_guarded``).
RULE = {"margin": 99.0, "rel": 0.04, "fit": 0.3, "between": 8, "min_post": 0, "no_ident": True}
# Guards frozen on dev for the shipped emissions (cost 0 dev catches on either model):
# no jump after a restart segment that also fits the skipped ayah's ending (rhyme), and
# undecoded audio before the jump at most 0.2 x the skipped ayah's length (dev max 0.17).
RULE_GUARDED = {**RULE, "no_restart_pre": True, "restart_alt": 0.3, "between_rel": 0.2}


@njit(cache=True)
def span_costs(q, ref, ws, cost, max_ratio, slack):
    """C[i, e]: edit cost of the whole query against words i..e of ``ref``
    (``ws`` = char start of each word, plus the end). inf where pruned."""
    nw = ws.shape[0] - 1
    n = q.shape[0]
    out = np.full((nw, nw), np.inf)
    prev = np.empty(ref.shape[0] + 1, dtype=np.float64)
    cur = np.empty(ref.shape[0] + 1, dtype=np.float64)
    lim = int(n * max_ratio + slack)
    for i in range(nw):
        r0 = ws[i]
        m = min(ref.shape[0] - r0, lim)
        if m <= 0:
            continue
        for j in range(m + 1):
            prev[j] = j
        for a in range(1, n + 1):
            cur[0] = a
            qa = q[a - 1]
            for j in range(1, m + 1):
                c = prev[j - 1] + cost[qa, ref[r0 + j - 1]]
                u = prev[j] + 1.0
                if u < c:
                    c = u
                lf = cur[j - 1] + 1.0
                if lf < c:
                    c = lf
                cur[j] = c
            for j in range(m + 1):
                prev[j] = cur[j]
        for e in range(i, nw):
            L = ws[e + 1] - r0
            if L > m:
                break
            out[i, e] = prev[L]
    return out


@njit(cache=True)
def global_cost(q, r, cost):
    n, m = q.shape[0], r.shape[0]
    prev = np.empty(m + 1, dtype=np.float64)
    cur = np.empty(m + 1, dtype=np.float64)
    for j in range(m + 1):
        prev[j] = j
    for a in range(1, n + 1):
        cur[0] = a
        for j in range(1, m + 1):
            c = prev[j - 1] + cost[q[a - 1], r[j - 1]]
            u = prev[j] + 1.0
            if u < c:
                c = u
            lf = cur[j - 1] + 1.0
            if lf < c:
                c = lf
            cur[j] = c
        for j in range(m + 1):
            prev[j] = cur[j]
    return prev[m]


def segments(tokens: list, gap: int, min_chars: int) -> list[dict]:
    """Token stream -> pause-delimited segments (chars, first/last frame, emit time)."""
    toks = sorted(tokens, key=lambda t: t[1])
    segs: list[dict] = []
    for sym, frame, t in toks:
        if segs and frame - segs[-1]["f1"] < gap:
            s = segs[-1]
            s["text"] += sym
            s["f1"] = frame
            s["t"] = max(s["t"], float(t))
        else:
            segs.append({"text": sym, "f0": frame, "f1": frame, "t": float(t)})
    out: list[dict] = []
    for s in segs:
        if out and len(s["text"]) < min_chars:
            out[-1]["text"] += s["text"]
            out[-1]["f1"] = s["f1"]
            out[-1]["t"] = max(out[-1]["t"], s["t"])
        else:
            out.append(s)
    if len(out) > 1 and len(out[0]["text"]) < min_chars:
        out[1]["text"] = out[0]["text"] + out[1]["text"]
        out[1]["f0"] = out[0]["f0"]
        out.pop(0)
    return out


def _text_key(corpus: Corpus, k: tuple[int, int]) -> tuple[str, ...] | None:
    ws = corpus.keys.get(k)
    return tuple(ws) if ws else None


def ident_guarded(corpus: Corpus, s: int, a: int) -> bool:
    """Skipping this ayah leaves a take that reads as an in-order or repeated recitation: its text
    equals an adjacent ayah's, or the ayahs on either side of it are equal (109:4 between 109:3 and
    109:5). Exact diacritic-free word keys. Near-equal pairs such as 94:5/94:6 (``fa-``) stay
    flaggable: dev skips of 94:5 and 109:3 are caught correctly."""
    me = _text_key(corpus, (s, a))
    prev, nxt = _text_key(corpus, (s, a - 1)), _text_key(corpus, (s, a + 1))
    if me is not None and me in (prev, nxt):
        return True
    return prev is not None and prev == nxt


def window_of(row: dict, corpus: Corpus, mode: str, back: int, ahead: int) -> list[tuple[int, int]]:
    if mode == "expected":
        vs = row.get("expected_verses") or []
        surahs = {int(v["surah"]) for v in vs}
        if len(surahs) != 1:
            return []
        s = surahs.pop()
        ayahs = [int(v["ayah"]) for v in vs]
        return [(s, a) for a in range(min(ayahs), max(ayahs) + 1) if corpus.has((s, a))]
    vs = row.get("verses") or []
    if not vs:
        return []
    s, a = int(vs[0][0]), int(vs[0][1])
    return [(s, x) for x in range(max(1, a - back), a + ahead + 1) if corpus.has((s, x))]


def _chain(costs, pre, seg_len, words_ayah, wlen_cum, ayah_first, ayah_last, p, free: bool):
    """DP over segments. State: last word read (index nw = nothing yet).
    Returns (best cost, path) where path is per segment ('g'|'p'|(i, e))."""
    nw = len(words_ayah)
    INF = np.inf
    # transition cost T[prev_state, start word]; prev_state nw = start of take
    T = np.full((nw + 1, nw), INF)
    T[nw, :] = 0.0
    skipped = {}
    for pe in range(nw):
        nxt = pe + 1
        for i in range(nw):
            if i <= pe:
                T[pe, i] = p["restart"]
                continue
            full = [x for x in range(len(ayah_first)) if ayah_first[x] >= nxt and ayah_last[x] < i]
            gap_chars = wlen_cum[i] - wlen_cum[nxt]
            if not full:
                T[pe, i] = p["wcost"] * gap_chars
            elif free:
                full_chars = sum(wlen_cum[ayah_last[x] + 1] - wlen_cum[ayah_first[x]] for x in full)
                T[pe, i] = p["jump"] + p["wcost"] * (gap_chars - full_chars)
                skipped[(pe, i)] = full
    cur = np.full(nw + 1, INF)
    cur[nw] = 0.0
    back: list[tuple] = []
    for k, C in enumerate(costs):
        A = (cur[:, None] + T).min(axis=0)          # best cost to start at word i
        Aarg = (cur[:, None] + T).argmin(axis=0)
        M = A[:, None] + C                            # start i, end e
        best_i = M.argmin(axis=0)
        new = np.full(nw + 1, INF)
        new[:nw] = M[best_i, np.arange(nw)]
        bp: list = [None] * (nw + 1)
        for e in range(nw):
            if np.isfinite(new[e]):
                bp[e] = ("m", int(best_i[e]), int(Aarg[best_i[e]]))
        g = p["garbage"] * seg_len[k]
        for s in range(nw + 1):
            if cur[s] + g < new[s]:
                new[s] = cur[s] + g
                bp[s] = ("g", s)
        if cur[nw] + pre[k] < new[nw]:
            new[nw] = cur[nw] + pre[k]
            bp[nw] = ("p", nw)
        back.append(bp)
        cur = new
    end = int(np.argmin(cur))
    best = float(cur[end])
    path: list = []
    s = end
    for k in range(len(costs) - 1, -1, -1):
        b = back[k][s]
        if b is None:
            return best, [], skipped
        if b[0] == "m":
            path.append((b[1], s))
            s = b[2]
        else:
            path.append(b[0])
            s = b[1]
    path.reverse()
    return best, path, skipped


def detect_take(corpus: Corpus, row: dict, mode: str, p: dict) -> dict | None:
    """Candidates for one take; thresholds ``margin`` / ``fit`` are applied by the caller."""
    win = window_of(row, corpus, mode, p["back"], p["ahead"])
    toks = row.get("tokens") or []
    if len(win) < 3 or len(toks) < 4:
        return None
    segs = segments(toks, int(p["gap"]), int(p["min_chars"]))
    if len(segs) < 2:
        return None
    words, words_ayah = [], []
    ayah_first, ayah_last = [], []
    for x, k in enumerate(win):
        ayah_first.append(len(words))
        for w in corpus.ph[k]:
            words.append(w)
            words_ayah.append(x)
        ayah_last.append(len(words) - 1)
    ref = encode("".join(words))
    ws = np.cumsum([0] + [len(w) for w in words]).astype(np.int64)
    costs, pre, seg_len, qs = [], [], [], []
    for s in segs:
        q = encode(s["text"])
        qs.append(q)
        costs.append(span_costs(q, ref, ws, COST, 2.0, 12.0))
        pre.append(min(global_cost(q, encode(ph), COST) for ph in PREAMBLES))
        seg_len.append(len(q))
    wlen_cum = [int(v) for v in ws]
    args = (costs, np.array(pre), seg_len, words_ayah, wlen_cum, ayah_first, ayah_last, p)
    c_in, _path_in, _ = _chain(*args, free=False)
    c_free, path, skipped = _chain(*args, free=True)
    out = {"margin": round(c_in - c_free, 3), "jumps": [], "segs": len(segs)}
    if not path or not np.isfinite(c_free):
        return out
    matched = [(k, st) for k, st in enumerate(path) if isinstance(st, tuple)]
    for n, ((k0, (i0, e0)), (k1, (i1, e1))) in enumerate(zip(matched, matched[1:])):
        full = skipped.get((e0, i1))
        if not full:
            continue
        # the pre-jump segment went back onto words already read and ends an ayah: a restart
        pre_restart = any(e_prev >= i0 for _k, (_i, e_prev) in matched[:n]) and e0 == ayah_last[words_ayah[e0]]
        between_garbage = sum(seg_len[k] for k in range(k0 + 1, k1))
        later = {words_ayah[w] for _k, (a, b) in matched if _k >= k1 for w in range(a, b + 1)}
        fit0 = costs[k0][i0, e0] / max(1, seg_len[k0])
        fit1 = costs[k1][i1, e1] / max(1, seg_len[k1])
        for x in full:
            out["jumps"].append({
                "surah": win[x][0], "ayah": win[x][1], "fit0": round(float(fit0), 3),
                "fit1": round(float(fit1), 3), "between": int(between_garbage),
                "read_later": bool(x in later), "at": round(float(segs[k1]["t"]), 2),
                "post_chars": int(seg_len[k1]), "pre_chars": int(seg_len[k0]),
                "pre_restart": bool(pre_restart),
                # best fit of the pre-jump segment as the ending of the skipped ayah (per char)
                "pre_on_skipped": round(float(min(costs[k0][i, ayah_last[x]]
                                                  for i in range(ayah_first[x], ayah_last[x] + 1))
                                              / max(1, seg_len[k0])), 3),
                "skip_chars": int(wlen_cum[ayah_last[x] + 1] - wlen_cum[ayah_first[x]]),
                "ident": ident_guarded(corpus, win[x][0], win[x][1]),
            })
    return out


_CORPUS: list[Corpus] = []


def _ident(s: int, a: int) -> bool:
    """Guard for candidate files written before ``ident`` was recorded."""
    if not _CORPUS:
        _CORPUS.append(Corpus())
    return ident_guarded(_CORPUS[0], int(s), int(a))


def flags_of(cand: dict | None, rule: dict) -> list[dict]:
    """Thresholds: absolute ``margin``, or ``rel`` margin per char of the segment after the jump."""
    if not cand or cand["margin"] <= 0:
        return []
    out = []
    for j in cand["jumps"]:
        if cand["margin"] < rule["margin"] and cand["margin"] / max(1, j["post_chars"]) < rule.get("rel", 99):
            continue
        if j["read_later"] or j["between"] > rule.get("between", 0):
            continue
        if max(j["fit0"], j["fit1"]) > rule["fit"]:
            continue
        if j["post_chars"] < rule.get("min_post", 0):
            continue
        if rule.get("no_restart_pre") and j.get("pre_restart") \
                and j.get("pre_on_skipped", 0) <= rule.get("restart_alt", 99):
            continue
        if "between_rel" in rule and j["between"] > rule["between_rel"] * j.get("skip_chars", 0):
            continue
        if rule.get("no_ident") and (j["ident"] if "ident" in j else _ident(j["surah"], j["ayah"])):
            continue
        out.append({"kind": "possible_skipped_ayah", "surah": j["surah"], "ayah": j["ayah"], "word": 0,
                    "atSeconds": j["at"], "source": "ayah_order"})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("detect",))
    ap.add_argument("--run", type=Path, required=True, help="tracker_correction_eval dir (rows with tokens)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="help-acted,help-clean,tlog-dev,v1")
    ap.add_argument("--passage", choices=("expected", "tracker"), default="tracker")
    ap.add_argument("--params", default="{}", help="JSON overrides of DEFAULTS (segmentation / chain costs)")
    args = ap.parse_args(argv)
    p = {**DEFAULTS, **json.loads(args.params)}
    corpus = Corpus()
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
                cand = detect_take(corpus, row, args.passage, p)
                fh.write(json.dumps({"id": row["id"], "split": row.get("split"), "cand": cand}) + "\n")
        print(name, flush=True)


if __name__ == "__main__":
    main()
