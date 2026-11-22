"""What the listeners heard in a review window, phoneme by phoneme: the review page's and the case view's "What the
listeners heard" panel (docs/benchmarks/lab-review.md, GET /api/phonemes).

The words in the window come from the session's `said` (what was actually said, planted words included, each with its
phonemes and times) plus its omitted-word and skipped-passage events (absent from `said`: shown where they should be
missing, with the reference phonemes they would have had). Each listener's greedy phonemes (`<id>.<listener>.<variant>
.evidence.json`, frame `start` * frameMs) in the window are aligned against the expected phonemes in one edit alignment,
where a heard phoneme may only pair with an expected one of a word whose span is within TOL_S of it. Unpaired heard
phonemes go by time to the word or omitted slot they fall in, else to the gap between words.

Phonemes are the reference notation (ref-v2.json, quran-transcript, the muaalem inventory): one character each.
Pure logic, standard library only; tested by tools/test_lab_phonemes.py.
"""

import bisect
import re

LISTENERS = (("student", "spliced-clean-int8", "The app's ear"), ("teacher", "teacher", "The reference ear"))
TOL_S = 0.12  # seconds a heard phoneme may sit outside the word it pairs with (3 frames)
LOW_CONF = 0.6  # greedy confidence below which a phoneme is shown dimmed
LIKE = 0.5  # similarity at which heard phonemes count as the word
MAX_WINDOW_S = 120
PLANTED = {"wrong": "wrong_word", "added": "added_word", "omitted": "omitted_word"}
FOCUS_ROLE = {"wrong_word": "wrong", "added_word": "added", "omitted_word": "omitted", "skipped_passage": "omitted",
              "repeat": "repeat", "restart": "repeat"}
AUDIO = re.compile(r"^(?P<dir>(?:[A-Za-z0-9._-]+/)*sessions)/(?P<session>[A-Za-z0-9._-]+)\.(?P<variant>phone|studio)\.wav$")


# ---- files ---------------------------------------------------------------------------------------------------------

def session_of(audio_path: str) -> tuple[str, str, str] | None:
    """(sessions dir, session id, variant) of a session's audio path (`<set>/sessions/<id>.<variant>.wav`), else None."""
    m = AUDIO.match(audio_path or "")
    return (m["dir"], m["session"], m["variant"]) if m and ".." not in audio_path else None


def session_rel(d: str, session: str) -> str:
    return f"{d}/{session}.json"


def evidence_rel(d: str, session: str, listener: str, variant: str) -> str:
    return f"{d}/{session}.{listener}.{variant}.evidence.json"


def key(at: dict | None) -> str | None:
    return f"{at['surah']}:{at['ayah']}:{at['word']}" if at else None


class Ref:
    """Reference words from ref-v2.json: key `s:a:w` -> (text, phonemes in context), and reading order."""

    def __init__(self, doc: dict):
        self.words, self.order = {}, []
        for a in doc.get("ayahs", []):
            o, p = a["o"], a["p"]
            for i, w in enumerate(a["w"]):
                k = f"{a['s']}:{a['a']}:{i + 1}"
                self.words[k] = (w, p[o[i]:o[i + 1] if i + 1 < len(o) else len(p)])
                self.order.append(k)
        self.at = {k: i for i, k in enumerate(self.order)}

    def run(self, start: str, n: int) -> list[tuple[str, str]]:
        """`n` words in reading order from `start` (fewer at the end of the Quran, none for an unknown key)."""
        i = self.at.get(start)
        return [] if i is None else [self.words[k] for k in self.order[i:i + n]]


# ---- the window ----------------------------------------------------------------------------------------------------

def window_columns(sdoc: dict, ref: Ref | None, a: float, b: float) -> list[dict]:
    """The words of the window in reading order: every `said` word overlapping [a, b), and every omitted word or skipped
    passage whose slot touches it. Each: text, role (ref, wrong, added, repeat, omitted), planted, at, startS, endS,
    expected phonemes (what should be heard given the label: nothing for an omission), and `reference` (an omission's
    phonemes as it would have been said; for a wrong word, the original word's text and phonemes)."""
    cols = []
    for w in sdoc.get("said", []):
        if not (w["endS"] > a and w["startS"] < b):
            continue
        c = {"kind": "word", "text": w.get("text", ""), "role": w.get("role", "ref"), "planted": w.get("role") in PLANTED,
             "at": key(w.get("at")), "startS": w["startS"], "endS": w["endS"], "expected": w.get("phonemes") or ""}
        if c["role"] == "wrong" and ref and c["at"] in ref.words:
            t, p = ref.words[c["at"]]
            c["reference"] = {"text": t, "phonemes": p}
        cols.append(c)
    for e in sdoc.get("events", []):
        if e.get("errorClass") not in ("omitted_word", "skipped_passage") or not (e["endS"] >= a and e["startS"] <= b):
            continue
        text = e.get("expected") or ""
        run = ref.run(key(e.get("location")), len(text.split())) if ref and text else []
        cols.append({"kind": "word", "text": text, "role": "omitted", "planted": True, "errorClass": e["errorClass"],
                     "at": key(e.get("location")), "startS": e["startS"], "endS": e["endS"], "expected": "",
                     "reference": {"text": text, "phonemes": "".join(p for _, p in run)}})
    return sorted(cols, key=lambda c: (c["startS"], c["endS"]))


def heard_in(ev: dict, lo: float, hi: float, starts: list[int] | None = None) -> list[dict]:
    """The listener's greedy phonemes with a start time in [lo, hi): {p, t (s), conf}. `starts`: the evidence's sorted
    start frames, when cached."""
    fm, ps = ev["frameMs"], ev.get("phonemes", [])
    starts = starts if starts is not None else [p["start"] for p in ps]
    i = bisect.bisect_left(starts, lo * 1000 / fm - 1e-6)
    out = []
    for p in ps[i:]:
        t = p["start"] * fm / 1000
        if t >= hi:
            break
        if t >= lo:
            out.append({"p": p["p"], "t": round(t, 3), "conf": p.get("conf")})
    return out


def span_of(cols: list[dict], a: float, b: float) -> tuple[float, float]:
    """The stretch whose heard phonemes are shown: the window, widened to cover its first and last words whole."""
    return (min([a] + [c["startS"] for c in cols[:1]]), max([b] + [c["endS"] for c in cols[-1:]]))


# ---- alignment -----------------------------------------------------------------------------------------------------

INF = float("inf")
OUTSIDE = 1 / 64  # extra cost of a pair heard just outside its word's span (exact in binary, so ties compare equal)


def align(expected: list[tuple[str, int]], heard: list[dict], spans: list[tuple[float, float]], tol: float = TOL_S) -> list[dict]:
    """Edit alignment of expected phonemes (char, column) against heard ones in time order: a match costs 0, a
    substitution 1 (only within `tol` of the expected phoneme's column span), a missing or extra phoneme 1. Ops in
    order: same / sub {e, h, col, conf, t}, del {e, col}, ins {h, conf, t}."""
    n, m = len(expected), len(heard)
    near = lambda i, j: spans[expected[i][1]][0] - tol <= heard[j]["t"] <= spans[expected[i][1]][1] + tol

    def pair(i, j):
        """Cost of pairing, INF beyond `tol`; just outside the span costs a little more, so of equal alignments the
        phoneme stays with the word it was heard in."""
        if not near(i, j):
            return INF
        a, b = spans[expected[i][1]]
        return (expected[i][0] != heard[j]["p"]) + (0 if a <= heard[j]["t"] < b else OUTSIDE)

    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + pair(i - 1, j - 1), D[i - 1][j] + 1, D[i][j - 1] + 1)
    out, i, j = [], n, m
    while i or j:
        if i and j and near(i - 1, j - 1):
            same = expected[i - 1][0] == heard[j - 1]["p"]
            if D[i][j] == D[i - 1][j - 1] + pair(i - 1, j - 1):
                h = heard[j - 1]
                out.append({"op": "same" if same else "sub", "e": expected[i - 1][0], "col": expected[i - 1][1], "h": h["p"], "conf": h["conf"], "t": h["t"]})
                i, j = i - 1, j - 1
                continue
        if j and D[i][j] == D[i][j - 1] + 1:
            h = heard[j - 1]
            out.append({"op": "ins", "h": h["p"], "conf": h["conf"], "t": h["t"]})
            j -= 1
        else:
            out.append({"op": "del", "e": expected[i - 1][0], "col": expected[i - 1][1]})
            i -= 1
    return out[::-1]


def owner(t: float, cols: list[dict]) -> tuple[str, int]:
    """Where an unpaired heard phoneme at `t` goes: ('col', k) inside column k's span, else ('gap', k) after column k
    (-1 before the first)."""
    k = -1
    for i, c in enumerate(cols):
        if c["startS"] <= t < c["endS"] or (c["startS"] == c["endS"] == t):
            return "col", i
        if c["startS"] <= t:
            k = i
    return "gap", k


def similarity(a: str, b: str) -> float:
    """1 - edit distance / the longer length (1 for two empty strings)."""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = row
    return 1 - prev[-1] / max(len(a), len(b))


def status(col: dict, ops: list[dict]) -> str:
    heard = any("h" in o for o in ops)
    if col.get("role") == "omitted":
        return "heard" if heard else "absent"
    if not heard:
        return "nothing"
    return "match" if all(o["op"] == "same" for o in ops) else "differs"


def listen(cols: list[dict], heard: list[dict]) -> tuple[list[list[dict]], dict[int, list[dict]]]:
    """One listener: per column its ops, and per gap (keyed by the column before it, -1 for the start) the phonemes
    heard there."""
    expected = [(ch, k) for k, c in enumerate(cols) for ch in c["expected"]]
    spans = [(c["startS"], c["endS"]) for c in cols]
    per, gaps = [[] for _ in cols], {}
    for o in align(expected, heard, spans):
        if "col" in o:
            per[o.pop("col")].append(o)
            continue
        where, k = owner(o["t"], cols)
        (per[k] if where == "col" else gaps.setdefault(k, [])).append(o)
    return per, gaps


def cell(col: dict, ops: list[dict]) -> dict:
    heard = "".join(o["h"] for o in ops if "h" in o)
    out = {"heard": heard, "ops": ops, "status": status(col, ops)}
    if col.get("reference"):
        out["likeReference"] = round(similarity(heard, col["reference"]["phonemes"]), 2)
    if col["expected"]:
        out["likeExpected"] = round(similarity(heard, col["expected"]), 2)
    return out


# ---- plain words ---------------------------------------------------------------------------------------------------

def focus_of(cols: list[dict], label_type: str | None, a: float, b: float) -> list[int]:
    """The columns the item's label is about: the planted (or repeated) word nearest the window's centre, a whole run of
    repeated words, or none for a clean label."""
    role = FOCUS_ROLE.get(label_type or "")
    if label_type == "clean":
        return []
    cand = [k for k, c in enumerate(cols) if c["role"] == role] if role else [k for k, c in enumerate(cols) if c["planted"]]
    if not cand:
        return []
    if role == "repeat":
        return cand
    mid = (a + b) / 2
    return [min(cand, key=lambda k: abs((cols[k]["startS"] + cols[k]["endS"]) / 2 - mid))]


def heard_as(cols: list[dict], cells: list[dict], focus: list[int]) -> str:
    """How one listener heard the focus words: absent / anyway / partial (omissions); planted / original / nothing /
    unclear (said words)."""
    heard = "".join(cells[k]["heard"] for k in focus)
    exp = "".join(cols[k]["expected"] for k in focus)
    ref = "".join((cols[k].get("reference") or {}).get("phonemes", "") for k in focus)
    if cols[focus[0]]["role"] == "omitted":
        return "absent" if not heard else "anyway" if ref and similarity(heard, ref) >= LIKE else "partial"
    if not heard:
        return "nothing"
    se, sr = similarity(heard, exp), similarity(heard, ref) if ref else 0.0
    if se >= LIKE and se >= sr:
        return "planted"
    return "original" if sr >= LIKE else "unclear"


CONSONANT = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")


def differences(cells: list[dict], focus: list[int]) -> str:
    """What a listener got wrong in the focus words, in plain words: consonants missing or heard as another, then any
    vowel or length change ("" when every sound matched)."""
    ops = [o for k in focus for o in cells[k]["ops"]]
    miss = [o["e"] for o in ops if o["op"] == "del" and o.get("e") in CONSONANT]
    swap = [(o["e"], o["h"]) for o in ops if o["op"] == "sub" and (o.get("e") in CONSONANT or o.get("h") in CONSONANT)]
    soft = any(o["op"] in ("del", "sub", "ins") for o in ops) and not (miss or swap) or any(
        o["op"] in ("del", "sub") and o.get("e") not in CONSONANT and o.get("h") not in CONSONANT for o in ops)
    bits = [f"no «{e}»" for e in miss] + [f"«{h}» for «{e}»" for e, h in swap] + (["a vowel or length different"] if soft else [])
    return ", ".join(bits)


def _say(how: str, word: str, diff: str = "") -> str:
    """word: the focus words in plain form, e.g. «بِمَا» or "the skipped ayah"; diff: `differences` of a word heard."""
    heard = f"heard most of {word} ({diff})" if diff else f"heard {word} (as described)"
    return {"absent": "heard nothing there (as expected)", "anyway": f"heard {word} after all (unexpected)", "partial": "picked up a few sounds there",
            "planted": heard, "original": "heard the mushaf's word instead (unexpected)", "nothing": "heard nothing there",
            "unclear": f"heard {word} only partly" + (f" ({diff})" if diff else "")}[how]


def summary(cols: list[dict], listeners: list[dict], focus: list[int], t0: float = 0.0, skipped: bool = False) -> dict:
    """{text, elsewhere}: one plain sentence on how each listener heard the label's word(s), and how many other words
    each heard differently. t0: the clip's start in the session, so times read as seconds into the clip."""
    have = [x for x in listeners if x["found"]]
    if not have:
        return {"text": "No listener evidence for this session", "elsewhere": ""}
    name = {x["key"]: x["name"] for x in listeners}
    lower = lambda s: s[0].lower() + s[1:]
    others = [k for k, c in enumerate(cols) if k not in focus and c["role"] != "omitted"]
    differ = {x["key"]: sum(x["cells"][k]["status"] != "match" for k in others) for x in have}
    if not focus:
        if not any(differ.values()):
            text = "Both ears heard every word as expected" if len(have) == 2 else f"{name[have[0]['key']]} heard every word as expected"
        else:
            text = ", ".join(f"{lower(name[x['key']])} {differ[x['key']]}" if i else f"{name[x['key']]} heard {differ[x['key']]} of {len(others)} words slightly differently"
                             for i, x in enumerate(have)) + " (usually a vowel or an ending)"
        missing = [x for x in listeners if not x["found"]]
        return {"text": text + "".join(f"; no evidence for {lower(x['name'])}" for x in missing), "elsewhere": ""}
    role, at = cols[focus[0]]["role"], cols[focus[0]]["startS"]
    hows = {x["key"]: heard_as(cols, x["cells"], focus) for x in have}
    words = " ".join(cols[k].get("text", "") for k in focus).split()
    word = "the skipped ayah" if skipped else f"«{' '.join(words[:3])}{'…' if len(words) > 3 else ''}»"
    says = {x["key"]: _say(hows[x["key"]], word, differences(x["cells"], focus) if hows[x["key"]] in ("planted", "unclear") else "") for x in have}
    if len(have) == 2 and len(set(says.values())) == 1:
        how = next(iter(hows.values()))
        text = ("neither ear heard anything there (as expected)" if how == "absent"
                else "neither ear heard anything there" if how == "nothing" else f"both ears {next(iter(says.values()))}")
    else:
        parts = [f"{lower(name[x['key']])} {says[x['key']]}" for x in have]
        parts += [f"no evidence for {lower(x['name'])}" for x in listeners if not x["found"]]
        text = "; ".join(parts)
    here = f"Where {word} should be missing" if role == "omitted" else f"At {word}"
    text = f"{here} ({max(0.0, at - t0):.1f} s into the clip), {text}"
    bits = [f"{lower(name[x['key']])} heard {differ[x['key']]} of {len(others)} words slightly differently" if i == 0 else f"{lower(name[x['key']])} {differ[x['key']]}"
            for i, x in enumerate(x for x in have if differ[x["key"]])]
    elsewhere = f"In the rest of the clip, {', '.join(bits)} (usually a vowel or an ending)" if bits else ""
    return {"text": text, "elsewhere": elsewhere}


# ---- the response --------------------------------------------------------------------------------------------------

def view(sdoc: dict, ref: Ref | None, evidence: dict[str, dict | None], a: float, b: float, label_type: str | None = None,
         files: dict[str, str] | None = None, starts: dict[str, list[int]] | None = None) -> dict:
    """The /api/phonemes response. evidence: listener key (student, teacher) -> parsed evidence or None."""
    cols = window_columns(sdoc, ref, a, b)
    lo, hi = span_of(cols, a, b)
    listeners, gaps = [], {}
    for k, lid, nm in LISTENERS:
        ev = evidence.get(k)
        x = {"key": k, "id": lid, "name": nm, "found": bool(ev), **({"file": files[k]} if files and k in files else {})}
        if ev:
            per, g = listen(cols, heard_in(ev, lo - TOL_S, hi + TOL_S, (starts or {}).get(k)))
            x["cells"] = [cell(c, ops) for c, ops in zip(cols, per)]
            x["frameMs"] = ev["frameMs"]
            for gk, ops in g.items():
                # The span is widened by TOL_S so edge words keep their phonemes; unpaired ones out there are neighbours'.
                if ops := [o for o in ops if lo <= o["t"] < hi]:
                    gaps.setdefault(gk, {})[k] = ops
        else:
            x["message"] = f"No evidence for {nm[0].lower()}{nm[1:]}" + (f" ({files[k]})" if files and k in files else "")
        listeners.append(x)
    focus = focus_of(cols, label_type, a, b)
    out_cols = []
    for gk in range(-1, len(cols)):
        if gk >= 0:
            c = cols[gk]
            out_cols.append({**c, "focus": gk in focus, "heard": {x["key"]: x["cells"][gk] for x in listeners if x["found"]}})
        if gk in gaps:
            s = cols[gk]["endS"] if gk >= 0 else lo
            e = cols[gk + 1]["startS"] if gk + 1 < len(cols) else hi
            out_cols.append({"kind": "gap", "startS": s, "endS": e, "expected": "", "focus": False,
                             "heard": {x["key"]: {"heard": "".join(o["h"] for o in gaps[gk].get(x["key"], [])), "ops": gaps[gk].get(x["key"], []),
                                                   "status": "extra" if gaps[gk].get(x["key"]) else "nothing"} for x in listeners if x["found"]}})
    s = summary(cols, listeners, focus, a, label_type == "skipped_passage")
    return {"found": True, "window": {"startS": a, "endS": b}, "span": {"startS": lo, "endS": hi}, "summary": s["text"], "elsewhere": s["elsewhere"],
            "listeners": [{k: v for k, v in x.items() if k != "cells"} for x in listeners], "columns": out_cols,
            "lowConf": LOW_CONF, "tolS": TOL_S}
