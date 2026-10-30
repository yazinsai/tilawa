"""Automated test set checks from the teacher's evidence of the built audio (.scratch/lab, ticket 04); run by modal/lab_check.py.

Per session and audio variant, using what the labels say was recited (`said`, with planted edits applied):

- **Planted edits** (every splice, including repeats and restarts): over the window from the word before to the word
  after, the teacher's CTC score of what was said minus that of the original text (`llr`, nats), and the teacher's
  greedy decode's canon edit distance to each. The teacher must prefer the planted version by LLR_MIN, its greedy decode
  must be closer to it, and it must hear the planted version cleanly (local canon PER <= LOCAL_PER_MAX).
- **Clean stretches** (runs of unplanted words between seams): canon PER of the greedy decode against the said
  phonemes, cut at the teacher's forced alignment of the whole session, within CLEAN_PER_MAX (twice the teacher's gold
  canon PER) or, for stretches under SHORT_PHONEMES canon phonemes, at most SHORT_EDITS edits.
- **Timings:** that forced alignment's span of each ayah occurrence lies within TIMING_S of the labelled one.
- **Reciter:** the session's reader is a held-out reciter of `lab-reciters-v1.json` (not in any training source).

Canon PER is the gold-set metric (`per_breakdown.canon`: ں→ن, ۾→م, drop ڇ, collapse runs). Failures are reported with
reasons; nothing is patched.
"""

import json
import math
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

import lab_sources as L
from e1_report import LOG_FLOOR, TOKEN, viterbi
from per_breakdown import canon

ROOT = Path(__file__).resolve().parent.parent
VERSION = 1
TEACHER_GOLD_PER = json.loads((ROOT / "docs/benchmarks/listener-target-v1.json").read_text())["teacherGoldPer"] if (ROOT / "docs/benchmarks/listener-target-v1.json").exists() else 0.0339
LLR_MIN = 3.0
LOCAL_PER_MAX = 0.25
CLEAN_PER_MAX = 2 * TEACHER_GOLD_PER
SHORT_PHONEMES, SHORT_EDITS = 30, 2
TIMING_S = 0.3
PAD_S = 0.1
GAP_SLACK_S = 0.02  # an omission's interval runs from the end of the word before to the start of the next
LLR_CAP = 999.0  # the original can be longer than the window (a skipped ayah): no path, infinitely worse
EDITS = ("wrong_word", "omitted_word", "added_word", "skipped_passage", "repeat", "restart")
THRESHOLDS = {"llrMin": LLR_MIN, "localPerMax": LOCAL_PER_MAX, "cleanPerMax": round(CLEAN_PER_MAX, 4), "teacherGoldPer": round(TEACHER_GOLD_PER, 4),
              "shortPhonemes": SHORT_PHONEMES, "shortEdits": SHORT_EDITS, "timingS": TIMING_S, "padS": PAD_S}


def log_probs(ev: dict) -> np.ndarray:
    """[T, V+1] log posteriors from the evidence's top-k (the rest at LOG_FLOOR)."""
    T, k = ev["nFrames"], ev["posteriors"]["k"]
    ids = np.asarray(ev["posteriors"]["ids"], dtype=np.int64).reshape(T, k)
    p = np.asarray(ev["posteriors"]["probs"], dtype=np.float64).reshape(T, k)
    lp = np.full((T, len(TOKEN) + 1), LOG_FLOOR)
    np.put_along_axis(lp, ids, np.log(np.maximum(p, 1e-12)), axis=1)
    return lp


def word_phonemes(ref: dict) -> dict[str, str]:
    """`s:a:w` -> the word's slice of the continuous ayah phonemes (ref-v2 `p` / `o`)."""
    out = {}
    for a in ref["ayahs"]:
        p, o = a["p"], a["o"]
        for i in range(len(o)):
            out[f"{a['s']}:{a['a']}:{i + 1}"] = p[o[i] : (o[i + 1] if i + 1 < len(o) else len(p))]
    return out


def items(session: dict, ref: dict) -> list[dict]:
    """What was said in time order: `said` plus a basmala `non_quran` span as its own item."""
    out = [dict(x) for x in session["said"]]
    basmala = ref["ayahs"][0]["p"]
    for sp in session["spans"]:
        if sp["kind"] == "non_quran" and sp.get("label") == "basmala":
            out.append({"startS": sp["startS"], "endS": sp["endS"], "at": None, "phonemes": basmala, "role": "basmala"})
    return sorted(out, key=lambda x: (x["startS"], x["endS"]))


def align(lp: np.ndarray, its: list[dict], fs: float) -> list[tuple[float, float]] | None:
    """Forced alignment of every item's phonemes over the whole session (runs collapsed, garbage blank, madd repeats
    stretched, as the sources were aligned): each item's (start, end) in seconds."""
    p = "".join(x["phonemes"] for x in its)
    o, k = [], 0
    for x in its:
        o.append(k)
        k += len(x["phonemes"])
    seq, offs = L.collapse(p, o)
    _, spans = viterbi(L.garbage(lp), seq)
    if spans is None:
        return None
    ext = L.extend(spans, [TOKEN[c] for c in seq], lp.argmax(axis=1), len(lp))
    return [(a * fs, b * fs) for a, b, _ in L.word_spans(offs, ext, [1.0] * len(seq))]


def greedy_in(ev: dict, f0: int, f1: int) -> str:
    return "".join(q["p"] for q in ev["phonemes"] if f0 <= q["start"] < f1)


def score(lp: np.ndarray, f0: int, f1: int, phonemes: str) -> float:
    s, _ = viterbi(L.garbage(lp[f0:f1]), L.squeeze(phonemes))
    return s


def edit_check(session: dict, x: dict, its: list[dict], words: dict[str, str], ev: dict, lp: np.ndarray, fs: float) -> dict:
    """One planted edit: does the teacher hear what was planted rather than the original text?"""
    mine = [i for i, it in enumerate(its) if it.get("op") == x["id"]]
    if mine:
        a, b = its[mine[0]]["startS"], its[mine[-1]]["endS"]
        lo, hi = mine[0] - 1, mine[-1] + 1
    else:
        a, b = x["startS"], x["endS"]
        lo = max((i for i, it in enumerate(its) if it["endS"] <= a + GAP_SLACK_S), default=None)
        hi = min((i for i, it in enumerate(its) if it["startS"] >= b - GAP_SLACK_S), default=None)
    out = {"id": f"{session['id']}-{x['id']}", "session": session["id"], "splice": x["id"], "class": "mutashabihat" if x.get("mutashabihat") else x["type"],
           "type": x["type"], "location": x["location"], "startS": x["startS"], "endS": x["endS"]}
    if lo is None or hi is None or lo < 0 or hi >= len(its):
        return {**out, "pass": False, "reasons": ["no recited word on both sides of the edit"]}
    prev, nxt = its[lo], its[hi]
    planted = "".join(its[i]["phonemes"] for i in mine)
    if x["type"] == "wrong_word":
        orig = words[f"{x['location']['surah']}:{x['location']['ayah']}:{x['location']['word']}"]
    elif x["type"] in ("omitted_word", "skipped_passage"):
        orig = "".join(words[w] for w in x["removed"])
    else:
        orig = ""
    h_said, h_orig = prev["phonemes"] + planted + nxt["phonemes"], prev["phonemes"] + orig + nxt["phonemes"]
    f0, f1 = max(0, math.floor((prev["startS"] - PAD_S) / fs)), min(len(lp), math.ceil((nxt["endS"] + PAD_S) / fs))
    llr = min(LLR_CAP, score(lp, f0, f1, h_said) - score(lp, f0, f1, h_orig))
    g = canon(greedy_in(ev, f0, f1))
    d_said, d_orig = Levenshtein.distance(g, canon(h_said)), Levenshtein.distance(g, canon(h_orig))
    local = d_said / max(1, len(canon(h_said)))
    reasons = []
    if not llr >= LLR_MIN:
        reasons.append(f"teacher prefers the original (llr {llr:.1f})" if llr <= -LLR_MIN else f"teacher can't separate planted from original (llr {llr:.1f})")
    if not d_said < d_orig:
        reasons.append(f"greedy decode no closer to the planted words (edits {d_said} vs {d_orig} to the original)")
    if local > LOCAL_PER_MAX:
        reasons.append(f"teacher doesn't hear the planted version cleanly (local PER {local:.2f})")
    return {**out, "pass": not reasons, "reasons": reasons, "llr": round(float(llr), 2), "editsSaid": d_said, "editsOriginal": d_orig,
            "localPer": round(local, 3), "window": [round(f0 * fs, 3), round(f1 * fs, 3)], "greedy": g, "said": canon(h_said), "original": canon(h_orig)}


def stretches(session: dict, its: list[dict]) -> list[list[int]]:
    """Runs of unplanted Quran words with no seam inside, in time order (item indexes)."""
    from bisect import bisect_right

    seams = sorted({sg["startS"] for sg in session["segments"][1:]})
    out, cur, seg = [], [], None
    for i, it in enumerate(its):
        k = bisect_right(seams, it["startS"])
        straddles = k < len(seams) and seams[k] < it["endS"]
        if it["role"] != "ref" or straddles or k != seg:
            if cur:
                out.append(cur)
            cur = []
        seg = k
        if it["role"] == "ref" and not straddles:
            cur.append(i)
    if cur:
        out.append(cur)
    return out


def clean_check(session: dict, k: int, run: list[int], its: list[dict], spans: list[tuple[float, float]], ev: dict, fs: float) -> dict:
    ref_p = canon("".join(its[i]["phonemes"] for i in run))
    a, b = spans[run[0]][0], spans[run[-1]][1]
    g = canon(greedy_in(ev, round(a / fs), max(round(a / fs) + 1, round(b / fs))))
    edits = Levenshtein.distance(g, ref_p)
    per = edits / max(1, len(ref_p))
    ok = per <= CLEAN_PER_MAX or (len(ref_p) < SHORT_PHONEMES and edits <= SHORT_EDITS)
    first, last = its[run[0]], its[run[-1]]
    w = lambda it: f"{it['at']['surah']}:{it['at']['ayah']}:{it['at']['word']}"
    return {"id": f"{session['id']}-clean-{k}", "session": session["id"], "from": w(first), "to": w(last), "startS": first["startS"], "endS": last["endS"],
            "words": len(run), "phonemes": len(ref_p), "edits": edits, "per": round(per, 4), "pass": ok,
            "reasons": [] if ok else [f"teacher PER {per:.3f} over {len(ref_p)} phonemes (bar {CLEAN_PER_MAX:.3f})"],
            **({} if ok else {"greedy": g, "said": ref_p})}


def timing_check(session: dict, its: list[dict], spans: list[tuple[float, float]]) -> dict:
    """Each ayah occurrence (consecutive items of one ayah; added words join the ayah they're in) against its labels."""
    groups, key = [], None
    for i, it in enumerate(its):
        k = (it["at"]["surah"], it["at"]["ayah"], it["role"] in ("repeat", "restart")) if it["at"] else key if it["role"] == "added" else ("basmala",)
        if not groups or k != key:
            groups.append([])
        groups[-1].append(i)
        key = k
    bad, ds = [], []
    for g in groups:
        d0 = spans[g[0]][0] - its[g[0]]["startS"]
        d1 = spans[g[-1]][1] - its[g[-1]]["endS"]
        if max(abs(d0), abs(d1)) > TIMING_S:
            it = its[g[0]]
            bad.append({"at": f"{it['at']['surah']}:{it['at']['ayah']}" if it["at"] else it["role"], "startS": it["startS"], "dStartS": round(d0, 3), "dEndS": round(d1, 3)})
    for (a, b), it in zip(spans, its):
        if b > a:
            ds.append(abs(a - it["startS"]))
    q = [round(float(v), 3) for v in np.percentile(ds, [50, 90, 99])] if ds else None
    return {"pass": not bad, "occurrences": len(groups), "off": bad, "wordStartAbsDeltaS": q,
            "reasons": [f"{len(bad)} ayah occurrence(s) off by more than {TIMING_S} s"] if bad else []}


def reciter_check(session: dict, reciters: dict) -> dict:
    held = {r["readId"]: r for r in reciters["heldOut"]}
    excluded = {r["readId"]: r for r in reciters["excluded"]}
    rid, split = int(session["meta"]["readId"]), session["meta"]["split"]
    reasons = []
    if rid in excluded:
        reasons.append(f"read {rid} is excluded: {excluded[rid].get('reasons') or excluded[rid].get('reason')}")
    if rid not in held:
        reasons.append(f"read {rid} is not a held-out reciter")
    elif held[rid]["split"] != split:
        reasons.append(f"read {rid} is {held[rid]['split']}, session is {split}")
    return {"pass": not reasons, "readId": rid, "reciter": session["meta"]["reciter"], "reasons": reasons}


def check_session(session: dict, ev: dict, ref: dict, variant: str) -> dict:
    """Every check for one session's teacher evidence on one audio variant."""
    if abs(ev["nFrames"] * ev["frameMs"] / 1000 - session["durationS"]) > 0.1:
        raise ValueError(f"{session['id']}: evidence {ev['nFrames']} frames for {session['durationS']} s")
    fs = ev["frameMs"] / 1000
    lp = log_probs(ev)
    its = items(session, ref)
    spans = align(lp, its, fs)
    if spans is None:
        raise ValueError(f"{session['id']} {variant}: forced alignment failed")
    words = word_phonemes(ref)
    edits = [edit_check(session, x, its, words, ev, lp, fs) for x in session["splices"] if x["type"] in EDITS]
    clean = [clean_check(session, k, run, its, spans, ev, fs) for k, run in enumerate(stretches(session, its))]
    tot = sum(c["phonemes"] for c in clean)
    return {"session": session["id"], "variant": variant, "edits": edits, "clean": clean, "timing": timing_check(session, its, spans),
            "cleanPer": round(sum(c["edits"] for c in clean) / max(1, tot), 4), "cleanPhonemes": tot}


def review_with_failures(queue: dict, sessions: dict[str, dict], failed: list[dict], root: str) -> tuple[dict, dict]:
    """The listening queue with every failed planted edit in it: a failed edit not yet queued replaces a passing item of
    its kind (splices, or the repeat/restart cases), of its class when there is one, last first. Other ids are kept."""
    from lab_sessions import review_repeat, review_splice

    items = [dict(it) for it in queue["items"]]
    bad = {f["id"]: f for f in failed}
    cls_of = {}
    for s in sessions.values():
        for x in s["splices"]:
            cls_of[f"{s['id']}-{x['id']}"] = "mutashabihat" if x.get("mutashabihat") else x["type"]
    nonerror = lambda t: t in ("repeat", "restart")
    replaced, queued, missing = [], [], []
    for f in sorted(failed, key=lambda f: (f["session"], int(f["splice"].lstrip("x")))):
        have = next((it for it in items if it["id"] == f["id"]), None)
        if have is None:
            slots = [i for i, it in enumerate(items) if it["id"] not in bad and it["id"] in cls_of
                     and nonerror(cls_of[it["id"]]) == nonerror(f["type"]) and (it["kind"] == "splice") != nonerror(f["type"])]
            same = [i for i in slots if cls_of[items[i]["id"]] == f["class"]]
            if not (same or slots):
                missing.append(f["id"])
                continue
            i = (same or slots)[-1]
            s = sessions[f["session"]]
            x = next(x for x in s["splices"] if x["id"] == f["splice"])
            replaced.append({"out": items[i]["id"], "in": f["id"]})
            items[i] = (review_repeat if nonerror(f["type"]) else review_splice)(s, x, root)
            have = items[i]
        else:
            queued.append(f["id"])
        have["check"] = {"pass": False, "reasons": f["reasons"]}
    return {**queue, "items": items}, {"replaced": replaced, "alreadyQueued": queued, "notQueued": missing}


def summarize(results: list[dict]) -> dict:
    """Pass/fail counts per variant and class from check_session outputs."""
    out = {}
    for v in sorted({r["variant"] for r in results}):
        rs = [r for r in results if r["variant"] == v]
        by = {}
        for e in (e for r in rs for e in r["edits"]):
            c = by.setdefault(e["class"], {"n": 0, "pass": 0, "fail": 0})
            c["n"] += 1
            c["pass" if e["pass"] else "fail"] += 1
        cl = [c for r in rs for c in r["clean"]]
        ph = sum(c["phonemes"] for c in cl)
        out[v] = {"sessions": len(rs), "edits": dict(sorted(by.items())),
                  "mistakes": {k: sum(c[k] for cls, c in by.items() if cls not in ("repeat", "restart")) for k in ("n", "pass", "fail")},
                  "nonErrors": {k: sum(c[k] for cls, c in by.items() if cls in ("repeat", "restart")) for k in ("n", "pass", "fail")},
                  "clean": {"n": len(cl), "pass": sum(c["pass"] for c in cl), "fail": sum(not c["pass"] for c in cl), "phonemes": ph,
                            "per": round(sum(c["edits"] for c in cl) / max(1, ph), 4)},
                  "timing": {"sessions": len(rs), "pass": sum(r["timing"]["pass"] for r in rs), "occurrencesOff": sum(len(r["timing"]["off"]) for r in rs)}}
    return out
