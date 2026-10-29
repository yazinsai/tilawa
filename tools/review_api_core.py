"""Commercial ASR APIs as the second independent model in two-model triage (.scratch/lab/issues/12-review-asr-calibration.md,
17-two-model-triage.md; docs/benchmarks/lab-review-asr-v1.md, section "Commercial APIs").

Soniox and Speechmatics transcribe each window label-blind (modal/review_api.py). This module reads their transcripts
against the labelled text and the window's alternative text, scores them on your listening verdicts, pairs each with
Nemotron's likelihood arm and picks the winner. Everything here was fixed before any API output existed.

Pure logic, standard library only; tested by tools/test_review_api.py.
"""

import math
from collections import Counter

import review_asr_core as core

VENDORS = ("soniox", "speechmatics")
READINGS = ("judge", "opinion", "opinionRasm")  # judge decides pairing and triage; the opinions are the ticket 12 rules
JUDGE_MIN, JUDGE_SHARE, JUDGE_MAX_DIST = 2, 0.5, 0.5
BOILERPLATE = ("اشتركوا", "القناة", "ترجمة", "نانسي", "قنقر", "موسيقى")
EXTRA_MAX = 3  # more extra heard words than this in one window is a hallucination (round 2's automatic bucket 5)
BUCKETS = {1: "orthography", 2: "edge word", 3: "mishearing", 4: "canonicalisation", 5: "hallucination"}


# ---- transcripts ---------------------------------------------------------------------------------------------------

def soniox_transcript(doc: dict) -> str:
    """The transcript text of a Soniox GET /v1/transcriptions/{id}/transcript response (tokens joined if text is absent)."""
    if not doc:
        return ""
    if isinstance(doc.get("text"), str):
        return doc["text"].strip()
    return "".join(t.get("text", "") for t in doc.get("tokens") or []).strip()


def speechmatics_transcript(doc: dict) -> str:
    """The transcript of a Speechmatics json-v2 response: each result's top alternative; punctuation that attaches to
    the previous word is glued to it, everything else is separated by the language pack's word delimiter."""
    if not doc:
        return ""
    sep = ((doc.get("metadata") or {}).get("language_pack_info") or {}).get("word_delimiter", " ")
    out = ""
    for r in doc.get("results") or []:
        alts = r.get("alternatives") or []
        if not alts:
            continue
        w = alts[0].get("content", "")
        if r.get("type") == "punctuation" and r.get("attaches_to", "previous") == "previous" or not out:
            out += w
        else:
            out += sep + w
    return out.strip()


# ---- the judge: which text does the transcript fit? ----------------------------------------------------------------

def skeleton(text: str) -> str:
    """The text's rasm skeleton with no spaces (core.rasm per word, hamza kept), so a word heard split or joined costs nothing."""
    return "".join(core.rasm(t) for t, _ in core.tokens(text))


def fit_dist(heard: str, text: str) -> int:
    """Letter edit distance from `heard` to the best-fitting contiguous stretch of `text`: the text's ends are free, so
    words the model didn't write at the window edges cost nothing (they are shared by both candidate texts, and
    charging them lets the alignment absorb the edit next to them). Every heard letter counts."""
    prev = [0] * (len(text) + 1)
    for i, ch in enumerate(heard, 1):
        cur = [i]
        for j, cj in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != cj)))
        prev = cur
    return min(prev)


def judge(transcript: str, claimed: str, alternative: str | None) -> dict:
    """Is the label-blind transcript closer to the claimed text than to the alternative?

    Distances are fit_dist on space-free rasm skeletons. gap = d(alternative) - d(claimed), so a positive gap favours
    the claimed text; |gap| <= D = the edit distance between the two texts. match: gap >= need, with need =
    max(JUDGE_MIN, JUDGE_SHARE * D), i.e. the transcript lies on the claimed side of the midpoint by at least 2 letters;
    mismatch: gap <= -need; otherwise unsure. Also unsure: an empty transcript, one whose distance to the nearer text
    is over JUDGE_MAX_DIST of its own length (a hallucination can't vouch for either text), or one shorter than
    JUDGE_MAX_DIST of the nearer text (it heard too little of the window)."""
    t, c = skeleton(transcript), skeleton(claimed)
    if alternative is None:
        return {"decision": "unsure", "reason": "no alternative text", "claimed": None, "alternative": None, "diff": None}
    a = skeleton(alternative)
    if not t:
        return {"decision": "unsure", "reason": "empty transcript", "claimed": None, "alternative": None, "diff": None}
    dc, da, d = fit_dist(t, c), fit_dist(t, a), core.lev(c, a)
    gap, need = da - dc, max(JUDGE_MIN, JUDGE_SHARE * d)
    near, ref = (dc, c) if dc <= da else (da, a)
    rel, cover = near / len(t), len(t) / max(1, len(ref))
    out = {"claimed": dc, "alternative": da, "diff": gap, "need": need, "between": d, "relDist": round(rel, 3), "coverage": round(cover, 3),
           "unit": "letters"}
    if rel > JUDGE_MAX_DIST:
        return {"decision": "unsure", "reason": "transcript far from both texts", **out}
    if cover < JUDGE_MAX_DIST:
        return {"decision": "unsure", "reason": "transcript covers too little of the window", **out}
    if gap >= need:
        return {"decision": "match", "reason": None, **out}
    if gap <= -need:
        return {"decision": "mismatch", "reason": None, **out}
    return {"decision": "unsure", "reason": "too close to call", **out}


def readings(transcript: str | None, claimed: str, alternative: str | None) -> dict:
    """Every reading of one transcript for one claimed text: the judge against the alternative, and the two ticket 12
    opinion rules against the claimed text alone (round-1 compare, rasm with no edge context)."""
    if transcript is None:
        miss = {"decision": "unsure", "reason": "no transcript"}
        return {k: miss for k in READINGS}
    o, r = core.compare(claimed, transcript), core.compare_rasm(claimed, transcript)
    return {"judge": judge(transcript, claimed, alternative),
            "opinion": {"decision": o["decision"], "reasons": o["reasons"], "ops": o["ops"]},
            "opinionRasm": {"decision": r["decision"], "reasons": r["reasons"]}}


# ---- failure taxonomy ----------------------------------------------------------------------------------------------

def edit_zone(words: list[dict], x: dict | None) -> set[int]:
    """Indices into the window's words (the claimed text's words, in order) of the planted words and their neighbours:
    every word overlapping the edit's span, plus one word each side; for an omission or skip (nothing planted), the
    two words around the gap."""
    if not x:
        return set()
    hit = [k for k, w in enumerate(words) if w["endS"] > x["startS"] - 0.02 and w["startS"] < x["endS"] + 0.02 and w.get("role") != "ref"]
    if not hit:
        mid = (x["startS"] + x["endS"]) / 2
        k = sum((w["startS"] + w["endS"]) / 2 < mid for w in words)
        return {i for i in (k - 1, k) if 0 <= i < len(words)}
    return {i for i in range(hit[0] - 1, hit[-1] + 2) if 0 <= i < len(words)}


def bucket(claimed: str, transcript: str, opinion: dict, revert: str | None = None, zone: set[int] | None = None, phone: bool = False) -> dict:
    """Root cause of a round-1 opinion rejection on a correct window, by round 2's automatic rule in this order:
    5 hallucination (boilerplate, or more than EXTRA_MAX extra heard words); 4 canonicalisation (an edited window whose
    text with the edit undone matches the transcript); 1 orthography (equal space-free skeletons, or the rasm rule
    matches); 2 edge word (only edge reasons); 3 mishearing. Tagged with whether a failing word is at the edit
    (planted word or its neighbour) and whether the window is a phone copy."""
    words = [t for t, _ in core.tokens(transcript)]
    extra = [o for o in opinion.get("ops", []) if o.get("status") in ("extra", "edge-extra")]
    if any(b in transcript for b in BOILERPLATE) or len(extra) > EXTRA_MAX:
        b = 5
    elif revert and core.compare(revert, transcript)["decision"] == "match":
        b = 4
    elif skeleton(claimed) == skeleton(transcript) or core.compare_rasm(claimed, transcript)["decision"] == "match":
        b = 1
    elif all(r.startswith("edge-") for r in opinion.get("reasons", [])):
        b = 2
    else:
        b = 3
    bad = [o for o in opinion.get("ops", []) if o.get("status") in ("missing", "extra", "wrong", "doubtful", "edge-missing", "edge-extra")]
    at = bool(zone) and any(math.floor(o["ri"]) in zone or math.ceil(o["ri"]) in zone for o in bad if "ri" in o)
    return {"bucket": b, "name": BUCKETS[b], "atEdit": at, "phone": phone, "heardWords": len(words)}


# ---- scoring on your verdicts --------------------------------------------------------------------------------------

def reading_stats(rows: list[dict], get) -> dict:
    """rows: {id, truth (True = you heard the label as right), ...}; get(row) -> decision. Agreement: match on your ok
    items. On your bad items, flagged = not match (it would reach you), mismatch = called wrong."""
    ok, bad = [r for r in rows if r["truth"]], [r for r in rows if not r["truth"]]
    d = {r["id"]: get(r) for r in rows}
    return {"agreement": core.rate(sum(d[r["id"]] == "match" for r in ok), len(ok)),
            "okDecisions": dict(Counter(d[r["id"]] for r in ok)),
            "badFlagged": core.rate(sum(d[r["id"]] != "match" for r in bad), len(bad)),
            "badMismatch": core.rate(sum(d[r["id"]] == "mismatch" for r in bad), len(bad)),
            "badDecisions": {r["id"]: d[r["id"]] for r in bad},
            "okRejected": sorted(r["id"] for r in ok if d[r["id"]] != "match")}


def control_stats(rows: list[dict], get) -> dict:
    """Synthetic controls: rows {id, oneEditEdit, oneEditWords}; get(row, k) -> decision on the control text k in
    oneEdit / shifted. Flagged = not match. One-word = single-word controls that are not repeats, restarts or skips."""
    single = [r for r in rows if r["oneEditWords"] == 1 and r["oneEditEdit"] not in ("repeat", "restart", "skipped_passage")]
    fl = lambda rs, k: core.rate(sum(get(r, k) != "match" for r in rs), len(rs))
    return {"oneEdit": fl(rows, "oneEdit"), "oneWord": fl(single, "oneEdit"), "shifted": fl(rows, "shifted"),
            "oneEditMisses": sorted(r["id"] for r in rows if get(r, "oneEdit") == "match"),
            "repeatMisses": sorted(r["id"] for r in rows if r["oneEditEdit"] in ("repeat", "restart") and get(r, "oneEdit") == "match")}


def pairing(rows: list[dict], api) -> dict:
    """Two-model triage on your verdicts: Nemotron's likelihood decision (row['nemotron']) with the API's decision
    api(row). Auto-accepted = both match; every other item reaches you (spot checks come on top)."""
    import lab_triage_core as T

    out = {r["id"]: T.outcome(r["nemotron"], api(r)) for r in rows}
    ok, bad = [r for r in rows if r["truth"]], [r for r in rows if not r["truth"]]
    return {"okAccepted": core.rate(sum(out[r["id"]] == "auto-accepted" for r in ok), len(ok)),
            "badAccepted": sorted(r["id"] for r in bad if out[r["id"]] == "auto-accepted"),
            "badAcceptedRate": core.rate(sum(out[r["id"]] == "auto-accepted" for r in bad), len(bad)),
            "queued": sum(o != "auto-accepted" for o in out.values()), "items": len(rows),
            "okOutcomes": dict(Counter(out[r["id"]] for r in ok)), "badOutcomes": {r["id"]: out[r["id"]] for r in bad},
            "bothMismatchOk": sorted(r["id"] for r in ok if out[r["id"]] == "both-mismatch")}


def pick_winner(cands: dict[str, dict]) -> str:
    """The preregistered rule. cands: vendor -> {badAccepted (count), okAccepted (count), badFlagged, oneEditFlagged
    (counts), costPerHour}. Fewest of your bad items auto-accepted, then most of your ok items auto-accepted, then
    sensitivity (your bad items flagged, then synthetic one-edit controls flagged), then lower cost."""
    return min(sorted(cands), key=lambda v: (cands[v]["badAccepted"], -cands[v]["okAccepted"], -cands[v]["badFlagged"],
                                             -cands[v]["oneEditFlagged"], cands[v]["costPerHour"]))


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Your verdicts on byte-identical clips with the same text (carried over between test sets) count once: the
    earliest is kept. Returns (kept, dropped)."""
    seen, kept, dropped = {}, [], []
    for r in rows:
        k = (r["audioSha256"], r["claimed"])
        if k in seen:
            dropped.append({**r, "duplicateOf": seen[k]})
        else:
            seen[k] = r["id"]
            kept.append(r)
    return kept, dropped
