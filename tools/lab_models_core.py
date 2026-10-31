"""What the triage models heard for one review item: the review page's "What the models heard" panel
(docs/benchmarks/lab-review.md, GET /api/models).

A queue item's per-model records come from the item itself when the triage build embedded them (`triage.<model>`
with a `transcript`), else from the test set's triage outputs (docs/benchmarks/lab-triage-v1.md): the pairing's
per-case file for the decisions and scores, and the models' own transcript records. Each model's decision and score
are put in plain words, and its label-blind transcript is diffed word by word against the labelled text on the rasm
skeleton (review_asr_core.compare_rasm, the letters the Soniox judge reads).

Pure logic, standard library only; tested by tools/test_lab_models.py.
"""

import math

import lab_plain as plain
import review_asr_core as core

MODELS = ("nemotron", "soniox", "fastconformer")
NAMES = {"nemotron": "Nemotron", "soniox": "Soniox", "fastconformer": "FastConformer"}
ROLES = {m: plain.ROLE[m in plain.WRITES] for m in MODELS}
# What a model's transcript is for, when the decision doesn't come from it.
HEARD_NOTE = {m: "A rough write-down only: it invents and drops words, sometimes in another language. Its verdict above doesn't come from it."
              for m in ("nemotron", "fastconformer")}
# Per pairing: the file with every case's decisions (relative to /lab/data/<testSet>). Soniox's transcript is in it.
PAIRING_FILE = {"soniox": "triage-soniox-v1.json", "fastconformer": "triage-v1.json"}
TRANSCRIPTS = {"nemotron": "triage/asr-nemotron-3.5-transcribe.json", "fastconformer": "triage/triage-v1.fastconformer.json"}
MARGIN = 2.0  # nats, lab_triage_core.MARGINS (the likelihood arms' committed margin)
DECISION = {"match": "as described", "mismatch": "not as described", "unsure": "can't tell"}
OUTCOME = {"auto-accepted": "Both checks were happy with the description", "both-mismatch": "Both checks think the description is wrong",
           "disagree": "The two checks disagree", "unsure": "A check couldn't decide"}
REASON = {"both-mismatch": "both checks think the description is wrong", "disagree": "the two checks disagree", "unsure": "a check couldn't decide",
          "seam": "a check that the edit can't be heard", "spot": "a random check of a case both checks were happy with"}
ALT_NAME = {"edit": "this other version, with the change undone", "repeat": "this other version, said only once",
            "clean": "a made-up test version (nobody recited it; it's only there to compare against)"}
NONE = "No model output for this item"


# ---- lookup --------------------------------------------------------------------------------------------------------

def group_of(item: dict) -> str:
    t = (item.get("label") or {}).get("type")
    return "clean" if t == "clean" else "repeat" if t in ("repeat", "restart") else "edit"


def pairing(item: dict) -> tuple[str, ...]:
    """The models whose decisions put this item in its queue (its `triage` field), Nemotron first."""
    tri = item.get("triage") or {}
    return tuple(m for m in MODELS if isinstance(tri.get(m), dict))


_index: dict[int, tuple[object, dict]] = {}


def by_id(doc: dict | None) -> dict[str, dict]:
    """A triage file's cases by id, built once per parsed document."""
    if not doc:
        return {}
    hit = _index.get(id(doc))
    if hit and hit[0] is doc:
        return hit[1]
    idx = {c["id"]: c for c in doc.get("cases", [])}
    if len(_index) >= 16:  # a rewritten file is a new document; drop the old ones' indexes with them
        _index.clear()
    _index[id(doc)] = (doc, idx)
    return idx


def same_window(case: dict | None, item: dict) -> bool:
    """The case scored the clip and text this item plays (ids carry over between queues, windows may not)."""
    return bool(case) and case.get("audio") == item.get("audio") and case.get("claimed") == item.get("text")


def transcript_of(model: str, case_id: str, case: dict, load) -> str | None:
    if model == "soniox":
        return case.get("transcript")
    doc = load(TRANSCRIPTS[model]) if model in TRANSCRIPTS else None
    rec = ((doc or {}).get("items") or {}).get(case_id) or {}
    return rec.get("transcript")


def records(item: dict, load) -> dict | None:
    """{source, outcome, reasons, alternative, alternativeOp, group, models: [(model, record, inRule)]} for a queue item, or
    None when no triage output covers its window. load(rel) -> parsed JSON under /lab/data/<testSet> or None.

    Embedded (the item's `triage.<model>` has a `transcript`): read as it is. Otherwise the pairing's case file, and the
    models' transcript records. Soniox is added after the pairing when the item's queue used FastConformer and the test
    set also has a Soniox triage of the same window (inRule False)."""
    tri, pair = item.get("triage") or {}, pairing(item)
    if pair and all("transcript" in tri[m] for m in pair):
        return {"source": "queue", "outcome": tri.get("outcome"), "reasons": tri.get("reasons") or [], "alternative": tri.get("alternative"),
                "alternativeOp": tri.get("alternativeOp"), "group": group_of(item), "models": [(m, tri[m], True) for m in pair]}
    files = [PAIRING_FILE[m] for m in pair if m in PAIRING_FILE] or [PAIRING_FILE["soniox"], PAIRING_FILE["fastconformer"]]
    case, used = None, None
    for f in files:
        c = by_id(load(f)).get(item["id"])
        if same_window(c, item):
            case, used = c, f
            break
    if not case:
        return None
    models = pair or tuple(m for m in MODELS if isinstance(case.get(m), dict))
    out = [(m, {**case[m], "transcript": transcript_of(m, item["id"], case, load)}, True) for m in models if isinstance(case.get(m), dict)]
    if "soniox" not in models:
        s = by_id(load(PAIRING_FILE["soniox"])).get(item["id"])
        if same_window(s, item) and isinstance(s.get("soniox"), dict):
            out.append(("soniox", {**s["soniox"], "transcript": s.get("transcript")}, False))
    return {"source": used, "outcome": tri.get("outcome") or case.get("outcome"), "reasons": tri.get("reasons") or [],
            "alternative": case.get("alternative"), "alternativeOp": case.get("alternativeOp"), "group": case.get("group") or group_of(item),
            "models": out}


# ---- word diff -----------------------------------------------------------------------------------------------------

MERGE_COST, MERGE_MAX = 0.1, 3  # words heard split, joined or split elsewhere (up to 3 a side), when the space-free skeletons are identical


def _align(ref: list[tuple[str, str]], hyp: list[tuple[str, str]]) -> list[dict]:
    """Word edit alignment of (word, skeleton) lists: a pair costs 1 - similarity, a missing or extra word 1, a p:q
    regrouping with identical space-free skeletons MERGE_COST per word beyond a pair (من كلا heard as منك لا), so the
    smallest regrouping wins. No free words at the edges
    (unlike the rule's core.align)."""
    n, m = len(ref), len(hyp)
    sim = lambda a, b: 1.0 if a == b else 1 - core.lev(a, b) / max(len(a), len(b), 1)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if not i and not j:
                continue
            c = []
            if i and j:
                c.append((D[i - 1][j - 1] + 1 - sim(ref[i - 1][1], hyp[j - 1][1]), "pair", 1, 1))
            for p in range(1, min(MERGE_MAX, i) + 1):
                for q in range(1, min(MERGE_MAX, j) + 1):
                    if p + q > 2 and "".join(s for _, s in ref[i - p:i]) == "".join(s for _, s in hyp[j - q:j]):
                        c.append((D[i - p][j - q] + MERGE_COST * (p + q - 2), "merge", p, q))
            if i:
                c.append((D[i - 1][j] + 1, "del", 1, 0))
            if j:
                c.append((D[i][j - 1] + 1, "ins", 0, 1))
            D[i][j], *B[i][j] = min(c, key=lambda x: x[0])
    out, i, j = [], n, m
    while i or j:
        op, p, q = B[i][j]
        r, h = " ".join(w for w, _ in ref[i - p:i]), " ".join(w for w, _ in hyp[j - q:j])
        if op == "pair":
            out.append({"op": "same" if ref[i - 1][1] == hyp[j - 1][1] else "sub", "ref": r, "heard": h})
        elif op == "merge":
            out.append({"op": "same", "ref": r, "heard": h})
        elif op == "del":
            out.append({"op": "del", "ref": r})
        else:
            out.append({"op": "ins", "heard": h})
        i, j = i - p, j - q
    return out[::-1]


def word_diff(text: str, transcript: str | None) -> dict:
    """The transcript against the labelled text, word by word on the rasm skeleton (core.rasm: vowels, alif spellings
    and hamza seats don't count; a word heard split or joined matches). Segments in reading order: same {ref, heard},
    sub {ref, heard}, del {ref} (labelled, not heard), ins {heard} (heard, not labelled). `edge` marks a missing or
    extra word before the first or after the last word heard as labelled: a neighbouring sound the window caught, or a
    labelled word cut off at the window's end. A word heard differently is never an edge. counts: per op, edges excluded."""
    tok = lambda s: [(w, sk) for w, _ in core.tokens(s) if (sk := core.rasm(w))]
    segs = _align(tok(text), tok(transcript)) if transcript else []
    core_at = [k for k, s in enumerate(segs) if s["op"] == "same"]
    for k, s in enumerate(segs):
        if s["op"] in ("ins", "del") and core_at and (k < core_at[0] or k > core_at[-1]):
            s["edge"] = True
    counts = {k: sum(s["op"] == k and not s.get("edge") for s in segs) for k in ("same", "sub", "del", "ins")}
    counts["edge"] = sum(bool(s.get("edge")) for s in segs)
    return {"segments": segs, "counts": counts, "summary": diff_summary(counts, bool(transcript))}


def diff_summary(c: dict, heard: bool) -> str:
    if not heard:
        return "Nothing written down to compare"
    parts = [f"{c['sub']} changed" if c["sub"] else "", f"{c['del']} not written down" if c["del"] else "", f"{c['ins']} extra" if c["ins"] else ""]
    n, e = c["sub"] + c["del"] + c["ins"], c.get("edge", 0)
    edges = f"{e} word{'s' if e != 1 else ''} at the clip's edge (cut off there, ignore {'them' if e != 1 else 'it'})"
    if not n:
        return f"Wrote down exactly our text, plus {edges}" if e else "Wrote down exactly our text"
    return f"{n} word{'s' if n != 1 else ''} differ{'s' if n == 1 else ''} from our text: " + ", ".join(p for p in parts if p) + (f"; plus {edges}" if e else "")


# ---- plain words ---------------------------------------------------------------------------------------------------

def _n(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def score_line(r: dict) -> tuple[str, str]:
    """(the numbers behind a check's verdict in plain words, how they are measured) for one model's record."""
    d, dec = r.get("diff"), r.get("decision")
    if r.get("unit") == "letters" or "reason" in r:  # a transcript reading (the judge leaves out the unit when it had nothing to read)
        how = "Counted on letters only, without vowel marks. The more the two versions differ where it wrote, the bigger the lead it needs."
        if r.get("claimed") is None or d is None:
            return {"empty transcript": "It wrote down nothing", "no alternative text": "There was no other version to compare"}.get(r.get("reason"), "Nothing usable written down"), how
        letters = lambda n: f"{n} letter{'' if n == 1 else 's'}"
        dist = f"What it wrote is {letters(r['claimed'])} away from our text and {letters(r['alternative'])} from the other version"
        need = r.get("need") if (d or 0) >= 0 else r.get("needAgainst", r.get("need"))
        lead = f"it needed a lead of at least {letters(math.ceil(need))}" if need is not None else ""
        if r.get("reason") == "transcript far from both texts":
            return f"{dist}: too far from both to say", how
        if r.get("reason") == "transcript covers too little of the window":
            return f"{dist}, but it caught too little of the clip", how
        return f"{dist}; {lead}" if lead else dist, how
    how = f"How much better one version matches the sound than the other. {_n(MARGIN)} is the minimum to count; above 20 is very sure."
    if d is None:
        return {"match": "Only the version as described could be matched to the sound", "mismatch": "The version as described couldn't be matched to the sound"}.get(dec, "Neither version could be scored"), how
    side = "our text" if d >= 0 else "the other version"
    return f"Prefers {side} by {abs(d):.1f} (it needs {_n(MARGIN)}; above 20 is very sure)", how


def why_line(outcome: str | None, reasons: list[str]) -> str:
    """Why the item is in the queue, in plain words; with no reasons (a queue the triage didn't build), what the
    triage of the same window found."""
    if not reasons:
        o = OUTCOME.get(outcome, "")
        return f"The computer checks of this same clip: {o[:1].lower()}{o[1:]}" if o else ""
    head = f"Why you're hearing this: {REASON.get(reasons[0], reasons[0])}"
    return head + "".join(f", and {REASON.get(r, r)}" for r in reasons[1:])


def model_view(model: str, r: dict, text: str, in_rule: bool = True, n: int | None = None, other: str = "as the other version") -> dict:
    """n: the check's number on the page (lab_plain.checks), None for a check that didn't decide this item."""
    score, how = score_line(r)
    dec = r.get("decision", "unsure")
    t = r.get("transcript")
    return {"model": model, "name": f"{NAMES[model]} ({ROLES[model]})", "short": f"Check {n}" if n else "Another check", "vendor": NAMES[model],
            "role": ROLES[model], "verdict": plain.check_line(model, r, other),
            "decision": {"key": dec, "text": DECISION.get(dec, dec)}, "inRule": in_rule, "transcript": t,
            "heardNote": HEARD_NOTE.get(model), "score": score, "how": how, "diff": word_diff(text, t),
            **({"reason": r["reason"]} if r.get("reason") else {})}


def view(item: dict, load) -> dict:
    """The /api/models response for one queue item."""
    rec = records(item, load)
    if not rec:
        return {"id": item["id"], "found": False, "message": NONE, "models": []}
    alt = rec["alternative"]
    return {"id": item["id"], "found": True, "source": rec["source"],
            "outcome": {"key": rec["outcome"], "text": OUTCOME.get(rec["outcome"], rec["outcome"] or "")}, "reasons": rec["reasons"],
            "why": why_line(rec["outcome"], rec["reasons"]),
            "alternative": {"text": alt, "name": ALT_NAME.get(rec["group"], "the alternative"), "op": rec["alternativeOp"]} if alt else None,
            "models": [model_view(m, r, item.get("text", ""), inr, k + 1 if inr else None, plain.other_version(item))
                       for k, (m, r, inr) in enumerate(rec["models"])]}
