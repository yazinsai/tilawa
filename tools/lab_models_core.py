"""What the triage models heard for one review item: the review page's "What the models heard" panel
(docs/benchmarks/lab-review.md, GET /api/models).

A queue item's per-model records come from the item itself when the triage build embedded them (`triage.<model>`
with a `transcript`), else from the test set's triage outputs (docs/benchmarks/lab-triage-v1.md): the pairing's
per-case file for the decisions and scores, and the models' own transcript records. Each model's decision and score
are put in plain words, and its label-blind transcript is diffed word by word against the labelled text on the rasm
skeleton (review_asr_core.compare_rasm, the letters the Soniox judge reads).

Pure logic, standard library only; tested by tools/test_lab_models.py.
"""

import review_asr_core as core

MODELS = ("nemotron", "soniox", "fastconformer")
NAMES = {"nemotron": "Nemotron", "soniox": "Soniox", "fastconformer": "FastConformer"}
ROLES = {"nemotron": "checks how well the label fits the sound", "soniox": "transcribes the words",
         "fastconformer": "checks how well the label fits the sound"}
# What a model's transcript is for, when the decision doesn't come from it.
HEARD_NOTE = {m: "Its own transcript, only used to spell the two texts it scores against the sound." for m in ("nemotron", "fastconformer")}
# Per pairing: the file with every case's decisions (relative to /lab/data/<testSet>). Soniox's transcript is in it.
PAIRING_FILE = {"soniox": "triage-soniox-v1.json", "fastconformer": "triage-v1.json"}
TRANSCRIPTS = {"nemotron": "triage/asr-nemotron-3.5-transcribe.json", "fastconformer": "triage/triage-v1.fastconformer.json"}
MARGIN = 2.0  # nats, lab_triage_core.MARGINS (the likelihood arms' committed margin)
DECISION = {"match": "fits", "mismatch": "doesn’t fit", "unsure": "unsure"}
OUTCOME = {"auto-accepted": "Both models accepted the label", "both-mismatch": "Both models think the label is wrong",
           "disagree": "The two models disagree", "unsure": "A model couldn't tell"}
REASON = {"both-mismatch": "both models think the label is wrong", "disagree": "the two models disagree", "unsure": "a model couldn't tell",
          "seam": "a seam spot check", "spot": "a random spot check of labels both models accepted"}
ALT_NAME = {"edit": "the text without the planted mistake", "repeat": "the text without the repeat", "clean": "a one-word change of the text"}
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

MERGE_COST = 0.1  # one word heard as two or two as one, when the space-free skeletons are identical


def _align(ref: list[tuple[str, str]], hyp: list[tuple[str, str]]) -> list[dict]:
    """Word edit alignment of (word, skeleton) lists: a pair costs 1 - similarity, a missing or extra word 1, a 1:2 or
    2:1 merge with identical skeletons MERGE_COST. No free words at the edges (unlike the rule's core.align)."""
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
            if i > 1 and j and ref[i - 2][1] + ref[i - 1][1] == hyp[j - 1][1]:
                c.append((D[i - 2][j - 1] + MERGE_COST, "merge", 2, 1))
            if i and j > 1 and ref[i - 1][1] == hyp[j - 2][1] + hyp[j - 1][1]:
                c.append((D[i - 1][j - 2] + MERGE_COST, "merge", 1, 2))
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
    sub {ref, heard}, del {ref} (labelled, not heard), ins {heard} (heard, not labelled). `edge` marks what lies
    before the first or after the last word heard as labelled: a neighbouring sound the window caught, or a labelled
    word cut off at the window's end. counts: per op, edges excluded."""
    tok = lambda s: [(w, sk) for w, _ in core.tokens(s) if (sk := core.rasm(w))]
    segs = _align(tok(text), tok(transcript)) if transcript else []
    core_at = [k for k, s in enumerate(segs) if s["op"] == "same"]
    for k, s in enumerate(segs):
        if core_at and (k < core_at[0] or k > core_at[-1]):
            s["edge"] = True
    counts = {k: sum(s["op"] == k and not s.get("edge") for s in segs) for k in ("same", "sub", "del", "ins")}
    counts["edge"] = sum(bool(s.get("edge")) for s in segs)
    return {"segments": segs, "counts": counts, "summary": diff_summary(counts, bool(transcript))}


def diff_summary(c: dict, heard: bool) -> str:
    if not heard:
        return "No transcript to compare"
    parts = [f"{c['sub']} changed" if c["sub"] else "", f"{c['del']} not heard" if c["del"] else "", f"{c['ins']} extra" if c["ins"] else ""]
    n, e = c["sub"] + c["del"] + c["ins"], c.get("edge", 0)
    edges = f"{e} word{'s' if e != 1 else ''} at the window's edges"
    if not n:
        return f"Matches the label apart from {edges}" if e else "Every labelled word heard, nothing extra"
    return f"{n} word{'s' if n != 1 else ''} differ{'s' if n == 1 else ''}: " + ", ".join(p for p in parts if p) + (f"; also {edges}" if e else "")


# ---- plain words ---------------------------------------------------------------------------------------------------

def _n(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def score_line(r: dict) -> tuple[str, str]:
    """(score, how it was measured) for one model's record."""
    d, dec = r.get("diff"), r.get("decision")
    if r.get("unit") == "letters" or "reason" in r:  # a transcript reading (the judge leaves out the unit when it had nothing to read)
        how = "Letters of the rasm skeleton between what it heard and each text; the gap needed grows with how far apart the texts are where it heard them."
        if r.get("claimed") is None or d is None:
            return {"empty transcript": "It heard nothing", "no alternative text": "No alternative text to compare"}.get(r.get("reason"), "No usable transcript"), how
        dist = f"{r['claimed']} letter{'' if r['claimed'] == 1 else 's'} off the label, {r['alternative']} off the alternative"
        need, against = r.get("need"), r.get("needAgainst", r.get("need"))
        if r.get("reason") == "transcript far from both texts":
            return f"What it heard fits neither text: {dist}", how
        if r.get("reason") == "transcript covers too little of the window":
            return f"It heard too little of the window: {dist}", how
        if dec == "match":
            return f"{dist} · label closer by {d}, needs {_n(need)}", how
        if dec == "mismatch":
            return f"{dist} · alternative closer by {-d}, needs {_n(against)}", how
        return f"{dist} · too close to call: {'label' if d >= 0 else 'alternative'} closer by {abs(d)}, needs {_n(need if d >= 0 else against)}", how
    how = f"Log-likelihood of each text given the sound, in nats; {_n(MARGIN)} needed either way."
    if d is None:
        return {"match": "Only the label can fit this audio", "mismatch": "The label can't fit this audio"}.get(dec, "Couldn't score either text"), how
    if dec == "match":
        return f"label fits better by {d:.1f}", how
    if dec == "mismatch":
        return f"alternative fits better by {-d:.1f}", how
    return f"too close to call: {'label' if d >= 0 else 'alternative'} ahead by {abs(d):.1f}, needs {_n(MARGIN)}", how


def why_line(outcome: str | None, reasons: list[str]) -> str:
    """Why the item is in the queue, in plain words; with no reasons (a queue the triage didn't build), what the
    triage of the same window found."""
    if not reasons:
        o = OUTCOME.get(outcome, "")
        return f"Triage of the same window: {o[:1].lower()}{o[1:]}" if o else ""
    head = f"In the queue because {REASON.get(reasons[0], reasons[0])}"
    return head + "".join(f" · also {REASON.get(r, r)}" for r in reasons[1:])


def model_view(model: str, r: dict, text: str, in_rule: bool = True) -> dict:
    score, how = score_line(r)
    dec = r.get("decision", "unsure")
    t = r.get("transcript")
    return {"model": model, "name": f"{NAMES[model]} ({ROLES[model]})", "short": NAMES[model], "role": ROLES[model],
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
            "models": [model_view(m, r, item.get("text", ""), inr) for m, r, inr in rec["models"]]}
