"""Two-model triage of a lab test set (.scratch/lab/issues/17-two-model-triage.md, spec decisions 16-17).

Every case of the test set is scored by two independent likelihood arms, each comparing the labelled text (claimed)
with one alternative text over the same audio window:

- planted edit (splice): what was said vs the original text at that spot (the edit undone, review_asr_core.revert_text);
- repeat or restart: with vs without the repeated words (also revert_text);
- clean stretch: the labelled text vs a one-word substitution or insertion (review_asr_core.perturb_text, seeded per
  case id; see CLEAN_OPS).

Nemotron 3.5 Transcribe (LM pairwise, plain spellings) and FastConformer (CTC pairwise, diacritised spellings) each say
match / mismatch / unsure with their committed 2-nat margin. Both match -> auto-accepted; both mismatch ->
both-mismatch (you confirm); one each -> disagree; anything else -> unsure. Everything but auto-accepted goes to the
listening queue, plus a fixed seeded spot check of the auto-accepted cases (SPOT_EDITS edits and repeats, SPOT_CLEAN
clean windows) and a seeded seam spot check of the planted edits (SEAM_SPOT of them, at most SEAM_CAP).

Pure logic, standard library only; run by modal/lab_triage.py, tested by tools/test_lab_triage.py.
"""

import math
import random
from collections import Counter

import lab_sessions as LS
import review_asr_core as core

VERSION = 1
MARGINS = {"nemotron": core.LIK_MARGIN, "fastconformer": core.LIK_MARGIN}  # nats, both arms' committed calibration margin
# A fixed stratified spot check, not a share: lab-v1.1's 10% of auto-accepted cases put 144 random items in a
# 257-item queue, mostly clean windows both models had already accepted.
SPOT_EDITS, SPOT_CLEAN = 15, 15  # auto-accepted edits and repeats; auto-accepted clean windows
SEAM_SPOT, SEAM_CAP, SEED = 0.10, 20, 17
# From lab-v1.4 the seam spot check has two seeded pools: splices that put donor audio in (wrong and added words,
# mutashabihat swaps: 3 of lab-v1.3's 12 seam spot checks were bad, every bad splice so far was one of these) and the
# rest (omissions, skips, repeats and restarts, never bad), each a share of its planted edits up to a cap.
INSERTED = ("wrong_word", "added_word", "mutashabihat")
SEAM_POOLS = {"inserted": (0.20, 30), "other": (SEAM_SPOT, SEAM_CAP)}
BAD_PER = 40  # at most 1 bad verdict per this many queue items (spec decision 16 scaled)
CLEAN_TARGET_S, CLEAN_MIN_WORDS = 8.0, 3
# No deletions: a peaky CTC model drops a clearly heard word for a few blank frames' cost, less than one imperfect
# spelling costs, so FastConformer preferred the shorter text on 3 of 19 clean deletions of the lab-v1.1 smoke run
# (13-21 nats, the word in its own greedy output) and on none of 32 substitutions and insertions.
CLEAN_OPS = ("sub", "ins")
OUTCOMES = ("auto-accepted", "both-mismatch", "disagree", "unsure")
REASONS = ("both-mismatch", "disagree", "unsure", "seam", "spot")
TITLES = {
    "both-mismatch": "Both models think the label is wrong — please confirm",
    "disagree": "The two models disagree",
    "unsure": "A model couldn't tell — please check",
    "seam": "Seam spot check",
    "spot": "Random spot check",
}
ALSO = {"seam": "a seam spot check", "spot": "a random spot check"}
ALT_NAME = {"edit": "the text without the planted mistake", "repeat": "the text without the repeat", "clean": "a one-word change of the text"}
MODEL_NAME = {"nemotron": "Nemotron", "fastconformer": "FastConformer", "soniox": "Soniox", "speechmatics": "Speechmatics"}
MODELS = ("nemotron", "fastconformer")


# ---- cases ---------------------------------------------------------------------------------------------------------

def edit_class(x: dict) -> str:
    return "mutashabihat" if x.get("mutashabihat") else x["type"]


def clean_runs(s: dict) -> list[list[dict]]:
    """Runs of unplanted words with no seam inside (tools/lab_checks.py `stretches` on the labels): the k-th run here is
    checks-v1.json's `<session>-clean-<k>`."""
    from bisect import bisect_right

    seams = sorted({g["startS"] for g in s["segments"][1:]})
    out, cur, seg = [], [], None
    for w in sorted(s["said"], key=lambda w: (w["startS"], w["endS"])):
        k = bisect_right(seams, w["startS"])
        straddles = k < len(seams) and seams[k] < w["endS"]
        if w["role"] != "ref" or straddles or k != seg:
            if cur:
                out.append(cur)
            cur = []
        seg = k
        if w["role"] == "ref" and not straddles:
            cur.append(w)
    if cur:
        out.append(cur)
    return out


def split_run(run: list[dict], target: float = CLEAN_TARGET_S, min_words: int = CLEAN_MIN_WORDS) -> list[list[dict]]:
    """A clean run cut at word boundaries into round(duration / target) windows of about equal length (by word middle);
    a window with fewer than min_words words joins the one before it (or after, for the first)."""
    a, b = run[0]["startS"], run[-1]["endS"]
    n = max(1, round((b - a) / target))
    groups = [[] for _ in range(n)]
    for w in run:
        groups[min(n - 1, int(((w["startS"] + w["endS"]) / 2 - a) / max(b - a, 1e-9) * n))].append(w)
    out = []
    for g in (g for g in groups if g):
        if out and (len(g) < min_words or len(out[-1]) < min_words):
            out[-1] = out[-1] + g
        else:
            out.append(g)
    return out


def clean_item(s: dict, words: list[dict], case_id: str, root: str) -> dict:
    a, b = LS.whole_words(s, words[0]["startS"], words[-1]["endS"])
    return {"id": case_id, "kind": "case", "title": "Recited as written, no mistake",
            "audio": {"path": f"{root}/{s['meta']['audio']['studio']}", "startS": a, "endS": b},
            "text": LS.said_text(s, a, b), "label": {"type": "clean"},
            "context": f"session {s['id']} at {words[0]['startS']:.1f} s · {s['meta']['reciter']}"}


def perturbation(case_id: str, text: str, pool: list[str], seed: int = SEED) -> dict:
    """The clean window's alternative: one interior word substituted by a clearly different word of similar length,
    or a foreign word inserted (CLEAN_OPS), the operation and position seeded by the case id alone. Falls back to a
    deletion only when the session has no suitable foreign word."""
    rng = random.Random(f"{seed}:{case_id}:perturb")
    return core.perturb_text(text, pool, rng.choice(CLEAN_OPS), rng)


def session_cases(s: dict, root: str, seed: int = SEED) -> list[dict]:
    """Every case of one session, studio copy: each planted edit (the review_splice / review_repeat window, ids
    `<session>-<edit>` as in the v1 listening check) and each clean window (`<session>-clean-<k>-<j>`)."""
    out = []
    for x in s["splices"]:
        rep = x["type"] in ("repeat", "restart")
        item = LS.review_repeat(s, x, root) if rep else LS.review_splice(s, x, root)
        alt = core.revert_text(item, s)
        out.append({"id": item["id"], "session": s["id"], "group": "repeat" if rep else "edit", "class": edit_class(x),
                    "linked": LS.linked(x), "item": item, "claimed": item["text"],
                    "alternative": alt and alt["text"], "alternativeOp": alt and alt["op"]})
    pool = sorted({w["text"] for w in s["said"]})
    for k, run in enumerate(clean_runs(s)):
        for j, words in enumerate(split_run(run)):
            item = clean_item(s, words, f"{s['id']}-clean-{k}-{j}", root)
            alt = perturbation(item["id"], item["text"], pool, seed)
            out.append({"id": item["id"], "session": s["id"], "group": "clean", "class": "clean", "linked": False, "item": item,
                        "claimed": item["text"], "alternative": alt["text"], "alternativeOp": alt["op"]})
    return out


def reviewed_clean_cases(queue: dict, sessions: dict[str, dict], have: set[str], seed: int = SEED) -> list[dict]:
    """Clean windows of an earlier listening queue (the v1 check) that aren't triage cases, scored the same way so the
    comparison with your verdicts covers all of them. Studio or phone, as they were heard."""
    out = []
    for it in queue.get("items", []):
        if it["id"] in have or it["label"]["type"] != "clean" or core.session_id(it) not in sessions:
            continue
        s = sessions[core.session_id(it)]
        alt = perturbation(it["id"], it["text"], sorted({w["text"] for w in s["said"]}), seed)
        out.append({"id": it["id"], "session": s["id"], "group": "clean", "class": "clean", "linked": False, "item": it,
                    "claimed": it["text"], "alternative": alt["text"], "alternativeOp": alt["op"], "reviewedOnly": True})
    return out


# ---- decisions -----------------------------------------------------------------------------------------------------

def pairwise(claimed: float | None, alternative: float | None, margin: float) -> dict:
    """match / mismatch / unsure from two scores. A side with no score (no CTC path through the window, e.g. a skipped
    ayah put back into a few seconds of audio) counts as infinitely worse; neither side scored is unsure."""
    if claimed is None and alternative is None:
        d, diff = "unsure", None
    elif alternative is None:
        d, diff = "match", None
    elif claimed is None:
        d, diff = "mismatch", None
    else:
        diff = round(claimed - alternative, 2)
        d = core.prefer(claimed, alternative, margin)
    return {"decision": d, "claimed": claimed, "alternative": alternative, "diff": diff}


def outcome(a: str, b: str) -> str:
    if a == b == "match":
        return "auto-accepted"
    if a == b == "mismatch":
        return "both-mismatch"
    if {a, b} == {"match", "mismatch"}:
        return "disagree"
    return "unsure"


def spot_sample(ids, n: int, seed: int, tag: str) -> list[str]:
    """n seeded ids (all of them when fewer), independent of their order."""
    pool = sorted(set(ids))
    return sorted(random.Random(f"{seed}:{tag}").sample(pool, min(len(pool), n)))


def seam_count(n: int, frac: float = SEAM_SPOT, cap: int = SEAM_CAP) -> int:
    """Seam spot checks for n planted edits: a share rounded up, at most cap."""
    return min(cap, math.ceil(frac * n))


def max_bad(n: int, per: int = BAD_PER) -> int:
    """The acceptance bar scaled to the queue: at most 1 bad per `per` items, never below 1."""
    return max(1, n // per)


# ---- queue ---------------------------------------------------------------------------------------------------------

def _model_line(name: str, r: dict, group: str) -> str:
    m = MODEL_NAME[name]
    alt = ALT_NAME[group]
    if r.get("unit") == "letters":
        return _transcript_line(m, r, alt)
    if r["diff"] is None:
        return {"match": f"{m}: only the label can fit this audio", "mismatch": f"{m}: the label can't fit this audio",
                "unsure": f"{m}: couldn't score either text"}[r["decision"]]
    d = r["diff"]
    if r["decision"] == "match":
        return f"{m}: the label fits better than {alt} by {d:.1f}"
    if r["decision"] == "mismatch":
        return f"{m}: {alt} fits better than the label by {-d:.1f}"
    return f"{m}: can't separate the label from {alt} ({d:+.1f})"


def _transcript_line(m: str, r: dict, alt: str) -> str:
    """A label-blind transcript read by review_api_core.judge (letter distances on the rasm skeleton)."""
    if r["diff"] is None:
        return f"{m}: no usable transcript"
    if r.get("reason") == "transcript far from both texts":
        return f"{m}: what it heard fits neither text"
    if r.get("reason") == "transcript covers too little of the window":
        return f"{m}: it heard too little of the window"
    d = r["diff"]
    if r["decision"] == "match":
        return f"{m}: what it heard is closer to the label than to {alt}, by {d} letters"
    if r["decision"] == "mismatch":
        return f"{m}: what it heard is closer to {alt} than to the label, by {-d} letters"
    return f"{m}: what it heard can't separate the label from {alt} ({d:+d} letters)"


def why(case: dict, reasons: list[str], spot: tuple[int, int] = (SPOT_EDITS, SPOT_CLEAN), models: tuple[str, str] = MODELS) -> str:
    """The plain sentence at the start of a queue item's context."""
    parts = []
    if reasons[0] in ("both-mismatch", "disagree", "unsure"):
        nats = [case[m].get("unit") != "letters" for m in models]
        lines = [_model_line(m, case[m], case["group"]) + (" nats" if n and not all(nats) and case[m]["diff"] is not None else "")
                 for m, n in zip(models, nats)]
        parts.append("; ".join(lines) + (" (nats)" if all(nats) else ""))
    if "seam" in reasons:
        parts.append("Seam: listen to the join (a click, a jump in level or room sound, a clipped start, an unnatural join)")
    if reasons[0] == "spot":
        n, what = (spot[1], "clean windows") if case["group"] == "clean" else (spot[0], "edits and repeats")
        parts.append(f"Both models accepted this label; {n} accepted {what} are picked at random to check by ear")
    return " · ".join(parts)


def model_summary(r: dict) -> dict:
    """A model's record as a queue item carries it: decision and diff always; a transcript reading's letter distances,
    margins and reason, and the model's label-blind transcript, when it has them (the review page's model panel)."""
    keep = ("reason", "need", "needAgainst", "unit", "transcript") + (("claimed", "alternative") if r.get("unit") == "letters" else ())
    return {"decision": r["decision"], "diff": r["diff"], **{k: r[k] for k in keep if r.get(k) is not None}}


def queue_item(case: dict, reasons: list[str], carried: dict | None = None, spot: tuple[int, int] = (SPOT_EDITS, SPOT_CLEAN),
               models: tuple[str, str] = MODELS) -> dict:
    it = case["item"]
    title = TITLES[reasons[0]] + "".join(f" · also {ALSO[r]}" for r in reasons[1:] if r in ALSO)
    ctx = " · ".join(x for x in (why(case, reasons, spot, models), it.get("title"), it.get("context")) if x)
    tri = {"reasons": reasons, "outcome": case["outcome"], "class": case["class"],
           **{m: model_summary(case[m]) for m in models},
           "alternative": case["alternative"], "alternativeOp": case["alternativeOp"]}
    if carried:
        tri["carriedOver"] = carried
    return {**{k: v for k, v in it.items() if k != "title"}, "title": title, "context": ctx, "triage": tri}


def reasons_for(case: dict, spot: set[str], seam: set[str]) -> list[str]:
    out = [case["outcome"]] if case["outcome"] != "auto-accepted" else []
    if case["id"] in seam:
        out.append("seam")
    if case["id"] in spot:
        out.append("spot")
    return out


def carried_over(item: dict, reviews: dict[str, dict], earlier: dict[str, dict]) -> dict | None:
    """An earlier verdict on this id (verdicts are keyed by test set and id, so the review page shows it already). Also
    says whether the earlier queue played the same window and text."""
    r = reviews.get(item["id"])
    if not r:
        return None
    prev = earlier.get(item["id"])
    same = bool(prev) and prev["audio"] == item["audio"] and prev["text"] == item["text"]
    return {"verdict": r["verdict"], "note": r.get("note", ""), "at": r.get("at"), "queue": r.get("queue"), "sameWindow": same,
            "reviewer": r.get("reviewer")}


def seam_pool(case: dict) -> str:
    return "inserted" if case["class"] in INSERTED else "other"


def seam_sample(planted: list[dict], seam_spot, seed: int, test_set: str) -> set[str]:
    """Seam spot check ids. seam_spot: (share, cap) over every planted edit (lab-v1.1 to lab-v1.3), or {pool: (share,
    cap)} drawn per seam_pool, each pool seeded on its own."""
    if isinstance(seam_spot, dict):
        out = set()
        for pool, (share, cap) in sorted(seam_spot.items()):
            ids = [c["id"] for c in planted if seam_pool(c) == pool]
            out |= set(spot_sample(ids, seam_count(len(ids), share, cap), seed, f"{test_set}:seam:{pool}"))
        return out
    ids = [c["id"] for c in planted]
    return set(spot_sample(ids, seam_count(len(ids), *seam_spot), seed, f"{test_set}:seam"))


def triage(cases: list[dict], scores: dict[str, dict[str, dict]], test_set: str, spot: tuple[int, int] = (SPOT_EDITS, SPOT_CLEAN),
           seam_spot: tuple[float, int] | dict = (SEAM_SPOT, SEAM_CAP), seed: int = SEED, reviews: dict[str, dict] | None = None,
           earlier: dict[str, dict] | None = None, models: tuple[str, str] = MODELS) -> dict:
    """cases: session_cases (+ reviewed_clean_cases, flagged reviewedOnly). scores: model -> case id -> {claimed,
    alternative} (None when the model gave no score), or a decision already made ({decision, diff, ...}, e.g. a
    transcript read by review_api_core.judge); any of them may carry the model's label-blind `transcript`, which the
    case and its queue item keep. spot: (edits and repeats, clean windows) drawn from the
    auto-accepted cases; seam_spot: (share, cap) of the planted edits, or per pool (seam_sample). Returns {cases,
    queue, counts}; reviewedOnly cases are decided but never counted or queued."""
    reviews, earlier = reviews or {}, earlier or {}
    out = []
    for c in cases:
        r = {**{k: c[k] for k in c if k != "item"}, "item": c["item"]}
        for m in models:
            sc = scores[m].get(c["id"]) if c["alternative"] is not None else None
            r[m] = (sc if sc and "decision" in sc else pairwise(sc["claimed"], sc["alternative"], MARGINS[m]) if sc
                    else {"decision": "unsure", "claimed": None, "alternative": None, "diff": None})
            heard = (scores[m].get(c["id"]) or {}).get("transcript")
            if heard is not None and "transcript" not in r[m]:
                r[m] = {**r[m], "transcript": heard}
        r["outcome"] = outcome(r[models[0]]["decision"], r[models[1]]["decision"])
        out.append(r)
    real = [c for c in out if not c.get("reviewedOnly")]
    accepted = [c for c in real if c["outcome"] == "auto-accepted"]
    spot_ids = set(spot_sample([c["id"] for c in accepted if c["group"] != "clean"], spot[0], seed, f"{test_set}:spot:edit"))
    spot_ids |= set(spot_sample([c["id"] for c in accepted if c["group"] == "clean"], spot[1], seed, f"{test_set}:spot:clean"))
    seam_ids = seam_sample([c for c in real if c["group"] != "clean"], seam_spot, seed, test_set)
    items = []
    for c in real:
        c["spotCheck"], c["seamSpotCheck"] = c["id"] in spot_ids, c["id"] in seam_ids
        c["queued"] = reasons_for(c, spot_ids, seam_ids)
        if c["queued"]:
            items.append(queue_item(c, c["queued"], carried_over(c["item"], reviews, earlier), spot, models))
    items.sort(key=lambda it: (REASONS.index(it["triage"]["reasons"][0]), it["id"]))
    queue = {"version": 1, "testSet": test_set, "maxBad": max_bad(len(items)), "items": items}
    return {"cases": out, "queue": queue, "counts": counts(real, items, models), "spotIds": sorted(spot_ids), "seamIds": sorted(seam_ids)}


def confirm_queue(queue: dict) -> dict:
    """The both-mismatch items of a triage queue on their own, to hear first: diagnostic, so no bad item is allowed."""
    return {**queue, "maxBad": 0, "items": [it for it in queue["items"] if it["triage"]["reasons"][0] == "both-mismatch"]}


def counts(cases: list[dict], items: list[dict], models: tuple[str, str] = MODELS) -> dict:
    by = {}
    for c in cases:
        row = by.setdefault(c["class"], {"n": 0, **{o: 0 for o in OUTCOMES}})
        row["n"] += 1
        row[c["outcome"]] += 1
    groups = {}
    for c in cases:
        row = groups.setdefault(c["group"], {"n": 0, **{o: 0 for o in OUTCOMES}})
        row["n"] += 1
        row[c["outcome"]] += 1
    first = Counter(it["triage"]["reasons"][0] for it in items)
    anyr = Counter(r for it in items for r in it["triage"]["reasons"])
    per_model = {m: dict(Counter(c[m]["decision"] for c in cases)) for m in models}
    return {"cases": len(cases), "outcomes": dict(Counter(c["outcome"] for c in cases)), "byClass": dict(sorted(by.items())),
            "byGroup": groups, "perModel": per_model,
            "queue": {"items": len(items), "byFirstReason": {r: first.get(r, 0) for r in REASONS},
                      "byAnyReason": {r: anyr.get(r, 0) for r in REASONS},
                      "seamByPool": dict(sorted(Counter(seam_pool(it["triage"]) for it in items if "seam" in it["triage"]["reasons"]).items())),
                      "carriedOver": sum("carriedOver" in it["triage"] for it in items), "maxBad": max_bad(len(items))}}


def compare_reviewed(cases: list[dict], queue: dict, reviews: dict[str, dict], models: tuple[str, str] = MODELS) -> dict:
    """Your earlier verdicts against the triage outcome of the same ids (same window and text where `sameWindow`)."""
    by = {c["id"]: c for c in cases}
    rows = {}
    for it in queue.get("items", []):
        c, v = by.get(it["id"]), reviews.get(it["id"], {}).get("verdict")
        if not c or not v:
            continue
        rows[it["id"]] = {"verdict": v, "outcome": c["outcome"], "class": c["class"], **{m: c[m]["decision"] for m in models},
                          **{f"{m}Diff": c[m]["diff"] for m in models}, "reviewedOnly": bool(c.get("reviewedOnly")),
                          "sameWindow": c["item"]["audio"] == it["audio"] and c["item"]["text"] == it["text"]}
    ok = [r for r in rows.values() if r["verdict"] == "ok"]
    return {"n": len(rows), "ok": len(ok), "outcomesOnOk": dict(Counter(r["outcome"] for r in ok)),
            "falseAlarms": sorted(i for i, r in rows.items() if r["verdict"] == "ok" and r["outcome"] == "both-mismatch"), "items": rows}
