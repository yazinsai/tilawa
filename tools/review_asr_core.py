"""Pure logic for the independent audio opinion (.scratch/lab/issues/12-review-asr-calibration.md, docs/agents/audio-review.md).

The ASR models in modal/review_asr.py transcribe each review window without seeing its text or label. This module then
compares the transcript with the labelled text word by word, builds the negative controls, scores calibration and picks
the pre-screen subset. Standard library only; tested by tools/test_review_asr.py.

The rule below was fixed before the first calibration run on lab-v1.1.
"""

import math
import random
import re
import unicodedata

OK_SIM, WRONG_SIM = 0.75, 0.5  # word similarity: >= OK_SIM matches, < WRONG_SIM is a different word, between is doubtful
MERGE_COST, MERGE_SIM = 0.15, 0.9  # one word heard as two (or two as one), e.g. يَـٰٓأَيُّهَا vs "يا أيها"; must match closely
EDGE_FREE = 1  # heard words allowed before the first / after the last labelled word (a neighbouring sound)
DECISIONS = ("match", "mismatch", "unsure")

_DROP = re.compile("[\u0610-\u061a\u064b-\u065f\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed\u0640]")
_MAP = str.maketrans({"\u0671": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "\u0672": "ا", "\u0673": "ا", "ؤ": "و", "ئ": "ي", "ى": "ي",
                      "ة": "ه", "ء": None, "ی": "ي", "ک": "ك"})
_LETTERS = re.compile("[^\u0621-\u064a]")


def norm_variants(word: str) -> set[str]:
    """Undiacritised forms of one word. The superscript alef (Uthmani ـٰ) is tried as ا and as nothing, so ذَٰلِكَ
    matches both ذالك and ذلك."""
    w = unicodedata.normalize("NFC", word)
    out = set()
    for dagger in ("ا", ""):
        v = _DROP.sub("", w.replace("\u0670", dagger)).translate(_MAP)
        if v := _LETTERS.sub("", v):
            out.add(v)
    return out


def norm_words(text: str) -> list[str]:
    """One normalised form per word, for comparing whole texts."""
    return [min(v) for _, v in tokens(text)]


def tokens(text: str) -> list[tuple[str, set[str]]]:
    """(original word, its normalised variants) for every word that has Arabic letters."""
    out = []
    for t in re.split(r"[\s\u060c\u061b\u061f.,!?;:«»()\[\]\"'\-]+", text or ""):
        if v := norm_variants(t):
            out.append((t, v))
    return out


def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: set[str], b: set[str]) -> float:
    """Best 1 - edit distance / longer length over the variants of two words."""
    return max(1 - lev(x, y) / max(len(x), len(y)) for x in a for y in b)


def _cat(a: set[str], b: set[str]) -> set[str]:
    return {x + y for x in a for y in b}


def align(ref: list[tuple[str, set[str]]], hyp: list[tuple[str, set[str]]]) -> list[dict]:
    """Word alignment of the labelled text (ref) with the transcript (hyp). EDGE_FREE transcript words before the first and
    after the last labelled word are free (the window may catch a neighbouring sound). Ops: pair, merge (1 ref : 2 hyp or 2 : 1),
    delete (ref word not heard), insert (heard word not in the text)."""
    n, m, inf = len(ref), len(hyp), float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        D[0][j], B[0][j] = float(max(0, j - EDGE_FREE)), ("lead",)
    for i in range(1, n + 1):
        for j in range(m + 1):
            cands = [(D[i - 1][j] + 1, ("del",))]
            if j:
                s = sim(ref[i - 1][1], hyp[j - 1][1])
                cands += [(D[i - 1][j - 1] + 1 - s, ("pair", s)), (D[i][j - 1] + 1, ("ins",))]
            if j > 1 and (s := sim(ref[i - 1][1], _cat(hyp[j - 2][1], hyp[j - 1][1]))) >= MERGE_SIM:
                cands.append((D[i - 1][j - 2] + 1 - s + MERGE_COST, ("merge12", s)))
            if i > 1 and j and (s := sim(_cat(ref[i - 2][1], ref[i - 1][1]), hyp[j - 1][1])) >= MERGE_SIM:
                cands.append((D[i - 2][j - 1] + 1 - s + MERGE_COST, ("merge21", s)))
            D[i][j], B[i][j] = min(cands, key=lambda c: c[0])
    j = min(range(m + 1), key=lambda k: D[n][k] + max(0, m - k - EDGE_FREE))
    ops = [{"op": "trail", "hyp": hyp[k][0], "ri": n - 0.5, "free": k - j < EDGE_FREE} for k in range(m - 1, j - 1, -1)]
    i = n
    while i > 0 or j > 0:
        b = B[i][j]
        if b[0] == "lead":
            ops += [{"op": "lead", "hyp": hyp[k][0], "ri": -0.5, "free": j - k <= EDGE_FREE} for k in range(j - 1, -1, -1)]
            break
        if b[0] == "del":
            ops.append({"op": "del", "ref": ref[i - 1][0], "ri": i - 1})
            i -= 1
        elif b[0] == "ins":
            ops.append({"op": "ins", "hyp": hyp[j - 1][0], "ri": i - 0.5})
            j -= 1
        elif b[0] == "pair":
            ops.append({"op": "pair", "ref": ref[i - 1][0], "hyp": hyp[j - 1][0], "sim": round(b[1], 3), "ri": i - 1})
            i, j = i - 1, j - 1
        elif b[0] == "merge12":
            ops.append({"op": "merge", "ref": ref[i - 1][0], "hyp": f"{hyp[j - 2][0]} {hyp[j - 1][0]}", "sim": round(b[1], 3), "ri": i - 1})
            i, j = i - 1, j - 2
        else:
            ops.append({"op": "merge", "ref": f"{ref[i - 2][0]} {ref[i - 1][0]}", "hyp": hyp[j - 1][0], "sim": round(b[1], 3), "ri": i - 1})
            i, j = i - 2, j - 1
    return ops[::-1]


def compare(text: str, transcript: str, before: str = "", after: str = "") -> dict:
    """match / mismatch / unsure for one window, with the word alignment.

    mismatch: inside the window a labelled word is not heard, a heard word (2+ letters) is not in the text, or a word is
    heard as a different word (similarity < WRONG_SIM). unsure: only doubtful words (between WRONG_SIM and OK_SIM),
    one-letter extras, or labelled words missing at the very start or end of the window (cut-off or swallowed by the
    model). match: every labelled word heard, nothing extra inside.

    `before` / `after`: words known to be just outside the window when the audio given to the model was widened. They
    are aligned too but never count; only the labelled words and extras between them decide."""
    pre, mid, post, hyp = tokens(before), tokens(text), tokens(after), tokens(transcript)
    ref, nb, ne = pre + mid + post, len(pre), len(pre) + len(mid)
    if not mid:
        return {"decision": "unsure", "reasons": ["no labelled words"], "ops": []}
    if not hyp:
        return {"decision": "unsure", "reasons": ["empty transcript"], "ops": []}
    ops = align(ref, hyp)
    core = [k for k, o in enumerate(ops) if o["op"] in ("pair", "merge", "ins")]
    first, last = (core[0], core[-1]) if core else (len(ops), -1)
    wrong, doubt = [], []
    for k, o in enumerate(ops):
        if o["op"] == "del":
            o["status"] = "edge-missing" if k < first or k > last else "missing"
        elif o["op"] == "ins" or not o.get("free", True):
            o["status"] = "extra" if min(len(v) for v in norm_variants(o["hyp"])) >= 2 else "extra-letter"
        elif o["op"] in ("pair", "merge"):
            o["status"] = "ok" if o["sim"] >= OK_SIM else "doubtful" if o["sim"] >= WRONG_SIM else "wrong"
        else:
            o["status"] = "outside"
        if (pre or post) and o["status"] != "outside":
            r = o["ri"]
            if o["op"] in ("ins", "lead", "trail") and not nb - 0.5 <= r <= ne - 0.5 or o["op"] not in ("ins", "lead", "trail") and not nb <= r < ne:
                o["status"] = "context"
            elif o["op"] in ("ins", "lead", "trail") and r in (nb - 0.5, ne - 0.5) and o["status"] == "extra":
                o["status"] = "edge-extra"
        if o["status"] in ("missing", "extra", "wrong"):
            wrong.append(o)
        elif o["status"] in ("edge-missing", "edge-extra", "extra-letter", "doubtful"):
            doubt.append(o)
    reasons = [f"{o['status']}: {o.get('ref', '')}{' / ' if 'ref' in o and 'hyp' in o else ''}{o.get('hyp', '')}" for o in wrong + doubt]
    decision = "mismatch" if wrong else "unsure" if doubt else "match"
    return {"decision": decision, "reasons": reasons, "ops": ops,
            "refWords": len(mid), "heardWords": len(hyp), "okWords": sum(o.get("status") == "ok" for o in ops)}


def combine(decisions: list[str]) -> str:
    """Consensus of several models: all match -> match; any mismatch and none match -> mismatch; otherwise unsure."""
    if decisions and all(d == "match" for d in decisions):
        return "match"
    if "mismatch" in decisions and "match" not in decisions:
        return "mismatch"
    return "unsure"


_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    """The generation-loop guardrail from the Nemotron 3.5 Transcribe model card, applied to its reported transcripts."""
    return _BLOCK_RUN.sub(lambda m: m.group(1), _CHAR_RUN.sub(lambda m: m.group(1), text))


# Likelihood arm (Nemotron 3.5 Transcribe): the model scores given candidate texts against the audio instead of
# transcribing. The candidates are exposed to the model, so this is a likelihood comparison, not an independent opinion.

LIK_MARGIN = 2.0  # nats; fixed before the first likelihood run
MAX_SPELLINGS = 8

_MARKS = re.compile("[\u0610-\u061a\u064b-\u0653\u0656-\u065f\u06d6-\u06ed\u0640]")


def plain_spellings(word: str) -> list[str]:
    """Modern undiacritised spellings of one Uthmani word, e.g. ٱلصَّلَوٰةِ -> الصلاة, ءَامَنُوا۟ -> آمنوا. A superscript alef
    that is not after ى or و is tried both as ا and as nothing (ٱلْكِتَـٰبُ -> الكتاب, ذَٰلِكَ -> ذلك)."""
    w = re.sub("\u0650([\u0651\u0652]?)ى", "\u0650\\1ي", unicodedata.normalize("NFD", word))
    w = unicodedata.normalize("NFC", _MARKS.sub("", w).replace("\u0671", "ا"))
    w = re.sub("ى(?=[\u0621-\u064a])", "ي", w.replace("ى\u0670", "ى").replace("و\u0670", "ا"))
    out = []
    for dagger in ("ا", "") if "\u0670" in w else ("",):
        v = w.replace("\u0670", dagger).replace("\u0654ا", "آ").replace("\u0654", "أ").replace("ءا", "آ")
        v = re.sub("[^\u0621-\u064a]", "", v)
        if v and v not in out:
            out.append(v)
    return out


def plain_candidates(text: str, transcript: str = "") -> list[str]:
    """Spellings of a candidate text in the model's writing system, for scoring. A word that has a normalised form in
    common with a transcript word takes that word's spelling (the same word, as the model writes it); other words try
    their plain_spellings. At most MAX_SPELLINGS combinations, varying the earliest ambiguous words."""
    heard = tokens(transcript)
    per = []
    for w, v in tokens(text):
        same = next((t for t, tv in heard if tv & v), None)
        per.append([re.sub("[^\u0621-\u064a]", "", unicodedata.normalize("NFC", same))] if same else plain_spellings(w) or [w])
    out = [[]]
    for opts in per:
        out = [o + [s] for o in out for s in opts][:MAX_SPELLINGS] if len(out) * len(opts) <= MAX_SPELLINGS else [o + [opts[0]] for o in out]
    return [" ".join(o) for o in out]


def plain_transcript(transcript: str) -> str:
    """The transcript's Arabic words without punctuation, scored the same way as the candidates."""
    return " ".join(re.sub("[^\u0621-\u064a]", "", unicodedata.normalize("NFC", t)) for t, _ in tokens(transcript))


def prefer(claimed: float, alternative: float, margin: float = LIK_MARGIN) -> str:
    """Pairwise rule: does the claimed text fit the audio better than the alternative by at least `margin` nats?"""
    d = claimed - alternative
    return "match" if d >= margin else "mismatch" if d <= -margin else "unsure"


def near_best(claimed: float, best: float, margin: float = LIK_MARGIN) -> str:
    """Absolute rule: the claimed text is within `margin` nats of the model's own best hypothesis."""
    return "match" if claimed - best >= -margin else "mismatch"


# Negative controls: the same audio paired with text that is wrong by construction.

def _session_words(session: dict, a: float, b: float) -> list[dict]:
    return [w for w in session["said"] if w["endS"] > a and w["startS"] < b]


def revert_text(item: dict, session: dict) -> dict | None:
    """The window's text with its planted edit undone, i.e. what the label would say if the edit had not been made:
    a wrong word back to the expected one, an added word or repeat removed, an omitted word or skipped passage put back.
    The audio still has the edit, so a good opinion says mismatch. None when the edit can't be located in the window."""
    edit_id = item["id"][len(session["id"]) + 1:]
    x = next((x for x in session["splices"] if x["id"] == edit_id), None)
    a, b = item["audio"]["startS"], item["audio"]["endS"]
    ws = _session_words(session, a, b)
    if x is None or " ".join(w["text"] for w in ws) != item["text"]:
        return None
    words = [w["text"] for w in ws]
    hit = [k for k, w in enumerate(ws) if w["role"] != "ref" and w["endS"] > x["startS"] and w["startS"] < x["endS"]]
    exp = (x.get("expected") or "").split()
    if x["type"] == "wrong_word" and hit:
        new = words[: hit[0]] + exp + words[hit[-1] + 1:]
        op = "planted word back to the expected one"
    elif x["type"] in ("added_word", "repeat", "restart") and hit:
        new = [w for k, w in enumerate(words) if k not in hit]
        op = "planted extra word(s) removed" if x["type"] == "added_word" else f"{x['type']} removed"
    elif x["type"] in ("omitted_word", "skipped_passage") and exp:
        mid = (x["startS"] + x["endS"]) / 2
        k = sum((w["startS"] + w["endS"]) / 2 < mid for w in ws)
        if not 0 < k < len(words):
            return None
        new = words[:k] + exp + words[k:]
        op = "omitted word put back" if x["type"] == "omitted_word" else "skipped passage put back"
    else:
        return None
    return {"text": " ".join(new), "control": "revert", "edit": x["type"], "op": op,
            "words": len(exp) if x["type"] in ("omitted_word", "skipped_passage", "wrong_word") else len(hit)}


def perturb_text(text: str, pool: list[str], op: str, rng: random.Random) -> dict:
    """One interior word of `text` substituted, deleted or a foreign word inserted ('sub', 'del', 'ins'). Pool words are
    clearly different from the word they replace (similarity < WRONG_SIM) and of similar length."""
    words = text.split()
    k = rng.randrange(1, len(words) - 1) if len(words) > 2 else len(words) // 2
    here = norm_variants(words[k]) or {""}
    nl = min(len(v) for v in here)
    cands = sorted({p for p in pool if (pv := norm_variants(p)) and abs(min(len(v) for v in pv) - nl) <= 2
                    and sim(pv, here) < WRONG_SIM and all(sim(pv, norm_variants(w) or {""}) < WRONG_SIM for w in words)})
    if op == "del" or not cands:
        return {"text": " ".join(words[:k] + words[k + 1:]), "control": "perturb", "edit": "del", "op": f"text drops {words[k]}", "words": 1}
    w = rng.choice(cands)
    if op == "sub":
        return {"text": " ".join(words[:k] + [w] + words[k + 1:]), "control": "perturb", "edit": "sub", "op": f"{words[k]} claimed as {w}", "words": 1}
    return {"text": " ".join(words[:k] + [w] + words[k:]), "control": "perturb", "edit": "ins", "op": f"text claims extra {w}", "words": 1}


def controls(queue: dict, sessions: dict[str, dict], seed: int = 12) -> dict[str, dict]:
    """Per item id: {'oneEdit': ..., 'shifted': ...}. oneEdit undoes the planted edit (splices, repeats, restarts) or, for
    clean windows, plants a one-word substitution / deletion / insertion in the text (cycling, seeded). shifted is the
    text of the next item whose words differ (a gross mismatch)."""
    rng = random.Random(f"{seed}:controls")
    items = queue["items"]
    pool = sorted({w for it in items for w in it["text"].split()})
    out, cycle = {}, ["sub", "ins", "del"]
    n_clean = 0
    for i, it in enumerate(items):
        s = sessions.get(session_id(it))
        one = revert_text(it, s) if s and it["label"]["type"] != "clean" else None
        if one is None:
            one = perturb_text(it["text"], pool, cycle[n_clean % 3], rng)
            n_clean += 1
        mine = norm_words(it["text"])
        k = next(k for k in range(1, len(items)) if norm_words(items[(i + k) % len(items)]["text"]) != mine)
        other = items[(i + k) % len(items)]
        out[it["id"]] = {"oneEdit": one, "shifted": {"text": other["text"], "control": "shifted", "edit": "other item", "op": f"text of {other['id']}"}}
    return out


def session_id(item: dict) -> str:
    """lab-v1.1/sessions/lab-v1.1-e20.studio.wav -> lab-v1.1-e20"""
    return item["audio"]["path"].rsplit("/", 1)[-1].rsplit(".", 2)[0]


# Calibration statistics.

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def score(decisions: dict[str, str], truth: dict[str, bool]) -> dict:
    """decisions: id -> match/mismatch/unsure; truth: id -> True when the text really matches the audio.
    specificity = matching windows accepted (match); sensitivity = wrong texts called mismatch; flagged = wrong texts not
    accepted (mismatch or unsure), the number that matters for a pre-screen since unsure goes to the listener."""
    pos = [i for i, t in truth.items() if t]
    neg = [i for i, t in truth.items() if not t]
    count = lambda ids, d: sum(decisions[i] == d for i in ids)
    return {"positives": {d: count(pos, d) for d in DECISIONS}, "negatives": {d: count(neg, d) for d in DECISIONS},
            "specificity": rate(count(pos, "match"), len(pos)), "falseAlarm": rate(count(pos, "mismatch"), len(pos)),
            "sensitivity": rate(count(neg, "mismatch"), len(neg)), "flagged": rate(len(neg) - count(neg, "match"), len(neg))}


# Pre-screen.

def prescreen(queue: dict, decisions: dict[str, str], spot: float = 0.10, seed: int = 12, seams: str = "listen") -> dict:
    """The queue the listener still has to hear: every mismatch and unsure item, a seeded `spot` share (rounded up) of the
    auto-cleared matches, and, with seams='listen', every splice (the ASR checks words, not whether a seam sounds natural).
    Items keep the ticket 06 format; their context gets the reason they are in the queue."""
    items = queue["items"]
    why = {}
    for it in items:
        d = decisions.get(it["id"], "unsure")
        if d != "match":
            why[it["id"]] = f"ASR {d}"
        elif seams == "listen" and it["kind"] == "splice":
            why[it["id"]] = "seam (ASR heard the right words; listen to the join)"
    cleared = sorted(it["id"] for it in items if it["id"] not in why)
    rng = random.Random(f"{seed}:{queue['testSet']}:spot")
    for i in rng.sample(cleared, min(len(cleared), math.ceil(spot * len(cleared)))):
        why[i] = "spot check (ASR heard the labelled words)"
    keep = []
    for it in items:
        if it["id"] in why:
            ctx = f"{it['context']} · " if it.get("context") else ""
            keep.append({**it, "context": f"{ctx}pre-screen: {why[it['id']]}"})
    return {**queue, "items": keep, "prescreen": {"from": len(items), "kept": len(keep), "spot": spot, "seed": seed, "seams": seams,
            "reasons": {k: sum(v.startswith(k) for v in why.values()) for k in ("ASR mismatch", "ASR unsure", "seam", "spot check")}}}
