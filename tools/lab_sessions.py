"""Pure logic for test set lab-v1.1, spliced sessions (.scratch/lab, ticket 03); run by modal/lab_build_sessions.py.

Window choice, splice boundaries, the plant schedule, output-timeline arithmetic and labels (BenchSession, splice
records, manifest, review queue), plus the sample-level render and phone degradation. Standard library only except
`render` and `degrade`, which import numpy (and scipy) when called.

Terms: a *rec* is one teacher-aligned mp3quran surah recording (`align.json`); a *token* is one word occurrence in the
output, taken from a rec; a *piece* is a run of tokens that are consecutive in one rec, so audio is only ever cut
between pieces, at a *seam*. A *boundary* is the gap between two neighbouring words of a rec, with what each side
needs from the other to sound right when joined to something else.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261028
TEST_SET = "lab-v1.1"
SR = 16_000
XFADE = 160  # samples: the 10 ms crossfade centred on each seam
EDGE_FADE = 320  # 20 ms fade in and out at the session ends
SESSION_S = (180.0, 480.0)
TARGET_S = (200.0, 440.0)
SPLIT_SESSIONS = 20
SLICES = {"ayah1": 4, "strict": 3, "muta": 3}  # per split; the rest are normal starts
CLEAN = {"ayah1": 1, "strict": 1, "normal": 3}  # per split
EXCLUDED_READS = {135: "timings drift against the audio in every surah (lab-sources-v1)"}
STRICT_MIN = 6  # a strict start needs more than 5 further words to become unique
STRICT_SURAHS = (26, 54, 55, 77)  # refrain surahs in every reader's pool, preferred for strict starts
NORMAL_MAX = 5
PAUSE_DB = -12.0  # an ayah boundary is a pause when its quietest frame is this far below the ayah level
INNER_PAUSE = (0.3, -15.0)  # an in-ayah gap at least this long and quiet is a pause too
LINK_DB = -3.0  # a linked (no pause) boundary is cut only when its quietest frame is at least this quiet
LEN_RATIO_MAX = 1.05  # an ayah whose teacher decode is this much longer than its text holds unlabelled speech (a natural repeat)
AYAH_PER_MAX = 0.10  # nor is an ayah the teacher matches less well than this (squeezed PER of its source alignment)
CUT_SLACK_S = 0.02  # the cut must lie in the aligned gap, give or take this
MIN_SCORE = 0.5  # teacher word score either side of a seam
DONOR_SCORE = 0.7
MAX_GAIN_DB = 6.0
FIRST_PLANT_S = 20.0
EDIT_GAP = (8, 6.0)  # words and seconds between any two edits
DONOR_DIST_S = 30.0  # a donor from inside the session window must be this far from the plant
CLASS_WEIGHTS = {"wrong_word": 0.3, "omitted_word": 0.35, "added_word": 0.15, "skipped_passage": 0.12}  # mutashabihat swaps are placed first
MUTA_MAX = {True: 6, False: 3}  # swaps per session, focus sessions first
SNR_DB = (8.0, 22.0)
HIGHPASS_HZ = 120.0
LOWPASS_HZ = 7000.0
EVIDENCE = "sessions/{id}.{listener}.{variant}.evidence.json"

V = "\u064e\u064f\u0650"  # fatha, damma, kasra
LONG = ("اا", "ۦۦ", "ۥۥ")
THROAT = set("ءهعحغخ")
WEAK_ONSET = THROAT  # glottal, pharyngeal and h-like heads: a cut inside linked speech clips their quiet onset
QALQALA = "ڇ"
# Hafs idgham across a word boundary: a closed final consonant merges into these first letters (and into itself).
ASSIMILATES = {"م": "ب", "ت": "دط", "د": "ت", "ط": "ت", "ذ": "ظ", "ث": "ذ", "ل": "ر", "ب": "م", "ق": "ك"}


class PlanError(ValueError):
    pass


# ---- reference ----------------------------------------------------------------------------------------------------

class Ref:
    """ref-v2 words in mushaf order (the order of `loadReferenceIndex`, so `disambiguation-v1.json` indexes it)."""

    def __init__(self, asset: dict):
        from help_import import skeleton

        self.version = asset["version"]
        self.words, self.idx, self.ayah_first, self.ayah_words = [], {}, {}, {}
        for a in asset["ayahs"]:
            p, o, key = a["p"], a["o"], (a["s"], a["a"])
            self.ayah_first[key] = len(self.words)
            self.ayah_words[f"{a['s']}:{a['a']}"] = a["w"]
            for i, w in enumerate(a["w"]):
                self.idx[(a["s"], a["a"], i + 1)] = len(self.words)
                self.words.append({"s": a["s"], "a": a["a"], "w": i + 1, "n": len(a["w"]), "text": w, "skel": skeleton(w),
                                   "cont": p[o[i]:(o[i + 1] if i + 1 < len(o) else len(p))], "iso": a["iso"][i]})

    def ref(self, i: int) -> dict:
        w = self.words[i]
        return {"surah": w["s"], "ayah": w["a"], "word": w["w"]}

    def name(self, i: int) -> str:
        w = self.words[i]
        return f"{w['s']}:{w['a']}:{w['w']}"


def tail_kind(cont: str, iso: str, text: str = "") -> tuple[str, str] | None:
    """How a word ends inside an ayah, if its continuous ending is its pause form plus at most a short vowel.

    `vowel`: the pause form plus a case or helping vowel (or a ta marbuta read as t plus a vowel). `long`: ends in a
    long vowel left as it is. `closed`: ends in a consonant left as it is, or a tanween said in full (izhar, so the next
    word starts with a throat letter). Anything else (madd shortened by the next word, nasalised, assimilated or
    merged endings) is None. Runs are squeezed first: the pause form models ghunna and madd lengths differently, and a
    madd lengthened by a following hamza is refused by `joins`.
    """
    if not cont:
        return None
    s, c = squeeze(iso.rstrip(QALQALA)), squeeze(cont)
    k = min(len(s), 4)
    if c[-1] in V and c[:-1].endswith(s[-k:]):
        return ("vowel", c[-2])
    if s.endswith("ه") and text.rstrip("\u064b-\u0652\u06e2\u06ed ").endswith("ة") and c[-1:] in V and c[-2:-1] == "ت" and c[:-2].endswith(s[-k:-1]):
        return ("vowel", "ت")
    if c[-1:] == "ن" and c[-2:-1] in V and len(c) > 2:
        core = s[:-2] if c[-2] == "\u064e" and s.endswith("\u064eا") else s  # fath tanween pauses on a long a
        if core and c[:-2].endswith(core[-min(len(core), 4):]):
            return ("closed", "ن")
    if c.endswith(s[-k:]):
        if s.endswith(tuple(squeeze(x) for x in LONG)):
            return ("long", s[-1])
        if s[-1] not in V:
            return ("closed", s[-1])
    return None


def head_kind(cont: str, iso: str) -> tuple[str, str] | None:
    """How a word starts inside an ayah: `firm` as in its pause form, `wasl` with only the connecting hamza dropped."""
    if not cont:
        return None
    if cont[:2] == iso[:2]:
        return ("firm", cont[0])
    if iso[:1] == "ء" and iso[1:2] in V and cont[:2] == iso[2:4]:
        return ("wasl", cont[0])
    return None


def said_form(w: dict, pause_before: bool, pause_after: bool) -> str | None:
    """A word's phonemes as said in its source: the continuous form, with the pause form's tail where the reciter
    stopped after it (waqf) and its head where they started again after a stop (a connecting hamza said). Ayah edges
    are already in pause form in the continuous phonemes. None when the two forms can't be combined (the head is
    changed by the word before)."""
    cont, iso = w["cont"], w["iso"]
    head = pause_before and w["w"] > 1
    tail = pause_after and w["w"] < w["n"]
    if not (head or tail):
        return cont
    hk = head_kind(cont, iso)
    if hk is None:
        return None
    if head and tail:
        return iso
    if tail:
        return iso if hk[0] == "firm" else iso[2:]
    return cont if hk[0] == "firm" else iso[:2] + cont


def joins(t: dict | None, h: dict | None, restart: bool = False) -> bool:
    """Whether the word before boundary `t` can be followed by the word after boundary `h` with no audible seam.

    The tail was produced before a head of `t`'s kind, so it only joins a head of the same kind; a pause (an ayah end
    or a real stop) joins another pause or a firm head. A restart breaks off anywhere and starts again after a pause.
    """
    if not t or not h:
        return False
    if restart:
        return h["kind"] == "pause"
    if t["kind"] == "pause":
        return h["kind"] == "pause" or h["head"][0] == "firm"
    if h["kind"] == "pause" or h["head"][0] != t["head"][0]:
        return False
    tk, last = t["tail"]
    f = h["head"][1]
    if tk == "long":
        return f != "ء" and f != last
    if tk == "closed":
        return f in THROAT if last == "ن" else f not in ASSIMILATES.get(last, "") + last
    return True


def twin_words(ref: Ref) -> dict[int, list[int]]:
    """Host word -> the same position in each twin ayah: same word count, every other word equal (letter skeletons).

    Swapping the host word for its twin's word recites the twin ayah instead: a mutashabihat slip.
    """
    groups = defaultdict(list)
    by_ayah = defaultdict(list)
    for i, w in enumerate(ref.words):
        by_ayah[(w["s"], w["a"])].append(i)
    for key, ids in by_ayah.items():
        sk = [ref.words[i]["skel"] for i in ids]
        for j in range(len(ids)):
            if len(ids) >= 2:
                groups[(len(ids), j, tuple(sk[:j]), tuple(sk[j + 1:]))].append(ids[j])
    out = defaultdict(list)
    for g in groups.values():
        for a in g:
            for b in g:
                if ref.words[a]["skel"] != ref.words[b]["skel"]:
                    out[a].append(b)
    return dict(out)


# ---- recordings and boundaries --------------------------------------------------------------------------------------

def refused_ayah(a: dict) -> str | None:
    """Why an included source ayah is still kept out of sessions: its teacher decode runs long (an unlabelled natural
    repeat the alignment's garbage blank absorbed) or matches its text loosely."""
    if a["excluded"]:
        return None
    if (a.get("lenRatio") or 1.0) >= LEN_RATIO_MAX:
        return "long"
    if (a.get("per") or 0.0) > AYAH_PER_MAX:
        return "per"
    return None


class Rec:
    """One aligned surah recording: words in order with their ayah's verdict and level.

    Ayahs `refused_ayah` names, and those in `refuse` (ayah numbers the automated checks failed), count as excluded.
    """

    def __init__(self, align: dict, ref: Ref, refuse: dict | None = None):
        self.read, self.surah, self.split, self.reciter = align["readId"], align["surah"], align["split"], align["reciter"]
        self.key = f"{self.read}/{self.surah:03d}"
        self.audio, self.duration = align["audio"], align["durationS"]
        self.basmala = align.get("basmala")
        ay = {a["ayah"]: a for a in align["ayahs"]}
        self.refused = {a: why for a, why in ((a, refused_ayah(x)) for a, x in ay.items()) if why}
        self.refused.update({a: "check" for a in (refuse or {}) if a in ay and not ay[a]["excluded"]})
        self.words = []
        for w in align["words"]:
            a = ay[w["ayah"]]
            self.words.append({**w, "ref": ref.idx[(w["surah"], w["ayah"], w["word"])], "excluded": a["excluded"] or w["ayah"] in self.refused,
                               "level": a.get("levelDb")})
        self.pos = {(w["ayah"], w["word"]): k for k, w in enumerate(self.words)}
        self.ayahs = []  # (ayah, first k, last k) for aligned ayahs in order
        for k, w in enumerate(self.words):
            if w["word"] == 1:
                self.ayahs.append([w["ayah"], k, k])
            elif self.ayahs and self.ayahs[-1][0] == w["ayah"]:
                self.ayahs[-1][2] = k
        self._b = {}

    def usable(self, k: int, score: float = MIN_SCORE) -> bool:
        w = self.words[k]
        return not w["excluded"] and w["endS"] > w["startS"] and (w["score"] or 0) >= score

    def pause_after(self, k: int) -> bool:
        """The reciter stops after word k: an ayah end, or an in-ayah gap as long and quiet as INNER_PAUSE."""
        if k + 1 >= len(self.words) or self.words[k + 1]["ayah"] != self.words[k]["ayah"]:
            return True
        w, n = self.words[k], self.words[k + 1]
        return n["startS"] - w["endS"] >= INNER_PAUSE[0] and w["cutDb"] is not None and w["cutDb"] <= INNER_PAUSE[1]

    def said(self, k: int, ref: Ref) -> str | None:
        """Word k's phonemes as this reciter said them (`said_form`)."""
        return said_form(ref.words[self.words[k]["ref"]], k == 0 or self.pause_after(k - 1), self.pause_after(k))

    def boundary(self, k: int, ref: Ref) -> dict | None:
        """The gap after word k, when audio can be cut there: kinds either side, cut time and depth."""
        if k in self._b:
            return self._b[k]
        self._b[k] = b = self._boundary(k, ref)
        return b

    def _boundary(self, k: int, ref: Ref) -> dict | None:
        if not 0 <= k < len(self.words) - 1:
            return None
        w, n = self.words[k], self.words[k + 1]
        cut, db = w["cutAfterS"], w["cutDb"]
        if cut is None or not (self.usable(k) and self.usable(k + 1)):
            return None
        rw, rn = ref.words[w["ref"]], ref.words[n["ref"]]
        if not (w["endS"] - CUT_SLACK_S <= cut <= n["startS"] + CUT_SLACK_S):
            return None
        same = n["ayah"] == w["ayah"] and n["word"] == w["word"] + 1
        nxt = n["ayah"] == w["ayah"] + 1 and n["word"] == 1 and w["word"] == rw["n"]
        base = {"k": k, "cut": cut, "db": db}
        if nxt:
            return {**base, "kind": "pause"} if db <= PAUSE_DB else None
        if not same:
            return None
        if n["startS"] - w["endS"] >= INNER_PAUSE[0] and db <= INNER_PAUSE[1]:
            return {**base, "kind": "pause"}
        t, h = tail_kind(rw["cont"], rw["iso"], rw["text"]), head_kind(rn["cont"], rn["iso"])
        if db > LINK_DB or not t or not h or h[1] in WEAK_ONSET or not joins({"kind": "link", "tail": t, "head": h}, {"kind": "link", "head": h}):
            return None
        return {**base, "kind": "link", "tail": t, "head": h}


def runs(rec: Rec) -> list[list[int]]:
    """Maximal runs of consecutive included ayahs (indexes into rec.ayahs) whose words all have time."""
    out, cur = [], []
    for i, (a, k0, k1) in enumerate(rec.ayahs):
        ok = all(not rec.words[k]["excluded"] and rec.words[k]["endS"] > rec.words[k]["startS"] for k in range(k0, k1 + 1))
        if ok and cur and rec.ayahs[cur[-1]][0] == a - 1:
            cur.append(i)
        else:
            if cur:
                out.append(cur)
            cur = [i] if ok else []
    if cur:
        out.append(cur)
    return out


def start_cut(rec: Rec, i: int) -> float | None:
    """Session start before ayah index i: in the pause before it, or just before the basmala or ayah 1."""
    a, k0, _ = rec.ayahs[i]
    if a == 1:
        b = rec.basmala
        t = (b["startS"] if b and b.get("present") else rec.words[k0]["startS"]) - 0.15
        return max(0.0, t)
    if k0 == 0:
        return None
    p = rec.words[k0 - 1]
    if p["ayah"] != a - 1 or p["cutAfterS"] is None or p["cutDb"] > PAUSE_DB or not (p["endS"] <= p["cutAfterS"] <= rec.words[k0]["startS"] + CUT_SLACK_S):
        return None
    return p["cutAfterS"]


def end_cut(rec: Rec, i: int) -> float | None:
    _, _, k1 = rec.ayahs[i]
    w = rec.words[k1]
    if w["cutAfterS"] is None:
        return min(rec.duration - 0.02, w["endS"] + 0.6) if k1 == len(rec.words) - 1 else None
    if w["cutDb"] > PAUSE_DB or w["cutAfterS"] < w["endS"]:
        return None
    return w["cutAfterS"]


# ---- windows ------------------------------------------------------------------------------------------------------

def window_candidates(rec: Rec, ref: Ref, dis: list[int], kind: str, target: float, twins: dict, pool_refs: set) -> list[dict]:
    """One window per allowed start ayah: the end ayah whose source length is nearest the target."""
    out = []
    for run in runs(rec):
        for si, i in enumerate(run):
            a, k0, _ = rec.ayahs[i]
            if (kind == "ayah1") != (a == 1):
                continue
            k = dis[rec.words[k0]["ref"]]
            if kind == "strict" and k < STRICT_MIN or kind != "strict" and not 0 <= k <= NORMAL_MAX:
                continue
            t0 = start_cut(rec, i)
            if t0 is None:
                continue
            best = None
            for j in run[si:]:
                t1 = end_cut(rec, j)
                if t1 is None:
                    continue
                d = t1 - t0
                if d > TARGET_S[1] + 20:
                    break
                if d >= TARGET_S[0] - 10 and (best is None or abs(d - target) < abs(best[1] - target)):
                    best = (j, d, t1)
            if best is None:
                continue
            j, d, t1 = best
            ks = range(k0, rec.ayahs[j][2] + 1)
            muta = sum(1 for kk in ks for t in twins.get(rec.words[kk]["ref"], ()) if t in pool_refs)
            out.append({"rec": rec.key, "first": i, "last": j, "startS": t0, "endS": t1, "sourceS": d, "dis": k, "muta": muta,
                        "startAyah": a, "endAyah": rec.ayahs[j][0]})
    return out


def allocate(readers: dict[str, list[int]], seed: int = SEED) -> list[dict]:
    """Session slots per split: who recites, which start slice, whether clean. Readers get one session each, then
    the extra sessions go to a seeded shuffle of them."""
    slots = []
    for split in sorted(readers):
        rng = random.Random(f"{seed}:alloc:{split}")
        rs = sorted(readers[split])
        rng.shuffle(rs)
        order = rs + rs[: SPLIT_SESSIONS - len(rs)]
        kinds = [k for k, n in SLICES.items() for _ in range(n)]
        kinds += ["normal"] * (SPLIT_SESSIONS - len(kinds))
        rng.shuffle(kinds)
        for r, kind in zip(order, kinds):
            slots.append({"split": split, "read": r, "kind": kind})
        clean = dict(CLEAN)
        for s in [s for s in slots if s["split"] == split]:
            group = "normal" if s["kind"] in ("normal",) else s["kind"]
            s["clean"] = clean.get(group, 0) > 0
            if s["clean"]:
                clean[group] -= 1
    return slots


def muta_sites(rec: Rec, k0: int, k1: int, ref: Ref, donors: "Donors", twins: dict) -> int:
    """Words in rec[k0..k1] that can be swapped for their twin ayah's word from this reader with clean seams."""
    n = 0
    for k in range(k0 + 1, k1):
        refs = twins.get(rec.words[k]["ref"])
        bA, bB = (rec.boundary(k - 1, ref), rec.boundary(k, ref)) if refs else (None, None)
        if bA and bB and any(joins(bA, hb) and joins(tb, bB) for r in refs for _, _, hb, tb in donors.by_ref.get(r, ())):
            n += 1
    return n


def choose_windows(slots: list[dict], recs: dict[str, Rec], ref: Ref, dis: list[int], twins: dict, donors_for, seed: int = SEED) -> list[dict]:
    """A window per slot. Scarce slices are placed first; a reader never gets the same surah twice and surahs used
    less often are preferred. A slot whose slice has no window for its reader swaps reader with a later slot.
    Mutashabihat-focus slots take the window with the most clean twin swaps for their reader."""
    by_read = defaultdict(list)
    for r in recs.values():
        by_read[r.read].append(r)
    pool = {r: {w["ref"] for x in rs for w in x.words if not w["excluded"]} for r, rs in by_read.items()}
    used_surah, taken = Counter(), set()
    rank = {"strict": 0, "muta": 1, "ayah1": 2, "normal": 3}
    order = sorted(range(len(slots)), key=lambda i: (rank[slots[i]["kind"]], i))
    out = [None] * len(slots)
    for n, i in enumerate(order):
        s = slots[i]
        rng = random.Random(f"{seed}:window:{s['split']}:{i}")
        target = rng.uniform(*TARGET_S)
        candidates_for = lambda read: [c for r in sorted(by_read[read], key=lambda r: r.surah) if (read, r.surah) not in taken
                                       for c in window_candidates(r, ref, dis, "normal" if s["kind"] == "muta" else s["kind"], target, twins, pool[read])]
        cands = candidates_for(s["read"])
        if not cands:
            for j in order[n + 1:]:
                o = slots[j]
                if o["split"] == s["split"] and o["read"] != s["read"] and (c2 := candidates_for(o["read"])):
                    s["read"], o["read"], cands = o["read"], s["read"], c2
                    break
        if not cands:
            # Nobody left to swap with: the split's reader with the fewest sessions so far who still has a window.
            load = Counter(x["read"] for x in out if x)
            for read in sorted({r.read for r in recs.values() if r.split == s["split"]} - {s["read"]}, key=lambda r: (load[r], r)):
                if c2 := candidates_for(read):
                    s["read"], cands = read, c2
                    break
        if not cands:
            raise PlanError(f"no {s['kind']} window for read {s['read']}")
        surah = lambda c: recs[c["rec"]].surah
        if s["kind"] == "muta":
            def count(cs: list[dict], read: int) -> list[dict]:
                for c in cs:
                    if c["muta"]:
                        r = recs[c["rec"]]
                        c["muta"] = muta_sites(r, r.ayahs[c["first"]][1], r.ayahs[c["last"]][2], ref, donors_for(read), twins)
                return cs
            cands = count(cands, s["read"])
            best = max(c["muta"] for c in cands)
            for j in order[n + 1:]:
                o = slots[j]
                if best >= 3:
                    break
                if o["split"] == s["split"] and o["kind"] == "normal" and o["read"] != s["read"]:
                    c2 = count(candidates_for(o["read"]), o["read"])
                    if c2 and max(c["muta"] for c in c2) > best:
                        s["read"], o["read"], cands = o["read"], s["read"], c2
                        best = max(c["muta"] for c in c2)
        if s["kind"] == "strict":
            key = lambda c: (surah(c) not in STRICT_SURAHS, -min(c["dis"], 12), used_surah[surah(c)], abs(c["sourceS"] - target))
        elif s["kind"] == "muta":
            key = lambda c: (-min(c["muta"], 12), used_surah[surah(c)], abs(c["sourceS"] - target))
        else:
            key = lambda c: (used_surah[surah(c)], abs(c["sourceS"] - target))
        cands.sort(key=lambda c: (key(c), c["rec"], c["first"]))
        top = [c for c in cands if key(c)[:-1] == key(cands[0])[:-1]][:12]
        c = rng.choice(top)
        used_surah[surah(c)] += 1
        taken.add((s["read"], surah(c)))
        out[i] = {**s, **c}
    return out


# ---- edits --------------------------------------------------------------------------------------------------------

def squeeze(s: str) -> str:
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class Donors:
    """Every word of one reader that can be lifted out whole: included ayah, good score, a cuttable boundary either side,
    phonemes as said known (`Rec.said`), and not refused by the automated checks (`refuse`: (rec key, ayah, word))."""

    def __init__(self, recs: list[Rec], ref: Ref, refuse: set | None = None):
        self.items = []
        for r in sorted(recs, key=lambda r: r.surah):
            for k in range(1, len(r.words) - 1):
                w = r.words[k]
                if not r.usable(k, DONOR_SCORE) or (r.key, w["ayah"], w["word"]) in (refuse or ()) or r.said(k, ref) is None:
                    continue
                hb, tb = r.boundary(k - 1, ref), r.boundary(k, ref)
                if hb and tb:
                    self.items.append((r, k, hb, tb))
        self.by_ref = defaultdict(list)
        for it in self.items:
            self.by_ref[it[0].words[it[1]]["ref"]].append(it)

    def find(self, t_in: dict, h_out: dict, host: Rec, near_s: float, window: tuple[float, float], ok, rng: random.Random, refs=None):
        """A donor word whose head joins the tail at `t_in` and whose tail joins the head at `h_out`.

        Same recording first (outside the window, or far from the plant inside it), then the reader's other surahs.
        Among the best-joined few, a seeded pick.
        """
        pool = [it for r in refs for it in self.by_ref.get(r, ())] if refs is not None else self.items
        cands = []
        for r, k, hb, tb in pool:
            w = r.words[k]
            if r is host and window[0] - 1 < w["endS"] and w["startS"] < window[1] + 1 and abs(w["startS"] - near_s) < DONOR_DIST_S:
                continue
            if not (joins(t_in, hb) and joins(tb, h_out) and ok(r, k)):
                continue
            cands.append((0 if r is host else 1, max(hb["db"], tb["db"]), r.surah, k, (r, k, hb, tb)))
        if not cands:
            return None
        cands.sort(key=lambda c: c[:4])
        best = [c for c in cands if c[0] == cands[0][0]][:8]
        return rng.choice(best)[4]


def forms_other(ref: Ref, ayahs: list[str], recited: list[str]) -> bool:
    """`help_import.forms_other_ayah`, except that ayahs with the same text as the prompt (refrains) are not
    candidates: reciting a refrain is not a slip into another ayah, the slip has to make the words another ayah."""
    from help_import import forms_other_ayah

    same = {tuple(ref.ayah_words[a]) for a in ayahs}
    words = {k: v if k in ayahs or tuple(v) not in same else ["\u0000"] * len(v) for k, v in ref.ayah_words.items()}
    return forms_other_ayah(words, ayahs, recited)


def plan_edits(win: dict, recs: dict[str, Rec], ref: Ref, donors: Donors, twins: dict, dis: list[int], seed: int = SEED,
               refuse_sites: set | None = None) -> list[dict]:
    """The planted mistakes and the one repeat or restart of a session, in source terms.

    Mistakes go one per 2-3 ayahs from the first ayah after the disambiguation point (and 20 s in) to two ayahs before
    the end, at least EDIT_GAP apart. Each takes a class weighted by CLASS_WEIGHTS among those with a clean site in
    that ayah. Clean sessions get only the repeat or restart. A site in `refuse_sites` ((rec key, ayah, word, class)
    of the edit's location, from failed automated checks) is never planted again.
    """
    rng = random.Random(f"{seed}:edits:{win['id']}")
    rec = recs[win["rec"]]
    ay = rec.ayahs[win["first"]: win["last"] + 1]
    k_first, k_last = ay[0][1], ay[-1][2]
    window = (win["startS"], win["endS"])
    w = rec.words
    dpoint = k_first + max(0, win["dis"])
    edits, muta_left = [], MUTA_MAX[win["kind"] == "muta"]
    refused = lambda k, cls: (rec.key, w[k]["ayah"], w[k]["word"], cls) in (refuse_sites or ())

    def far(k0: int, k1: int) -> bool:
        return all(k1 + EDIT_GAP[0] <= e["k0"] or e["k1"] + EDIT_GAP[0] <= k0 for e in edits) and \
            all(w[k1]["endS"] + EDIT_GAP[1] <= w[e["k0"]]["startS"] or w[e["k1"]]["endS"] + EDIT_GAP[1] <= w[k0]["startS"] for e in edits)

    def other(cls: str, words_of_ayah: list[str], key: str, prompt: list[str] | None = None) -> bool:
        return forms_other(ref, prompt or [key], words_of_ayah)

    def ayah_text(i: int) -> list[str]:
        a = rec.ayahs[i]
        return [ref.words[w[k]["ref"]]["text"] for k in range(a[1], a[2] + 1)]

    def sites(ai: int, cls: str) -> list[dict]:
        a, k0, k1 = ay[ai]
        key = f"{rec.surah}:{a}"
        text = ayah_text(win["first"] + ai)
        out = []
        if cls == "skipped_passage":
            b0, b1 = rec.boundary(k0 - 1, ref), rec.boundary(k1, ref)
            n = k1 - k0 + 1
            prev, nxt = ayah_text(win["first"] + ai - 1), ayah_text(win["first"] + ai + 1)
            if b0 and b1 and b0["kind"] == b1["kind"] == "pause" and 3 <= n <= 30 and text != prev and text != nxt and far(k0, k1) \
                    and w[k1]["endS"] - w[k0]["startS"] <= 25 and not refused(k0, cls):
                ctx = [f"{rec.surah}:{a - 1}", key, f"{rec.surah}:{a + 1}"]
                rec_words = prev + nxt
                out.append({"cls": cls, "k0": k0, "k1": k1, "seam": (b0, b1), "remove": list(range(k0, k1 + 1)),
                            "other": other(cls, rec_words, key, ctx)})
            return out
        for k in range(k0, k1 + 1):
            if k <= k_first or k >= k_last or refused(k, cls):
                continue
            rw = ref.words[w[k]["ref"]]
            bA, bB = rec.boundary(k - 1, ref), rec.boundary(k, ref)  # A|B and B|C around word k (B)
            if cls == "omitted_word":
                if not (bA and bB and joins(bA, bB) and len(squeeze(rw["cont"])) >= 3 and far(k, k)):
                    continue
                if rw["skel"] in (ref.words[w[k - 1]["ref"]]["skel"], ref.words[w[k + 1]["ref"]]["skel"]):
                    continue
                said = text[: k - k0] + text[k - k0 + 1:]
                out.append({"cls": cls, "k0": k, "k1": k, "seam": (bA, bB), "remove": [k], "other": other(cls, said, key)})
            elif cls in ("wrong_word", "mutashabihat"):
                if not (bA and bB and far(k, k)):
                    continue
                if cls == "mutashabihat":
                    refs = twins.get(w[k]["ref"])
                    if not refs:
                        continue
                    ok = lambda r, kk: True
                else:
                    refs = None
                    near = {ref.words[w[j]["ref"]]["skel"] for j in (k - 1, k, k + 1)}
                    sq = squeeze(rw["cont"])

                    def ok(r, kk, near=near, sq=sq):
                        x = ref.words[r.words[kk]["ref"]]
                        xs = squeeze(x["cont"])
                        return x["skel"] not in near and 0.5 <= len(xs) / max(1, len(sq)) <= 2.0 and levenshtein(xs, sq) >= max(2, len(sq) // 3)
                d = donors.find(bA, bB, rec, w[k]["startS"], window, ok, rng, refs)
                if not d:
                    continue
                x = ref.words[d[0].words[d[1]]["ref"]]
                said = text[: k - k0] + [x["text"]] + text[k - k0 + 1:]
                fo = other(cls, said, key)
                if cls == "mutashabihat" and not fo:
                    continue
                out.append({"cls": cls, "k0": k, "k1": k, "seam": (bA, bB), "remove": [k], "donor": d, "other": fo})
            elif cls == "added_word":
                if not (bA and far(k - 1, k)):
                    continue
                near = {ref.words[w[j]["ref"]]["skel"] for j in (k - 1, k)}

                def ok(r, kk, near=near):
                    x = ref.words[r.words[kk]["ref"]]
                    return x["skel"] not in near and 3 <= len(squeeze(x["cont"])) <= 10
                d = donors.find(bA, bA, rec, w[k]["startS"], window, ok, rng)
                if not d:
                    continue
                x = ref.words[d[0].words[d[1]]["ref"]]
                said = text[: k - k0] + [x["text"]] + text[k - k0:]
                out.append({"cls": cls, "k0": k, "k1": k, "seam": (bA, bA), "remove": [], "donor": d, "before": k, "other": other(cls, said, key)})
        return out

    # The repeat or restart first, in a random ayah of the middle of the session, so plants keep their distance from it.
    want = "restart" if rng.random() < 0.4 else "repeat"
    order = list(range(max(1, len(ay) // 6), max(2, len(ay) - len(ay) // 6)))
    rng.shuffle(order)
    for kind in (want, "repeat" if want == "restart" else "restart"):
        options = []
        for ai in order:
            a, k0, k1 = ay[ai]
            if ai >= len(ay) - 1 or w[k0]["startS"] - win["startS"] < 10:
                continue
            start = rec.boundary(k0 - 1, ref)
            for k in range(k0 + 1, k1):
                t = rec.boundary(k, ref)
                if not t:
                    continue
                if kind == "repeat":
                    for back in (2, 3, 1):
                        j = k - back + 1
                        if j > k0 and joins(t, rec.boundary(j - 1, ref)) and not refused(j, "repeat"):
                            options.append({"cls": "repeat", "k0": j, "k1": k, "seam": (t, rec.boundary(j - 1, ref)), "db": t["db"]})
                            break
                elif start and joins(t, start, restart=True) and not refused(k0, "restart"):
                    options.append({"cls": "restart", "k0": k0, "k1": k, "seam": (t, start), "db": t["db"]})
            if options:
                break
        if options:
            edits.append(min(options, key=lambda o: (o["db"], o["k0"])))
            break
    if not win["clean"]:
        first = next((i for i, (_, k0, k1) in enumerate(ay) if k0 > dpoint and w[k0]["startS"] - win["startS"] >= FIRST_PLANT_S), len(ay))
        # Mutashabihat swaps go first, wherever a twin site is: they are the scarce slice.
        for ai in range(first, len(ay) - 2):
            if muta_left <= 0:
                break
            found = sites(ai, "mutashabihat")
            if found:
                found.sort(key=lambda s: (max(b["db"] for b in s["seam"]), s["k0"]))
                edits.append(found[0])
                muta_left -= 1
        ai = first
        while ai <= len(ay) - 3:
            if any(w[e["k0"]]["ayah"] == ay[ai][0] for e in edits):
                ai += 1
                continue
            classes = [c for c in CLASS_WEIGHTS if c != "mutashabihat"]
            got = None
            while classes and not got:
                c = rng.choices(classes, weights=[CLASS_WEIGHTS[x] for x in classes])[0]
                found = sites(ai, c)
                classes.remove(c)
                if found:
                    found.sort(key=lambda s: (max(b["db"] for b in s["seam"]), s["k0"]))
                    got = rng.choice(found[:3])
            if got:
                edits.append(got)
                ai += 2
            else:
                ai += 1
    return sorted(edits, key=lambda e: e["k0"])


def tokens_for(win: dict, rec: Rec, edits: list[dict]) -> list[dict]:
    """The output word sequence: host words with each edit applied; every token after a cut names the edit that made it."""
    ay = rec.ayahs[win["first"]: win["last"] + 1]
    by_k = {}
    for n, e in enumerate(edits):
        e["id"] = f"x{n + 1}"
        by_k.setdefault(e["k0"], []).append(e)
    out, k, pending = [], ay[0][1], None
    host = lambda kk, **kw: {"rec": rec.key, "k": kk, "at": rec.words[kk]["ref"], "role": "ref", **kw}
    while k <= ay[-1][2]:
        es = by_k.get(k, [])
        e = es[0] if es else None
        if e and e["cls"] in ("repeat", "restart"):
            out += [host(kk) for kk in range(k, e["k1"] + 1)]
            out += [host(kk, role=e["cls"], op=e["id"], **({"seam": e["id"], "brk": e["cls"]} if kk == k else {})) for kk in range(k, e["k1"] + 1)]
            k = e["k1"] + 1
            continue
        if e and e["cls"] in ("omitted_word", "skipped_passage"):
            k = e["k1"] + 1
            pending = e["id"]
            continue
        if e and e["cls"] == "added_word":
            r, dk = e["donor"][:2]
            out.append({"rec": r.key, "k": dk, "at": None, "role": "added", "op": e["id"], "seam": e["id"]})
            out.append(host(k, seam=e["id"]))
            k += 1
            continue
        if e and e["cls"] in ("wrong_word", "mutashabihat"):
            r, dk = e["donor"][:2]
            out.append({"rec": r.key, "k": dk, "at": rec.words[k]["ref"], "role": "wrong", "op": e["id"], "seam": e["id"]})
            k += 1
            pending = e["id"]
            continue
        out.append(host(k, **({"seam": pending} if pending else {})))
        pending = None
        k += 1
    return out


# ---- timeline -----------------------------------------------------------------------------------------------------

def pieces_of(tokens: list[dict], recs: dict[str, Rec], start_s: float, end_s: float, host: Rec) -> list[dict]:
    """Group tokens into pieces cut at the boundaries between them; output offsets in samples, crossfades centred."""
    groups = []
    for t in tokens:
        g = groups[-1] if groups else None
        if g and g[-1]["rec"] == t["rec"] and g[-1]["k"] + 1 == t["k"] and not t.get("brk") and "seam" not in t:
            g.append(t)
        else:
            groups.append([t])
    out, off = [], 0
    for n, g in enumerate(groups):
        r = recs[g[0]["rec"]]
        k0, k1 = g[0]["k"], g[-1]["k"]
        a = start_s if n == 0 else r.words[k0 - 1]["cutAfterS"]
        b = end_s if n == len(groups) - 1 else r.words[k1]["cutAfterS"]
        if a is None or b is None:
            raise PlanError(f"no cut around {r.key} word {k0}-{k1}")
        s0, s1 = round(a * SR), round(b * SR)
        if s1 - s0 < EDGE_FADE or s0 - (XFADE if n else 0) < 0 or s1 + (XFADE if n < len(groups) - 1 else 0) > round(r.duration * SR):
            raise PlanError(f"piece {r.key} {a}-{b} too short or at the file edge")
        gain = 0.0
        if r is not host or g[0]["role"] in ("wrong", "added"):
            prev = out[-1]["tokens"][-1] if out else None
            lvl = recs[prev["rec"]].words[prev["k"]]["level"] if prev else None
            don = r.words[k0]["level"]
            gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, lvl - don)) if lvl is not None and don is not None else 0.0
        out.append({"rec": r.key, "audio": r.audio, "src": [s0, s1], "out": off, "len": s1 - s0, "gainDb": round(gain, 2), "tokens": g,
                    "seam": g[0].get("seam") if n else None})
        off += s1 - s0
    for p in out[1:]:
        if not p["seam"]:
            raise PlanError(f"seam at {p['out']} without an edit")
    return out


def token_times(pieces: list[dict], recs: dict[str, Rec]) -> list[tuple[float, float]]:
    """Output (startS, endS) per token, in order: source word times shifted by their piece, clipped to it."""
    out = []
    for p in pieces:
        r, (s0, s1), off = recs[p["rec"]], p["src"], p["out"]
        for t in p["tokens"]:
            w = r.words[t["k"]]
            a = min(max(round(w["startS"] * SR), s0), s1) - s0 + off
            b = min(max(round(w["endS"] * SR), s0), s1) - s0 + off
            if b - a < SR // 100:
                raise PlanError(f"{r.key} word {t['k']} clipped to {b - a} samples")
            out.append((round(a / SR, 4), round(b / SR, 4)))
    return out


def label(win: dict, rec: Rec, edits: list[dict], tokens: list[dict], pieces: list[dict], recs: dict[str, Rec], ref: Ref, phone: dict) -> dict:
    """BenchSession (sessions.ts) plus `said`, `splices` and `meta`; times in the output timeline."""
    times = token_times(pieces, recs)
    sid = win["id"]
    dur = round(sum(p["len"] for p in pieces) / SR, 4)
    toks = [{**t, "startS": a, "endS": b} for t, (a, b) in zip(tokens, times)]
    spans, words, nonerr = [], [], []
    for t in toks:
        if t["at"] is None:
            continue
        cur = spans[-1] if spans else None
        new = cur is None or t.get("brk") or (cur["kind"] != "recitation" and t["role"] not in ("repeat", "restart"))
        if new:
            cur = {"id": f"span-{len(spans) + 1}", "kind": t.get("brk") or "recitation", "from": ref.ref(t["at"]), "startS": t["startS"]}
            spans.append(cur)
        cur["to"], cur["endS"], cur["_last"] = ref.ref(t["at"]), t["endS"], t["at"]
        t["span"] = cur["id"]
        word = {"span": cur["id"], "at": ref.ref(t["at"]), "startS": t["startS"], "endS": t["endS"]}
        if t["role"] == "wrong":
            x = ref.words[recs[t["rec"]].words[t["k"]]["ref"]]
            word["said"] = {"text": x["text"], "from": ref.ref(recs[t["rec"]].words[t["k"]]["ref"])}
        words.append(word)
        if t["role"] in ("repeat", "restart"):
            nonerr.append({"span": cur["id"], "at": ref.ref(t["at"]), "kind": t["role"], "startS": t["startS"], "endS": t["endS"]})
    for s in spans:
        s.pop("_last")
    b = rec.basmala
    if win["startAyah"] == 1 and b and b.get("present"):
        s0 = pieces[0]["src"][0]
        bs, be = max(0, round(b["startS"] * SR) - s0) / SR, (round(b["endS"] * SR) - s0) / SR
        be = min(be, spans[0]["startS"])
        if be - bs > 0.2:
            spans.insert(0, {"id": "basmala", "kind": "non_quran", "label": "basmala", "startS": round(bs, 4), "endS": round(be, 4)})
    by_op = defaultdict(list)
    for i, t in enumerate(toks):
        if t.get("op"):
            by_op[t["op"]].append(i)
    seam_tok = {}
    for i, t in enumerate(toks):
        if t.get("seam"):
            seam_tok.setdefault(t["seam"], []).append(i)
    events, splices = [], []
    piece_of = {}
    n = 0
    for pi, p in enumerate(pieces):
        for _ in p["tokens"]:
            piece_of[n] = pi
            n += 1
    for e in edits:
        cls = e["cls"]
        ev_cls = "wrong_word" if cls == "mutashabihat" else cls
        first_after = seam_tok[e["id"]][-1] if cls not in ("repeat", "restart") else by_op[e["id"]][0]
        seams = []
        for i in seam_tok.get(e["id"], []):
            pi = piece_of[i]
            if pieces[pi]["tokens"][0] is not tokens[i] or pi == 0:
                continue
            pa, pb = pieces[pi - 1], pieces[pi]
            ra, rb = recs[pa["rec"]], recs[pb["rec"]]
            ka, kb = pa["tokens"][-1]["k"], pb["tokens"][0]["k"]
            ba, bb = ra.boundary(ka, ref), rb.boundary(kb - 1, ref)
            seams.append({"outS": round(pb["out"] / SR, 4), "xfadeS": XFADE / SR,
                          "out": {"audio": ra.audio, "srcS": round(pa["src"][1] / SR, 4), "after": ref.name(ra.words[ka]["ref"]), "cutDb": ba and ba["db"],
                                  "boundary": ba and ba["kind"], "tail": ba and ba.get("tail") and ba["tail"][0]},
                          "in": {"audio": rb.audio, "srcS": round(pb["src"][0] / SR, 4), "before": ref.name(rb.words[kb]["ref"]), "cutDb": bb and bb["db"],
                                 "boundary": bb and bb["kind"], "head": bb and bb.get("head") and bb["head"][0]},
                          "gainDb": pb["gainDb"]})
        rec_w = rec.words
        rec_text = lambda ks: " ".join(ref.words[rec_w[k]["ref"]]["text"] for k in ks)
        if cls in ("repeat", "restart"):
            idx = by_op[e["id"]]
            loc, expected, said = ref.ref(rec_w[e["k0"]]["ref"]), rec_text(range(e["k0"], e["k1"] + 1)), rec_text(range(e["k0"], e["k1"] + 1))
            start, end = toks[idx[0]]["startS"], toks[idx[-1]]["endS"]
        else:
            ev = {"span": None, "errorClass": ev_cls, "state": "confirmed", "location": None, "startS": None, "endS": None, "forms_other_ayah": bool(e["other"])}
            if cls in ("wrong_word", "mutashabihat", "added_word"):
                i = by_op[e["id"]][0]
                ev["startS"], ev["endS"] = toks[i]["startS"], toks[i]["endS"]
                nxt = toks[i + 1]
                ev["span"] = nxt["span"] if cls == "added_word" else toks[i]["span"]
                ev["location"] = ref.ref(nxt["at"]) if cls == "added_word" else ref.ref(toks[i]["at"])
                d = e["donor"]
                xr = d[0].words[d[1]]["ref"]
                ev["expected"] = "" if cls == "added_word" else rec_text([e["k0"]])
                ev["observed"] = ref.words[xr]["text"]
                ev["said"] = d[0].said(d[1], ref)
                ev["donor"] = ref.name(xr)
            else:
                i = first_after
                ev["startS"], ev["endS"] = toks[i - 1]["endS"], toks[i]["startS"]
                ev["span"] = toks[i]["span"]
                if toks[i - 1].get("span") != ev["span"]:
                    raise PlanError(f"{sid} {e['id']}: omission across spans")
                ev["location"] = ref.ref(rec_w[e["k0"]]["ref"])
                if cls == "skipped_passage":
                    ev["through"] = ref.ref(rec_w[e["k1"]]["ref"])
                ev["expected"], ev["observed"] = rec_text(range(e["k0"], e["k1"] + 1)), ""
            ev["splice"] = e["id"]
            events.append(ev)
            loc, expected, said, start, end = ev["location"], ev["expected"], ev["observed"], ev["startS"], ev["endS"]
        donor = None
        if e.get("donor"):
            r, k = e["donor"][:2]
            dw = r.words[k]
            donor = {"audio": r.audio, "rec": r.key, "word": ref.name(dw["ref"]), "srcStartS": dw["startS"], "srcEndS": dw["endS"],
                     "cutS": [r.words[k - 1]["cutAfterS"], dw["cutAfterS"]], "cutDb": [r.words[k - 1]["cutDb"], dw["cutDb"]], "sameRecording": r is rec}
        src_k = e["k0"] if cls != "added_word" else e["k0"]
        splices.append({"id": e["id"], "type": ev_cls if cls not in ("repeat", "restart") else cls, "mutashabihat": cls == "mutashabihat",
                        "formsOtherAyah": bool(e.get("other")), "location": loc, "expected": expected, "said": said,
                        "startS": start, "endS": end, "host": {"audio": rec.audio, "word": ref.name(rec_w[src_k]["ref"]), "srcS": rec_w[src_k]["startS"]},
                        "removed": [ref.name(rec_w[k]["ref"]) for k in e.get("remove", [])], "donor": donor, "seams": seams})
    said = [{"startS": t["startS"], "endS": t["endS"], "at": ref.ref(t["at"]) if t["at"] is not None else None,
             "from": {"audio": recs[t["rec"]].audio, **ref.ref(recs[t["rec"]].words[t["k"]]["ref"])},
             "text": ref.words[recs[t["rec"]].words[t["k"]]["ref"]]["text"],
             "phonemes": recs[t["rec"]].said(t["k"], ref) or ref.words[recs[t["rec"]].words[t["k"]]["ref"]]["cont"],
             "role": t["role"], **({"op": t["op"]} if t.get("op") else {})} for t in toks]
    segments = [{"id": f"seg-{n + 1}", "speaker": speaker(rec), "audio": p["audio"], "split": "eval", "timing": "alignment",
                 "startS": round(p["out"] / SR, 4), "endS": round((p["out"] + p["len"]) / SR, 4),
                 "srcStartS": round(p["src"][0] / SR, 4), "srcEndS": round(p["src"][1] / SR, 4), "gainDb": p["gainDb"]} for n, p in enumerate(pieces)]
    counts = Counter(("mutashabihat" if s["mutashabihat"] else s["type"]) for s in splices)
    return {"version": 1, "id": sid, "track": "recitation", "speaker": speaker(rec), "source": "stitched",
            "audio": f"sessions/{sid}.phone.wav", "durationS": dur, "context": {"kind": "none"},
            "segments": segments, "spans": spans, "words": words, "events": events, "nonErrors": nonerr,
            "said": said, "splices": splices,
            "meta": {"testSet": TEST_SET, "split": win["split"], "readId": rec.read, "reciter": rec.reciter, "surah": rec.surah,
                     "ayahs": [win["startAyah"], win["endAyah"]], "startWord": ref.name(rec.words[rec.ayahs[win["first"]][1]]["ref"]),
                     "disambiguationWords": win["dis"], "slice": win["kind"], "strict": win["kind"] == "strict", "ayah1": win["startAyah"] == 1,
                     "clean": not events, "counts": dict(sorted(counts.items())), "mutashabihat": counts.get("mutashabihat", 0),
                     "formsOtherAyah": sum(e["forms_other_ayah"] for e in events),
                     "audio": {"studio": f"sessions/{sid}.studio.wav", "phone": f"sessions/{sid}.phone.wav"},
                     "phone": phone, "sourceWindowS": [round(win["startS"], 3), round(win["endS"], 3)], "seed": SEED}}


def speaker(rec: Rec) -> str:
    return f"mp3quran-{rec.read}"


def validate(s: dict, ref: Ref) -> list[str]:
    """Python mirror of the core `validateSession` rules this builder can break (the core check runs on Modal too)."""
    out = []
    rank = lambda r: ref.idx.get((r["surah"], r["ayah"], r["word"]))
    last, spans = 0.0, {}
    for p in s["spans"]:
        if p["startS"] < last or p["endS"] <= p["startS"] or p["endS"] > s["durationS"] + 1e-6:
            out.append(f"span {p['id']} unordered or empty")
        if p["kind"] != "non_quran" and rank(p["from"]) > rank(p["to"]):
            out.append(f"span {p['id']} reversed")
        spans[p["id"]] = p
        last = p["endS"]
    inside = lambda t: (sp := spans.get(t["span"])) and sp["kind"] != "non_quran" and sp["startS"] <= t["startS"] <= t["endS"] <= sp["endS"]
    last, prev = 0.0, {}
    for w in s["words"]:
        sp = spans.get(w["span"])
        if not inside(w) or w["endS"] <= w["startS"] or w["startS"] < last:
            out.append(f"word {w['at']} at {w['startS']} outside its span, empty or overlapping")
        elif not rank(sp["from"]) <= rank(w["at"]) <= rank(sp["to"]) or rank(w["at"]) <= prev.get(w["span"], -1):
            out.append(f"word {w['at']} out of order in {w['span']}")
        prev[w["span"]] = rank(w["at"]) if w["span"] in spans else -1
        last = w["endS"]
    for e in s["events"] + s["nonErrors"]:
        if not inside(e):
            out.append(f"truth at {e['startS']} outside its span")
    last = 0.0
    for g in s["segments"]:
        if g["startS"] < last or g["speaker"] != s["speaker"]:
            out.append(f"segment {g['id']} out of order")
        last = g["endS"]
    return out


# ---- phone copy ---------------------------------------------------------------------------------------------------

def phone_params(sid: str, duration_s: float, bank: dict, seed: int = SEED) -> dict:
    """RIR, noise clips (looped in order until they cover the session) and SNR for one session's phone copy."""
    rng = random.Random(f"{seed}:phone:{sid}")
    rir = rng.randrange(len(bank["rirs"]))
    noises, total = [], 0.0
    while total < duration_s + 1:
        i = rng.randrange(len(bank["noises"]))
        noises.append(i)
        total += bank["noises"][i]["durationS"]
    return {"rir": rir, "rirName": bank["rirs"][rir]["name"], "noise": noises, "noiseNames": [bank["noises"][i]["name"] for i in noises],
            "snrDb": round(rng.uniform(*SNR_DB), 2), "highpassHz": HIGHPASS_HZ, "lowpassHz": LOWPASS_HZ}


def render(pieces: list[dict], load) -> "np.ndarray":
    """Studio audio (float32): each piece gain-scaled and laid at its offset; seams crossfaded over XFADE samples
    centred on the seam (equal-gain raised cosine), session ends faded over EDGE_FADE. `load(audio)` gives float32."""
    import numpy as np

    total = sum(p["len"] for p in pieces)
    out = np.zeros(total + 2 * XFADE, dtype=np.float64)
    h = XFADE // 2
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(XFADE) + 0.5) / XFADE)
    for n, p in enumerate(pieces):
        x = load(p["audio"])
        s0, s1 = p["src"]
        lo, hi = (s0 - h if n else s0), (s1 + h if n < len(pieces) - 1 else s1)
        seg = x[lo:hi].astype(np.float64) * 10 ** (p["gainDb"] / 20)
        if n:
            seg[:XFADE] *= ramp
        else:
            seg[:EDGE_FADE] *= np.linspace(0, 1, EDGE_FADE)
        if n < len(pieces) - 1:
            seg[-XFADE:] *= ramp[::-1]
        else:
            seg[-EDGE_FADE:] *= np.linspace(1, 0, EDGE_FADE)
        at = p["out"] - (h if n else 0) + XFADE
        out[at: at + len(seg)] += seg
    return out[XFADE: XFADE + total].astype(np.float32)


def degrade(x, rir, noise_clips: list, snr_db: float, highpass: float = HIGHPASS_HZ, lowpass: float = LOWPASS_HZ):
    """Phone copy: room impulse response from its direct path (so no delay), phone band, then noise at `snr_db`
    against the active speech level, peak-limited to -1 dBFS."""
    import numpy as np
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(rir, dtype=np.float64)
    r = r[int(np.argmax(np.abs(r))):]
    r = r / np.sqrt(np.sum(r * r))
    y = fftconvolve(x, r)[: len(x)]
    y *= rms(x) / max(rms(y), 1e-9)
    y = sosfiltfilt(butter(4, [highpass, lowpass], btype="band", fs=SR, output="sos"), y)
    fade = np.sin(np.linspace(0, np.pi / 2, XFADE)) ** 2
    clips = []
    for c in noise_clips:
        c = np.asarray(c, dtype=np.float64) - np.mean(c)
        c[:XFADE] *= fade
        c[-XFADE:] *= fade[::-1]
        clips.append(c)
    noise = np.concatenate(clips)
    noise = noise[: len(y)] if len(noise) >= len(y) else np.resize(noise, len(y))
    speech = active_rms(y)
    noise *= speech / max(rms(noise), 1e-9) / 10 ** (snr_db / 20)
    out = y + noise
    peak = np.max(np.abs(out))
    if peak > 0.89:
        out *= 0.89 / peak
    return out.astype(np.float32)


def rms(x) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def active_rms(x, hop: int = SR // 100) -> float:
    """RMS over the loudest half of 10 ms frames: the speech level, ignoring pauses."""
    import numpy as np

    n = len(x) // hop
    e = np.mean(np.square(np.asarray(x[: n * hop], dtype=np.float64).reshape(n, hop)), axis=1)
    top = np.sort(e)[n // 2:]
    return float(np.sqrt(np.mean(top))) if len(top) else rms(x)


def pcm16(x) -> bytes:
    """16 kHz mono PCM16 WAV bytes, deterministic header."""
    import io
    import wave

    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768), -32768, 32767)).astype("<i2").tobytes())
    return buf.getvalue()


# ---- whole set ----------------------------------------------------------------------------------------------------

def plan_set(catalog: dict, aligns: dict[str, dict], ref: Ref, dis: list[int], bank: dict, seed: int = SEED, only: int = 0,
             refusals: dict | None = None) -> dict:
    """Every session of the set: windows, edits, pieces and labels. `aligns` maps catalog `align` paths to their JSON.
    `refusals` (`refusals_from_checks`) keeps source ayahs, edit sites and donor words that failed the automated checks
    out of the plan, so a rebuild replaces them instead of patching them.

    A session whose plan breaks a rule (output length, validation) stops the build with PlanError.
    """
    rf = refusals or {}
    ayahs_out = defaultdict(set)
    for x in rf.get("ayahs", ()):
        ayahs_out[x["rec"]].add(x["ayah"])
    sites_out = {(x["rec"], x["ayah"], x["word"], x["class"]) for x in rf.get("sites", ())}
    donors_out = {(x["rec"], x["ayah"], x["word"]) for x in rf.get("donors", ())}
    recs = {}
    for it in catalog["items"]:
        if it["readId"] in EXCLUDED_READS or it["align"] not in aligns:
            continue
        a = aligns[it["align"]]
        r = Rec(a, ref, ayahs_out.get(f"{a['readId']}/{a['surah']:03d}"))
        recs[r.key] = r
    twins = twin_words(ref)
    readers = defaultdict(set)
    for r in recs.values():
        if any(len(run) for run in runs(r)):
            readers[r.split].add(r.read)
    slots = allocate({k: sorted(v) for k, v in readers.items()}, seed)
    donors = {}

    def donors_for(read: int) -> Donors:
        if read not in donors:
            donors[read] = Donors([r for r in recs.values() if r.read == read], ref, donors_out)
        return donors[read]

    wins = choose_windows(slots, recs, ref, dis, twins, donors_for, seed)
    wins.sort(key=lambda w: (w["split"], w["read"], recs[w["rec"]].surah, w["first"]))
    count = Counter()
    for w in wins:
        count[w["split"]] += 1
        w["id"] = f"{TEST_SET}-{w['split'][0]}{count[w['split']]:02d}"
    if only:
        wins = [w for w in wins if w["clean"] is False][: only - 1] + [w for w in wins if w["clean"]][:1]
    sessions, plans = [], []
    for w in wins:
        rec = recs[w["rec"]]
        edits = plan_edits(w, recs, ref, donors_for(rec.read), twins, dis, seed, sites_out)
        while True:
            toks = tokens_for(w, rec, edits)
            pieces = pieces_of(toks, recs, w["startS"], w["endS"], rec)
            dur = sum(p["len"] for p in pieces) / SR
            cuts = [e for e in edits if e["cls"] in ("skipped_passage", "omitted_word")]
            if dur >= SESSION_S[0] or not cuts:
                break
            # Too short once the skips are taken out: give back the longest one.
            edits.remove(max(cuts, key=lambda e: (rec.words[e["k1"]]["endS"] - rec.words[e["k0"]]["startS"], -e["k0"])))
        if not SESSION_S[0] <= dur <= SESSION_S[1]:
            raise PlanError(f"{w['id']}: {dur:.1f} s outside {SESSION_S}")
        s = label(w, rec, edits, toks, pieces, recs, ref, phone_params(w["id"], dur, bank, seed))
        problems = validate(s, ref)
        if problems:
            raise PlanError(f"{w['id']}: {problems[:3]}")
        sessions.append(s)
        plans.append({"id": w["id"], "pieces": [{k: p[k] for k in ("rec", "audio", "src", "out", "len", "gainDb")} for p in pieces],
                      "phone": s["meta"]["phone"], "samples": sum(p["len"] for p in pieces)})
    why = Counter(v for r in recs.values() for v in r.refused.values())
    return {"sessions": sessions, "plans": plans,
            "refused": {"ayahs": dict(sorted(why.items())), "sites": len(sites_out), "donors": len(donors_out)}}


def rec_key(audio: str) -> str:
    """`sources/mp3quran/<read>/<surah>.wav` -> the Rec key `<read>/<surah>`."""
    read, surah = audio.split("/")[-2:]
    return f"{read}/{int(surah.split('.')[0]):03d}"


def refusals_from_checks(checks: dict, sessions: dict[str, dict], previous: dict | None = None, iteration: int = 1) -> tuple[dict, list[dict]]:
    """The builder's side of the automated checks (decision 16): every failed case becomes a refusal in source terms, so
    the next plan rebuilds it elsewhere or drops it. A failed edit refuses its site for its class and its donor word;
    a failed clean stretch or timing refuses its source ayahs. Cumulative over iterations; returns (doc, new entries)."""
    doc = {"version": 1, "kind": "lab-refusals", "testSet": checks["testSet"], "ayahs": [], "sites": [], "donors": [], "unmapped": []}
    for k in ("ayahs", "sites", "donors", "unmapped"):
        doc[k] = [dict(x) for x in (previous or {}).get(k, [])]
    have = {k: {json.dumps({f: x[f] for f in x if f not in ("reason", "from", "iteration")}, sort_keys=True) for x in doc[k]} for k in doc if isinstance(doc[k], list)}
    new = []

    def add(kind: str, entry: dict, reason: str, src: str):
        key = json.dumps(entry, sort_keys=True)
        if key in have[kind]:
            return
        have[kind].add(key)
        x = {**entry, "reason": reason, "from": src, "iteration": iteration}
        doc[kind].append(x)
        new.append({"kind": kind, **x})

    for f in checks.get("failed", []):
        s = sessions[f["session"]]
        x = next(x for x in s["splices"] if x["id"] == f["splice"])
        host = f"{s['meta']['readId']}/{s['meta']['surah']:03d}"
        why = "; ".join(f["reasons"])
        cls = "mutashabihat" if x.get("mutashabihat") else x["type"]
        add("sites", {"rec": host, "ayah": x["location"]["ayah"], "word": x["location"]["word"], "class": cls}, why, f["id"])
        if x.get("donor"):
            _, a, w = map(int, x["donor"]["word"].split(":"))
            add("donors", {"rec": x["donor"].get("rec") or rec_key(x["donor"]["audio"]), "ayah": a, "word": w}, why, f["id"])
    for c in checks.get("failedClean", []):
        s = sessions[c["session"]]
        host = f"{s['meta']['readId']}/{s['meta']['surah']:03d}"
        for a in range(int(c["from"].split(":")[1]), int(c["to"].split(":")[1]) + 1):
            add("ayahs", {"rec": host, "ayah": a}, "; ".join(c["reasons"]), c["id"])
    for t in checks.get("failedTiming", []):
        s = sessions[t["session"]]
        host = f"{s['meta']['readId']}/{s['meta']['surah']:03d}"
        for o in t["off"]:
            src = f"{t['session']}-timing-{o['startS']}"
            why = f"ayah occurrence off by {o['dStartS']:+.2f} s start, {o['dEndS']:+.2f} s end"
            if ":" in str(o["at"]):
                add("ayahs", {"rec": host, "ayah": int(o["at"].split(":")[1])}, why, src)
            else:
                add("unmapped", {"session": t["session"], "at": o["at"], "startS": o["startS"]}, why, src)
    for k in ("ayahs", "sites", "donors"):
        doc[k].sort(key=lambda x: json.dumps({f: x[f] for f in x if f not in ("reason", "from", "iteration")}, sort_keys=True))
    return doc, new


def manifest(sessions: list[dict]) -> dict:
    """Shaped for `sessions-cli --manifest`: `session` inline (audio = the phone copy, relative to the manifest), and
    `evidence` a pattern ticket 05 fills per listener and variant (EVIDENCE)."""
    return {"version": 1, "testSet": TEST_SET, "reference": "ref-v2", "seed": SEED, "headline": "phone", "variants": ["phone", "studio"],
            "evidencePattern": EVIDENCE,
            "sessions": [{"id": s["id"], "split": s["meta"]["split"], "reciter": s["meta"]["reciter"], "readId": s["meta"]["readId"],
                          "surah": s["meta"]["surah"], "durationS": s["durationS"],
                          "slices": {"start": s["meta"]["slice"] if s["meta"]["slice"] != "muta" else "normal", "strict": s["meta"]["strict"],
                                     "ayah1": s["meta"]["ayah1"], "clean": s["meta"]["clean"], "mutashabihat": s["meta"]["mutashabihat"],
                                     "formsOtherAyah": s["meta"]["formsOtherAyah"]},
                          "audio": s["meta"]["audio"], "label": f"sessions/{s['id']}.json",
                          "evidence": EVIDENCE.replace("{id}", s["id"]).replace("{variant}", "phone"),
                          "session": {k: v for k, v in s.items() if k not in ("said", "splices", "meta")}} for s in sessions]}


def summary(sessions: list[dict], extra: dict | None = None) -> dict:
    """Counts for docs/benchmarks/lab-v1.json: per split, slice and class; durations; reciters."""
    def block(ss):
        cls = Counter()
        for s in ss:
            for sp in s["splices"]:
                cls["mutashabihat" if sp["mutashabihat"] else sp["type"]] += 1
        planted = sum(len(s["events"]) for s in ss)
        mins = sum(s["durationS"] for s in ss) / 60
        return {"sessions": len(ss), "reciters": len({s["meta"]["readId"] for s in ss}), "minutes": round(mins, 1),
                "durationS": [round(min((s["durationS"] for s in ss), default=0), 1), round(max((s["durationS"] for s in ss), default=0), 1)],
                "ayah1": sum(s["meta"]["ayah1"] for s in ss), "strict": sum(s["meta"]["strict"] for s in ss),
                "clean": sum(s["meta"]["clean"] for s in ss), "planted": planted,
                "formsOtherAyah": sum(e["forms_other_ayah"] for s in ss for e in s["events"]),
                "plantedPerAyah": round(planted / max(1, sum(s["meta"]["ayahs"][1] - s["meta"]["ayahs"][0] + 1 for s in ss if not s["meta"]["clean"])), 3),
                "classes": dict(sorted(cls.items())), "words": sum(len(s["words"]) for s in ss)}
    out = {"version": 1, "testSet": TEST_SET, "seed": SEED, "all": block(sessions),
           "splits": {sp: block([s for s in sessions if s["meta"]["split"] == sp]) for sp in sorted({s["meta"]["split"] for s in sessions})},
           "reciters": sorted({(s["meta"]["split"], s["meta"]["readId"], s["meta"]["reciter"]) for s in sessions}),
           "surahs": dict(sorted(Counter(s["meta"]["surah"] for s in sessions).items()))}
    out["reciters"] = [{"split": a, "readId": b, "reciter": c, "sessions": sum(s["meta"]["readId"] == b for s in sessions)} for a, b, c in out["reciters"]]
    if extra:
        out.update(extra)
    return out


REVIEW_TITLES = {"wrong_word": "A wrong word put in", "omitted_word": "A word left out", "added_word": "An extra word put in",
                 "skipped_passage": "An ayah skipped", "mutashabihat": "A word swapped for the one in a look-alike ayah",
                 "repeat": "A few words said twice (not a mistake)", "restart": "The ayah started again (not a mistake)"}


def linked(x: dict) -> bool:
    """A splice with a cut inside continuous speech (no pause either side of the seam): the riskiest kind."""
    return any("link" in (sm.get("out", {}).get("boundary"), sm.get("in", {}).get("boundary")) for sm in x["seams"])


def said_text(s: dict, a: float, b: float) -> str:
    return " ".join(w["text"] for w in s["said"] if w["endS"] > a and w["startS"] < b)


def _where(at: dict) -> str:
    return f"{at['surah']}:{at['ayah']}, word {at['word']}"


def review_splice(s: dict, x: dict, root: str = TEST_SET) -> dict:
    """A planted edit as a review item: 2.5 s either side, with the host recording before and after its first seam."""
    a, b = max(0.0, x["startS"] - 2.5), min(s["durationS"], x["endS"] + 2.5)
    h = x["seams"][0]["out"]
    cls = "mutashabihat" if x["mutashabihat"] else x["type"]
    how = "joined mid-phrase, no pause" if linked(x) else "joined at a pause"
    return {"id": f"{s['id']}-{x['id']}", "kind": "splice", "title": f"{REVIEW_TITLES[cls]} ({how})",
            "audio": {"path": f"{root}/{s['meta']['audio']['studio']}", "startS": round(a, 3), "endS": round(b, 3)},
            "before": {"path": h["audio"], "startS": round(max(0.0, h["srcS"] - 3), 3), "endS": round(h["srcS"], 3)},
            "after": {"path": h["audio"], "startS": round(h["srcS"], 3), "endS": round(h["srcS"] + 3.2, 3)},
            "text": said_text(s, a, b), "label": {"type": x["type"], "expected": x["expected"], "said": x["said"]},
            "context": f"{_where(x['location'])} · session {s['id']} at {x['startS']:.1f} s · {s['meta']['reciter']}"
                       + (f" · word from {x['donor']['word']}" if x.get("donor") else "") + (" · forms another ayah" if x["formsOtherAyah"] else "")}


def review_repeat(s: dict, x: dict, root: str = TEST_SET) -> dict:
    """A planted repeat or restart as a review case, 3 s either side."""
    a, b = max(0.0, x["startS"] - 3.0), min(s["durationS"], x["endS"] + 3.0)
    return {"id": f"{s['id']}-{x['id']}", "kind": "case", "title": REVIEW_TITLES[x["type"]],
            "audio": {"path": f"{root}/{s['meta']['audio']['studio']}", "startS": round(a, 3), "endS": round(b, 3)},
            "text": said_text(s, a, b), "label": {"type": x["type"], "expected": x["expected"], "said": x["said"]},
            "context": f"{_where(x['location'])} · session {s['id']} at {x['startS']:.1f} s · {s['meta']['reciter']}"}


def review_queue(sessions: list[dict], ref: Ref, seed: int = SEED) -> dict:
    """The v1 listening check (ticket 07, lab-review.md): 20 splices with the original before and after, then 20 cases
    (clean stretches, repeats and restarts, phone copies). Ids are `<session>-<edit>` or `<session>-<kind>-<time>`.
    Splices are spread over the classes and sessions, linked seams (cut inside continuous speech) first; skipped
    passages always sit between pauses."""
    rng = random.Random(f"{seed}:review")
    root = sessions[0]["meta"]["testSet"] if sessions else TEST_SET
    sp = [(s, x) for s in sessions for x in s["splices"] if x["type"] not in ("repeat", "restart")]
    want = {"wrong_word": 5, "omitted_word": 4, "added_word": 4, "skipped_passage": 3, "mutashabihat": 4}
    by = defaultdict(list)
    for s, x in sp:
        by["mutashabihat" if x["mutashabihat"] else x["type"]].append((s, x))
    picked = []
    for cls, n in want.items():
        pool = sorted(by[cls], key=lambda p: (p[0]["id"], p[1]["id"]))
        rng.shuffle(pool)
        pool.sort(key=lambda p: (not linked(p[1]), sum(q[0]["id"] == p[0]["id"] for q in picked)))
        picked += pool[:n]
    rest = [p for p in sp if p not in picked]
    rng.shuffle(rest)
    rest.sort(key=lambda p: not linked(p[1]))
    picked += rest[: 20 - len(picked)]
    items = [review_splice(s, x, root) for s, x in picked]
    cases = []
    reps = [(s, x) for s in sessions for x in s["splices"] if x["type"] in ("repeat", "restart")]
    rng.shuffle(reps)
    for s, x in reps[:6]:
        cases.append(review_repeat(s, x, root))

    def clean_windows(s, length=8.0):
        busy = [(x["startS"] - 3, x["endS"] + 3) for x in s["splices"]]
        out, t = [], 5.0
        while t + length < s["durationS"] - 2:
            if not any(a < t + length and t < b for a, b in busy) and said_text(s, t, t + length):
                out.append(t)
            t += length
        return out

    order = sorted(sessions, key=lambda s: s["id"])
    rng.shuffle(order)
    for variant, n in (("studio", 7), ("phone", 7)):
        k = 0
        for s in order:
            if k == n:
                break
            if any(c["id"].startswith(s["id"] + "-") for c in cases):
                continue
            ws = clean_windows(s)
            if not ws:
                continue
            t = rng.choice(ws)
            cases.append({"id": f"{s['id']}-{variant}-{int(t)}", "kind": "case",
                          "title": "Recited as written, no mistake" + (" (phone copy: room echo and noise)" if variant == "phone" else ""),
                          "audio": {"path": f"{root}/{s['meta']['audio'][variant]}", "startS": round(t, 3), "endS": round(t + 8.0, 3)},
                          "text": said_text(s, t, t + 8.0), "label": {"type": "clean"},
                          "context": f"session {s['id']} at {t:.1f} s · {s['meta']['reciter']}"
                                     + (f" · phone copy, SNR {s['meta']['phone']['snrDb']} dB" if variant == "phone" else "")})
            k += 1
    return {"version": 1, "testSet": root, "maxBad": 1, "items": items + cases[:20]}
