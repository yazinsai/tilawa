"""Structured local margin and training-only recognition corruption search.

The recurrence has the same four arc families, restart source, prices, tie rules
and ayah recentering as src/local.ts. Features match src/trajectory.ts. No test
set input or absolute location feature enters the fitted model.
"""
import bisect
import collections
import math
import random

import numpy as np

FEATURES = ["insert", "consume", "restart", "delete", "acoustic", "heardVowel", "heardNasal",
            "heardEmphatic", "writtenVowel", "writtenNasal", "writtenEmphatic", "exact", "soft",
            "durationRun", "wordStart", "wordEnd", "wordLength", "wordMovement", "movementLength",
            "ayahCrossing", "phoneMovement", "restartExact", "restartSoft", "restartNasal"]
VOWELS = set("َُِاۦۥ")
SHORT = set("َُِ")
NASAL = set("نمں۾")
EMPHATIC = set("صضطظق")
HELD = set("اۦۥں۾")
CONFUSABLE = {x for pair in ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"]
              for x in (pair, pair[::-1])}


def sub_cost(ref, obs):
    return 0. if ref == obs else .5 if ref + obs in CONFUSABLE or ref in SHORT and obs in SHORT else 1.


class Reference:
    def __init__(self, data):
        self.ref, self.words = data["phonemes"], data["words"]
        self.starts = [w["start"] for w in self.words]
        self.ayah_start, self.ayah_end, self.word_ayah = [], [], []
        self.entries = set()
        for i, w in enumerate(self.words):
            prev = self.words[i - 1] if i else None
            if not prev or (prev["surah"], prev["ayah"]) != (w["surah"], w["ayah"]):
                self.ayah_start.append(w["start"])
            if prev and w["surah"] != prev["surah"] and w["surah"] != 9:
                self.entries.add(w["start"])
            self.word_ayah.append(len(self.ayah_start) - 1)
            if len(self.ayah_end) < len(self.ayah_start):
                self.ayah_end.append(w["end"])
            else:
                self.ayah_end[-1] = w["end"]
        self.word_starts = set(self.starts)
        self.delete = np.array([self.indel(p, self.run(i)) for i, p in enumerate(self.ref)])
        self.start_array = np.array(self.starts)
        self.end_array = np.array([w["end"] for w in self.words])
        self.surah_array = np.array([w["surah"] for w in self.words])
        self.ayah_array = np.array([w["ayah"] for w in self.words])
        self.symbols = np.array(list(self.ref))
        self.runs = np.array([self.run(i) for i in range(len(self.ref))])
        self.vowels = np.isin(self.symbols, list(VOWELS))
        self.nasals = np.isin(self.symbols, list(NASAL))
        self.emphatics = np.isin(self.symbols, list(EMPHATIC))
        self.duration = (np.r_[False, self.symbols[1:] == self.symbols[:-1]]
                         | np.r_[self.symbols[:-1] == self.symbols[1:], False])

    def wi(self, at):
        return max(0, bisect.bisect_right(self.starts, at) - 1)

    def window(self, best):
        a = self.word_ayah[self.wi(max(0, best - 1))]
        return self.ayah_start[max(0, a - 2)], self.ayah_end[min(len(self.ayah_end) - 1, a + 2)]

    def run(self, at):
        a, b = at, at + 1
        while a > 0 and self.ref[a - 1] == self.ref[at]:
            a -= 1
        while b < len(self.ref) and self.ref[b] == self.ref[at]:
            b += 1
        return b - a

    @staticmethod
    def indel(p, run):
        long = p in HELD and run >= 2 or p in set("نم") and run >= 3
        return .2 if long else .5 if p in SHORT else 1.

    def insertion(self, at, p):
        before = self.run(at - 1) if at > 0 and self.ref[at - 1] == p else 0
        after = self.run(at) if at < len(self.ref) and self.ref[at] == p else 0
        return self.indel(p, max(before, after) + 1)

    def features(self, kind, source, destinations, p, prices):
        x = np.zeros((len(destinations), len(FEATURES)))
        x[:, kind] = 1
        x[:, 4] = prices
        x[:, 5:8] = [p in VOWELS, p in NASAL, p in EMPHATIC]
        at = np.minimum(destinations, len(self.ref) - 1) if kind == 0 else destinations - 1
        wi = np.maximum(0, np.searchsorted(self.start_array, at, side="right") - 1)
        source_wi = self.wi(max(0, source - 1))
        movement = wi - source_wi if kind == 2 else np.zeros(len(wi))
        exact = (self.symbols[at] == p) & (kind != 3)
        soft = np.array([sub_cost(r, p) < 1 for r in self.symbols[at]]) & ~exact & (kind != 3)
        x[:, 8:17] = np.column_stack([self.vowels[at], self.nasals[at], self.emphatics[at], exact, soft,
                                     self.duration[at], at == self.start_array[wi], destinations == self.end_array[wi],
                                     np.log1p(self.end_array[wi] - self.start_array[wi]) / 5])
        x[:, 17] = np.tanh(movement / 8)
        x[:, 18] = np.log1p(np.abs(movement)) / 5
        if kind == 2:
            x[:, 19] = (self.surah_array[wi] != self.surah_array[source_wi]) | (self.ayah_array[wi] != self.ayah_array[source_wi])
            x[:, 20] = np.tanh((destinations - 1 - source) / 16)
            x[:, 21] = exact
            x[:, 22] = soft
            x[:, 23] = p in NASAL
        return x

    def replay(self, phones, start, weights):
        lo, hi = self.window(start)
        c = np.full(hi - lo + 1, np.inf)
        c[start - lo] = 0
        x = np.zeros((len(c), len(FEATURES)))
        origins = np.zeros(len(c), dtype=int)
        origins[start - lo] = start
        best = start
        for p in phones:
            destinations = np.arange(lo, hi + 1)
            ins = np.array([self.insertion(at, p) for at in destinations])
            ix = self.features(0, best, destinations, p, ins)
            bc, bx, bo = c + 10 * (ins + ix @ weights), x + ix, origins.copy()
            sub = np.array([sub_cost(self.ref[at - 1], p) for at in destinations[1:]])
            entry = np.array([3. if at - 1 in self.entries else 0. for at in destinations[1:]])
            sx = self.features(1, best, destinations[1:], p, sub)
            sc = c[:-1] + 10 * (sub + entry + sx @ weights)
            consume = sc <= bc[1:]
            bc[1:] = np.where(consume, sc, bc[1:])
            bx[1:] = np.where(consume[:, None], x[:-1] + sx, bx[1:])
            bo[1:] = np.where(consume, origins[:-1], bo[1:])
            jump = np.array([at - 1 != best and at - 1 in self.word_starts for at in destinations[1:]])
            source_surah = self.words[self.wi(max(0, best - 1))]["surah"]
            crossing = np.array([0. if self.words[self.wi(at - 1)]["surah"] == source_surah or e else 3.
                                 for at, e in zip(destinations[1:], entry)])
            jx = self.features(2, best, destinations[1:], p, sub)
            jc = c[best - lo] + 20 + np.abs(destinations[1:] - 1 - best) * .001 + 10 * (sub + entry + crossing + jx @ weights)
            take = jump & (jc < bc[1:])
            bc[1:] = np.where(take, jc, bc[1:])
            bx[1:] = np.where(take[:, None], x[best - lo] + jx, bx[1:])
            bo[1:] = np.where(take, destinations[1:] - 1, bo[1:])
            # Reference-only closure. A tie retains the base path rather than deleting.
            dx = self.features(3, best, destinations[1:], p, self.delete[lo:hi])
            dc = 10 * (self.delete[lo:hi] + entry + dx @ weights)
            accumulated = np.r_[0., np.cumsum(dc)]
            base_values = bc - accumulated
            previous = np.r_[np.inf, np.minimum.accumulate(base_values)[:-1]]
            record = base_values <= previous
            ancestor = np.maximum.accumulate(np.where(record, np.arange(len(c)), 0))
            cumulative_x = np.vstack([np.zeros(len(FEATURES)), np.cumsum(dx, axis=0)])
            c = base_values[ancestor] + accumulated
            x = bx[ancestor] + cumulative_x - cumulative_x[ancestor]
            origins = bo[ancestor]
            tied = np.flatnonzero(c == c.min())
            chosen = tied[np.argmin(np.abs(lo + tied - best - 1))]
            best = lo + int(chosen)
            c -= c[chosen]
            new_lo, new_hi = self.window(best)
            if (lo, hi) != (new_lo, new_hi):
                nc = np.full(new_hi - new_lo + 1, np.inf)
                nx = np.zeros((len(nc), len(FEATURES)))
                no = np.zeros(len(nc), dtype=int)
                a, b = max(lo, new_lo), min(hi, new_hi) + 1
                nc[a - new_lo:b - new_lo] = c[a - lo:b - lo]
                nx[a - new_lo:b - new_lo] = x[a - lo:b - lo]
                no[a - new_lo:b - new_lo] = origins[a - lo:b - lo]
                lo, hi, c, x, origins = new_lo, new_hi, nc, nx, no
        return {"lo": lo, "cost": c / 10, "features": x, "origin": origins, "best": best}

    def margin(self, example, weights, phones=None):
        r = self.replay(example["phones"] if phones is None else phones, example["start"], weights)
        ats = np.arange(r["lo"], r["lo"] + len(r["cost"]))
        valid = np.isfinite(r["cost"])
        target = (ats >= example["target"][0]) & (ats < example["target"][1]) & valid
        # Inner opponent is a different location reached through an existing word restart.
        rival = ~target & valid & (r["origin"] != example["start"])
        if not target.any() or not rival.any():
            return None
        true_at = np.flatnonzero(target)[np.argmin(r["cost"][target])]
        wrong_at = np.flatnonzero(rival)[np.argmin(r["cost"][rival])]
        return float(r["cost"][true_at] - r["cost"][wrong_at]), r["features"][true_at] - r["features"][wrong_at]


def edits(reference, heard):
    """Deterministic full-transcript alignment for estimating the corruption support."""
    c = np.zeros((len(heard) + 1, len(reference) + 1))
    op = np.zeros(c.shape, dtype=np.uint8)
    c[0] = np.arange(len(reference) + 1)
    c[:, 0] = np.arange(len(heard) + 1)
    op[0, 1:] = 2
    op[1:, 0] = 1
    for i, obs in enumerate(heard, 1):
        for j, ref in enumerate(reference, 1):
            choices = [c[i - 1, j - 1] + (ref != obs), c[i - 1, j] + 1, c[i, j - 1] + 1]
            op[i, j] = np.argmin(choices)
            c[i, j] = min(choices)
    i, j, out = len(heard), len(reference), []
    while i or j:
        k = int(op[i, j])
        if k == 0:
            if reference[j - 1] != heard[i - 1]:
                out.append(("sub", reference[j - 1], heard[i - 1]))
            i, j = i - 1, j - 1
        elif k == 1:
            out.append(("insert", reference[j - 1] if j else "", heard[i - 1]))
            i -= 1
        else:
            # Expanded duration repetitions are genuine obligations, not recognition-error support.
            if not (j > 1 and reference[j - 1] == reference[j - 2] or j < len(reference) and reference[j - 1] == reference[j]):
                out.append(("delete", reference[j - 1], ""))
            j -= 1
    return out


def corruption_channel(recordings, reference):
    counts = collections.Counter()
    rates = []
    for r in recordings:
        if r.get("split") != "train":
            raise ValueError("Non-training recording supplied")
        transcript = "".join(reference.ref[reference.words[w]["start"]:reference.words[w]["end"]] for w in r["path"])
        observed = edits(transcript, r["phones"])
        counts.update(observed)
        rates.append(len(observed) / max(1, len(transcript)))
    # Predeclared minimum empirical support and conservative two-edit cap, selected without practice.
    patterns = sorted((pattern, count) for pattern, count in counts.items() if count >= 2)
    if not patterns:
        raise ValueError("No supported recognition corruptions")
    return {"patterns": patterns, "rate": float(np.quantile(rates, .5)), "max_edits": 2}


def corrupt(phones, pattern, rng):
    kind, source, target = pattern
    sites = [i for i, p in enumerate(phones) if p == source]
    if kind == "insert" and source == "":
        sites = [0]
    if not sites:
        return None
    i = rng.choice(sites)
    if kind == "sub":
        return phones[:i] + target + phones[i + 1:]
    if kind == "delete":
        return phones[:i] + phones[i + 1:]
    return phones[:i] + target + phones[i:]


def inner_search(reference, example, weights, channel, seed, adversarial, width=8, on_selection=None):
    """Both arms score exactly width proposals per depth; only selection differs."""
    rng = random.Random(seed)
    selection_rng = random.Random(seed + 1)
    patterns, counts = zip(*channel["patterns"])
    current = example["phones"]
    evaluations = 0
    budget = min(channel["max_edits"], max(1, math.ceil(len(current) * channel["rate"])))
    for _ in range(budget):
        proposals = [current]
        for _ in range(width - 1):
            proposal = None
            for _ in range(32):
                pattern = rng.choices(patterns, weights=counts)[0]
                proposal = corrupt(current, pattern, rng)
                if proposal and proposal != current:
                    break
            proposals.append(proposal or current)
        scores = [reference.margin(example, weights, p) for p in proposals]
        evaluations += len(scores)
        valid = [i for i, score in enumerate(scores) if score is not None]
        if not valid:
            continue
        # Random arm still computes all margins, using the identical proposal budget and architecture.
        chosen = max(valid, key=lambda i: scores[i][0]) if adversarial else selection_rng.choice(valid)
        current = proposals[chosen]
    if on_selection is not None:
        on_selection(current)
    return reference.margin(example, weights, current), evaluations + 1
