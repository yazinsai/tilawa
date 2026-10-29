"""Scripts for scripted-error recordings (.scratch/lab/issues/18-scripted-recordings.md, docs/benchmarks/lab-recordings.md).

    python3 tools/lab_scripts.py batch                      # docs/benchmarks/lab-scripts-v1.json, the default batch
    python3 tools/lab_scripts.py batch --pool my-pool.json --n 12 --seed 7 --batch rec-v2 --out /tmp/rec-v2.json
    python3 tools/lab_scripts.py show rec-v1-03             # one script of the default batch as plain text

A script is a passage of 3-6 minutes at a normal pace with planted mistakes the reciter performs on purpose: about one
per 2-3 ayahs, each a wrong word (a twin ayah's word, a word that sits between the same neighbours elsewhere in the
Quran, or a look-alike word), an added word, an omitted word, or a self-correction (say a wrong word, stop, go back and
say it right). A few scripts are clean. The script records, per word, what is to be said, so a take's labels are known
by construction. Standard library only; deterministic for a seed.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_sessions as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261029
BATCH = "rec-v1"
OUT = ROOT / "docs/benchmarks/lab-scripts-v1.json"
SCRIPT_VERSION = 1
# Seconds of recitation at a normal (not slow murattal) pace: per ref-v2 phoneme, madd and ghunna lengths included, plus
# a breath per ayah. About 31 s for al-Fatiha, 3:16 for an-Naba, 4:55 for al-Mulk. An estimate; takes record the real length.
PACE = {"perPhonemeS": 0.105, "perAyahS": 0.8}
MINUTES = (3.0, 6.0)
TARGET_MINUTES = (3.2, 5.0)
# Where passages come from: surahs the reciter is likely to know. Juz Amma passages are runs of whole surahs; other
# passages start at an ayah 1 and run whole surahs when they fit, else stop at an ayah end.
POOL = [
    {"tag": "juz-amma", "surahs": list(range(78, 115)), "weight": 4},
    {"tag": "juz-tabarak", "surahs": list(range(67, 78)), "weight": 3},
    {"tag": "long", "surahs": [18, 36, 55, 56, 19, 2], "weight": 3},
]
CLEAN_SCRIPTS = 2  # per batch of 10
FIRST_PLANT = (2, 20.0)  # no mistake in the first two ayahs or the first 20 s: the listener has to find the passage first
GAP = (8, 10.0)  # words and estimated seconds between two plants (Juz Amma ayahs are short: 2 ayahs can be 8 s)
AYAH_STEP = (2, 3)  # the next plant is 2 or 3 ayahs after the last
# Spliced deletions sound natural, spliced insertions don't: recordings lean on wrong and added words.
CLASS_WEIGHTS = {"wrong_word": 0.5, "added_word": 0.3, "omitted_word": 0.2}
ADDED_MAX = 0.35  # added words are at most this share of a script's mistakes
SELF_CORRECTIONS = (1, 1, 2)  # per planted script, a seeded pick
NO_BASMALA = (1, 9)  # al-Fatiha's basmala is its first ayah; at-Tawba has none
BASMALA = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
SURAHS = ("Al-Fatiha Al-Baqara Al-Imran An-Nisa Al-Maida Al-Anam Al-Araf Al-Anfal At-Tawba Yunus Hud Yusuf Ar-Rad Ibrahim "
          "Al-Hijr An-Nahl Al-Isra Al-Kahf Maryam Ta-Ha Al-Anbiya Al-Hajj Al-Muminun An-Nur Al-Furqan Ash-Shuara An-Naml "
          "Al-Qasas Al-Ankabut Ar-Rum Luqman As-Sajda Al-Ahzab Saba Fatir Ya-Sin As-Saffat Sad Az-Zumar Ghafir Fussilat "
          "Ash-Shura Az-Zukhruf Ad-Dukhan Al-Jathiya Al-Ahqaf Muhammad Al-Fath Al-Hujurat Qaf Adh-Dhariyat At-Tur An-Najm "
          "Al-Qamar Ar-Rahman Al-Waqia Al-Hadid Al-Mujadila Al-Hashr Al-Mumtahana As-Saff Al-Jumua Al-Munafiqun At-Taghabun "
          "At-Talaq At-Tahrim Al-Mulk Al-Qalam Al-Haqqa Al-Maarij Nuh Al-Jinn Al-Muzzammil Al-Muddaththir Al-Qiyama Al-Insan "
          "Al-Mursalat An-Naba An-Naziat Abasa At-Takwir Al-Infitar Al-Mutaffifin Al-Inshiqaq Al-Buruj At-Tariq Al-Ala "
          "Al-Ghashiya Al-Fajr Al-Balad Ash-Shams Al-Lail Ad-Duha Ash-Sharh At-Tin Al-Alaq Al-Qadr Al-Bayyina Az-Zalzala "
          "Al-Adiyat Al-Qaria At-Takathur Al-Asr Al-Humaza Al-Fil Quraish Al-Maun Al-Kawthar Al-Kafirun An-Nasr Al-Masad "
          "Al-Ikhlas Al-Falaq An-Nas").split()
CLASS_NAMES = {"wrong_word": "wrong word", "omitted_word": "word left out", "added_word": "word added", "self_correction": "self-correction"}


# ---- reference ----------------------------------------------------------------------------------------------------

class Quran:
    """ref-v2 words (`lab_sessions.Ref`) with ayah lists, estimated durations and neighbour indexes for substitutes."""

    def __init__(self, asset: dict):
        self.ref = ref = S.Ref(asset)
        self.ayahs = defaultdict(list)  # surah -> ayah numbers
        self.p = {}
        for a in asset["ayahs"]:
            self.ayahs[a["s"]].append(a["a"])
            self.p[(a["s"], a["a"])] = a["p"]
        self.basmala = {"p": asset["ayahs"][0]["p"], "o": asset["ayahs"][0]["o"]}
        self.twins = S.twin_words(ref)
        self.freq = Counter(w["skel"] for w in ref.words)
        sk = lambda i: ref.words[i]["skel"] if 0 <= i < len(ref.words) else "^"
        same = lambda i, j: 0 <= j < len(ref.words) and (ref.words[i]["s"], ref.words[i]["a"]) == (ref.words[j]["s"], ref.words[j]["a"])
        self.sk = lambda i, d: sk(i + d) if same(i, i + d) else "|"
        self.mid = defaultdict(list)    # (left, right) -> words between them
        self.after = defaultdict(list)  # left -> words after it
        self.before = defaultdict(list)  # right -> words before it
        for i in range(len(ref.words)):
            l, r = self.sk(i, -1), self.sk(i, 1)
            self.mid[(l, r)].append(i)
            self.after[l].append(i)
            self.before[r].append(i)
        self.by_skel = defaultdict(list)
        for i, w in enumerate(ref.words):
            self.by_skel[w["skel"]].append(i)

    def words_of(self, s: int, a: int) -> list[int]:
        k0 = self.ref.idx[(s, a, 1)]
        return list(range(k0, k0 + self.ref.words[k0]["n"]))

    def seconds(self, s: int, a: int) -> float:
        return PACE["perPhonemeS"] * len(self.p[(s, a)]) + PACE["perAyahS"]

    def basmala_s(self) -> float:
        return PACE["perPhonemeS"] * len(self.basmala["p"]) + PACE["perAyahS"]

    def name(self, i: int) -> str:
        return self.ref.name(i)


def surah_name(s: int) -> str:
    return SURAHS[s - 1]


def distinct(a: str, b: str) -> int:
    return S.levenshtein(S.squeeze(a), S.squeeze(b))


def where(q: Quran, i: int) -> str:
    w = q.ref.words[i]
    return f"{w['s']}:{w['a']}"


# ---- passages -----------------------------------------------------------------------------------------------------

def passage_seconds(q: Quran, ayahs: list[tuple[int, int]]) -> float:
    return sum(q.seconds(s, a) for s, a in ayahs) + sum(q.basmala_s() for s, a in ayahs if a == 1 and s not in NO_BASMALA)


def passage(q: Quran, surahs: list[int], used: set[int], target_s: float, rng: random.Random) -> list[tuple[int, int]] | None:
    """Consecutive ayahs for one script from a seeded unused start surah: whole surahs while they fit under the cap,
    until the target; a surah longer than the cap gives its opening ayahs up to the target. None when no start works."""
    lo, hi = MINUTES[0] * 60, MINUTES[1] * 60
    starts = [s for s in surahs if s not in used]
    rng.shuffle(starts)
    for s0 in starts:
        out, run = [], s0
        while run in surahs and run not in used and passage_seconds(q, out) < target_s:
            nxt = out + [(run, a) for a in q.ayahs[run]]
            if passage_seconds(q, nxt) > hi:
                break
            out, run = nxt, run + 1
        if not out:
            for a in q.ayahs[s0]:
                out.append((s0, a))
                if passage_seconds(q, out) >= target_s:
                    break
        if passage_seconds(q, out) >= lo:
            return out
    return None


# ---- plants -------------------------------------------------------------------------------------------------------

def substitutes(q: Quran, g: int, rng: random.Random, lookalikes: bool = True) -> list[dict]:
    """Plausible wrong words for word g, best first: the twin ayah's word, a word found elsewhere between the same
    neighbours, a look-alike word. Never a neighbour's word, and always said differently."""
    ref, w = q.ref, q.ref.words[g]
    near = {q.sk(g, d) for d in (-1, 0, 1)}
    seen, out = set(), []

    def add(j: int, kind: str, why: str, rank: tuple) -> None:
        x = ref.words[j]
        if x["skel"] in near or x["skel"] in seen or x["text"] == w["text"]:
            return
        if distinct(x["cont"], w["cont"]) < 1 or not 0.5 <= len(S.squeeze(x["cont"])) / max(1, len(S.squeeze(w["cont"]))) <= 2.0:
            return
        seen.add(x["skel"])
        out.append({"kind": kind, "word": j, "why": why, "rank": rank})

    for j in sorted(q.twins.get(g, [])):
        add(j, "twin", f"the look-alike ayah {where(q, j)} has it here", (0, 0))
    l, r = q.sk(g, -1), q.sk(g, 1)
    if l != "|" and r != "|":
        for j in q.mid[(l, r)]:
            ctx = (q.sk(j, -2) == q.sk(g, -2)) + (q.sk(j, 2) == q.sk(g, 2))
            add(j, "context", f"{where(q, j)} has it between the same words", (1, -ctx, -q.freq[ref.words[j]["skel"]]))
    else:  # first or last word of the ayah: two words of context on the one side
        for j in q.mid[(l, r)]:
            if (l == "|" and q.sk(j, 2) == q.sk(g, 2)) or (r == "|" and q.sk(j, -2) == q.sk(g, -2)):
                add(j, "context", f"{where(q, j)} has it in the same place", (1, 0, -q.freq[ref.words[j]["skel"]]))
    if lookalikes and not out and len(w["skel"]) >= 3:
        for d, f, sk, j in lookalike_words(q, w["skel"]):
            add(j, "confusable", "a look-alike word", (2, d, f))
    out.sort(key=lambda c: c["rank"])
    best = out[0]["rank"][:1] if out else None
    top = [c for c in out if c["rank"][:1] == best][:3]
    rng.shuffle(top)
    return top + [c for c in out if c not in top]


def lookalike_words(q: Quran, skel: str) -> list[tuple[int, int, str, int]]:
    """(letters changed, -frequency, skeleton, a word) of Quran words that look like `skel`: one letter changed, or a
    leading و / ف added or dropped. Two changed letters mostly give words no reciter would slip into."""
    out = []
    for sk, js in q.by_skel.items():
        if q.freq[sk] < 2 or sk == skel:
            continue
        if len(sk) == len(skel):
            if sum(a != b for a, b in zip(sk, skel)) == 1:
                out.append((1, -q.freq[sk], sk, js[0]))
        elif sk[1:] == skel and sk[0] in "وف" or skel[1:] == sk and skel[0] in "وف":
            out.append((1, -q.freq[sk], sk, js[0]))
    return sorted(out)[:20]


def additions(q: Quran, g: int, rng: random.Random) -> list[dict]:
    """Plausible added words before word g: a word found elsewhere between the same two words, else one that follows
    the word before (or precedes this word) elsewhere."""
    ref = q.ref
    l, r = q.sk(g, -1), q.sk(g, 0)
    near = {l, r}
    seen, out = set(), []

    def add(j: int, kind: str, why: str, rank: tuple) -> None:
        x = ref.words[j]
        if x["skel"] in near or x["skel"] in seen or not 3 <= len(S.squeeze(x["cont"])) <= 12:
            return
        seen.add(x["skel"])
        out.append({"kind": kind, "word": j, "why": why, "rank": rank})

    for j in q.mid[(l, r)]:
        add(j, "context", f"{where(q, j)} reads the two words with it between", (0, -q.freq[ref.words[j]["skel"]]))
    for j in q.after[l]:
        add(j, "after", f"{where(q, j)} goes on with it after the word before", (1, -q.freq[ref.words[j]["skel"]]))
    for j in q.before[r]:
        add(j, "before", f"{where(q, j)} has it before this word", (1, -q.freq[ref.words[j]["skel"]]))
    out.sort(key=lambda c: c["rank"])
    best = out[0]["rank"][:1] if out else None
    top = [c for c in out if c["rank"][:1] == best][:4]
    rng.shuffle(top)
    return top + [c for c in out if c not in top]


def omittable(q: Quran, g: int) -> bool:
    w = q.ref.words[g]
    return w["n"] >= 3 and 1 < w["w"] < w["n"] and len(S.squeeze(w["cont"])) >= 3 and w["skel"] not in (q.sk(g, -1), q.sk(g, 1))


def site_options(q: Quran, s: int, a: int, cls: str, rng: random.Random) -> list[dict]:
    """Every way to plant `cls` in ayah s:a, seeded order."""
    order = q.words_of(s, a)
    rng.shuffle(order)
    out = []
    for lookalikes in (False, True):  # look-alike words only where no word has a twin or context substitute
        for g in order:
            w = q.ref.words[g]
            if cls == "omitted_word" and omittable(q, g) and not lookalikes:
                out.append({"cls": cls, "word": g})
            elif cls in ("wrong_word", "self_correction") and w["n"] >= 2 and (cls == "wrong_word" or w["w"] > 1):
                subs = substitutes(q, g, rng, lookalikes)
                if subs:
                    out.append({"cls": cls, "word": g, "with": subs[0]})
            elif cls == "added_word" and w["w"] > 1 and not lookalikes:
                adds = additions(q, g, rng)
                if adds:
                    out.append({"cls": cls, "word": g, "with": adds[0]})
        if out:
            break
    rank = {"twin": 0, "context": 1, "after": 2, "before": 2, "confusable": 3}
    out.sort(key=lambda o: rank[o["with"]["kind"]] if o.get("with") else 0)
    return out


def plan(q: Quran, ayahs: list[tuple[int, int]], clean: bool, rng: random.Random) -> list[dict]:
    """The plants of one script, in order: sites every 2-3 ayahs from the third ayah and 20 s in to the one before the
    last, GAP apart; one or two self-corrections, the rest drawn by CLASS_WEIGHTS with added words capped."""
    if clean:
        return []
    t, starts = 0.0, []
    for s, a in ayahs:
        if a == 1 and s not in NO_BASMALA:
            t += q.basmala_s()
        starts.append(t)
        t += q.seconds(s, a)
    sites, i = [], FIRST_PLANT[0]
    while i < len(ayahs) - 1:
        if starts[i] >= FIRST_PLANT[1]:
            sites.append(i)
            i += rng.choice(AYAH_STEP)
        else:
            i += 1
    n_sc = min(rng.choice(SELF_CORRECTIONS), max(0, len(sites) - 2))
    mid = sites[len(sites) // 4: max(len(sites) // 4 + 1, 3 * len(sites) // 4)]
    sc = set(rng.sample(mid, min(n_sc, len(mid))))
    plants, last = [], None
    for i in sites:
        s, a = ayahs[i]
        want = ["self_correction"] if i in sc else None
        if not want:
            mistakes = [p for p in plants if p["cls"] != "self_correction"]
            added_ok = sum(p["cls"] == "added_word" for p in mistakes) + 1 <= ADDED_MAX * (len(mistakes) + 1)
            classes = [c for c in CLASS_WEIGHTS if c != "added_word" or added_ok]
            first = rng.choices(classes, weights=[CLASS_WEIGHTS[c] for c in classes])[0]
            want = [first] + [c for c in classes if c != first]
        for cls in want:
            opts = [o for o in site_options(q, s, a, cls, rng) if last is None or far(q, last, o["word"], starts, ayahs)]
            if opts:
                plants.append({**opts[0], "ayahIndex": i})
                last = opts[0]["word"]
                break
    return plants


def far(q: Quran, g0: int, g1: int, starts: list[float], ayahs: list[tuple[int, int]]) -> bool:
    w0, w1 = q.ref.words[g0], q.ref.words[g1]
    i0, i1 = ayahs.index((w0["s"], w0["a"])), ayahs.index((w1["s"], w1["a"]))
    t0 = starts[i0] + q.seconds(w0["s"], w0["a"]) * w0["w"] / w0["n"]
    t1 = starts[i1] + q.seconds(w1["s"], w1["a"]) * (w1["w"] - 1) / w1["n"]
    return g1 - g0 >= GAP[0] and t1 - t0 >= GAP[1]


# ---- the script ---------------------------------------------------------------------------------------------------

def in_context(x: dict, first: bool, last: bool) -> str:
    """Word x's phonemes said at another place (first or last word of the ayah there, or neither): its continuous form
    when that doesn't depend on its own neighbours (same ayah edges, an unassimilated head and tail), else its pause
    form, without the connecting hamza inside an ayah."""
    cont, iso = x["cont"], x["iso"]
    src_first, src_last = x["w"] == 1, x["w"] == x["n"]
    head = src_first if first else not src_first and S.head_kind(cont, iso) is not None
    tail = src_last if last else not src_last and S.tail_kind(cont, iso, x["text"]) is not None
    if head and tail:
        return cont
    return iso[2:] if not first and x["text"].startswith("\u0671") and iso[:1] == "ء" else iso


def said_items(q: Quran, ayahs: list[tuple[int, int]], plants: list[dict], rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    """(said, display ayahs, plant records). `said` is what is to be recited, one item per word in time order, with
    its reference word (`at`, null for an added word or the basmala), text, phonemes and role:
    ref, wrong, added, slip (the wrong word of a self-correction), retake (the words said again to fix it), basmala."""
    ref = q.ref
    by_word = {p["word"]: p for p in plants}
    said, display, records = [], [], []
    num = 0
    for ai, (s, a) in enumerate(ayahs):
        if a == 1 and s not in NO_BASMALA:
            said.append({"at": None, "text": BASMALA, "phonemes": q.basmala["p"], "role": "basmala"})
            display.append({"key": f"{s}:0", "surah": s, "ayah": 0, "basmala": True, "tokens": [{"text": BASMALA}], "plants": []})
        row = {"key": f"{s}:{a}", "surah": s, "ayah": a, "tokens": [], "plants": []}
        ids = q.words_of(s, a)
        ayah_said = []
        for g in ids:
            w = ref.words[g]
            p = by_word.get(g)
            at = ref.ref(g)
            if not p:
                said.append({"at": at, "text": w["text"], "phonemes": w["cont"], "role": "ref"})
                row["tokens"].append({"text": w["text"]})
                ayah_said.append(w["text"])
                continue
            num += 1
            pid = f"m{num}"
            row["plants"].append(pid)
            cls = p["cls"]
            rec = {"id": pid, "n": num, "class": cls, "location": at, "ayah": f"{s}:{a}", "expected": w["text"]}
            if cls == "omitted_word":
                row["tokens"].append({"text": w["text"], "mark": "omitted", "plant": pid})
                rec |= {"said": "", "instruction": f"Leave out «{w['text']}».", "why": "a word left out"}
            elif cls in ("wrong_word", "self_correction"):
                x = ref.words[p["with"]["word"]]
                ph = in_context(x, w["w"] == 1, w["w"] == w["n"])
                rec |= {"said": x["text"], "saidPhonemes": ph if cls == "wrong_word" else x["iso"], "saidFrom": ref.ref(p["with"]["word"]),
                        "kind": p["with"]["kind"], "why": f"«{x['text']}»: {p['with']['why']}"}
                if cls == "wrong_word":
                    said.append({"at": at, "text": x["text"], "phonemes": ph, "role": "wrong", "plant": pid})
                    row["tokens"].append({"text": x["text"], "mark": "wrong", "plant": pid, "original": w["text"]})
                    rec["instruction"] = f"Say «{x['text']}» instead of «{w['text']}»."
                    ayah_said.append(x["text"])
                else:
                    back = 2 if w["w"] > 1 and rng.random() < 0.5 else 1
                    retake = list(range(g - back + 1, g + 1))
                    said.append({"at": at, "text": x["text"], "phonemes": x["iso"], "role": "slip", "plant": pid})
                    for k, r in enumerate(retake):
                        rw = ref.words[r]
                        ph = S.said_form(rw, k == 0, False) or rw["cont"]
                        said.append({"at": ref.ref(r), "text": rw["text"], "phonemes": ph, "role": "retake", "plant": pid})
                    again = " ".join(ref.words[r]["text"] for r in retake)
                    row["tokens"] += [{"text": x["text"], "mark": "slip", "plant": pid, "original": w["text"]}, {"text": "↺", "mark": "back", "plant": pid}]
                    row["tokens"] += [{"text": ref.words[r]["text"], "mark": "retake", "plant": pid} for r in retake]
                    rec |= {"retake": [ref.ref(r) for r in retake], "instruction": f"Say «{x['text']}» instead of «{w['text']}», stop, then go back and say «{again}» correctly."}
                    ayah_said.append(w["text"])
                    rec["kind"] = p["with"]["kind"]
            elif cls == "added_word":
                x = ref.words[p["with"]["word"]]
                ph = in_context(x, False, False)
                said.append({"at": None, "text": x["text"], "phonemes": ph, "role": "added", "plant": pid})
                said.append({"at": at, "text": w["text"], "phonemes": w["cont"], "role": "ref"})
                row["tokens"] += [{"text": x["text"], "mark": "added", "plant": pid}, {"text": w["text"]}]
                rec |= {"said": x["text"], "saidPhonemes": ph, "saidFrom": ref.ref(p["with"]["word"]), "kind": p["with"]["kind"],
                        "expected": "", "before": w["text"], "instruction": f"Add «{x['text']}» before «{w['text']}».", "why": f"«{x['text']}»: {p['with']['why']}"}
                ayah_said += [x["text"], w["text"]]
            if cls != "self_correction":
                rec["formsOtherAyah"] = None  # filled once the ayah is complete
            records.append(rec)
        for rec in records:
            if rec["ayah"] == f"{s}:{a}" and rec.get("formsOtherAyah", False) is None:
                rec["formsOtherAyah"] = S.forms_other(ref, [f"{s}:{a}"], [w for w in ayah_said if w])
        display.append(row)
    return said, display, records


def labels(records: list[dict], said: list[dict]) -> dict:
    """The expected BenchSession truth, before timing: events (planted mistakes) and non-errors (self-corrections)."""
    events, non_errors = [], []
    for r in records:
        if r["class"] == "self_correction":
            for at in r["retake"]:
                non_errors.append({"kind": "self_correction", "at": at, "plant": r["id"]})
            continue
        e = {"plant": r["id"], "errorClass": r["class"], "state": "confirmed", "location": r["location"], "expected": r["expected"],
             "observed": r["said"], "forms_other_ayah": bool(r["formsOtherAyah"])}
        if r.get("saidPhonemes"):
            e["said"] = r["saidPhonemes"]
        events.append(e)
    return {"events": events, "nonErrors": non_errors}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def script(q: Quran, sid: str, tag: str, ayahs: list[tuple[int, int]], clean: bool, seed: int, plants: list[dict] | None = None) -> dict:
    """One script; `plants` (as `plan` returns them) overrides the seeded plan."""
    if plants is None:
        plants = plan(q, ayahs, clean, random.Random(f"{seed}:{sid}:plants"))
    said, display, records = said_items(q, ayahs, plants, random.Random(f"{seed}:{sid}:said"))
    (s0, a0), (s1, a1) = ayahs[0], ayahs[-1]
    surahs = sorted({s for s, _ in ayahs})
    title = " – ".join(surah_name(s) for s in (surahs[0], surahs[-1]) if s) if len(surahs) > 1 else surah_name(s0)
    rng_s = f"{s0}:{a0}–{a1}" if s0 == s1 else f"{s0}:{a0}–{s1}:{a1}"
    whole = all({a for ss, a in ayahs if ss == s} == set(q.ayahs[s]) for s in surahs)
    body = {"id": sid, "version": SCRIPT_VERSION, "tag": tag, "title": title, "range": rng_s, "wholeSurahs": whole,
            "from": {"surah": s0, "ayah": a0}, "to": {"surah": s1, "ayah": a1}, "ayahCount": len(ayahs),
            "estimatedS": round(passage_seconds(q, ayahs), 1), "clean": clean,
            "counts": dict(sorted(Counter(r["class"] for r in records).items())),
            "plants": records, "ayahs": display, "said": said, "labels": labels(records, said)}
    return {**body, "sha": digest(body)}


def batch(asset: dict, n: int = 10, seed: int = SEED, name: str = BATCH, pool: list[dict] | None = None, clean: int | None = None) -> dict:
    """`n` scripts from the pool, tags allotted by weight (largest remainder), no surah used twice; `clean` of them
    (CLEAN_SCRIPTS per 10 by default) carry no mistakes. Juz Amma first, so the first script is a short familiar one."""
    q = Quran(asset)
    pool = pool or POOL
    rng = random.Random(f"{seed}:{name}")
    total = sum(p["weight"] for p in pool)
    raw = [n * p["weight"] / total for p in pool]
    alloc = [int(x) for x in raw]
    for i in sorted(range(len(pool)), key=lambda i: -(raw[i] - alloc[i]))[: n - sum(alloc)]:
        alloc[i] += 1
    slots = [p["tag"] for p, k in zip(pool, alloc) for _ in range(k)]
    order = []
    while slots:  # interleave tags in pool order
        for p in pool:
            if p["tag"] in slots:
                slots.remove(p["tag"])
                order.append(p["tag"])
    n_clean = round(n * CLEAN_SCRIPTS / 10) if clean is None else clean
    clean_at = set(rng.sample(range(1, n), min(n_clean, n - 1)))
    used, out = set(), []
    for k, tag in enumerate(order):
        p = next(x for x in pool if x["tag"] == tag)
        target = rng.uniform(*TARGET_MINUTES) * 60
        ayahs = passage(q, p["surahs"], used, target, rng)
        if not ayahs:
            continue
        used |= {s for s, _ in ayahs}
        out.append(script(q, f"{name}-{len(out) + 1:02d}", tag, ayahs, k in clean_at, seed))
    return {"version": 1, "kind": "lab-scripts", "batch": name, "seed": seed, "reference": asset["version"], "pace": PACE,
            "pool": pool, "scripts": out, "summary": summary(out)}


def summary(scripts: list[dict]) -> dict:
    cls = Counter(r["class"] for s in scripts for r in s["plants"])
    kinds = Counter(r.get("kind", "-") for s in scripts for r in s["plants"] if r["class"] in ("wrong_word", "added_word", "self_correction"))
    return {"scripts": len(scripts), "clean": sum(s["clean"] for s in scripts), "estimatedMinutes": round(sum(s["estimatedS"] for s in scripts) / 60, 1),
            "ayahs": sum(s["ayahCount"] for s in scripts), "classes": dict(sorted(cls.items())),
            "mistakes": sum(n for c, n in cls.items() if c != "self_correction"),
            "formsOtherAyah": sum(bool(r.get("formsOtherAyah")) for s in scripts for r in s["plants"]),
            "substituteKinds": dict(sorted(kinds.items()))}


def text(s: dict) -> str:
    """One script as plain text: the passage with marks, then the numbered instructions."""
    mark = {"wrong": "[{}]", "added": "+[{}]", "omitted": "-({})", "slip": "[{}]", "back": "{}", "retake": "{}"}
    lines = [f"{s['id']} · {s['title']} {s['range']} · about {s['estimatedS'] / 60:.1f} min · " + (", ".join(f"{n} {CLASS_NAMES[c]}" for c, n in s["counts"].items()) or "clean")]
    for row in s["ayahs"]:
        toks = " ".join(mark[t["mark"]].format(t["text"]) if t.get("mark") else t["text"] for t in row["tokens"])
        lines.append(f"{row['key']:>7}  {toks}" + (f"   ({', '.join(p.lstrip('m') for p in row['plants'])})" if row["plants"] else ""))
    for p in s["plants"]:
        lines.append(f"  {p['n']}. {p['ayah']} {CLASS_NAMES[p['class']]}: {p['instruction']} ({p['why']})" + (" · forms another ayah" if p.get("formsOtherAyah") else ""))
    return "\n".join(lines)


E2E = {"surah": 77, "ayahs": (1, 20), "plants": [  # (class, word, said from): every said word is in Al-Mursalat
    ("wrong_word", (77, 3, 2), (77, 4, 2)), ("added_word", (77, 7, 3), (77, 15, 2)), ("self_correction", (77, 10, 3), (77, 9, 3)),
    ("omitted_word", (77, 14, 4), None), ("wrong_word", (77, 18, 3), (77, 15, 3))]}


def e2e_batch(asset: dict) -> dict:
    """The end-to-end test batch (speaker `test`): Al-Mursalat 77:1-20 with one plant of each class whose said words
    come from the same surah, so modal/lab_recordings.py can cut a synthetic take from one mp3quran reader; and the
    same passage clean."""
    q = Quran(asset)
    s0 = E2E["surah"]
    ayahs = [(s0, a) for a in range(E2E["ayahs"][0], E2E["ayahs"][1] + 1)]
    plants = []
    for cls, at, frm in E2E["plants"]:
        g = q.ref.idx[at]
        p = {"cls": cls, "word": g, "ayahIndex": at[1] - E2E["ayahs"][0]}
        if frm:
            p["with"] = {"kind": "context", "word": q.ref.idx[frm], "why": f"{frm[0]}:{frm[1]} has it", "rank": (0,)}
        plants.append(p)
    out = [script(q, "rec-test-77", "e2e", ayahs, False, SEED, plants), script(q, "rec-test-77c", "e2e", ayahs, True, SEED, [])]
    return {"version": 1, "kind": "lab-scripts", "batch": "test", "seed": SEED, "reference": asset["version"], "pace": PACE, "pool": [],
            "scripts": out, "summary": summary(out)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch")
    b.add_argument("--n", type=int, default=10)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--batch", default=BATCH)
    b.add_argument("--pool", help="JSON list of {tag, surahs, weight}")
    b.add_argument("--clean", type=int)
    b.add_argument("--out", default=str(OUT))
    e = sub.add_parser("e2e", help="the end-to-end test batch (Al-Mursalat, speaker test)")
    e.add_argument("--out", default="/tmp/lab-scripts-test.json")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.add_argument("--file", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "batch":
        asset = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
        pool = json.loads(Path(a.pool).read_text()) if a.pool else None
        out = batch(asset, a.n, a.seed, a.batch, pool, a.clean)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(out["summary"], ensure_ascii=False))
        for s in out["scripts"]:
            print(f"  {s['id']} {s['title']} {s['range']}: {s['estimatedS'] / 60:.1f} min, {s['ayahCount']} ayahs, {dict(s['counts']) or 'clean'}")
    elif a.cmd == "e2e":
        out = e2e_batch(json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text()))
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(text(out["scripts"][0]))
    else:
        doc = json.loads(Path(a.file).read_text())
        print(text(next(s for s in doc["scripts"] if s["id"] == a.id)))


if __name__ == "__main__":
    main()
