"""A review item in plain words for a listener who isn't an engineer (the review page's heading, ticket 20): what to
listen for, why the item is in the queue, where in the clip and the Quran it is, what each computer check concluded,
and where in the Arabic text the change sits (a gap for a left-out word or ayah, a mark for a changed, extra or
repeated word). Built from the queue item alone (label, text, triage, audio and the builder's context line), so
every existing queue reads this way without being rebuilt.

Pure logic, standard library only; tested by tools/test_lab_plain.py.
"""

import difflib
import re

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

# Which check is which: the sound-matching check first, then the one that writes down what it hears.
CHECKS = ("nemotron", "fastconformer", "soniox")
WRITES = {"soniox"}  # a label-blind transcript read letter by letter; the others score both texts against the sound
ROLE = {False: "matches both versions against the sound", True: "writes down what it hears, without seeing our text"}
WEAK, CLEAR = 5.0, 20.0  # sound-match margins (its decision needs 2): below WEAK reads "weak", above CLEAR "strong"
TAGS = {"both-mismatch": "Both computer checks think this is wrong: please confirm",
        "mismatch": "The computer check thinks this is wrong: please confirm",
        "disagree": "The two computer checks disagree",
        "unsure": "A computer check couldn't decide",
        "seam": "Check the edit can't be heard",
        "spot": "Random check of a case both computer checks were happy with"}
SEAM = ("Also listen at the cut: a click, a sudden change in loudness or room sound, a word that starts clipped, "
        "or a join that doesn't sound like one breath.")
NOT_A_MISTAKE = "That's allowed, not a mistake."

LOCATION = re.compile(r"(\d+):(\d+), word (\d+) · session (\S+) at ([\d.]+) s · ([^·]+?)(?: · |$)")
CLEAN_AT = re.compile(r"session (\S+) at ([\d.]+) s · ([^·]+?)(?: · |$)")
DONOR = re.compile(r"word from (\d+):(\d+):(\d+)")


def q(s: str) -> str:
    return f"«{s}»"


def short(words: list[str] | str, n: int = 3) -> str:
    ws = words.split() if isinstance(words, str) else words
    return " ".join(ws[:n]) + ("…" if len(ws) > n else "")


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def surah(n: int) -> str:
    return SURAHS[n - 1] if 1 <= n <= len(SURAHS) else f"Surah {n}"


def kind(item: dict) -> str:
    t = item.get("label", {}).get("type")
    return "mutashabihat" if (item.get("triage") or {}).get("class") == "mutashabihat" else t


# ---- where the change sits in the text ------------------------------------------------------------------------------

def marks(item: dict) -> list[dict]:
    """Where the label's change sits among the text's words (split on whitespace, as the page splits it): {kind: gap,
    at, words} before word `at` for words left out; {kind: mark, from, to, words} for words changed, added or said
    again. From the triage's alternative text (the label's change undone); without it, the said word found in the text."""
    text = (item.get("text") or "").split()
    tri, lab = item.get("triage") or {}, item.get("label") or {}
    t = lab.get("type")
    if t == "clean" or not text:
        return []
    alt = (tri.get("alternative") or "").split()
    if alt and alt != text:
        out = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, text, alt, autojunk=False).get_opcodes():
            if op == "insert":
                out.append({"kind": "gap", "at": i1, "words": " ".join(alt[j1:j2])})
            elif op in ("delete", "replace"):
                out.append({"kind": "mark", "from": i1, "to": i2, "words": " ".join(text[i1:i2])})
        return out
    said = (lab.get("said") or "").split()
    for i in range(len(text) - len(said) + 1) if said else ():
        if text[i: i + len(said)] == said:
            return [{"kind": "mark", "from": i, "to": i + len(said), "words": " ".join(said)}]
    return []


# ---- the listening question ----------------------------------------------------------------------------------------

def ask(item: dict, ms: list[dict]) -> str:
    lab, text = item.get("label") or {}, (item.get("text") or "").split()
    k, exp, said = kind(item), lab.get("expected") or "", lab.get("said") or ""
    gap = next((m for m in ms if m["kind"] == "gap"), None)
    mark = next((m for m in ms if m["kind"] == "mark"), None)
    before = lambda i: text[i - 1] if 0 < i <= len(text) else None
    after = lambda i: text[i] if 0 <= i < len(text) else None
    if k == "omitted_word":
        a, b = (before(gap["at"]), after(gap["at"])) if gap else (None, None)
        where = f" between {q(a)} and {q(b)}" if a and b else f" after {q(a)}" if a else f" before {q(b)}" if b else ""
        return f"Listen for: {q(exp)} should be missing{where}."
    if k == "skipped_passage":
        a, b = (before(gap["at"]), after(gap["at"])) if gap else (None, None)
        jump = f"goes straight from {q(a)} to {q(b)}, " if a and b else ""
        return f"Listen for: the reciter {jump}leaving out the ayah {q(short(exp))}."
    if k == "added_word":
        a = before(mark["from"]) if mark else None
        return f"Listen for: an extra {q(said)}{f' after {q(a)}' if a else ''}."
    if k in ("wrong_word", "mutashabihat"):
        line = f"Listen for: {q(said)} said where the mushaf has {q(exp)}."
        return line + (" It's the wording of a look-alike ayah." if k == "mutashabihat" else "")
    if k == "repeat":
        return f"Listen for: {q(short(said or exp, 5))} said twice. {NOT_A_MISTAKE}"
    if k == "restart":
        return f"Listen for: the reciter goes back and starts the ayah again from {q(short(said or exp))}. {NOT_A_MISTAKE}"
    if k == "clean":
        return "Listen for: every word said as written, with nothing missing, extra or changed."
    return f"Listen for: {lab.get('type', 'the label')}."


# ---- where in the Quran and the clip -------------------------------------------------------------------------------

def where(item: dict) -> dict:
    """{line, footer} from the builder's context line: the surah, ayah and word, the time into the clip, how the cut
    was made; the session and reciter."""
    ctx, audio = item.get("context") or "", item.get("audio") or {}
    bits, foot = [], []
    m = LOCATION.search(ctx)
    c = None if m else CLEAN_AT.search(ctx)
    if m:
        s, a, w, sid, t, reciter = int(m[1]), int(m[2]), int(m[3]), m[4], float(m[5]), m[6].strip()
        bits.append(f"{surah(s)} {s}:{a}, {ordinal(w)} word")
    elif c:
        sid, t, reciter = c[1], float(c[2]), c[3].strip()
    else:
        return {"line": "", "footer": item.get("id", "")}
    into = t - float(audio.get("startS", 0))
    if m and 0 <= into <= float(audio.get("endS", 0)) - float(audio.get("startS", 0)) + 0.5:
        bits.append(f"about {into:.1f} s into the clip")
    if "(joined mid-phrase, no pause)" in ctx:
        bits.append("the edit is cut into flowing speech")
    elif "(joined at a pause)" in ctx:
        bits.append("the edit sits in a pause")
    if d := DONOR.search(ctx):
        bits.append(f"the put-in word is this reciter's own recitation of {surah(int(d[1]))} {d[1]}:{d[2]}")
    if "forms another ayah" in ctx:
        bits.append("the changed line is itself a real ayah, a typical look-alike slip")
    foot = [f"Reciter: {reciter}", f"session {sid}", f"item {item.get('id', '')}"]
    return {"line": " · ".join(bits), "footer": " · ".join(foot)}


# ---- what the computer checks concluded ------------------------------------------------------------------------------

def other_version(item: dict) -> str:
    """The version each check weighs the label against, as a phrase that fits "hears it …"."""
    tri, lab = item.get("triage") or {}, item.get("label") or {}
    op, k = tri.get("alternativeOp") or "", kind(item)
    if k == "omitted_word":
        return f"with {q(lab.get('expected', ''))} said"
    if k == "skipped_passage":
        return "with the skipped ayah said"
    if k == "added_word":
        return f"without {q(lab.get('said', ''))}"
    if k in ("wrong_word", "mutashabihat"):
        return f"with {q(lab.get('expected', ''))}, as the mushaf has it"
    if k in ("repeat", "restart"):
        return "said only once"
    if m := re.fullmatch(r"(\S+) claimed as (\S+)", op):
        return f"as a made-up test version with {q(m[2])} in place of {q(m[1])}"
    if m := re.fullmatch(r"text claims extra (\S+)", op):
        return f"as a made-up test version with an extra {q(m[1])}"
    return "as the other version"


def strength(model: str, r: dict) -> str:
    d = r.get("diff")
    if d is None:
        return ""
    if model in WRITES:
        need = r.get("need") if d >= 0 else r.get("needAgainst", r.get("need"))
        return "clearly" if need and abs(d) >= 2 * need else "just enough"
    return "weak" if abs(d) < WEAK else "clear" if abs(d) <= CLEAR else "strong"


def check_line(model: str, r: dict, other: str) -> str:
    """One check's conclusion in plain words, without its numbers."""
    dec, d, why = r.get("decision"), r.get("diff"), r.get("reason")
    if model in WRITES:
        if why == "transcript far from both texts":
            return "what it wrote down matches neither version"
        if why == "transcript covers too little of the window":
            return "it caught too little of the clip to say"
        if d is None:
            return "it gave nothing usable"
        if dec == "match":
            return f"wrote down the version as described ({strength(model, r)})"
        if dec == "mismatch":
            return f"wrote down the version {other} ({strength(model, r)})"
        return "what it wrote down could be either version"
    if d is None:
        return {"match": "only the version as described fits the sound", "mismatch": "the version as described can't fit the sound"}.get(dec, "couldn't score this")
    if dec == "match":
        return f"hears it as described ({strength(model, r)})"
    if dec == "mismatch":
        return f"hears it {other} ({strength(model, r)})"
    return "can't tell"


def checks(item: dict) -> list[dict]:
    """[{n, model, decision, text}] for the checks that decided the item (sound-match first)."""
    tri = item.get("triage") or {}
    other = other_version(item)
    ms = [m for m in CHECKS if isinstance(tri.get(m), dict) and tri[m].get("decision")]
    return [{"n": i + 1, "model": m, "decision": tri[m]["decision"], "role": ROLE[m in WRITES],
             "text": f"Check {i + 1} ({ROLE[m in WRITES]}): {check_line(m, tri[m], other)}"} for i, m in enumerate(ms)]


def tag(item: dict) -> str:
    tri = item.get("triage") or {}
    reasons = tri.get("reasons") or []
    if not reasons:
        return ""
    first = reasons[0]
    if first == "unsure" and any(isinstance(tri.get(m), dict) and tri[m].get("decision") == "mismatch" for m in CHECKS):
        return "One computer check says this is wrong, the other can't tell"
    return TAGS.get(first, "")


def plain(item: dict) -> dict:
    """The review page's plain heading for a queue item: {tag, ask, where, checks, seam, footer, marks}."""
    ms = marks(item)
    w = where(item)
    reasons = (item.get("triage") or {}).get("reasons") or []
    return {"tag": tag(item), "ask": ask(item, ms), "where": w["line"], "checks": checks(item),
            "seam": SEAM if "seam" in reasons else "", "footer": w["footer"], "marks": ms}
