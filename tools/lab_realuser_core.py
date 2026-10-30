"""Real-user clean set (.scratch/lab/issues/19-real-user-clean-set.md, docs/benchmarks/lab-realuser-v1.md).

Real non-professional recitation on phones with no planted mistakes, scored for false warnings per recited minute.
Three sources make one `realuser` slice:

- sobolev: sobolev210/quran-recitation-errors, app recordings cut into one chunk per ayah. Hafs only. A recording's
  chunks are joined back in chunk order. Chunks with reviewer tags stay, and their tags are kept as candidate real
  mistakes.
- retasy: RetaSy quranic_audio_dataset, the `correct` Quranic clips that are Hafs or don't say. The Aya text is mapped
  to surah:ayah. One reciter's clips are joined in mushaf order, one pass per repeat of an ayah.
- tlog: the tlog-v2 clean-match test split (held out by speaker cluster), contiguous chains of one cluster's clips.

Every clip is decoded by the teacher and checked against its label before it is joined (`verify_clip`). A joined
session is a take of a clean script built from its clips (`script`): each clip's words, with an optional isti'adha and
basmala before it. It is checked by `lab_recordings_core.check_take` and labelled by `session`.

Layout on the tilawa-lab volume:

    /lab/data/realuser/clips/<source>/<clip>.wav   16 kHz mono PCM16 of every clip
    /lab/data/realuser/clips/manifest-v1.json      the clips as a decodable set (teacher evidence beside each WAV)
    /lab/data/realuser/clips-v1.json               every clip: source, speaker, label, verification, why dropped
    /lab/data/realuser/splits-v1.json              speaker -> practice / exam per source, fixed before any scoring
    /lab/data/realuser-v1/sessions/<id>.*          the joined sessions (phone WAV, BenchSession, checks, evidence)
    /lab/data/realuser-v1/manifest-v1.json         test set realuser-v1: phone only, slices.source "realuser"
    /lab/data/realuser-v1/review/flagged.json      practice ayahs heard differently from their label

Pure logic; numpy and rapidfuzz only where audio or evidence is touched.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict

TEST_SET = "realuser-v1"
ROOT = "realuser"
SOURCES = ("sobolev", "retasy", "tlog")
SEED = 20261030
MIN_S = 5.0  # lab_recordings.MIN_S: shorter takes are joined first
SESSION_MAX_S = 300.0
TLOG_TARGET_S = 90 * 60.0  # about 1.5 h of tlog clips, sampled as whole chains
TLOG_CHAIN_MAX_S = 120.0
SPEAKER_MAX_S = 6 * 60.0  # per speaker, so one prolific reciter doesn't make a source
GAP_S = 0.5  # silence between joined clips
EDGE_S = 0.25  # silence at either end of a session
FADE_S = 0.02
GAIN_MAX_DB = 10.0  # clips of one session are level-matched to its median clip, at most this much
CLIP_S = (0.5, 60.0)
PER_OK = 0.15  # lab_recordings_core.AYAH_PER_MAX: a clip the teacher hears this close to its label is as labelled
PER_DROP = 0.4  # further than this from its label and from every other ayah: not the labelled text
RELABEL_MARGIN = 0.1
EARLY_S = 10.0  # the report splits false warnings in a session's first 10 s from the rest
RETASY_SURAHS: dict[str, int | str] = {"Al-Faatihah": 1, "Ayat al-Kursi": "2:255", "Al-Asr": 103, "Al-Humazah": 104, "Al-Kauthar": 108, "Al-Kafiroon": 109,
                 "An-Nasr": 110, "Al-Masad": 111, "Al-Ikhlas": 112, "Al-Falaq": 113, "An-Nas": 114}
NO_BASMALA = (1, 9)
CHUNK = re.compile(r"chunk(\d+)\.wav$")
DIACRITIC = re.compile(r"[\u0610-\u061a\u0640\u064b-\u065f\u0670\u06d6-\u06ed]")
LETTERS = str.maketrans({"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ی": "ي", "ک": "ك", "ۀ": "ه", "ە": "ه"})
NON_LETTER = re.compile(r"[^\u0621-\u064a]")


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def norm(text: str) -> str:
    """Letters only, diacritics and spaces dropped, alef and ya forms merged: text written in different orthographies
    (RetaSy's, the app's, the Uthmani reference) compares letter for letter."""
    return NON_LETTER.sub("", DIACRITIC.sub("", str(text or "")).translate(LETTERS))


def at(s: int, a: int, w: int) -> dict:
    return {"surah": s, "ayah": a, "word": w}


# ---- clips --------------------------------------------------------------------------------------------------------
# A clip: {id, source, speaker, wav, durationS, from, to (word refs), order, origin, tags?, candidates?}. `order` sorts
# a speaker's clips in the order they are joined: chunk number for sobolev, mushaf order otherwise.

def clip_id(source: str, *parts) -> str:
    return f"{source}-" + "-".join(re.sub(r"[^A-Za-z0-9]+", "", str(p))[:12] for p in parts)


def sobolev_clips(rows: list[dict], ayah_words: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    """(clips, dropped) from metadata.jsonl rows: Hafs only, one clip per chunk; a chunk with reviewer tags carries them
    as candidates ({word, position, tags}), a candidate real mistake rather than clean audio."""
    clips, dropped = [], []
    for r in rows:
        if r.get("riwayah") != "Hafs":
            continue
        m = CHUNK.search(r["file_name"])
        s, a = int(r["surah"]), int(r["ayah"])
        words = ayah_words.get(f"{s}:{a}")
        cid = clip_id("sob", r["recording_id"][:8], f"c{m[1]}" if m else r["file_name"])
        if not m or not words:
            dropped.append({"id": cid, "source": "sobolev", "kind": "label", "why": "no chunk number" if not m else f"no ayah {s}:{a}"})
            continue
        cands = []
        for word, tags in (r.get("errors") or {}).items():
            pos = next((i + 1 for i, w in enumerate(words) if norm(w) == norm(word)), None)
            cands.append({"word": word, "position": pos, "tags": list(tags)})
        clips.append({"id": cid, "source": "sobolev", "speaker": f"sobolev/{r['recording_id']}", "file": r["file_name"],
                      "from": at(s, a, 1), "to": at(s, a, len(words)), "order": int(m[1]),
                      "origin": {"recording": r["recording_id"], "chunk": int(m[1]), "reviewer": r.get("reviewer_id"), "text": r.get("text")},
                      **({"candidates": cands, "tags": sorted({t for c in cands for t in c["tags"]})} if cands else {})})
    return clips, dropped


def ayah_texts(ayah_words: dict[str, list[str]]) -> dict[str, str]:
    return {k: norm("".join(ws)) for k, ws in ayah_words.items()}


def map_text(text: str, surah: int | str | None, ayah_words: dict[str, list[str]], texts: dict[str, str]) -> tuple[dict | None, dict | None, float]:
    """(from, to, distance) of the words `text` matches best: a whole ayah (in `surah` when known, or the one ayah
    `"s:a"` given), else a run of one ayah's words when the clip is a fragment (Ayat al-Kursi's clips are). Distance:
    letter edits over the matched text's length."""
    from rapidfuzz.distance import Levenshtein

    t = norm(text)
    if not t:
        return None, None, 1.0
    d = lambda ref: Levenshtein.distance(t, ref) / max(1, len(ref))
    if isinstance(surah, str):
        pool = [surah]
    else:
        pool = [k for k in texts if surah is None or k.split(":")[0] == str(surah)] or list(texts)
    best = min(pool, key=lambda k: (d(texts[k]), k))
    s, a = map(int, best.split(":"))
    out = (at(s, a, 1), at(s, a, len(ayah_words[best])), d(texts[best]))
    for k in pool[:8] if len(pool) <= 8 else [best]:
        ws = [norm(w) for w in ayah_words[k]]
        s, a = map(int, k.split(":"))
        for i in range(len(ws)):
            for j in range(i + 1, len(ws) + 1):
                x = d("".join(ws[i:j]))
                if x < out[2] - 1e-9:
                    out = (at(s, a, i + 1), at(s, a, j), x)
    return out[0], out[1], round(out[2], 3)


def retasy_clips(items: list[dict], ayah_words: dict[str, list[str]], max_text_dist: float = 0.15) -> tuple[list[dict], list[dict]]:
    """(clips, dropped) from /data/public/retasy/manifest.json items: `correct`, Quranic, Hafs or no qiraah given, one
    copy of identical audio, the Aya text mapped to its words. Clips of reciter `Unknown` (160 clips from different
    people, with no other speaker fields) are each their own speaker."""
    texts = ayah_texts(ayah_words)
    clips, dropped = [], []
    for it in items:
        if it["label"] != "correct" or not it["quranic"]:
            continue
        cid = clip_id("ret", it["id"][:12])
        kind, why = (("qiraah", f"qiraah {it['reciter_qiraah']}") if it["reciter_qiraah"] not in (None, "hafs") else
                     ("audio", "no audio") if not it.get("wav") else ("duplicate", f"same audio as {it['dup_of']}") if it.get("dup_of") else (None, None))
        if why:
            dropped.append({"id": cid, "source": "retasy", "kind": kind, "why": why})
            continue
        frm, to, dist = map_text(it["aya_text"], RETASY_SURAHS.get(it["surah"]), ayah_words, texts)
        if frm is None or dist > max_text_dist:
            dropped.append({"id": cid, "source": "retasy", "kind": "label", "why": f"Aya text matches no ayah ({dist:.2f} at best)"})
            continue
        spk = it["reciter_id"] if it["reciter_id"] and it["reciter_id"] != "Unknown" else f"unknown-{it['id']}"
        clips.append({"id": cid, "source": "retasy", "speaker": f"retasy/{spk}", "file": it["wav"], "from": frm, "to": to, "order": None,
                      "origin": {"item": it["id"], "surah": it["surah"], "text": it["aya_text"], "textDistance": dist, "qiraah": it["reciter_qiraah"],
                                 "createDate": it.get("create_date"), "gender": it.get("reciter_gender"), "age": it.get("reciter_age")}})
    return clips, dropped


def tlog_clips(rows: list[dict]) -> list[dict]:
    """Clips from tlog-v2 clean-match rows with a span label (text-only labels have no location)."""
    out = []
    for r in rows:
        if r["label"]["kind"] != "span":
            continue
        sp = r["label"]["span"]
        out.append({"id": clip_id("tlog", r["shard"].replace("clean-", ""), r["row"]), "source": "tlog", "speaker": f"tlog/{r['cluster']}",
                    "file": f"{r['shard']}:{r['row']}", "from": dict(sp["start"]), "to": dict(sp["end"]), "order": None, "durationS": r["durationS"],
                    "origin": {"shard": r["shard"], "row": r["row"], "path": r["path"], "cluster": r["cluster"], "per": r["per"]}})
    return out


# ---- splits and sessions ------------------------------------------------------------------------------------------

def splits(clips: list[dict], seed: int = SEED) -> dict:
    """speaker -> practice / exam: per source, the seeded first half of its speakers (rounded down) is exam."""
    out = {}
    for src in SOURCES:
        spk = sorted({c["speaker"] for c in clips if c["source"] == src}, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
        for i, s in enumerate(spk):
            out[s] = "exam" if i < len(spk) // 2 else "practice"
    return out


def chains(clips: list[dict], word_index, rng: random.Random) -> list[list[dict]]:
    """One speaker's clips as maximal runs where each starts at the word after the last one's end, one clip per start."""
    by = defaultdict(list)
    for c in clips:
        by[word_index(c["from"])].append(c)
    used, out = set(), []
    for start in sorted(by):
        if start in used:
            continue
        run, k = [], start
        while k in by and k not in used:
            c = rng.choice(by[k])
            used.add(k)
            run.append(c)
            k = word_index(c["to"]) + 1
        out.append(run)
    return out


def tlog_sample(clips: list[dict], word_index, target_s: float = TLOG_TARGET_S, chain_max_s: float = TLOG_CHAIN_MAX_S, seed: int = SEED) -> list[list[dict]]:
    """Chains of one cluster's contiguous clips, cut at `chain_max_s`, taken round-robin over the clusters in a seeded
    order (shorter chains first in each cluster) until `target_s` of audio; chains under MIN_S are not taken."""
    rng = random.Random(f"{seed}:tlog")
    by = defaultdict(list)
    for c in clips:
        by[c["speaker"]].append(c)
    queues = {}
    for spk in sorted(by):
        runs = []
        for ch in chains(by[spk], word_index, rng):
            cur = []
            for c in ch:
                if cur and sum(x["durationS"] for x in cur) + c["durationS"] > chain_max_s:
                    runs.append(cur)
                    cur = []
                cur.append(c)
            runs.append(cur)
        runs = [r for r in runs if sum(x["durationS"] for x in r) >= MIN_S]
        rng.shuffle(runs)
        queues[spk] = runs
    order = sorted(queues)
    rng.shuffle(order)
    out, total = [], 0.0
    while total < target_s and any(queues.values()):
        for spk in order:
            if queues[spk] and total < target_s:
                r = queues[spk].pop()
                out.append(r)
                total += sum(x["durationS"] for x in r)
    return out


def passes(clips: list[dict], ordinal) -> list[list[dict]]:
    """One speaker's clips dealt into passes, each strictly in mushaf order: a clip goes to the first pass that ends
    before it, so a reciter's repeats of an ayah land in separate passes."""
    out: list[list[dict]] = []
    for c in sorted(clips, key=lambda c: (ordinal(c["from"]), c["origin"].get("createDate") or "", c["id"])):
        o = ordinal(c["from"])
        p = next((p for p in out if ordinal(p[-1]["to"]) < o), None)
        if p is None:
            out.append([c])
        else:
            p.append(c)
    return out


def cut(run: list[dict], max_s: float = SESSION_MAX_S) -> list[list[dict]]:
    out, cur = [], []
    for c in run:
        if cur and sum(x["durationS"] for x in cur) + c["durationS"] > max_s:
            out.append(cur)
            cur = []
        cur.append(c)
    return out + [cur] if cur else out


def plan_sessions(clips: list[dict], split_of: dict[str, str], ordinal, word_index, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """(sessions, dropped). A session: {id, source, speaker, split, clips}. sobolev: a recording's chunks in chunk order.
    retasy: one pass per session; a pass under MIN_S joins the session before it. tlog: one sampled chain per session.
    Each speaker keeps sessions up to SPEAKER_MAX_S, in a seeded order. Groups still under MIN_S are dropped."""
    rng = random.Random(f"{seed}:sessions")
    by = defaultdict(list)
    for c in clips:
        by[(c["source"], c["speaker"])].append(c)
    groups, dropped = [], []
    for (src, spk), cs in sorted(by.items()):
        if src == "sobolev":
            runs = cut(sorted(cs, key=lambda c: c["order"]))
        elif src == "retasy":
            runs = []
            for p in passes(cs, ordinal):
                if sum(x["durationS"] for x in p) < MIN_S and runs:
                    runs[-1] = runs[-1] + p
                else:
                    runs.append(p)
            runs = [r for p in runs for r in cut(p)]
        else:  # tlog: the sampled chains (`chain`, from tlog_sample)
            byc = defaultdict(list)
            for c in cs:
                byc[c["chain"]].append(c)
            runs = [sorted(byc[k], key=lambda c: word_index(c["from"])) for k in sorted(byc)]
        short = [r for r in runs if sum(x["durationS"] for x in r) < MIN_S]
        dropped += [{"id": c["id"], "source": src, "kind": "short", "why": f"under {MIN_S:.0f} s even joined ({sum(x['durationS'] for x in r):.1f} s)"} for r in short for c in r]
        runs = [r for r in runs if r not in short]
        rng.shuffle(runs)
        kept, total = [], 0.0
        for r in runs:
            d = sum(x["durationS"] for x in r)
            if kept and total + d > SPEAKER_MAX_S:
                dropped += [{"id": c["id"], "source": src, "kind": "speaker_cap", "why": f"speaker over {SPEAKER_MAX_S / 60:.0f} min"} for c in r]
                continue
            kept.append(r)
            total += d
        groups += [(src, spk, r) for r in kept]
    sessions = []
    for src in SOURCES:
        mine = sorted((g for g in groups if g[0] == src), key=lambda g: (g[1], g[2][0]["id"]))
        for n, (_, spk, r) in enumerate(mine, 1):
            sessions.append({"id": f"{TEST_SET}-{src[:3]}-{n:03d}", "source": src, "speaker": spk, "split": split_of[spk], "clips": [c["id"] for c in r]})
    return sessions, dropped


# ---- clip verification --------------------------------------------------------------------------------------------

def best_span(heard: str, words: list[str], prefixes: list[str], reach: int = 2) -> tuple[int, int, float] | None:
    """The run of `words` (canon phonemes, in mushaf order) a clip's decode matches best: (first, last, PER) with
    `prefixes` allowed before it. The best partial alignment of the decode in the words' phonemes, its ends snapped to
    word boundaries within `reach` words."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    if not heard or not words:
        return None
    offs, k = [], 0
    for w in words:
        offs.append(k)
        k += len(w)
    text = "".join(words)
    if len(heard) < len(text):
        al = fuzz.partial_ratio_alignment(heard, text)
        a, b = al.dest_start, al.dest_end
    else:
        a, b = 0, len(text)
    word_at = lambda c: max(0, min(len(words) - 1, next((i for i in range(len(offs) - 1, -1, -1) if offs[i] <= c), 0)))
    i0, i1 = word_at(a), word_at(max(a, b - 1))
    best = None
    for i in range(max(0, i0 - reach), min(len(words), i0 + reach + 1)):
        for j in range(max(i, i1 - reach), min(len(words), i1 + reach + 1)):
            t = "".join(words[i : j + 1])
            x = min(Levenshtein.distance(heard, p + t) for p in prefixes) / max(1, len(t))
            if best is None or x < best[2] - 1e-9:
                best = (i, j, x)
    return best


def verify_clip(heard: str, expected: str, prefixes: list[str], alternatives: dict[str, str] | None = None,
                span: tuple[int, int, float] | None = None) -> dict:
    """The teacher's greedy decode of a clip (`heard`, canon phonemes) against its label (`expected`), allowing any of
    `prefixes` (an isti'adha, a basmala, both) said before it: ok within PER_OK. Else relabel, when within PER_OK and
    RELABEL_MARGIN better than the label: to `span` (the best run of words in the label's surah, `best_span`, as
    (first, last, PER); sobolev's chunks often hold more or less than the ayah they are labelled with), else to the
    best of `alternatives` (key -> canon phonemes, whole ayahs). Else unsure up to PER_DROP (kept, heard differently:
    a likely real mistake), else drop."""
    from rapidfuzz.distance import Levenshtein

    def per(t: str) -> float:
        return min(Levenshtein.distance(heard, p + t) for p in prefixes) / max(1, len(t))

    p = round(per(expected), 3)
    if p <= PER_OK:
        return {"status": "ok", "per": p}
    if span and span[2] <= PER_OK and span[2] <= p - RELABEL_MARGIN:
        return {"status": "relabel", "per": p, "span": [span[0], span[1]], "perTo": round(span[2], 3)}
    best, bp = None, 1.0
    for k, t in (alternatives or {}).items():
        if t and 0.5 <= len(heard) / max(1, len(t)) <= 3.0:
            x = per(t)
            if x < bp:
                best, bp = k, x
    bp = round(bp, 3)
    if best is not None and bp <= PER_OK and bp <= p - RELABEL_MARGIN:
        return {"status": "relabel", "per": p, "to": best, "perTo": bp}
    if p <= PER_DROP:
        return {"status": "unsure", "per": p, **({"best": best, "perBest": bp} if best else {})}
    return {"status": "drop", "per": p, **({"best": best, "perBest": bp} if best else {})}


# ---- the script and the labels ------------------------------------------------------------------------------------

def clip_items(ref, c: dict, k: int, optional: dict[str, dict], basmala: dict, separate: bool) -> list[dict]:
    """Said items of clip `c` (index k in the session): its words in their said form (a pause before the first word and
    after the last), with an optional isti'adha before a clip that is its own recording (or the first), and an optional
    basmala before one that starts at a surah's first ayah (or starts a separate recording, outside al-Fatiha and
    at-Tawba)."""
    import lab_sessions as S

    i0, i1 = ref.idx[(c["from"]["surah"], c["from"]["ayah"], c["from"]["word"])], ref.idx[(c["to"]["surah"], c["to"]["ayah"], c["to"]["word"])]
    out = []
    if separate or k == 0:
        out.append({**optional["istiadha"], "clip": k})
    s0, a0, w0 = c["from"]["surah"], c["from"]["ayah"], c["from"]["word"]
    if w0 == 1 and s0 not in NO_BASMALA and (a0 == 1 or separate or k == 0):
        out.append({"at": None, "text": basmala["text"], "phonemes": basmala["phonemes"], "role": "basmala", "clip": k})
    for i in range(i0, i1 + 1):
        w = ref.words[i]
        ph = S.said_form(w, i == i0, i == i1) or (w["iso"] if i == i0 == i1 else w["cont"])
        out.append({"at": ref.ref(i), "text": w["text"], "phonemes": ph, "role": "ref", "clip": k})
    return out


def script(ref, sid: str, clips: list[dict], optional: dict[str, dict], basmala: dict) -> dict:
    """A clean script for `lab_recordings_core.check_take`: every clip's said items in session order (`clip` is the clip's
    index), no plants. sobolev chunks are one recording; other sources' clips are separate recordings."""
    said = [x for k, c in enumerate(clips) for x in clip_items(ref, c, k, optional, basmala, c["source"] != "sobolev")]
    body = {"id": sid, "version": 1, "tag": "realuser", "said": said, "plants": [], "labels": {"events": [], "nonErrors": []},
            "clips": [c["id"] for c in clips]}
    return {**body, "sha": digest(body)}


def join_plan(durations: list[float], gap_s: float = GAP_S, edge_s: float = EDGE_S) -> list[tuple[float, float]]:
    """Each clip's (start, end) on the session clock: EDGE_S of silence first and last, GAP_S between clips."""
    out, t = [], edge_s
    for d in durations:
        out.append((round(t, 3), round(t + d, 3)))
        t += d + gap_s
    return out


def join_audio(waves: list, sr: int = 16_000):
    """The session audio (float32): clips level-matched to their median speech level (at most GAIN_MAX_DB either way),
    faded in and out over FADE_S, laid out by `join_plan`. Returns (audio, gains in dB)."""
    import numpy as np

    import lab_sessions as S

    levels = [20 * np.log10(max(S.active_rms(x), 1e-6)) for x in waves]
    mid = float(np.median(levels))
    gains = [float(np.clip(mid - lv, -GAIN_MAX_DB, GAIN_MAX_DB)) for lv in levels]
    plan = join_plan([len(x) / sr for x in waves])
    total = round((plan[-1][1] + EDGE_S) * sr)
    out = np.zeros(total, dtype=np.float32)
    f = round(FADE_S * sr)
    for x, g, (a, _) in zip(waves, gains, plan):
        y = np.asarray(x, dtype=np.float32) * np.float32(10 ** (g / 20))
        if len(y) > 2 * f:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            y[:f] *= ramp
            y[-f:] *= ramp[::-1]
        i = round(a * sr)
        out[i : i + len(y)] += y[: total - i]
    return np.clip(out, -1.0, 1.0), [round(g, 1) for g in gains]


def session(plan: dict, clips: list[dict], result: dict, audio: str, placed: list[tuple[float, float]], word_index) -> dict:
    """A joined session as a BenchSession (packages/core/src/bench/sessions.ts), source `recorded` (real device audio), no
    events. One span per clip's heard words: `repeat` when it starts at or before a word already recited (its
    re-recited words are `repeat` non-errors), else `recitation`. Kept optional items are non-Quran spans. `meta`
    carries the clips, the seams between them (contiguous or a jump), the ayahs heard differently and sobolev's
    reviewer tags as candidate real mistakes."""
    timed = result["said"]
    spans, words, non_errors = [], [], []
    top = -1
    for k, c in enumerate(clips):
        mine = [t for t in timed if t.get("clip") == k]
        for t in mine:
            if t["role"] in ("istiadha", "basmala", "closing") and t["endS"] > t["startS"]:
                spans.append({"id": f"{t['role']}-{len(spans) + 1}", "kind": "non_quran", "label": t["role"], "startS": t["startS"], "endS": t["endS"]})
        heard = [t for t in mine if t["at"] and t["endS"] > t["startS"]]
        if not heard:
            continue
        first = word_index(heard[0]["at"])
        sp = {"id": f"span-{len(spans) + 1}", "kind": "repeat" if first <= top else "recitation", "from": heard[0]["at"], "to": heard[-1]["at"],
              "startS": heard[0]["startS"], "endS": heard[-1]["endS"]}
        spans.append(sp)
        for t in heard:
            words.append({"span": sp["id"], "at": t["at"], "startS": t["startS"], "endS": t["endS"]})
            if word_index(t["at"]) <= top:
                non_errors.append({"span": sp["id"], "at": t["at"], "kind": "repeat", "startS": t["startS"], "endS": t["endS"]})
        top = max(top, word_index(heard[-1]["at"]))
    spans.sort(key=lambda x: x["startS"])
    for sp in spans:
        sp["startS"], sp["endS"] = round(sp["startS"], 3), round(sp["endS"], 3)
    seams = []
    for k in range(1, len(clips)):
        contiguous = word_index(clips[k]["from"]) == word_index(clips[k - 1]["to"]) + 1
        seams.append({"atS": round((placed[k - 1][1] + placed[k][0]) / 2, 3), "kind": "contiguous" if contiguous else "jump", "before": clips[k]["id"]})
    cands = []
    for k, c in enumerate(clips):
        for x in c.get("candidates", []):
            ws = [t for t in timed if t.get("clip") == k and t["at"] and (x["position"] is None or t["at"]["word"] == x["position"]) and t["endS"] > t["startS"]]
            a, b = (ws[0]["startS"], ws[-1]["endS"]) if ws else placed[k]
            cands.append({"clip": c["id"], "ayah": f"{c['from']['surah']}:{c['from']['ayah']}", "word": x["word"], "position": x["position"], "tags": x["tags"],
                          "startS": round(a, 3), "endS": round(b, 3)})
    flagged = [{k: a[k] for k in ("key", "startS", "endS", "per", "greedy", "said") if k in a} for a in result["ayahs"] if a["flag"]]
    meta = {"testSet": TEST_SET, "source": plan["source"], "speaker": plan["speaker"], "split": plan["split"],
            "clips": [{"id": c["id"], "startS": p[0], "endS": p[1], "from": c["from"], "to": c["to"], "verify": c.get("verify"),
                       **({"relabelledFrom": c["relabelledFrom"]} if c.get("relabelledFrom") else {}), "origin": c["origin"]} for c, p in zip(clips, placed)],
            "seams": seams, "resumesS": [x["atS"] for x in seams], "flagged": flagged, "candidates": cands,
            "check": {k: result.get(k) for k in ("coverage", "heard", "dropped", "durationS")}, "clean": True}
    return {"version": 1, "id": plan["id"], "track": "recitation", "speaker": plan["speaker"], "source": "recorded", "audio": audio,
            "durationS": result["durationS"], "context": {"kind": "none"}, "spans": spans, "words": words, "events": [], "nonErrors": non_errors,
            "said": [{k: t[k] for k in ("startS", "endS", "at", "text", "phonemes", "role", "clip")} for t in timed], "meta": meta}


def manifest(sessions: list[dict]) -> dict:
    """Test set realuser-v1, shaped like lab_recordings_core.manifest: phone only, `slices.source` "realuser" with the
    source it came from as `slices.origin`."""
    return {"version": 1, "testSet": TEST_SET, "reference": "ref-v2", "headline": "phone", "variants": ["phone"],
            "evidencePattern": "sessions/{id}.{listener}.{variant}.evidence.json",
            "sessions": [{"id": s["id"], "split": s["meta"]["split"], "speaker": s["speaker"], "source": "realuser", "durationS": s["durationS"],
                          "tags": ["realuser", s["meta"]["source"]], "slices": {"source": "realuser", "origin": s["meta"]["source"], "clean": True},
                          "audio": {"phone": s["audio"]}, "label": f"sessions/{s['id']}.json",
                          "session": {k: v for k, v in s.items() if k not in ("said", "meta")}} for s in sorted(sessions, key=lambda s: s["id"])]}


def join(base: dict, real: dict, prefix: str = "sessions/") -> dict:
    """A test set manifest with the realuser sessions appended (their files copied beside the base's under `prefix`)."""
    ids = {e["id"] for e in base["sessions"]}
    out = []
    for e in real["sessions"]:
        if e["id"] in ids:
            raise ValueError(f"session {e['id']} is already in {base['testSet']}")
        sess = dict(e["session"], audio=f"{prefix}{e['id']}.phone.wav")
        out.append({**e, "audio": {"phone": sess["audio"]}, "label": f"{prefix}{e['id']}.json", "session": sess})
    return {**base, "sessions": base["sessions"] + out, "realuser": {"from": real["testSet"], "sessions": len(out)}}


SEAM_EXCLUDE_S = (1.0, 3.0)  # a jump seam's window, before and after it


def seam_exclusions(sessions: list[dict], test_set: str = TEST_SET, extra: list[dict] = ()) -> dict:
    """A `LabExclusions` list (lab.ts) with a window around every jump seam: where joined clips skip ahead or go
    elsewhere, the joined audio reads as a skipped passage that no reciter made (a clip the source lacks, or another
    recording), so the engine's calls there are about the join. `extra`: review verdicts' windows to add."""
    a, b = SEAM_EXCLUDE_S
    out = [{"session": s["id"], "startS": round(max(0.0, x["atS"] - a), 3), "endS": round(min(s["durationS"], x["atS"] + b), 3), "verdict": "bad_splice",
            "note": f"join seam: the clips jump here (before {x['before']})", "item": f"seam-{s['id']}-{x['atS']}"}
           for s in sorted(sessions, key=lambda s: s["id"]) for x in s["meta"]["seams"] if x["kind"] == "jump"]
    return {"version": 1, "kind": "lab-flags", "testSet": test_set, "base": test_set, "excluded": out + list(extra)}


def flagged_queue(sessions: list[dict], root: str = TEST_SET) -> dict:
    """Practice ayahs the teacher hears differently from their label (likely real user mistakes) and practice
    sobolev reviewer tags, as a ticket 06 review queue. Exam sessions are never queued."""
    items = []
    for s in sessions:
        m = s["meta"]
        if m["split"] != "practice":
            continue
        wav = f"{root}/{s['audio']}"
        text = lambda a, b: " ".join(w["text"] for w in s["said"] if w["at"] and w["endS"] > a and w["startS"] < b)
        for f in m["flagged"]:
            a, b = round(max(0.0, f["startS"] - 0.5), 3), round(min(s["durationS"], f["endS"] + 0.5), 3)
            items.append({"id": f"{s['id']}-ayah-{f['key'].replace(':', '-')}-{int(f['startS'])}", "kind": "case", "title": "Heard differently from its label",
                          "audio": {"path": wav, "startS": a, "endS": b}, "text": text(a, b), "label": {"type": "clean"},
                          "context": f"{f['key']} · {m['source']} {m['speaker']} at {f['startS']:.1f} s · teacher PER {f['per']:.2f}: a real slip, or said as written?"})
        for c in m["candidates"]:
            a, b = round(max(0.0, c["startS"] - 1.0), 3), round(min(s["durationS"], c["endS"] + 1.0), 3)
            items.append({"id": f"{s['id']}-tag-{c['clip']}-{c['position'] or 0}", "kind": "case", "title": f"Reviewer tag: {', '.join(c['tags'])}",
                          "audio": {"path": wav, "startS": a, "endS": b}, "text": text(a, b), "label": {"type": "clean"},
                          "context": f"{c['ayah']} «{c['word']}» · sobolev reviewer tag {', '.join(c['tags'])}: a memorization mistake, or not?"})
    return {"version": 1, "testSet": root, "maxBad": max(1, len(items) // 40), "items": items}


def summary(found: dict[str, int], clips: list[dict], dropped: list[dict], sessions: list[dict]) -> dict:
    """Per source: clips found, kept, relabelled, dropped by reason; sessions, speakers and minutes per split."""
    out = {}
    for src in SOURCES:
        ss = [s for s in sessions if s["meta"]["source"] == src]
        out[src] = {"found": found.get(src, 0), "joined": sum(len(s["meta"]["clips"]) for s in ss),
                    "dropped": dict(Counter(d["kind"] for d in dropped if d["source"] == src).most_common()),
                    "relabelled": sum(c["source"] == src and bool(c.get("relabelledFrom")) for c in clips),
                    "unsure": sum(c["source"] == src and (c.get("verify") or {}).get("status") == "unsure" for c in clips),
                    "splits": {sp: {"sessions": sum(s["meta"]["split"] == sp for s in ss), "speakers": len({s["speaker"] for s in ss if s["meta"]["split"] == sp}),
                                    "minutes": round(sum(s["durationS"] for s in ss if s["meta"]["split"] == sp) / 60, 1),
                                    "clips": sum(len(s["meta"]["clips"]) for s in ss if s["meta"]["split"] == sp)} for sp in ("practice", "exam")}}
    return out
