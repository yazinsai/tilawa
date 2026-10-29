"""Pure logic for scripted-error recordings (.scratch/lab/issues/18-scripted-recordings.md, docs/benchmarks/lab-recordings.md).

Served by modal/lab_review.py (the /record page) and run by modal/lab_recordings.py (processing). Take layout on the
tilawa-lab volume, under /lab/data/recordings/:

    scripts/<batch>.json                    script batches (tools/lab_scripts.py)
    <speaker>/<script>/take-<n>.<ext>       the upload as the browser recorded it (webm, mp4, ogg or wav)
    <speaker>/<script>/take-<n>.json        upload metadata (speaker, script id/version/sha, device, time)
    <speaker>/<script>/take-<n>.result.json the checks and the verdict (ok, listen, redo)
    sessions/<id>.phone.wav                 the take as 16 kHz mono PCM16 (the test set's `phone` audio)
    sessions/<id>.json                      the take's BenchSession; <id>.teacher.phone.evidence.json and
                                            <id>.checks.json (the WAV's sha256) beside it
    review/takes.json                       the spot-check queue (ticket 06 format)
    manifest-v1.json                        accepted takes, shaped like a test set manifest

The numeric checks (`check_take`) need numpy and the lab checks' CTC helpers; everything else is standard library.
"""

import json
import math
import random
import re
from collections import Counter, defaultdict

SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ROOT = "recordings"
TEST_SET = "recordings"
EXTS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/aac": "aac"}
MAX_UPLOAD = 200 * 1024 * 1024
TAKE = re.compile(r"^take-(\d+)\.(webm|ogg|mp4|mp3|wav|aac)$")
LLR_MIN = 3.0  # nats: the teacher must prefer what the script says was said by this much (lab_checks.LLR_MIN)
LOCAL_PER_MAX = 0.25
AYAH_PER_MAX = 0.15  # an ayah the teacher hears this differently from the script holds an unscripted change (lab_sources.MAX_PER)
BASMALA_PER_MAX = 0.3  # a scripted basmala heard worse than this wasn't said; it is aligned again without it
COVERAGE_MIN = 0.9
PAD_S = 0.1
REDO_FLAGGED_AYAHS = 3
SPOT = {"performed": 3, "selfCorrections": 1, "clean": 2}  # per take; every unsure plant and flagged ayah is queued too
CLEAN_WINDOW_S = 8.0
SEED = 20261029


def safe(v: str, what: str) -> str:
    if not SAFE.match(v or ""):
        raise ValueError(f"{what} must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    return v


def ext_for(mime: str) -> str:
    base = (mime or "").split(";")[0].strip().lower()
    if base not in EXTS:
        raise ValueError(f"unsupported audio type {mime!r}")
    return EXTS[base]


def take_dir(speaker: str, script: str) -> str:
    return f"{ROOT}/{safe(speaker, 'speaker')}/{safe(script, 'script')}"


def next_take(names: list[str]) -> int:
    return max((int(m[1]) for n in names if (m := TAKE.match(n))), default=0) + 1


def take_files(speaker: str, script: str, n: int, ext: str) -> dict:
    d, sid = f"{take_dir(speaker, script)}/take-{n}", session_id(speaker, script, n)
    return {"upload": f"{d}.{ext}", "meta": f"{d}.json", "result": f"{d}.result.json", "wav": f"{ROOT}/sessions/{sid}.phone.wav",
            "session": f"{ROOT}/sessions/{sid}.json", "evidence": f"{ROOT}/sessions/{sid}.teacher.phone.evidence.json",
            "checks": f"{ROOT}/sessions/{sid}.checks.json"}


def session_id(speaker: str, script: str, n: int) -> str:
    return f"rec-{speaker}-{script}-t{n}"


def take_meta(form: dict, script: dict, n: int, ext: str, at: str, size: int) -> dict:
    """The upload's metadata record from the page's form fields, checked against the script it names."""
    speaker = safe(form.get("speaker", ""), "speaker")
    if form.get("scriptVersion") not in (None, "", str(script["version"])) or form.get("scriptSha") not in (None, "", script["sha"]):
        raise ValueError(f"the page has an old copy of {script['id']}; reload it")
    clip = lambda k, m=200: str(form.get(k) or "")[:m]
    num = lambda k: float(form[k]) if str(form.get(k) or "").replace(".", "", 1).isdigit() else None
    return {"version": 1, "speaker": speaker, "script": script["id"], "scriptVersion": script["version"], "scriptSha": script["sha"],
            "batch": form.get("batch") or None, "take": n, "ext": ext, "bytes": size, "mime": clip("mime", 80), "device": clip("device"),
            "userAgent": clip("userAgent", 300), "trackSettings": clip("trackSettings", 500), "recordedAt": clip("recordedAt", 40) or at,
            "uploadedAt": at, "clientDurationS": num("durationS"), "note": clip("note", 2000)}


# ---- checks -------------------------------------------------------------------------------------------------------

def items(script: dict, drop_basmala: set[int] | None = None) -> list[dict]:
    """What the script says is recited, one item per word (`said`), optionally without some basmala items (indexes)."""
    return [dict(x, i=i) for i, x in enumerate(script["said"]) if not (drop_basmala and i in drop_basmala)]


def plant_window(its: list[dict], plant: dict) -> tuple[int, int, list[int]] | None:
    """(item before, item after, the plant's own items) for one plant; an omission has no items of its own."""
    mine = [k for k, it in enumerate(its) if it.get("plant") == plant["id"]]
    if mine:
        lo, hi = mine[0] - 1, mine[-1] + 1
    else:  # omitted: the gap between the word before and the word after
        loc = plant["location"]
        nxt = next((k for k, it in enumerate(its) if it["at"] and it["role"] == "ref" and (it["at"]["surah"], it["at"]["ayah"]) == (loc["surah"], loc["ayah"])
                    and it["at"]["word"] == loc["word"] + 1), None)
        if nxt is None:
            return None
        lo, hi = nxt - 1, nxt
    if lo < 0 or hi >= len(its):
        return None
    return lo, hi, mine


def decide(llr: float, d_said: int, d_orig: int, local: float) -> tuple[str, list[str]]:
    """performed / not_performed / unsure from the teacher's LLR (said vs original), its greedy decode and local PER."""
    reasons = []
    if llr <= -LLR_MIN:
        return "not_performed", [f"the teacher hears the original text here (llr {llr:.1f})"]
    if llr < LLR_MIN:
        reasons.append(f"the teacher can't tell it from the original (llr {llr:.1f})")
    if not d_said < d_orig:
        reasons.append(f"the teacher's decode is no closer to the scripted words ({d_said} edits vs {d_orig} to the original)")
    if local > LOCAL_PER_MAX:
        reasons.append(f"the teacher doesn't hear the scripted words cleanly (local PER {local:.2f})")
    return ("unsure" if reasons else "performed"), reasons


def take_verdict(aligned: bool, plants: list[dict], ayahs: list[dict], coverage: float) -> tuple[str, list[str]]:
    """ok, listen (something for a human to hear) or redo (the take doesn't follow the script)."""
    if not aligned:
        return "redo", ["the recording doesn't match the script's text (alignment failed)"]
    why = [f"mistake {p['n']} ({p['ayah']}) not performed: {p['reasons'][0]}" for p in plants if p["status"] == "not_performed"]
    flagged = [a for a in ayahs if a["flag"]]
    if coverage < COVERAGE_MIN:
        why.append(f"only {coverage:.0%} of the script's words were found")
    if len(flagged) >= REDO_FLAGGED_AYAHS:
        why.append(f"{len(flagged)} ayahs sound different from the script")
    if why:
        return "redo", why
    listen = [f"mistake {p['n']} ({p['ayah']}): {p['reasons'][0]}" for p in plants if p["status"] == "unsure"]
    listen += [f"{a['key']} sounds different from the script (PER {a['per']:.2f})" for a in flagged]
    return ("listen", listen) if listen else ("ok", [])


def original_phonemes(script: dict, plant: dict, words: dict[str, str]) -> str:
    """The reference phonemes at a plant's spot: the word replaced (wrong word, self-correction, omitted) or nothing (added)."""
    if plant["class"] == "added_word":
        return ""
    loc = plant["location"]
    return words[f"{loc['surah']}:{loc['ayah']}:{loc['word']}"]


def check_take(script: dict, ev: dict, ref: dict) -> dict:
    """Align what the script says was said to the take's teacher evidence and check every plant and ayah.

    Each scripted basmala the teacher doesn't hear (greedy PER over BASMALA_PER_MAX) is dropped and the rest aligned again.
    """
    import numpy as np
    from rapidfuzz.distance import Levenshtein

    import lab_checks as C
    import lab_sources as L
    from per_breakdown import canon

    fs = ev["frameMs"] / 1000
    lp = C.log_probs(ev)
    words = C.word_phonemes(ref)
    drop: set[int] = set()
    for _ in range(2):
        its = items(script, drop)
        spans = C.align(lp, its, fs)
        if spans is None:
            break
        bad = set()
        for it, (a, b) in zip(its, spans):
            if it["role"] == "basmala":
                g = canon(C.greedy_in(ev, round(a / fs), max(round(a / fs) + 1, round(b / fs))))
                if Levenshtein.distance(g, canon(it["phonemes"])) / max(1, len(canon(it["phonemes"]))) > BASMALA_PER_MAX:
                    bad.add(it["i"])
        if not bad - drop:
            break
        drop |= bad
    duration = ev["nFrames"] * fs
    if spans is None:
        verdict, why = take_verdict(False, [], [], 0.0)
        return {"version": 1, "script": script["id"], "scriptSha": script["sha"], "durationS": round(duration, 3), "aligned": False,
                "verdict": verdict, "why": why, "plants": [], "ayahs": [], "said": [], "coverage": 0.0, "droppedBasmala": sorted(drop)}
    timed = [{**{k: v for k, v in it.items() if k != "i"}, "item": it["i"], "startS": round(a, 3), "endS": round(b, 3)} for it, (a, b) in zip(its, spans)]
    coverage = sum(t["endS"] > t["startS"] for t in timed) / max(1, len(timed))
    plants = []
    for p in script["plants"]:
        w = plant_window(timed, p)
        out = {"id": p["id"], "n": p["n"], "class": p["class"], "ayah": p["ayah"], "location": p["location"]}
        if w is None:
            plants.append({**out, "status": "unsure", "reasons": ["no scripted word on both sides of it"]})
            continue
        lo, hi, mine = w
        prev, nxt = timed[lo], timed[hi]
        # A self-correction is the slip and the words said again against the word said once, as written.
        said_mid, orig_mid = "".join(timed[k]["phonemes"] for k in mine), original_phonemes(script, p, words)
        h_said, h_orig = prev["phonemes"] + said_mid + nxt["phonemes"], prev["phonemes"] + orig_mid + nxt["phonemes"]
        f0, f1 = max(0, math.floor((prev["startS"] - PAD_S) / fs)), min(len(lp), math.ceil((nxt["endS"] + PAD_S) / fs))
        llr = min(C.LLR_CAP, C.score(lp, f0, f1, h_said) - C.score(lp, f0, f1, h_orig))
        g = canon(C.greedy_in(ev, f0, f1))
        d_said, d_orig = Levenshtein.distance(g, canon(h_said)), Levenshtein.distance(g, canon(h_orig))
        local = d_said / max(1, len(canon(h_said)))
        status, reasons = decide(float(llr), d_said, d_orig, local)
        plants.append({**out, "status": status, "reasons": reasons, "llr": round(float(llr), 2), "editsSaid": d_said, "editsOriginal": d_orig,
                       "localPer": round(local, 3), "window": [round(f0 * fs, 3), round(f1 * fs, 3)], "startS": timed[mine[0]]["startS"] if mine else prev["endS"],
                       "endS": timed[mine[-1]]["endS"] if mine else nxt["startS"], "greedy": g, "said": canon(h_said), "original": canon(h_orig)})
    ayahs = []
    groups = []
    for k, t in enumerate(timed):
        key = f"{t['at']['surah']}:{t['at']['ayah']}" if t["at"] else None
        if t["role"] == "basmala":
            continue
        if key is None and groups:  # an added word belongs to the ayah it is said in
            key = groups[-1]["key"]
        if not groups or groups[-1]["key"] != key:
            groups.append({"key": key, "items": []})
        groups[-1]["items"].append(k)
    for gr in groups:
        a, b = timed[gr["items"][0]]["startS"], timed[gr["items"][-1]]["endS"]
        said = canon("".join(timed[k]["phonemes"] for k in gr["items"]))
        g = canon(C.greedy_in(ev, round(a / fs), max(round(a / fs) + 1, round(b / fs))))
        per = Levenshtein.distance(g, said) / max(1, len(said))
        ayahs.append({"key": gr["key"], "startS": a, "endS": b, "per": round(per, 3), "flag": per > AYAH_PER_MAX,
                      **({"greedy": g, "said": said} if per > AYAH_PER_MAX else {})})
    verdict, why = take_verdict(True, plants, ayahs, coverage)
    return {"version": 1, "script": script["id"], "scriptSha": script["sha"], "durationS": round(duration, 3), "aligned": True, "verdict": verdict,
            "why": why, "coverage": round(coverage, 3), "droppedBasmala": sorted(drop), "plants": plants, "ayahs": ayahs, "said": timed,
            "thresholds": {"llrMin": LLR_MIN, "localPerMax": LOCAL_PER_MAX, "ayahPerMax": AYAH_PER_MAX, "basmalaPerMax": BASMALA_PER_MAX,
                           "coverageMin": COVERAGE_MIN, "redoFlaggedAyahs": REDO_FLAGGED_AYAHS}}


# ---- labels -------------------------------------------------------------------------------------------------------

def session(script: dict, result: dict, meta: dict, audio: str) -> dict:
    """The take as a BenchSession (packages/core/src/bench/sessions.ts) on the take's own clock, source `recorded`.

    One span per run of the recitation: a scripted basmala is a non-Quran span, and a self-correction's words said
    again are a `repeat` span with a `self_correction` non-error per word; the slip before it is a word at its place
    with what was said. Planted mistakes are events with the phonemes said (`said`), as in the spliced sessions."""
    timed = result["said"]
    spans, words, non_errors, events = [], [], [], []
    span_of = {}
    cur = None
    for k, t in enumerate(timed):
        if t["role"] == "basmala":
            spans.append({"id": f"basmala-{len(spans) + 1}", "kind": "non_quran", "label": "basmala", "startS": t["startS"], "endS": max(t["endS"], t["startS"] + 0.01)})
            cur = None
            continue
        if t["at"] is None:  # an added word sits in the run it is said in
            if cur is not None:
                span_of[k], cur["endS"] = cur["id"], max(cur["endS"], t["endS"])
            continue
        if t["endS"] <= t["startS"]:  # not heard: no word timing (coverage counts these)
            continue
        retake = t["role"] == "retake"
        if cur is None or (retake and cur["kind"] != "repeat") or (not retake and cur["kind"] == "repeat"):
            cur = {"id": f"span-{len(spans) + 1}", "kind": "repeat" if retake else "recitation", "from": t["at"], "to": t["at"], "startS": t["startS"], "endS": t["endS"]}
            spans.append(cur)
        cur["to"], cur["endS"] = t["at"], max(cur["endS"], t["endS"])
        span_of[k] = cur["id"]
        w = {"span": cur["id"], "at": t["at"], "startS": t["startS"], "endS": t["endS"]}
        if t["role"] in ("wrong", "slip"):
            w["said"] = {"text": t["text"]}
        words.append(w)
        if retake:
            non_errors.append({"span": cur["id"], "at": t["at"], "kind": "self_correction", "startS": t["startS"], "endS": t["endS"]})
    def placed(k: int) -> dict:
        """Item k's own interval, or (not heard) a zero-length one where the next heard word starts."""
        if k in span_of:
            return {"span": span_of[k], "startS": timed[k]["startS"], "endS": timed[k]["endS"]}
        j = next((j for j in range(k + 1, len(timed)) if j in span_of), None) or max(span_of)
        return {"span": span_of[j], "startS": timed[j]["startS"], "endS": timed[j]["startS"]}

    by_plant = {p["id"]: p for p in script["plants"]}
    checked = {p["id"]: p for p in result["plants"]}
    labels = {e["plant"]: e for e in script["labels"]["events"]}
    for pid, e in labels.items():
        p, c = by_plant[pid], checked.get(pid, {})
        mine = [k for k, t in enumerate(timed) if t.get("plant") == pid]
        ev = {"span": None, "errorClass": e["errorClass"], "state": e["state"], "location": e["location"], "forms_other_ayah": e["forms_other_ayah"],
              "expected": e["expected"], "observed": e["observed"], "plant": pid, "check": c.get("status")}
        if e.get("said"):
            ev["said"] = e["said"]
        if mine:  # wrong or added word
            ev |= placed(mine[0])
        else:  # omitted: the gap between the words either side (zero length when they touch)
            lo, hi, _ = plant_window(timed, p)
            if lo in span_of and span_of[lo] == span_of.get(hi):
                ev |= {"span": span_of[lo], "startS": timed[lo]["endS"], "endS": max(timed[lo]["endS"], timed[hi]["startS"])}
            else:
                x = placed(hi)
                ev |= {**x, "endS": x["startS"]}
        events.append(ev)
    sid = session_id(meta["speaker"], script["id"], meta["take"])
    for sp in spans:
        sp["startS"], sp["endS"] = round(sp["startS"], 3), round(sp["endS"], 3)
    return {"version": 1, "id": sid, "track": "recitation", "speaker": f"rec-{meta['speaker']}", "source": "recorded", "audio": audio,
            "durationS": result["durationS"], "context": {"kind": "none"}, "spans": spans, "words": words,
            "events": sorted(events, key=lambda e: e["startS"]), "nonErrors": non_errors,
            "said": [{k: t[k] for k in ("startS", "endS", "at", "text", "phonemes", "role")} | ({"plant": t["plant"]} if t.get("plant") else {}) for t in timed],
            "meta": {"testSet": TEST_SET, "script": script["id"], "scriptVersion": script["version"], "scriptSha": script["sha"], "take": meta["take"],
                     "speaker": meta["speaker"], "device": meta.get("device"), "recordedAt": meta.get("recordedAt"), "upload": meta.get("upload"),
                     "verdict": result["verdict"], "clean": not events, "counts": dict(sorted(Counter(p["class"] for p in script["plants"]).items())),
                     "formsOtherAyah": sum(e["forms_other_ayah"] for e in events), "range": script["range"], "title": script["title"]}}


def validate(s: dict, ref_words: set[tuple[int, int, int]]) -> list[str]:
    """Python mirror of the core `validateSession` rules a take can break (the core check runs in tools/test and on Modal)."""
    out = []
    rank = lambda r: (r["surah"], r["ayah"], r["word"])
    for r in [w["at"] for w in s["words"]] + [e["location"] for e in s["events"]]:
        if rank(r) not in ref_words:
            out.append(f"unknown word {r}")
    spans, last = {}, 0.0
    for p in s["spans"]:
        if p["startS"] < last - 1e-9 or p["endS"] <= p["startS"] or p["endS"] > s["durationS"] + 1e-6:
            out.append(f"span {p['id']} unordered, empty or past the end")
        if p["kind"] != "non_quran" and rank(p["from"]) > rank(p["to"]):
            out.append(f"span {p['id']} reversed")
        spans[p["id"]] = p
        last = p["endS"]
    inside = lambda t: (sp := spans.get(t["span"])) and sp["kind"] != "non_quran" and sp["startS"] - 1e-9 <= t["startS"] <= t["endS"] <= sp["endS"] + 1e-9
    last, prev = 0.0, {}
    for w in s["words"]:
        sp = spans.get(w["span"])
        if not inside(w) or w["endS"] <= w["startS"] or w["startS"] < last - 1e-9:
            out.append(f"word {w['at']} at {w['startS']} outside its span, empty or overlapping")
        elif not rank(sp["from"]) <= rank(w["at"]) <= rank(sp["to"]) or rank(w["at"]) <= prev.get(w["span"], (0, 0, 0)):
            out.append(f"word {w['at']} out of order in {w['span']}")
        prev[w["span"]] = rank(w["at"])
        last = w["endS"]
    for e in s["events"] + s["nonErrors"]:
        if not inside(e):
            out.append(f"truth at {e['startS']} outside its span")
    for e in s["events"]:
        if e["errorClass"] in ("wrong_word", "added_word") and not e.get("said"):
            out.append(f"{e['errorClass']} at {e['location']} lacks said phonemes")
    return out


# ---- review queue and manifest ------------------------------------------------------------------------------------

TITLES = {"wrong_word": "Scripted wrong word", "omitted_word": "Scripted word left out", "added_word": "Scripted extra word",
          "self_correction": "Scripted self-correction (not a mistake)"}
STATUS = {"performed": "the teacher hears it", "unsure": "the teacher isn't sure", "not_performed": "the teacher hears the original"}


def _text(s: dict, a: float, b: float) -> str:
    return " ".join(w["text"] for w in s["said"] if w["endS"] > a and w["startS"] < b and w["role"] != "basmala")


def _window(s: dict, a: float, b: float, pad: float = 2.5) -> tuple[float, float]:
    return round(max(0.0, a - pad), 3), round(min(s["durationS"], b + pad), 3)


def review_items(s: dict, result: dict, script: dict, wav: str, seed: int = SEED) -> list[dict]:
    """One take's spot check: every plant the teacher wasn't sure of, a seeded few it heard, one self-correction, every
    ayah that sounds different from the script, and two seeded clean windows."""
    rng = random.Random(f"{seed}:{s['id']}")
    by = {p["id"]: p for p in script["plants"]}
    meta = s["meta"]
    where = lambda at: f"{at['surah']}:{at['ayah']}, word {at['word']}"
    ctx = lambda t: f"{meta['speaker']} · {script['id']} take {meta['take']} at {t:.1f} s"
    out = []

    def plant_item(c: dict) -> dict:
        p = by[c["id"]]
        a, b = _window(s, c["startS"], c["endS"])
        typ = p["class"]
        label = {"type": typ, "expected": p["expected"] if typ != "added_word" else "", "said": p.get("said", "")}
        if typ == "self_correction":
            label = {"type": "self_correction", "expected": p["expected"], "said": p["said"]}
        return {"id": f"{s['id']}-{p['id']}", "kind": "case", "title": f"{TITLES[typ]} · {STATUS[c['status']]}",
                "audio": {"path": wav, "startS": a, "endS": b}, "text": _text(s, a, b), "label": label,
                "context": f"{where(p['location'])} · {ctx(c['startS'])} · {p['instruction']}" + (f" · llr {c['llr']:+.1f}" if "llr" in c else "")}

    checked = [c for c in result["plants"] if "startS" in c]
    unsure = [c for c in checked if c["status"] != "performed"]
    heard = [c for c in checked if c["status"] == "performed" and c["class"] != "self_correction"]
    sc = [c for c in checked if c["status"] == "performed" and c["class"] == "self_correction"]
    rng.shuffle(heard)
    rng.shuffle(sc)
    for c in sorted(unsure + heard[: SPOT["performed"]] + sc[: SPOT["selfCorrections"]], key=lambda c: c["startS"]):
        out.append(plant_item(c))
    for a in result["ayahs"]:
        if a["flag"]:
            x, y = _window(s, a["startS"], a["endS"], 0.25)
            out.append({"id": f"{s['id']}-ayah-{a['key'].replace(':', '-')}", "kind": "case", "title": "Sounds different from the script",
                        "audio": {"path": wav, "startS": x, "endS": y}, "text": _text(s, x, y), "label": {"type": "clean"},
                        "context": f"{a['key']} · {ctx(a['startS'])} · teacher PER {a['per']:.2f}: an unscripted slip, or a word said differently?"})
    busy = [(e["startS"] - 3, e["endS"] + 3) for e in s["events"]] + [(n["startS"] - 3, n["endS"] + 3) for n in s["nonErrors"]]
    busy += [(a["startS"], a["endS"]) for a in result["ayahs"] if a["flag"]]
    starts, t = [], 5.0
    while t + CLEAN_WINDOW_S < s["durationS"] - 2:
        if not any(a < t + CLEAN_WINDOW_S and t < b for a, b in busy) and _text(s, t, t + CLEAN_WINDOW_S):
            starts.append(t)
        t += CLEAN_WINDOW_S
    for t in sorted(rng.sample(starts, min(SPOT["clean"], len(starts)))):
        ws = [w for w in s["said"] if w["endS"] > t and w["startS"] < t + CLEAN_WINDOW_S and w["role"] != "basmala"]
        a, b = round(max(0.0, ws[0]["startS"] - 0.25), 3), round(min(s["durationS"], ws[-1]["endS"] + 0.25), 3)
        out.append({"id": f"{s['id']}-clean-{int(t)}", "kind": "case", "title": "Recited as scripted, no mistake", "audio": {"path": wav, "startS": a, "endS": b},
                    "text": _text(s, a, b), "label": {"type": "clean"}, "context": ctx(t)})
    return out


def current_takes(results: list[dict]) -> list[dict]:
    """The latest processed take per speaker and script that isn't a redo (`results`: {speaker, script, take, verdict, ...})."""
    best = {}
    for r in sorted(results, key=lambda r: r["take"]):
        if r["verdict"] != "redo":
            best[(r["speaker"], r["script"])] = r
    return [best[k] for k in sorted(best)]


def queue(items_by_take: list[list[dict]]) -> dict:
    flat = [it for its in items_by_take for it in its]
    return {"version": 1, "testSet": TEST_SET, "maxBad": max(1, len(flat) // 40), "items": flat}


def manifest(sessions: list[dict], splits: dict[str, str]) -> dict:
    """Accepted takes shaped like a test set manifest (tools/lab_sessions.manifest): the take is the `phone` variant (it
    is real device audio); there is no studio copy. `splits`: speaker -> practice / exam (default practice)."""
    return {"version": 1, "testSet": TEST_SET, "reference": "ref-v2", "headline": "phone", "variants": ["phone"],
            "evidencePattern": "sessions/{id}.{listener}.{variant}.evidence.json",
            "sessions": [{"id": s["id"], "split": splits.get(s["meta"]["speaker"], "practice"), "speaker": s["speaker"], "source": "recorded",
                          "script": s["meta"]["script"], "durationS": s["durationS"], "tags": ["recorded"],
                          "slices": {"source": "recorded", "clean": s["meta"]["clean"], "formsOtherAyah": s["meta"]["formsOtherAyah"]},
                          "audio": {"phone": s["audio"]}, "label": f"sessions/{s['id']}.json",
                          "session": {k: v for k, v in s.items() if k not in ("said", "meta")}} for s in sorted(sessions, key=lambda s: s["id"])]}


def join(base: dict, recorded: dict, prefix: str) -> dict:
    """A test set manifest with the recorded sessions appended as a `recorded` slice. Recorded paths are rewritten to
    `prefix` (where their files are copied, relative to the new manifest); ids must not clash."""
    ids = {e["id"] for e in base["sessions"]}
    out = []
    for e in recorded["sessions"]:
        if e["id"] in ids:
            raise ValueError(f"session {e['id']} is already in {base['testSet']}")
        sess = dict(e["session"], audio=f"{prefix}{e['id']}.phone.wav")
        out.append({**e, "audio": {"phone": sess["audio"]}, "label": f"{prefix}{e['id']}.json", "session": sess})
    return {**base, "sessions": base["sessions"] + out, "recorded": {"from": TEST_SET, "sessions": len(out)}}


def status_rows(scripts: list[dict], results: list[dict], speaker: str) -> dict:
    """Per script, this speaker's takes (newest first) for the record page."""
    by = defaultdict(list)
    for r in results:
        if r.get("speaker") == speaker:
            by[r["script"]].append(r)
    return {s["id"]: sorted(by[s["id"]], key=lambda r: -r["take"]) for s in scripts}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
