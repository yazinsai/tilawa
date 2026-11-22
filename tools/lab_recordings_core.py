"""Pure logic for scripted-error recordings (.scratch/lab/issues/18-scripted-recordings.md, docs/benchmarks/lab-recordings.md).

Served by modal/lab_review.py (the /record page) and run by modal/lab_recordings.py (processing). Take layout on the
tilawa-lab volume, under /lab/data/recordings/:

    scripts/<batch>.json                    script batches (tools/lab_scripts.py)
    <speaker>/<script>/take-<n>.<ext>       the upload as the browser recorded it (webm, mp4, ogg or wav)
    <speaker>/<script>/take-<n>.json        upload metadata (speaker, script id/version/sha, device, time)
    <speaker>/<script>/take-<n>.result.json the checks and the verdict (ok, listen, redo; refused for a fix that can't be joined)
    <speaker>/<script>/take-<n>.patch.<ext> a fix: the ayahs recorded again; take-<n>.wav is then the base take with it joined in
    sessions/<id>.phone.wav                 the take as 16 kHz mono PCM16 (the test set's `phone` audio)
    sessions/<id>.json                      the take's BenchSession; <id>.teacher.phone.evidence.json and
                                            <id>.checks.json (the WAV's sha256) beside it
    sessions/<id>-patch.phone.wav           a fix's patch as 16 kHz WAV, and its <id>-patch.teacher.phone.evidence.json
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
RESERVED = {"scripts", "sessions", "review"}  # directories under recordings/ that aren't speakers
TAKE = re.compile(r"^take-(\d+)\.(webm|ogg|mp4|mp3|wav|aac)$")
TAKE_ANY = re.compile(r"^take-(\d+)\.")  # any file of a take: a fix holds its number from upload, before its audio is composed
LLR_MIN = 3.0  # nats: the teacher must prefer what the script says was said by this much (lab_checks.LLR_MIN)
LOCAL_PER_MAX = 0.25
AYAH_PER_MAX = 0.15  # an ayah the teacher hears this differently from the script holds an unscripted change (lab_sources.MAX_PER)
BASMALA_PER_MAX = 0.3  # an optional item (basmala, isti'adha, closing) heard worse than this wasn't said; it is dropped
OPTIONAL = ("istiadha", "closing", "basmala")  # droppable roles, the unscripted ones dropped first
COVERAGE_MIN = 0.9
HEARD_MIN = 0.5  # the teacher's greedy phonemes over the script's: below this the take is silent or another passage
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
    if speaker.lower() in RESERVED:
        raise ValueError(f"speaker can't be {speaker!r}")
    return f"{ROOT}/{safe(speaker, 'speaker')}/{safe(script, 'script')}"


def next_take(names: list[str]) -> int:
    return max((int(m[1]) for n in names if (m := TAKE_ANY.match(n))), default=0) + 1


def take_files(speaker: str, script: str, n: int, ext: str) -> dict:
    d, sid = f"{take_dir(speaker, script)}/take-{n}", session_id(speaker, script, n)
    return {"upload": f"{d}.{ext}", "meta": f"{d}.json", "result": f"{d}.result.json", "wav": f"{ROOT}/sessions/{sid}.phone.wav",
            "session": f"{ROOT}/sessions/{sid}.json", "evidence": f"{ROOT}/sessions/{sid}.teacher.phone.evidence.json",
            "checks": f"{ROOT}/sessions/{sid}.checks.json"}


def session_id(speaker: str, script: str, n: int) -> str:
    return f"rec-{speaker}-{script}-t{n}"


def take_meta(form: dict, script: dict, n: int, ext: str, at: str, size: int, fix: dict | None = None) -> dict:
    """The upload's metadata record from the page's form fields, checked against the script it names. A fix
    (`fix_request`) uploads only its patch (`ext`); the take's own audio is composed as WAV, and its resume points
    (`resumesS`, `joinsS`) are filled in on the composed clock when it is joined."""
    speaker = safe(form.get("speaker", ""), "speaker")
    if form.get("scriptVersion") not in (None, "", str(script["version"])) or form.get("scriptSha") not in (None, "", script["sha"]):
        raise ValueError(f"the page has an old copy of {script['id']}; reload it")
    clip = lambda k, m=200: str(form.get(k) or "")[:m]
    num = lambda k: float(form[k]) if str(form.get(k) or "").replace(".", "", 1).isdigit() else None
    out = {"version": 1, "speaker": speaker, "script": script["id"], "scriptVersion": script["version"], "scriptSha": script["sha"],
           "batch": form.get("batch") or None, "take": n, "ext": ext, "bytes": size, "mime": clip("mime", 80), "device": clip("device"),
           "userAgent": clip("userAgent", 300), "trackSettings": clip("trackSettings", 500), "recordedAt": clip("recordedAt", 40) or at,
           "uploadedAt": at, "clientDurationS": num("durationS"), "resumesS": resumes(form.get("resumesS")), "note": clip("note", 2000)}
    if fix:
        out |= {"ext": "wav", "resumesS": [], "fix": {**fix, "patch": f"take-{n}.patch.{ext}", "patchExt": ext, "resumesS": out["resumesS"]}}
    return out


def resumes(raw) -> list[float]:
    """Take-clock times where a paused recording resumed: each is a join in the audio, like a splice seam."""
    try:
        xs = json.loads(raw or "[]")
    except ValueError:
        return []
    return sorted(round(float(x), 2) for x in xs[:200] if isinstance(x, (int, float)) and x >= 0) if isinstance(xs, list) else []


# ---- checks -------------------------------------------------------------------------------------------------------

def items(script: dict, drop: set[int] | None = None) -> list[dict]:
    """What the script says is recited, one item per word (`said`), optionally without some optional items (indexes)."""
    return [dict(x, i=i) for i, x in enumerate(script["said"]) if not (drop and i in drop)]


def plant_window(its: list[dict], plant: dict, open_ends: bool = False) -> tuple[int | None, int | None, list[int]] | None:
    """(item before, item after, the plant's own items) for one plant; an omission has no items of its own. With
    `open_ends` (a fix's patch, which starts and ends inside the script) a side past the ends is None."""
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
    if open_ends:
        lo, hi = (lo if lo >= 0 else None), (hi if hi < len(its) else None)
        return (lo, hi, mine) if mine or lo is not None else None
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


def take_verdict(aligned: bool, plants: list[dict], ayahs: list[dict], coverage: float, heard: float = 1.0) -> tuple[str, list[str]]:
    """ok, listen (something for a human to hear) or redo (the take doesn't follow the script)."""
    if heard < HEARD_MIN:
        return "redo", [f"the teacher hears almost none of the script ({heard:.0%} of its phonemes): is the microphone on?"]
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

    Optional items the teacher doesn't hear (greedy PER over BASMALA_PER_MAX) are dropped and the rest aligned again:
    the isti'adha and closing first, as a missing one can pull a basmala's window off its audio, then basmalas.
    """
    import numpy as np
    from rapidfuzz.distance import Levenshtein

    import lab_checks as C
    import lab_sources as L
    from per_breakdown import canon

    fs = ev["frameMs"] / 1000
    lp = C.log_probs(ev)
    words = C.word_phonemes(ref)
    heard = len(ev["phonemes"]) / max(1, len(L.squeeze("".join(x["phonemes"] for x in script["said"]))))
    drop: set[int] = set()
    for _ in range(4):
        its = items(script, drop)
        spans = C.align(lp, its, fs)
        if spans is None:
            break
        bad = {}
        for it, (a, b) in zip(its, spans):
            if it["role"] in OPTIONAL:
                g = canon(C.greedy_in(ev, round(a / fs), max(round(a / fs) + 1, round(b / fs))))
                if Levenshtein.distance(g, canon(it["phonemes"])) / max(1, len(canon(it["phonemes"]))) > BASMALA_PER_MAX:
                    bad[it["i"]] = it["role"]
        if not bad:
            break
        first = [i for i, r in bad.items() if r != "basmala"] or list(bad)
        drop |= set(first)
    duration = ev["nFrames"] * fs
    if spans is None:
        verdict, why = take_verdict(False, [], [], 0.0, heard)
        return {"version": 1, "script": script["id"], "scriptSha": script["sha"], "durationS": round(duration, 3), "aligned": False,
                "verdict": verdict, "why": why, "plants": [], "ayahs": [], "said": [], "coverage": 0.0, "heard": round(heard, 3), "dropped": sorted(drop)}
    timed = [{**{k: v for k, v in it.items() if k != "i"}, "item": it["i"], "startS": round(a, 3), "endS": round(b, 3)} for it, (a, b) in zip(its, spans)]
    coverage = sum(t["endS"] > t["startS"] for t in timed) / max(1, len(timed))
    plants = []
    for p in script["plants"]:
        w = plant_window(timed, p, open_ends="fix" in script)
        out = {"id": p["id"], "n": p["n"], "class": p["class"], "ayah": p["ayah"], "location": p["location"]}
        if w is None:
            plants.append({**out, "status": "unsure", "reasons": ["no scripted word on both sides of it"]})
            continue
        lo, hi, mine = w
        prev, nxt = timed[lo] if lo is not None else None, timed[hi] if hi is not None else None
        # A self-correction is the slip and the words said again against the word said once, as written.
        said_mid, orig_mid = "".join(timed[k]["phonemes"] for k in mine), original_phonemes(script, p, words)
        before, after = prev["phonemes"] if prev else "", nxt["phonemes"] if nxt else ""
        h_said, h_orig = before + said_mid + after, before + orig_mid + after
        a = prev["startS"] if prev else timed[mine[0]]["startS"]
        b = nxt["endS"] if nxt else timed[mine[-1]]["endS"] if mine else prev["endS"]
        f0, f1 = max(0, math.floor((a - PAD_S) / fs)), min(len(lp), math.ceil((b + PAD_S) / fs))
        llr = min(C.LLR_CAP, C.score(lp, f0, f1, h_said) - C.score(lp, f0, f1, h_orig))
        g = canon(C.greedy_in(ev, f0, f1))
        d_said, d_orig = Levenshtein.distance(g, canon(h_said)), Levenshtein.distance(g, canon(h_orig))
        local = d_said / max(1, len(canon(h_said)))
        status, reasons = decide(float(llr), d_said, d_orig, local)
        plants.append({**out, "status": status, "reasons": reasons, "llr": round(float(llr), 2), "editsSaid": d_said, "editsOriginal": d_orig,
                       "localPer": round(local, 3), "window": [round(f0 * fs, 3), round(f1 * fs, 3)],
                       "startS": timed[mine[0]]["startS"] if mine else prev["endS"], "endS": timed[mine[-1]]["endS"] if mine else nxt["startS"] if nxt else prev["endS"],
                       "greedy": g, "said": canon(h_said), "original": canon(h_orig)})
    ayahs = []
    groups = []
    for k, t in enumerate(timed):
        key = f"{t['at']['surah']}:{t['at']['ayah']}" if t["at"] else None
        if t["role"] in OPTIONAL:
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
    verdict, why = take_verdict(True, plants, ayahs, coverage, heard)
    return {"version": 1, "script": script["id"], "scriptSha": script["sha"], "durationS": round(duration, 3), "aligned": True, "verdict": verdict,
            "why": why, "coverage": round(coverage, 3), "heard": round(heard, 3), "dropped": sorted(drop), "plants": plants, "ayahs": ayahs, "said": timed,
            "thresholds": {"llrMin": LLR_MIN, "localPerMax": LOCAL_PER_MAX, "ayahPerMax": AYAH_PER_MAX, "basmalaPerMax": BASMALA_PER_MAX,
                           "coverageMin": COVERAGE_MIN, "heardMin": HEARD_MIN, "redoFlaggedAyahs": REDO_FLAGGED_AYAHS}}


# ---- fixing a passage ---------------------------------------------------------------------------------------------
# A take that fails in a few ayahs is fixed by recording only those ayahs again (a patch of whole ayahs A..B). The
# patch is checked against what the script says is said in A..B, then joined into the take: the take up to its pause
# before A, the patch, the take from its pause after B. Every join needs measured silence on both sides, by the
# lab_sessions weak-onset rule: JOIN_SILENCE_S of 10 ms frames below the room tone + 3 dB, or 30 dB under the speech
# level where the room is quieter than that. The cut and its crossfade sit inside that silence. The joined audio is a
# new take, checked again end to end like any other.

FIX_VERDICTS = ("redo", "listen")  # takes the page offers to fix
FIX_MERGE = 2  # failed ayahs at most this many ayahs apart are one passage to record again
JOIN_SILENCE_S = 0.08  # lab_sessions.ONSET_SILENCE_S
JOIN_KEEP_S = 0.4  # at most this much of each side's pause is kept at a join
JOIN_XFADE_S = 0.02
JOIN_REACH_S = (0.1, 0.05)  # a pause is looked for this far into the aligned word before it and the word after it
FIX_GAIN_MAX_DB = 6.0  # a patch this much louder or quieter than its take was recorded differently
PATCH_S = (1.0, 15 * 60.0)


class Refused(ValueError):
    """A fix that can't be joined: `why` for the page, `fix` what was found on the way."""

    def __init__(self, why: list[str], fix: dict | None = None):
        super().__init__("; ".join(why))
        self.why, self.fix = why, fix or {}


def ayah_keys(script: dict) -> list[str]:
    return [a["key"] for a in script["ayahs"] if not a.get("basmala")]


def passage(a: str, b: str) -> str:
    return a if a == b else f"{a}–{b}"


def said_ayahs(script: dict) -> list[str | None]:
    """The ayah each said item belongs to: its word's; an added word's and a basmala's, the next word's; none for the
    isti'adha and the closing."""
    out, nxt = [None] * len(script["said"]), None
    for i in range(len(script["said"]) - 1, -1, -1):
        x = script["said"][i]
        if x["at"]:
            nxt = f"{x['at']['surah']}:{x['at']['ayah']}"
            out[i] = nxt
        elif x["role"] in ("added", "basmala"):
            out[i] = nxt
    return out


def fix_items(script: dict, a: str, b: str) -> list[int]:
    """Indexes of the said items of ayahs a..b."""
    keys = ayah_keys(script)
    if a not in keys or b not in keys or keys.index(a) > keys.index(b):
        raise ValueError(f"{a}–{b} isn't a passage of {script['id']}")
    want = set(keys[keys.index(a) : keys.index(b) + 1])
    return [i for i, k in enumerate(said_ayahs(script)) if k in want]


def failed_ayahs(script: dict, result: dict) -> dict[str, list[str]]:
    """Ayah -> why the take fails there, in script order: planted mistakes not heard or unsure, an ayah that sounds
    different from the script, words the alignment didn't find."""
    why = defaultdict(list)
    for p in result.get("plants", []):
        if p["status"] != "performed":
            why[p["ayah"]].append(f"mistake {p['n']} {'not heard' if p['status'] == 'not_performed' else 'not sure'}")
    for a in result.get("ayahs", []):
        if a.get("flag") and a["key"]:
            why[a["key"]].append("sounds different from the script")
    for k, n in Counter(f"{t['at']['surah']}:{t['at']['ayah']}" for t in result.get("said", []) if t["at"] and t["endS"] <= t["startS"]).items():
        why[k].append(f"{n} word{'s' if n > 1 else ''} not found")
    order = {k: i for i, k in enumerate(ayah_keys(script))}
    return {k: why[k] for k in sorted(why, key=lambda k: order.get(k, -1)) if k in order}


def fix_ranges(script: dict, result: dict) -> list[dict]:
    """The passages to record again in a take that is redo or listen, {from, to, ayahs: {key: why}}: its failed ayahs,
    those at most FIX_MERGE ayahs apart in one passage. None for a take that didn't align (record it all again)."""
    if result.get("verdict") not in FIX_VERDICTS or not result.get("aligned"):
        return []
    keys, out = ayah_keys(script), []
    for k, why in failed_ayahs(script, result).items():
        if out and keys.index(k) - keys.index(out[-1]["to"]) <= FIX_MERGE:
            out[-1]["to"] = k
        else:
            out.append({"from": k, "to": k, "ayahs": {}})
        out[-1]["ayahs"][k] = why
    return out


def fix_request(script: dict, base_meta: dict | None, base_result: dict | None, a: str, b: str) -> dict:
    """A fix of take `base_meta` for ayahs a..b, when that take can take one: checked, aligned, this script version."""
    if not base_meta:
        raise Refused(["the take to fix doesn't exist"])
    n = base_meta["take"]
    if not base_result or not base_result.get("aligned") or base_result.get("verdict") in (None, "error", "refused"):
        raise Refused([f"take {n} has no checked alignment to join into: record the whole take again"])
    if base_meta["scriptSha"] != script["sha"]:
        raise Refused([f"take {n} was recorded from an older version of {script['id']}: record the whole take again"])
    try:
        fix_items(script, a, b)
    except ValueError as e:
        raise Refused([str(e)])
    return {"base": n, "from": a, "to": b}


def fix_script(script: dict, a: str, b: str) -> dict:
    """What a fix of ayahs a..b records, as a script for `check_take`: those ayahs' said items (`src`: the index in
    the whole script) and plants."""
    idx = fix_items(script, a, b)
    keys = {said_ayahs(script)[i] for i in idx}
    return {**{k: v for k, v in script.items() if k not in ("said", "plants", "labels", "ayahs")}, "fix": [a, b],
            "said": [dict(script["said"][i], src=i) for i in idx], "plants": [p for p in script["plants"] if p["ayah"] in keys]}


def patch_summary(r: dict) -> dict:
    """A patch's own check, as the page shows it."""
    return {**{k: r.get(k) for k in ("verdict", "why", "coverage", "heard", "dropped", "durationS")},
            "plants": [{k: p.get(k) for k in ("id", "n", "class", "ayah", "status", "reasons", "llr")} for p in r.get("plants", [])],
            "flagged": [a["key"] for a in r.get("ayahs", []) if a.get("flag")]}


def speech_level(db, lo: int = 0, hi: int | None = None) -> float:
    """dB of 10 ms frames lo..hi (`lab_sessions.frame_db`): the energy mean of the loudest half, the speech level
    (as lab_sessions.active_rms)."""
    xs = sorted(float(v) for v in db[lo:hi])
    top = xs[len(xs) // 2 :]
    return 10 * math.log10(sum(10 ** (v / 10) for v in top) / len(top)) if top else -100.0


def quiet_floor(db) -> float:
    """The dB a 10 ms frame of this recording is silent at or below: its room tone + 3 dB, or 30 dB under its speech
    level where the room is quieter (lab_sessions.onset_floor); never less than a pause's 12 dB under the speech
    (lab_sessions.PAUSE_DB), so a noisy room has no silence rather than silence everywhere."""
    import lab_sessions as S

    lvl = speech_level(db)
    return lvl + min(S.onset_floor(S.room_tone(db), lvl), S.PAUSE_DB)


def pauses(db, floor: float, lo: float, hi: float) -> list[tuple[float, float]]:
    """Runs of silent 10 ms frames (at most `floor` dB), at least JOIN_SILENCE_S long, inside lo..hi seconds."""
    i0, i1 = max(0, math.ceil(lo * 100 - 1e-6)), min(len(db), math.floor(hi * 100 + 1e-6))
    out, start = [], None
    for i in range(i0, i1 + 1):
        quiet = i < i1 and db[i] <= floor
        if quiet and start is None:
            start = i
        elif not quiet and start is not None:
            if (i - start) / 100 >= JOIN_SILENCE_S - 1e-9:
                out.append((start / 100, i / 100))
            start = None
    return out


def cut_in(run: tuple[float, float], keep: str) -> float:
    """Where to cut in a pause: its middle, keeping at most JOIN_KEEP_S of it beside the speech kept (`keep`: the
    speech `before` the pause or `after` it), the crossfade inside the pause."""
    a, b = run
    h = JOIN_XFADE_S / 2
    t = min((a + b) / 2, a + JOIN_KEEP_S) if keep == "before" else max((a + b) / 2, b - JOIN_KEEP_S)
    return round(min(max(t, a + h), b - h), 3)


def plan_fix(script: dict, fix: dict, take_said: list[dict], take_s: float, take_db, patch_said: list[dict], patch_s: float, patch_db) -> dict:
    """Where a patch joins its take. `fix`: {base, from, to}; `*_said`: timed items from `check_take`; `*_db`:
    `lab_sessions.frame_db` of each recording. The joins are the take's pause before ayah `from` and after `to`, and
    the patch's silence before its first word and after its last; each must be measured silence (`pauses`), else the
    fix is refused with the passage to record instead (`suggest`, None for the whole take). Then the patch's speech
    level against the take's: the gain that matches it, refused past FIX_GAIN_MAX_DB."""
    a, b, n = fix["from"], fix["to"], fix["base"]
    keys, idx = ayah_keys(script), fix_items(script, a, b)
    ka, kb = keys.index(a), keys.index(b)
    heard = [t for t in take_said if t["endS"] > t["startS"]]
    prev = next((t for t in reversed(heard) if t["item"] < idx[0]), None)
    nxt = next((t for t in heard if t["item"] > idx[-1]), None)
    inside = [t for t in heard if idx[0] <= t["item"] <= idx[-1]]
    first = inside[0]["startS"] if inside else nxt["startS"] if nxt else take_s
    last = inside[-1]["endS"] if inside else prev["endS"] if prev else 0.0
    r0, r1 = JOIN_REACH_S
    tf, pf = quiet_floor(take_db), quiet_floor(patch_db)
    before = pauses(take_db, tf, prev["endS"] - r0 if prev else 0.0, first + r1)
    after = pauses(take_db, tf, last - r0, nxt["startS"] + r1 if nxt else take_s)
    ph = [t for t in patch_said if t["endS"] > t["startS"]]
    p_before = pauses(patch_db, pf, 0.0, ph[0]["startS"] + r1) if ph else []
    p_after = pauses(patch_db, pf, ph[-1]["endS"] - r0, patch_s) if ph else []
    out = {"floorsDb": {"take": round(tf, 1), "patch": round(pf, 1)}}
    why, suggest = [], {"from": a, "to": b}
    if not before:
        why.append(f"no clean pause before {a} in take {n}: " + (f"record again from {keys[ka - 1]}" if ka else "record the whole take again"))
        suggest["from"] = keys[ka - 1] if ka else None
    if not after:
        why.append(f"no clean pause after {b} in take {n}: " + (f"record again up to {keys[kb + 1]}" if kb + 1 < len(keys) else "record the whole take again"))
        suggest["to"] = keys[kb + 1] if kb + 1 < len(keys) else None
    if not ph:
        why.append(f"none of {passage(a, b)} was heard in your fix")
    else:
        if not p_before:
            why.append(f"no silence before {a} at the start of your fix: after you start recording, wait a moment before you recite")
        if not p_after:
            why.append(f"no silence after {b} at the end of your fix: pause for a moment before you stop")
    if why:
        return {**out, "ok": False, "why": why, "suggest": None if None in suggest.values() else suggest}
    c1, c2 = cut_in(before[0], "before"), cut_in(after[-1], "after")
    p0, p1 = cut_in(p_before[-1], "after"), cut_in(p_after[0], "before")
    lt, lp = speech_level(take_db), speech_level(patch_db, round(p0 * 100), round(p1 * 100))
    gain = round(lt - lp, 1)
    out |= {"gainDb": gain, "levelsDb": {"take": round(lt, 1), "patch": round(lp, 1)}}
    if abs(gain) > FIX_GAIN_MAX_DB:
        return {**out, "ok": False, "suggest": suggest,
                "why": [f"your fix is {abs(gain):.1f} dB {'quieter' if gain > 0 else 'louder'} than take {n}: record it where you recorded the take, "
                        "as far from the microphone, or record the whole take again"]}
    if c1 > c2 or p1 - p0 < 2 * JOIN_XFADE_S:
        return {**out, "ok": False, "suggest": None, "why": [f"the pauses found don't leave a passage to replace ({c1}–{c2} s in take {n})"]}
    span = round(p1 - p0, 3)
    joins = [{"atS": c1, "before": a, "takeS": c1, "patchS": p0, "pauseS": round(c1 - before[0][0] + p_before[-1][1] - p0, 2),
              "silenceS": {"take": round(before[0][1] - before[0][0], 2), "patch": round(p_before[-1][1] - p_before[-1][0], 2)}},
             {"atS": round(c1 + span, 3), "after": b, "takeS": c2, "patchS": p1, "pauseS": round(p1 - p_after[0][0] + after[-1][1] - c2, 2),
              "silenceS": {"take": round(after[-1][1] - after[-1][0], 2), "patch": round(p_after[0][1] - p_after[0][0], 2)}}]
    return {**out, "ok": True, "why": [], "take": [c1, c2], "patch": [p0, p1], "joins": joins, "joinsS": [j["atS"] for j in joins],
            "durationS": round(c1 + span + take_s - c2, 3)}


def fix_resumes(base: list[float], patch: list[float], plan: dict) -> list[float]:
    """The composed take's resume points on its own clock: the base take's outside the passage replaced, the patch's
    inside the part kept, and the two joins."""
    (c1, c2), (p0, p1) = plan["take"], plan["patch"]
    shift = c1 + (p1 - p0) - c2
    out = [t for t in base if t < c1] + [t + shift for t in base if t > c2] + [c1 + t - p0 for t in patch if p0 < t < p1] + plan["joinsS"]
    return sorted({round(t, 2) for t in out})


def stitch(take, patch, plan: dict, sr: int = 16_000):
    """The composed audio (float32 at `sr`): `take` up to the first join, `patch` times its gain, `take` from the
    second join, with a JOIN_XFADE_S crossfade centred on each join (inside the pauses)."""
    import numpy as np

    h = round(JOIN_XFADE_S * sr / 2)
    c1, c2 = (round(t * sr) for t in plan["take"])
    p0, p1 = (round(t * sr) for t in plan["patch"])
    if not (h <= c1 <= c2 <= len(take) - h and h <= p0 and p0 + 2 * h <= p1 <= len(patch) - h):
        raise ValueError(f"joins out of range: take {c1}..{c2} of {len(take)}, patch {p0}..{p1} of {len(patch)}")
    x = np.asarray(take, dtype=np.float32)
    y = np.asarray(patch, dtype=np.float32)[p0 - h : p1 + h] * np.float32(10 ** (plan["gainDb"] / 20))
    up = (np.arange(2 * h, dtype=np.float32) + 0.5) / (2 * h)
    out = np.concatenate([x[: c1 - h], x[c1 - h : c1 + h] * (1 - up) + y[: 2 * h] * up, y[2 * h : len(y) - 2 * h],
                          y[len(y) - 2 * h :] * (1 - up) + x[c2 - h : c2 + h] * up, x[c2 + h :]])
    return np.clip(out, -1.0, 1.0)


def fix_files(speaker: str, script: str, n: int, ext: str) -> dict:
    d, sid = f"{take_dir(speaker, script)}/take-{n}", session_id(speaker, script, n)
    return {"patch": f"{d}.patch.{ext}", "patchWav": f"{ROOT}/sessions/{sid}-patch.phone.wav",
            "patchEvidence": f"{ROOT}/sessions/{sid}-patch.teacher.phone.evidence.json"}


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
        if t["role"] in OPTIONAL:
            spans.append({"id": f"{t['role']}-{len(spans) + 1}", "kind": "non_quran", "label": t["role"], "startS": t["startS"], "endS": max(t["endS"], t["startS"] + 0.01)})
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
                     "formsOtherAyah": sum(e["forms_other_ayah"] for e in events), "range": script["range"], "title": script["title"],
                     "resumesS": meta.get("resumesS") or [],
                     **({"joinsS": meta.get("joinsS", []), "fix": {k: meta["fix"][k] for k in ("base", "from", "to")}} if meta.get("fix") else {})}}


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
    return " ".join(w["text"] for w in s["said"] if w["endS"] > a and w["startS"] < b and w["role"] not in OPTIONAL)


def _window(s: dict, a: float, b: float, pad: float = 2.5) -> tuple[float, float]:
    return round(max(0.0, a - pad), 3), round(min(s["durationS"], b + pad), 3)


def review_items(s: dict, result: dict, script: dict, wav: str, seed: int = SEED) -> list[dict]:
    """One take's spot check: every plant the teacher wasn't sure of, a seeded few it heard, one self-correction, every
    ayah that sounds different from the script, every join of a fixed passage, and two seeded clean windows."""
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
    fx = meta.get("fix")
    for k, t in enumerate(meta.get("joinsS", [])):
        x, y = _window(s, t, t)
        out.append({"id": f"{s['id']}-join-{k + 1}", "kind": "case", "title": "Join of a fixed passage · should sound like one take",
                    "audio": {"path": wav, "startS": x, "endS": y}, "text": _text(s, x, y), "label": {"type": "clean"},
                    "context": f"{ctx(t)} · take {fx['base']} with {fx['from']}–{fx['to']} recorded again, joined {'before' if k == 0 else 'after'} it"})
    busy = [(e["startS"] - 3, e["endS"] + 3) for e in s["events"]] + [(n["startS"] - 3, n["endS"] + 3) for n in s["nonErrors"]]
    busy += [(a["startS"], a["endS"]) for a in result["ayahs"] if a["flag"]] + [(t - 3, t + 3) for t in meta.get("joinsS", [])]
    starts, t = [], 5.0
    while t + CLEAN_WINDOW_S < s["durationS"] - 2:
        if not any(a < t + CLEAN_WINDOW_S and t < b for a, b in busy) and _text(s, t, t + CLEAN_WINDOW_S):
            starts.append(t)
        t += CLEAN_WINDOW_S
    for t in sorted(rng.sample(starts, min(SPOT["clean"], len(starts)))):
        ws = [w for w in s["said"] if w["endS"] > t and w["startS"] < t + CLEAN_WINDOW_S and w["role"] not in OPTIONAL]
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
