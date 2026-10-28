"""Pure logic for the lab listening review (.scratch/lab/issues/06-review-ui.md, docs/benchmarks/lab-review.md).

Served by modal/lab_review.py; tested by tools/test_lab_review.py. Standard library only.
"""

import math
import random
import re
import struct

VERDICTS = {
    "ok": "Sounds right",
    "bad_label": "Bad label",
    "bad_audio": "Bad audio",
    "bad_splice": "Bad splice",
    "listener": "Listener fault",
    "tracker": "Tracker fault",
    "checker": "Checker fault",
}
BAD = {"bad_label", "bad_audio", "bad_splice"}
KINDS = {"splice", "case"}
LABELS = {"wrong_word", "omitted_word", "added_word", "skipped_passage", "repeat", "restart", "self_correction", "clean"}
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_NOTE = 20_000


def safe_rel(path: str) -> str:
    """A path relative to /lab/data with no way out of it."""
    parts = str(path or "").replace("\\", "/").split("/")
    if not path or path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"bad data path: {path!r}")
    return "/".join(parts)


def wav_info(f) -> tuple[bytes, int, int]:
    """(fmt chunk body, data offset, data size in bytes) of a RIFF/WAVE file object. A size past the end is clamped."""
    f.seek(0)
    head = f.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt = None
    while True:
        h = f.read(8)
        if len(h) < 8:
            raise ValueError("WAV has no data chunk")
        cid, size = h[:4], struct.unpack("<I", h[4:])[0]
        if cid == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk before fmt chunk")
            off = f.tell()
            return fmt, off, min(size, f.seek(0, 2) - off)
        if cid == b"fmt ":
            fmt = f.read(size)
            f.seek(size % 2, 1)
        else:
            f.seek(size + size % 2, 1)


def wav_bytes(fmt: bytes, data: bytes) -> bytes:
    fp, dp = b"\0" * (len(fmt) % 2), b"\0" * (len(data) % 2)
    riff = 4 + 8 + len(fmt) + len(fp) + 8 + len(data) + len(dp)
    return b"RIFF" + struct.pack("<I", riff) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + fp + b"data" + struct.pack("<I", len(data)) + data + dp


def wav_slice(f, start_s: float = 0.0, end_s: float | None = None) -> bytes:
    """A standalone WAV of [start_s, end_s) from a WAV file object, reading only that window. Any fixed-frame format (PCM, float)."""
    fmt, off, size = wav_info(f)
    if len(fmt) < 14:
        raise ValueError("WAV fmt chunk too short")
    rate, align = struct.unpack_from("<I", fmt, 4)[0], struct.unpack_from("<H", fmt, 12)[0]
    if not rate or not align:
        raise ValueError("WAV has no sample rate or frame size")
    n = size // align
    a = min(n, max(0, round(start_s * rate)))
    b = n if end_s is None else min(n, max(a, round(end_s * rate)))
    f.seek(off + a * align)
    return wav_bytes(fmt, f.read((b - a) * align))


def pcm16_wav(samples, rate: int) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return wav_bytes(fmt, struct.pack(f"<{len(samples)}h", *(max(-32767, min(32767, round(s * 32767))) for s in samples)))


def byte_range(header: str | None, total: int) -> tuple[int, int] | None:
    """Inclusive (first, last) byte for a single `Range: bytes=...` header; None serves the whole body. Raises ValueError if unsatisfiable."""
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header or "")
    if not m or m[1] == m[2] == "":
        return None
    if m[1] == "":
        n = int(m[2])
        if n == 0 or total == 0:
            raise ValueError("unsatisfiable range")
        return max(0, total - n), total - 1
    first = int(m[1])
    last = min(total - 1, int(m[2])) if m[2] else total - 1
    if first >= total or last < first:
        raise ValueError("unsatisfiable range")
    return first, last


def validate_queue(q) -> list[str]:
    """Problems with a review queue; empty when it can be reviewed."""
    if not isinstance(q, dict):
        return ["queue is not a JSON object"]
    out = []
    if q.get("version") != 1:
        out.append("version must be 1")
    if not SAFE.match(str(q.get("testSet", ""))):
        out.append("testSet must be a short name of letters, digits, '.', '_' or '-'")
    items = q.get("items")
    if not isinstance(items, list) or not items:
        return out + ["items must be a non-empty list"]
    seen = set()
    for n, it in enumerate(items):
        at = f"item {n} ({it.get('id') if isinstance(it, dict) else '?'})"
        if not isinstance(it, dict):
            out.append(f"{at}: not an object")
            continue
        if not isinstance(it.get("id"), str) or not it["id"]:
            out.append(f"{at}: id must be a non-empty string")
        elif it["id"] in seen:
            out.append(f"{at}: duplicate id")
        seen.add(it.get("id"))
        if it.get("kind") not in KINDS:
            out.append(f"{at}: kind must be 'splice' or 'case'")
        for k in ("audio", "before", "after"):
            if k == "audio" or k in it:
                out += [f"{at}: {k}: {p}" for p in clip_problems(it.get(k))]
        if not isinstance(it.get("text", ""), str):
            out.append(f"{at}: text must be a string")
        label = it.get("label")
        if not isinstance(label, dict) or label.get("type") not in LABELS:
            out.append(f"{at}: label.type must be one of {', '.join(sorted(LABELS))}")
    return out


def clip_problems(c) -> list[str]:
    if not isinstance(c, dict):
        return ["must be {path, startS, endS}"]
    out = []
    try:
        safe_rel(c.get("path"))
    except ValueError as e:
        out.append(str(e))
    s, e = c.get("startS"), c.get("endS")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (s, e)) or not 0 <= s < e:
        out.append("needs 0 <= startS < endS")
    elif e - s > 600:
        out.append("window longer than 10 minutes")
    return out


def empty_reviews(test_set: str, reviewer: str) -> dict:
    return {"version": 1, "testSet": test_set, "reviewer": reviewer, "items": {}}


def upsert(doc: dict, item_id: str, verdict: str, note: str, at: str, queue: str | None = None) -> dict:
    """Set the reviewer's verdict for one item, keeping any earlier different verdict or note in its history. Returns the record."""
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("item id required")
    note = "" if note is None else note
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ValueError("note must be a string under 20000 characters")
    old = doc["items"].get(item_id)
    rec = {"verdict": verdict, "note": note, "at": at, "queue": queue, "history": []}
    if old:
        rec["history"] = old.get("history", [])
        if (old["verdict"], old["note"]) != (verdict, note):
            rec["history"] = [*rec["history"], {k: old[k] for k in ("verdict", "note", "at")}]
    doc["items"][item_id] = rec
    return rec


def tally(queue: dict, reviews: dict) -> dict:
    """Progress against the acceptance bar: at most `maxBad` (default 1) bad-data verdicts across the queue."""
    ids = [it["id"] for it in queue["items"]]
    done = [reviews[i]["verdict"] for i in ids if i in reviews]
    bad, max_bad = sum(v in BAD for v in done), queue.get("maxBad", 1)
    state = "fail" if bad > max_bad else "pass" if len(done) == len(ids) else "open"
    return {"reviewed": len(done), "total": len(ids), "bad": bad, "maxBad": max_bad, "state": state}


FATIHA = "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ الرَّحْمَٰنِ الرَّحِيمِ مَالِكِ يَوْمِ الدِّينِ إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ".split()


def fixture(rate: int = 16000, seed: int = 7) -> dict[str, bytes | dict]:
    """A synthetic demo test set: tone 'words' (1 s each) standing in for recitation. Returns {relative path: WAV bytes or queue}."""
    rnd = random.Random(seed)

    def word(freq: float, timbre: int = 1) -> list[float]:
        n, on = rate, int(0.75 * rate)
        env = lambda i: min(1, i / (0.02 * rate), (on - i) / (0.02 * rate)) if i < on else 0
        return [0.3 * env(i) * sum(math.sin(2 * math.pi * freq * h * i / rate) / h for h in range(1, timbre + 1)) for i in range(n)]

    pitch = [220 * 2 ** (k / 12) for k in (0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19)]
    orig = [word(f) for f in pitch]
    donor = word(330, timbre=3)
    order = ["o0", "o1", "o2", "d0", "o4", "o5", "o4", "o5", "o6", "o8", "o9", "o10", "o11"]
    session = [s for k in order for s in (donor if k == "d0" else orig[int(k[1:])])]
    phone = [0.8 * s + 0.5 * (session[i - 2400] if i >= 2400 else 0) * 0.4 + rnd.gauss(0, 0.03) for i, s in enumerate(session)]
    for i in range(int(0.5 * rate), int(0.52 * rate)):
        phone[i] += 0.6 * rnd.choice((-1, 1))
    words = [FATIHA[int(k[1:])] if k[0] == "o" else "الْأَوَّلِينَ" for k in order]
    text = lambda a, b: " ".join(words[a:b])
    clip = lambda path, s, e: {"path": f"fixture-review/audio/{path}.wav", "startS": s, "endS": e}
    items = [
        {"id": "demo-1-wrong-3", "kind": "splice", "title": "Wrong word planted", "audio": clip("session", 2.0, 5.0),
         "before": clip("original", 1.0, 3.0), "after": clip("original", 4.0, 6.0), "text": text(2, 5),
         "label": {"type": "wrong_word", "expected": FATIHA[3], "said": "الْأَوَّلِينَ"}, "context": "Al-Fatiha 1:2, word 3 · demo session at 3.0 s"},
        {"id": "demo-1-repeat-4", "kind": "case", "title": "Repeat", "audio": clip("session", 4.0, 8.0), "text": text(4, 8),
         "label": {"type": "repeat", "expected": " ".join(FATIHA[4:6]), "said": " ".join(FATIHA[4:6])}, "context": "Al-Fatiha 1:3 · demo session at 6.0 s"},
        {"id": "demo-1-omit-7", "kind": "splice", "title": "Word left out", "audio": clip("session", 8.0, 10.5),
         "before": clip("original", 5.0, 7.0), "after": clip("original", 8.0, 10.0), "text": text(8, 10),
         "label": {"type": "omitted_word", "expected": FATIHA[7], "said": ""}, "context": "Al-Fatiha 1:4, word 2 · demo session at 9.0 s"},
        {"id": "demo-1-phone-clean", "kind": "case", "title": "Clean stretch, phone copy", "audio": clip("session-phone", 9.0, 13.0),
         "text": text(9, 13), "label": {"type": "clean"}, "context": "Al-Fatiha 1:4–1:5 · phone-degraded copy"},
        {"id": "demo-1-phone-start", "kind": "case", "title": "Session start, phone copy", "audio": clip("session-phone", 0.0, 3.0),
         "text": text(0, 3), "label": {"type": "clean"}, "context": "Al-Fatiha 1:2 · phone copy with a click at 0.5 s (try 'Bad audio')"},
    ]
    return {
        "fixture-review/audio/original.wav": pcm16_wav([s for w in orig for s in w], rate),
        "fixture-review/audio/session.wav": pcm16_wav(session, rate),
        "fixture-review/audio/session-phone.wav": pcm16_wav(phone, rate),
        "fixture-review/review/demo.json": {"version": 1, "testSet": "fixture-review", "items": items},
    }
