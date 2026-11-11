#!/usr/bin/env python3
"""Validate a Tilawa report artifact (report.json + audio.wav), schema tilawa.report/1.

Mirrors packages/core/src/report/validate.ts rule for rule (contract section 6, "Validation"); change both together.

    python3 tools/report_validate.py report.json audio.wav   # exit 0 when valid, 1 with one error per line
"""
import hashlib
import json
import math
import re
import struct
import sys

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
SCHEMA = re.compile(r"^tilawa\.report/(\d+)(\.\d+)?$")
SHA = re.compile(r"^[0-9a-f]{64}$")
CONTRIBUTOR = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_SOURCE_CHARS = 128
MAX_INCIDENTS = 100
MAX_NOTE_CHARS = 2000
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_AUDIO_MS = 300_000
SAMPLE_RATE = 16000
WAV_FMT = (16, 1, 1, 16000, 32000, 2, 16)
KINDS = {"false_warning", "missed_mistake", "wrong_position", "stumble", "unflagged_sample", "other"}
ANSWERS = {"yes", "no", "unsure", "skipped"}


def _obj(v):
    return isinstance(v, dict)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v, lo):
    return _num(v) and float(v).is_integer() and v >= lo


def _str(v, mx=math.inf):
    return isinstance(v, str) and 0 < len(v) <= mx


def _jslen(s):
    """JavaScript string length (UTF-16 code units), so limits match the TS validator."""
    return len(s.encode("utf-16-le")) // 2


def report_problems(r):
    e = []
    if not _obj(r):
        return ["report.json is not an object"]
    m = SCHEMA.match(r["schema"]) if isinstance(r.get("schema"), str) else None
    if not m:
        e.append("schema is not tilawa.report/<major>")
    elif m.group(1) != "1":
        e.append(f"unsupported report schema major {m.group(1)}")
    for k in ("reportId", "sessionId"):
        if not isinstance(r.get(k), str) or not UUID.match(r[k]):
            e.append(f"{k} is not a uuid")
    if not isinstance(r.get("createdAt"), str) or not ISO.match(r["createdAt"]):
        e.append("createdAt is not an ISO timestamp")
    if not (isinstance(r.get("source"), str) and 0 < _jslen(r["source"]) <= MAX_SOURCE_CHARS):
        e.append(f"source must be 1-{MAX_SOURCE_CHARS} chars")
    if "contributor" in r and (not isinstance(r["contributor"], str) or not CONTRIBUTOR.match(r["contributor"])):
        e.append("contributor must be 1-64 chars of [A-Za-z0-9_-]")

    engine = r.get("engine")
    if not _obj(engine):
        e.append("engine is missing")
    else:
        for k in ("sdkVersion", "modelId", "modelVersion", "referenceVersion"):
            if not _str(engine.get(k)):
                e.append(f"engine.{k} is missing")
        sha = engine.get("modelSha256", 0)
        if sha is not None and (not isinstance(sha, str) or not SHA.match(sha)):
            e.append("engine.modelSha256 must be null or a sha256 hex digest")
        if engine.get("eventSchema") != 1 or isinstance(engine.get("eventSchema"), bool):
            e.append("engine.eventSchema must be 1")
        for k in ("tier", "backend"):
            if k in engine and not isinstance(engine[k], str):
                e.append(f"engine.{k} must be a string")

    session = r.get("session")
    if not _obj(session):
        e.append("session is missing")
    else:
        if session.get("mode") not in ("free", "memorization"):
            e.append("session.mode must be free or memorization")
        p = session.get("passage", 0)
        if p is not None:
            if not (_obj(p) and _int(p.get("surah"), 1) and p["surah"] <= 114 and _int(p.get("fromAyah"), 1)
                    and _int(p.get("toAyah"), 1) and p["toAyah"] >= p["fromAyah"]):
                e.append("session.passage must be null or {surah, fromAyah, toAyah}")
        if not _obj(session.get("policy")):
            e.append("session.policy must be an object")
        if not isinstance(session.get("startedAt"), str) or not ISO.match(session["startedAt"]):
            e.append("session.startedAt is not an ISO timestamp")
        if not isinstance(session.get("vad"), bool):
            e.append("session.vad must be a boolean")

    audio = r.get("audio")
    span = None
    if not _obj(audio):
        e.append("audio is missing")
    else:
        if (audio.get("format") != "pcm_s16le" or not _num(audio.get("sampleRate")) or audio["sampleRate"] != 16000
                or not _num(audio.get("channels")) or audio["channels"] != 1):
            e.append("audio must be pcm_s16le, 16000 Hz, mono")
        ok_start = _num(audio.get("startMs")) and audio["startMs"] >= 0
        ok_dur = _num(audio.get("durationMs")) and 0 < audio["durationMs"] <= MAX_AUDIO_MS
        if not ok_start:
            e.append("audio.startMs must be a non-negative number")
        if not ok_dur:
            e.append(f"audio.durationMs must be in (0, {MAX_AUDIO_MS}]")
        if not isinstance(audio.get("sha256"), str) or not SHA.match(audio["sha256"]):
            e.append("audio.sha256 is not a sha256 hex digest")
        q = audio.get("requested")
        if not (_obj(q) and _num(q.get("startMs")) and _num(q.get("endMs")) and q["endMs"] >= q["startMs"]):
            e.append("audio.requested must be {startMs <= endMs}")
        ok_flags = isinstance(audio.get("truncatedStart"), bool) and isinstance(audio.get("truncatedEnd"), bool)
        if not ok_flags:
            e.append("audio.truncatedStart and audio.truncatedEnd must be booleans")
        if ok_start and ok_dur and ok_flags:
            span = (audio["startMs"], audio["startMs"] + audio["durationMs"], audio["truncatedStart"], audio["truncatedEnd"])

    incidents = r.get("incidents")
    if not isinstance(incidents, list):
        e.append("incidents must be an array")
    else:
        if len(incidents) > MAX_INCIDENTS:
            e.append(f"at most {MAX_INCIDENTS} incidents")
        for i, inc in enumerate(incidents):
            at = f"incidents[{i}]"
            if not _obj(inc):
                e.append(f"{at} is not an object")
                continue
            if not isinstance(inc.get("id"), str) or not UUID.match(inc["id"]):
                e.append(f"{at}.id is not a uuid")
            if inc.get("kind") not in KINDS:
                e.append(f"{at}.kind is unknown")
            if not (_num(inc.get("startMs")) and _num(inc.get("endMs")) and inc["endMs"] >= inc["startMs"]):
                e.append(f"{at} needs startMs <= endMs")
            elif span:
                start, end, trunc_start, trunc_end = span
                if not trunc_start and inc["startMs"] < start:
                    e.append(f"{at} starts before the audio")
                if not trunc_end and inc["endMs"] > end:
                    e.append(f"{at} ends after the audio")
            ids = inc.get("eventIds")
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                e.append(f"{at}.eventIds must be strings")
            m = inc.get("markedAt")
            if not (_obj(m) and _num(m.get("sessionMs")) and _num(m.get("wallMs"))):
                e.append(f"{at}.markedAt must be {{sessionMs, wallMs}}")
            c = inc.get("claim")
            if not _obj(c):
                e.append(f"{at}.claim is missing")
                continue
            if "answer" not in c or not (c["answer"] is None or c["answer"] in ANSWERS):
                e.append(f"{at}.claim.answer is unknown")
            for k in ("question", "text"):
                if k in c and not isinstance(c[k], str):
                    e.append(f"{at}.claim.{k} must be a string")
            if "note" in c and (not isinstance(c["note"], str) or _jslen(c["note"]) > MAX_NOTE_CHARS):
                e.append(f"{at}.claim.note must be at most {MAX_NOTE_CHARS} chars")

    obs = r.get("observations")
    if not _obj(obs) or not isinstance(obs.get("events"), list):
        e.append("observations.events must be an array")
    else:
        if any(not _obj(ev) or not isinstance(ev.get("type"), str) for ev in obs["events"]):
            e.append("observations.events entries need a type")
        if "evidence" in obs and not isinstance(obs["evidence"], list):
            e.append("observations.evidence must be an array")

    consent = r.get("consent")
    if not _obj(consent):
        e.append("consent is missing")
    else:
        if not _str(consent.get("policy")) or not _str(consent.get("version")):
            e.append("consent.policy and consent.version are required")
        if consent.get("send") is not True:
            e.append("consent.send must be true")
        if not isinstance(consent.get("training"), bool) or not isinstance(consent.get("publicDataset"), bool):
            e.append("consent.training and consent.publicDataset must be booleans")
    return e


def wav_problems(wav):
    if len(wav) < 44:
        return ["audio.wav is shorter than its 44-byte header"]
    e = []
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        e.append("audio.wav is not RIFF/WAVE")
    if wav[12:16] != b"fmt " or wav[36:40] != b"data":
        e.append("audio.wav does not have the canonical 44-byte header")
    fmt = struct.unpack_from("<IHHIIHH", wav, 16)
    if fmt != WAV_FMT:
        e.append(f"audio.wav fmt is {fmt}, expected {WAV_FMT} mono 16 kHz PCM16")
    if struct.unpack_from("<I", wav, 40)[0] != len(wav) - 44:
        e.append("audio.wav data size does not match the file")
    if struct.unpack_from("<I", wav, 4)[0] != len(wav) - 8:
        e.append("audio.wav RIFF size does not match the file")
    if (len(wav) - 44) % 2:
        e.append("audio.wav has an odd number of data bytes")
    return e


def audio_problems(r, wav):
    e = wav_problems(wav)
    samples = max(0, (len(wav) - 44) // 2)
    if samples > MAX_AUDIO_MS // 1000 * SAMPLE_RATE:
        e.append("audio.wav is longer than 300 s")
    audio = r.get("audio") if _obj(r) else None
    if _obj(audio) and _num(audio.get("durationMs")) and abs(samples / 16 - audio["durationMs"]) > 1:
        e.append(f"audio.wav holds {samples / 16} ms but audio.durationMs is {audio['durationMs']}")
    return e


def _reject_constant(name):
    raise ValueError(f"{name} is not JSON")  # JSON.parse rejects NaN/Infinity too


def validate(report_json, wav):
    """Every problem with a report.json (bytes or str) + audio.wav (bytes) pair; empty when valid."""
    data = report_json.encode() if isinstance(report_json, str) else report_json
    if len(data) > MAX_JSON_BYTES:
        return ["report.json is larger than 4 MiB"]
    try:
        r = json.loads(data, parse_constant=_reject_constant)
    except ValueError:
        return ["report.json is not valid JSON"]
    e = report_problems(r) + audio_problems(r, wav)
    sha = r.get("audio", {}).get("sha256") if _obj(r) and _obj(r.get("audio")) else None
    if isinstance(sha, str) and SHA.match(sha) and hashlib.sha256(wav).hexdigest() != sha:
        e.append("audio.sha256 does not match audio.wav")
    return e


def main(argv):
    if len(argv) != 3:
        print("usage: report_validate.py report.json audio.wav", file=sys.stderr)
        return 2
    with open(argv[1], "rb") as f:
        report_json = f.read()
    with open(argv[2], "rb") as f:
        wav = f.read()
    errors = validate(report_json, wav)
    for err in errors:
        print(err, file=sys.stderr)
    if not errors:
        print("ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
