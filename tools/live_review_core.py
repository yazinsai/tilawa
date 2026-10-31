"""Validation and portable issue records for live microphone sessions."""
import json
import math
import re

ID = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
MAX_AUDIO = 44 + 180 * 16000 * 2
MAX_REPORT = 16 * 1024 * 1024


def validate(report, audio):
    if not isinstance(report, dict) or report.get("schema") != 1:
        raise ValueError("Expected a version 1 live session")
    if not isinstance(report.get("id"), str) or not ID.fullmatch(report["id"]):
        raise ValueError("Invalid session ID")
    duration = report.get("durationMs")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 180000:
        raise ValueError("Record between 0 and 180 seconds")
    if not 44 < len(audio) <= MAX_AUDIO or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("Expected a WAV recording, at most 3 minutes")
    import struct
    if audio[12:16] != b"fmt " or struct.unpack_from("<IHHIIHH", audio, 16) != (16, 1, 1, 16000, 32000, 2, 16) or audio[36:40] != b"data":
        raise ValueError("Expected mono 16 kHz PCM16 WAV")
    if struct.unpack_from("<I", audio, 40)[0] != len(audio) - 44 or abs((len(audio) - 44) / 32 - duration) > 1:
        raise ValueError("Audio length does not match the session")
    flags = report.get("flags")
    if not isinstance(flags, list) or not 1 <= len(flags) <= 100:
        raise ValueError("Expected between 1 and 100 flags")
    for flag in flags:
        if not isinstance(flag, dict) or flag.get("component") not in ("listener", "tracker", "checker", "unsure"):
            raise ValueError("Invalid flag component")
        at = flag.get("timeMs")
        if not isinstance(at, (int, float)) or not math.isfinite(at) or not 0 <= at <= duration:
            raise ValueError("Flag is outside this recording")
        note = flag.get("note")
        if not isinstance(note, str) or not 1 <= len(note.strip()) <= 2000:
            raise ValueError("A flag needs a note of 1–2000 characters")
    if not isinstance(report.get("events"), list) or not isinstance(report.get("evidence"), list):
        raise ValueError("Session events and listener evidence are required")
    if not isinstance(report.get("build"), dict) or not isinstance(report.get("model"), dict):
        raise ValueError("Build and model identity are required")
    return report


def issue_markdown(report, flag, number):
    return (f"# Live mic: {flag['component']} at {flag['timeMs'] / 1000:.2f}s\n\n"
            "Status: needs-triage\n\n"
            f"Session: {report['id']}\nFlag: {number}\n"
            f"Build: {report['build'].get('revision', 'unknown')}\n"
            f"Model: {report['model'].get('modelVersion', 'unknown')}\n\n"
            f"## Report\n\n{flag['note']}\n\n"
            "## Evidence\n\nAudio: ../audio.wav\nTrace: ../report.json\n\n"
            f"Position at flag:\n\n```json\n{json.dumps(flag.get('position'), ensure_ascii=False)}\n```\n")
