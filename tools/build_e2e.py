# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Assemble end-to-end test sessions from everyayah per-ayah recordings.

Each session is one 16 kHz mono WAV plus a truth timeline of what plays when, built from the parts in
`tools/e2e_sessions.json`. Needs ffmpeg.

    uv run --python cpython-3.12-macos-aarch64-none tools/build_e2e.py
    modal run modal/decode_audio.py
    pnpm --filter @tilawa/core e2e
"""

import json
import subprocess
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "tools/e2e_sessions.json"
OUT = ROOT / ".scratch/e2e"
CACHE = OUT / "cache"
SR = 16_000
# A short gap between parts, as a reciter breathes between ayahs.
GAP_S = 0.4


def fetch(reciter: str, name: str) -> np.ndarray:
    path = CACHE / reciter / f"{name}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://everyayah.com/data/{reciter}/{name}.mp3", path)
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def build(session: dict) -> None:
    rng = np.random.default_rng(7)
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    t = 0.0

    def add(audio: np.ndarray, label: str) -> None:
        nonlocal t
        pieces.append(audio)
        truth.append({"label": label, "start_s": round(t, 3), "end_s": round(t + len(audio) / SR, 3)})
        t += len(audio) / SR
        gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
        pieces.append(gap)
        t += GAP_S

    for part in session["parts"]:
        if "ayahs" in part:
            s, a, b = part["ayahs"]
            for ayah in range(a, b + 1):
                add(fetch(session["reciter"], f"{s:03d}{ayah:03d}"), f"{s}:{ayah}")
        elif "say" in part:
            try:
                audio = fetch(session["reciter"], part["say"])
            except urllib.error.HTTPError:
                # Some reciters have no separate basmala file; their 1:1 is the same words.
                if part["say"] != "bismillah":
                    raise
                audio = fetch(session["reciter"], "001001")
            add(audio, part["say"])
        elif "silence" in part:
            add(np.zeros(int(part["silence"] * SR), dtype=np.float32), "silence")
        elif "noise" in part:
            # Babble-like noise: filtered white noise with a slow loudness wobble, loud enough to decode as junk.
            n = int(part["noise"] * SR)
            white = rng.standard_normal(n).astype(np.float32)
            smooth = np.convolve(white, np.ones(8) / 8, mode="same")
            wobble = 0.5 + 0.5 * np.sin(np.linspace(0, 12 * np.pi, n))
            add((0.15 * smooth * wobble).astype(np.float32), "noise")

    audio = np.concatenate(pieces)
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / f"{session['id']}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    meta = {k: session[k] for k in ("id", "about", "reciter", "mode")}
    (OUT / f"{session['id']}.truth.json").write_text(json.dumps({**meta, "durationS": round(t, 3), "truth": truth}, indent=1) + "\n")
    print(f"{session['id']}: {t:.1f} s, {len(truth)} parts")


def main() -> None:
    for session in json.loads(SESSIONS.read_text())["sessions"]:
        build(session)


if __name__ == "__main__":
    main()
