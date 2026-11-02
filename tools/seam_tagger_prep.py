"""Build phone-degraded ayah-seam clips for the seam tagger, from gold-excluded stored audio.

`neg` writes a phone copy of each real clip. `pos` splices one pause-bounded mushaf word into an ayah
seam of that clip, using the negative decode's timings, then phone-degrades the splice. The crossfade is
the practice set's 10 ms raised cosine (`lab_sessions.XFADE`).

    python tools/seam_tagger_prep.py stat
    python tools/seam_tagger_prep.py neg
    python tools/seam_tagger_prep.py pos
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import random
import sys

import numpy as np

SR = 16_000
FRAME_SAMPLES = 640  # 40 ms CTC frame
CONSONANTS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
ROOT = os.environ.get("SEAM_ROOT", f"/scratch/{os.environ.get('AUTOLAB_EXP', 'checker-learned-seam-token-tagger')}")
LIMIT = int(os.environ.get("SEAM_LIMIT", "800"))
MIN_S, MAX_S = 5.0, 20.0


def split_of(key: str) -> str:
    return "cal" if int(hashlib.md5(key.encode()).hexdigest()[:6], 16) % 5 == 0 else "train"


def shard_of(key: str) -> int:
    n = int(os.environ.get("LAB_SHARDS", "1"))
    return int(hashlib.md5(key.encode()).hexdigest()[:6], 16) % n


def mine() -> int:
    return int(os.environ.get("LAB_SHARD", "0"))


def mushaf(path: str):
    asset = json.load(open(path))
    parts = []
    words = []
    bounds = []
    pos = 0
    for ayah in asset["ayahs"]:
        bounds.append(pos)
        p = ayah["p"]
        parts.append(p)
        offs = list(ayah["o"]) + [len(p)]
        for i, (a, b) in enumerate(zip(offs, offs[1:])):
            words.append({"s": ayah["s"], "a": ayah["a"], "w": i + 1, "g0": pos + a, "g1": pos + b})
        pos += len(p)
    return "".join(parts), words, bounds


def locate(text: str, ref: str):
    hits = []
    start = 0
    while len(hits) < 3:
        i = text.find(ref, start)
        if i < 0:
            break
        hits.append(i)
        start = i + 1
    return hits


def banks():
    from lab_sessions import XFADE

    def clips(path):
        z = np.load(path)
        print(path, list(z.files), flush=True)
        if "data" in z.files and "offsets" in z.files:
            data, off = z["data"], z["offsets"]
            return [np.asarray(data[int(off[i]):int(off[i + 1])], dtype=np.float32) for i in range(len(off) - 1)]
        key = z.files[0]
        return [np.asarray(z[key][i], dtype=np.float32) for i in range(len(z[key]))]

    return clips("/student/aug/rir.npz"), clips("/student/aug/noise.npz"), XFADE


def degrade(wave: np.ndarray, rng: random.Random, rirs, noises) -> np.ndarray:
    from lab_sessions import SNR_DB, degrade as phone

    rir = rirs[rng.randrange(len(rirs))]
    noise = []
    total = 0
    while total < len(wave) + SR:
        c = noises[rng.randrange(len(noises))]
        noise.append(c)
        total += len(c)
    return phone(wave, rir, noise, rng.uniform(*SNR_DB))


def write_wav(path: str, wave: np.ndarray) -> None:
    from lab_sessions import pcm16

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(pcm16(np.asarray(wave, dtype=np.float32)))


def candidates(text: str, bounds: list[int]):
    import pyarrow.parquet as pq
    from student import exclusions, held_out

    exclude = "/app/docs/benchmarks/gold-per-v1-exclude.txt"
    if not os.path.exists(exclude):
        exclude = "/root/gold-per-v1-exclude.txt"
    gold = exclusions(open(exclude).read())
    out = []
    for f in sorted(glob.glob("/student/s1/*.parquet")):
        t = pq.read_table(f, columns=["shard", "row", "bucket", "split", "n_samples", "ref"]).to_pydict()
        for i, (sh, row, bk, sp, n, ref) in enumerate(zip(t["shard"], t["row"], t["bucket"], t["split"], t["n_samples"], t["ref"])):
            if bk != "clean-match" or sp != "train" or held_out(gold, sh, row, sp):
                continue
            if not (MIN_S * SR <= n <= MAX_S * SR) or len(ref) < 40:
                continue
            hits = locate(text, ref)
            if len(hits) != 1:
                continue
            g = hits[0]
            if not any(g < b < g + len(ref) for b in bounds):
                continue
            out.append({"file": f, "i": i, "key": f"{sh}:{row}", "ref": ref, "g": g, "n": int(n)})
    out.sort(key=lambda r: r["key"])
    return out[:LIMIT]


def meta_path(key: str, kind: str) -> str:
    return f"{ROOT}/meta/{kind}-{key.replace(':', '-')}.json"


def wav_path(key: str, kind: str) -> str:
    sid = f"{kind}-{key.replace(':', '-')}"
    return f"{ROOT}/wavs/{shard_of(key)}/{sid}.wav", sid


def load_rows(rows: list[dict]):
    import pyarrow.parquet as pq
    import soundfile as sf

    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(r["file"], []).append(r["i"])
    audio = {}
    for f, idxs in by.items():
        tab = pq.read_table(f, columns=["shard", "row", "flac"]).take(idxs).to_pydict()
        for sh, row, flac in zip(tab["shard"], tab["row"], tab["flac"]):
            wave, sr = sf.read(io.BytesIO(flac), dtype="float32")
            if sr != SR:
                raise RuntimeError(f"{sh}:{row} sample rate {sr}")
            audio[f"{sh}:{row}"] = np.asarray(wave, dtype=np.float32)
    return audio


def cmd_stat(text, bounds):
    rows = candidates(text, bounds)
    ncal = sum(split_of(r["key"]) == "cal" for r in rows)
    print(f"ayah-seam clips {len(rows)} cal {ncal} train {len(rows) - ncal} limit {LIMIT}", flush=True)


def cmd_neg(text, bounds):
    rows = [r for r in candidates(text, bounds) if shard_of(r["key"]) == mine()]
    print(f"shard {mine()} negatives {len(rows)}", flush=True)
    if not rows:
        return
    rirs, noises, _ = banks()
    audio = load_rows(rows)
    for r in rows:
        rng = random.Random(f"neg:{r['key']}")
        path, sid = wav_path(r["key"], "neg")
        if os.path.exists(path) and os.path.exists(meta_path(r["key"], "neg")):
            continue
        wave = degrade(audio[r["key"]], rng, rirs, noises)
        write_wav(path, wave)
        os.makedirs(f"{ROOT}/meta", exist_ok=True)
        json.dump({"id": r["key"], "split": split_of(r["key"]), "clean": True, "insert": None, "g": r["g"], "ref": r["ref"]},
                  open(meta_path(r["key"], "neg"), "w"), ensure_ascii=False)
    print(f"wrote negatives for shard {mine()}", flush=True)


def align(ref: str, hyp: str):
    """Map each reference phoneme to a hypothesis index, or -1."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (ri != hyp[j - 1]))
    out = [-1] * n
    i, j = n, m
    while i and j:
        sub = dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
        if dp[i, j] == sub:
            if ref[i - 1] == hyp[j - 1]:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def consonants(s: str) -> int:
    return sum(c in CONSONANTS for c in s)


def cmd_pos(text, words, bounds):
    from lab_sessions import XFADE

    rows = [r for r in candidates(text, bounds) if shard_of(r["key"]) == mine() and split_of(r["key"]) == "train"]
    print(f"shard {mine()} positive hosts {len(rows)}", flush=True)
    if not rows:
        return
    rirs, noises, _ = banks()
    audio = load_rows(rows)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(XFADE) + 0.5) / XFADE)
    made = 0
    for r in rows:
        sid = f"neg-{r['key'].replace(':', '-')}"
        ev_path = f"{ROOT}/evidence/{sid}.evidence.json"
        if not os.path.exists(ev_path):
            continue
        ev = json.load(open(ev_path))
        phones = ev["phonemes"]
        if len(phones) < 8:
            continue
        hyp = "".join(p["p"] for p in phones)
        mapping = align(r["ref"], hyp)

        def frame_of(ref_i: int) -> int | None:
            j = mapping[ref_i] if 0 <= ref_i < len(mapping) else -1
            if j < 0:
                return None
            return int(phones[j]["start"])

        seams = []
        for b in bounds:
            if not (r["g"] < b < r["g"] + len(r["ref"])):
                continue
            left_f = frame_of(b - r["g"] - 1)
            right_f = frame_of(b - r["g"])
            if left_f is None or right_f is None or right_f - left_f < 8:
                continue
            seams.append((b, left_f, right_f))
        if not seams:
            continue
        b, left_f, right_f = seams[len(seams) // 2]
        # Donor: a pause-bounded word in this clip, at least three words off the seam, two consonants, >= 0.26 s.
        donor = None
        for w in words:
            if w["g0"] < r["g"] or w["g1"] > r["g"] + len(r["ref"]):
                continue
            if abs(((w["g0"] + w["g1"]) // 2) - b) < 12:
                continue
            if consonants(text[w["g0"]:w["g1"]]) < 2:
                continue
            a = frame_of(w["g0"] - r["g"])
            c = frame_of(w["g1"] - r["g"] - 1)
            if a is None or c is None:
                continue
            if (c - a) * 0.04 < 0.26:
                continue
            pre = frame_of(w["g0"] - r["g"] - 1) if w["g0"] > r["g"] else a - 8
            post = frame_of(w["g1"] - r["g"]) if w["g1"] < r["g"] + len(r["ref"]) else c + 8
            if pre is None or post is None or a - pre < 4 or post - c < 4:
                continue
            donor = (pre, a, c, post)
            break
        if not donor:
            continue
        pre, a, c, post = donor
        host = audio[r["key"]]
        d0 = max(0, pre * FRAME_SAMPLES)
        d1 = min(len(host), post * FRAME_SAMPLES)
        word0 = a * FRAME_SAMPLES - d0
        word1 = c * FRAME_SAMPLES - d0
        if d1 - d0 < 2 * XFADE + 400 or word1 - word0 < 400:
            continue
        donor_audio = host[d0:d1].copy()
        at = int(((left_f + right_f) / 2) * FRAME_SAMPLES)
        at = max(XFADE, min(len(host) - XFADE, at))
        left, right = host[:at].copy(), host[at:].copy()
        d = donor_audio
        if len(left) < XFADE or len(right) < XFADE:
            continue
        left[-XFADE:] *= ramp[::-1]
        d = d.copy()
        d[:XFADE] *= ramp
        d[-XFADE:] *= ramp[::-1]
        right[:XFADE] *= ramp
        mid = d[XFADE:-XFADE]
        spliced = np.concatenate([left[:-XFADE], left[-XFADE:] + d[:XFADE], mid, d[-XFADE:] + right[:XFADE], right[XFADE:]])
        # The donor word's interior, past the head crossfade, in the spliced timeline.
        word_at = len(left) - XFADE + XFADE  # == len(left)
        # mid starts at len(left). The word inside the donor starts `word0` samples in, of which XFADE were the head fade.
        start_s = len(left) - XFADE + max(XFADE, word0)
        end_s = start_s + max(1, word1 - word0)
        rng = random.Random(f"pos:{r['key']}")
        wave = degrade(spliced.astype(np.float32), rng, rirs, noises)
        path, sid = wav_path(r["key"], "pos")
        write_wav(path, wave)
        insert = [start_s / FRAME_SAMPLES, end_s / FRAME_SAMPLES]
        json.dump({"id": r["key"] + "#ins", "split": "train", "clean": False, "insert": insert, "host": r["key"]},
                  open(meta_path(r["key"], "pos"), "w"), ensure_ascii=False)
        made += 1
    print(f"shard {mine()} spliced {made}", flush=True)


def crossfade(host, donor, at, ramp, xfade):
    """Insert `donor` at sample `at` with the practice set's raised-cosine crossfade. Returns (wave, word_sample_span) or None."""
    if len(donor) < 2 * xfade + 200 or at < xfade or at > len(host) - xfade:
        return None
    left, right, d = host[:at].copy(), host[at:].copy(), donor.copy()
    left[-xfade:] *= ramp[::-1]
    d[:xfade] *= ramp
    d[-xfade:] *= ramp[::-1]
    right[:xfade] *= ramp
    wave = np.concatenate([left[:-xfade], left[-xfade:] + d[:xfade], d[xfade:-xfade], d[-xfade:] + right[:xfade], right[xfade:]])
    # Donor interior, past the head fade, in the spliced timeline.
    start = len(left) + 0  # head fade ends at len(left); interior starts one fade in
    start = len(left) - xfade + xfade
    # The faded head is the first XFADE of the donor. Count the interior from there.
    start = (len(left) - xfade) + xfade
    end = start + (len(donor) - 2 * xfade)
    return wave.astype(np.float32), (start, end)


def cmd_hardneg(text, words, bounds):
    """Real-audio negatives the clean clips almost never produce as a shape-passing span.

    A restart copies the audio before the seam back into it (a whole-ayah or partial back-read).
    A fragment copies the middle of some other word, not a pause-bounded whole word.
    """
    from lab_sessions import XFADE

    rows = [r for r in candidates(text, bounds) if shard_of(r["key"]) == mine() and split_of(r["key"]) == "train"]
    print(f"shard {mine()} hard-negative hosts {len(rows)}", flush=True)
    if not rows:
        return
    rirs, noises, _ = banks()
    audio = load_rows(rows)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(XFADE) + 0.5) / XFADE)
    made = {"rep": 0, "noi": 0}
    for r in rows:
        sid = f"neg-{r['key'].replace(':', '-')}"
        ev_path = f"{ROOT}/evidence/{sid}.evidence.json"
        if not os.path.exists(ev_path):
            continue
        ev = json.load(open(ev_path))
        phones = ev["phonemes"]
        if len(phones) < 8:
            continue
        hyp = "".join(p["p"] for p in phones)
        mapping = align(r["ref"], hyp)

        def frame_of(ref_i: int):
            if not (0 <= ref_i < len(mapping)):
                return None
            j = mapping[ref_i]
            return None if j < 0 else int(phones[j]["start"])

        seam = None
        for b in bounds:
            if not (r["g"] < b < r["g"] + len(r["ref"])):
                continue
            left_f, right_f = frame_of(b - r["g"] - 1), frame_of(b - r["g"])
            if left_f is None or right_f is None or right_f - left_f < 8:
                continue
            seam = (b, left_f, right_f)
            break
        if not seam:
            continue
        b, left_f, right_f = seam
        host = audio[r["key"]]
        at = max(XFADE, min(len(host) - XFADE, int(((left_f + right_f) / 2) * FRAME_SAMPLES)))
        # Restart: the stretch from the previous ayah start (or 2 s back) up to the seam.
        back = None
        for bound in bounds:
            if r["g"] < bound <= b and b - bound > 8:
                back = bound
        src0 = frame_of((back - r["g"]) if back else 0)
        if src0 is None:
            src0 = max(0, left_f - 50)
        s0 = max(0, src0 * FRAME_SAMPLES)
        s1 = min(len(host), left_f * FRAME_SAMPLES)
        if s1 - s0 > 2 * XFADE + SR // 5:
            donor = host[s0:s1]
            # Keep at most ~2.5 s so the span stays a back-read, not the whole clip.
            if len(donor) > int(2.5 * SR):
                donor = donor[-int(2.5 * SR):]
            out = crossfade(host, donor, at, ramp, XFADE)
            if out:
                wave, (a, c) = out
                rng = random.Random(f"rep:{r['key']}")
                path, _ = wav_path(r["key"], "rep")
                write_wav(path, degrade(wave, rng, rirs, noises))
                json.dump({"id": r["key"] + "#rep", "split": "train", "clean": False, "label": "repeat",
                           "insert": [a / FRAME_SAMPLES, c / FRAME_SAMPLES], "host": r["key"]},
                          open(meta_path(r["key"], "rep"), "w"), ensure_ascii=False)
                made["rep"] += 1
        # Fragment: the middle of a word elsewhere, shorter than a whole pause-bounded word.
        for w in words:
            if w["g0"] < r["g"] + 4 or w["g1"] > r["g"] + len(r["ref"]) - 4:
                continue
            if abs(((w["g0"] + w["g1"]) // 2) - b) < 16:
                continue
            if w["g1"] - w["g0"] < 8:
                continue
            fa, fb = frame_of(w["g0"] - r["g"] + 2), frame_of(w["g1"] - r["g"] - 3)
            if fa is None or fb is None or fb - fa < 4:
                continue
            d0, d1 = fa * FRAME_SAMPLES, min(len(host), fb * FRAME_SAMPLES)
            if d1 - d0 < 2 * XFADE + 200:
                continue
            out = crossfade(host, host[d0:d1].copy(), at, ramp, XFADE)
            if not out:
                continue
            wave, (a, c) = out
            rng = random.Random(f"noi:{r['key']}")
            path, _ = wav_path(r["key"], "noi")
            write_wav(path, degrade(wave, rng, rirs, noises))
            json.dump({"id": r["key"] + "#noi", "split": "train", "clean": False, "label": "noise",
                       "insert": [a / FRAME_SAMPLES, c / FRAME_SAMPLES], "host": r["key"]},
                      open(meta_path(r["key"], "noi"), "w"), ensure_ascii=False)
            made["noi"] += 1
            break
    print(f"shard {mine()} hardneg {made}", flush=True)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "stat"
    ref = "/app/packages/core/assets/ref-v2.json"
    text, words, bounds = mushaf(ref)
    if phase == "stat":
        cmd_stat(text, bounds)
    elif phase == "neg":
        cmd_neg(text, bounds)
    elif phase == "pos":
        cmd_pos(text, words, bounds)
    elif phase == "hardneg":
        cmd_hardneg(text, words, bounds)
    else:
        raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main()
