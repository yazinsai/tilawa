"""Sample gold-excluded training clips and write studio plus phone-degraded wavs.

The phone copy uses the shipped training augmentation (room impulse, noise). Output stays
under /scratch so a later decode can estimate the listener confusion channel.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16_000
OUT = Path("/scratch/listener-step-span-noisy-channel-rank/channel")
N = 48
SEED = 0


def exclusions(text: str) -> set[tuple[str, int]]:
    return {(s, int(r)) for s, r in (line.rsplit(":", 1) for line in text.split() if line)}


def junctions() -> dict[str, set[str]]:
    ref = json.loads(Path("/app/packages/core/assets/ref-v2.json").read_text())
    ayahs = [a["p"] for a in ref["ayahs"]]
    ends: dict[str, set[str]] = {}
    for i in range(len(ayahs) - 1):
        a, b = ayahs[i], ayahs[i + 1]
        if len(a) < 8 or len(b) < 8:
            continue
        ends.setdefault(a[-8:], set()).add(b[:8])
    return ends


def crosses(ref: str, ends: dict[str, set[str]]) -> bool:
    for i in range(0, len(ref) - 16):
        nxt = ends.get(ref[i : i + 8])
        if nxt and ref[i + 8 : i + 16] in nxt:
            return True
    return False


def load_bank(path: str) -> dict:
    z = np.load(path)
    # student_train stores concatenated audio plus offsets. Accept either layout.
    if "data" in z and "offsets" in z:
        return {"data": z["data"], "offsets": z["offsets"]}
    keys = [k for k in z.files if k != "offsets"]
    if "offsets" in z:
        return {"data": z[keys[0]], "offsets": z["offsets"]}
    pieces = [z[k] for k in z.files]
    offsets = np.cumsum([0, *[len(p) for p in pieces]])
    return {"data": np.concatenate(pieces), "offsets": offsets}


def pick(bank: dict, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(len(bank["offsets"]) - 1))
    return np.asarray(bank["data"][bank["offsets"][k] : bank["offsets"][k + 1]], dtype=np.float32)


def reverb(wave: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    peak = int(np.argmax(np.abs(rir)))
    out = fftconvolve(wave, rir[peak:])[: len(wave)]
    scale = np.sqrt(np.mean(wave**2)) / max(1e-8, np.sqrt(np.mean(out**2)))
    return (out * scale).astype(np.float32)


def mix_noise(wave: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(len(noise)))
    n = np.resize(np.roll(noise, -start), len(wave))
    p_wave = float(np.mean(np.square(wave, dtype=np.float64)))
    p_noise = float(np.mean(np.square(n, dtype=np.float64)))
    if p_wave == 0 or p_noise == 0:
        return wave
    return (wave + n * np.sqrt(p_wave / (p_noise * 10 ** (snr_db / 10)))).astype(np.float32)


def main() -> None:
    ends = junctions()
    gold = exclusions(Path("/root/gold-per-v1-exclude.txt").read_text())
    files = sorted(Path("/student/s1").glob("clean-*.parquet"))
    chosen: list[tuple[str, int, str, str]] = []
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(files))
    for fi in order:
        if len(chosen) >= N:
            break
        f = files[int(fi)]
        t = pq.read_table(f, columns=["shard", "row", "split", "n_samples", "ref"]).to_pydict()
        idx = list(range(len(t["row"])))
        rng.shuffle(idx)
        for i in idx:
            if t["split"][i] != "train":
                continue
            n = int(t["n_samples"][i])
            if n < 4 * SR or n > 25 * SR:
                continue
            ref = t["ref"][i]
            if not isinstance(ref, str) or len(ref) < 40 or not crosses(ref, ends):
                continue
            key = (t["shard"][i], int(t["row"][i]))
            if key in gold:
                continue
            chosen.append((str(f), i, f"{key[0]}-{key[1]}", ref))
            if len(chosen) >= N:
                break
        print(f"scanned {f.name}: {len(chosen)}", flush=True)
    if len(chosen) < 8:
        raise SystemExit(f"only {len(chosen)} seam clips")
    rir = load_bank("/student/aug/rir.npz")
    noise = load_bank("/student/aug/noise.npz")
    wavs = OUT / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    refs = []
    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for path, row, name, ref in chosen:
        by_file.setdefault(path, []).append((row, name, ref))
    for path, rows in by_file.items():
        table = pq.read_table(path, columns=["flac"]).column("flac")
        for row, name, ref in rows:
            raw = table[row].as_py()
            wave, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            if sr != SR:
                raise SystemExit(f"unexpected rate {sr}")
            local = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
            phone = reverb(wave, pick(rir, local))
            phone = mix_noise(phone, pick(noise, local), float(local.uniform(5, 25)), local)
            sf.write(wavs / f"{name}.studio.wav", wave, SR, subtype="PCM_16")
            sf.write(wavs / f"{name}.phone.wav", np.clip(phone, -1, 1), SR, subtype="PCM_16")
            refs.append({"id": name, "ref": ref})
    (OUT / "refs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in refs))
    print(f"wrote {len(refs)} clips -> {wavs}", flush=True)


if __name__ == "__main__":
    main()
