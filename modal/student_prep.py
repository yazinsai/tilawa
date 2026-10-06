"""Student training data: tlog-v2 clean-match and deviant clips as 16 kHz FLAC joined with the teacher pass.

Reads the tlog shards cached on `tilawa-hf-cache`, `pass1/` on `tilawa-tlog`, and `manifest-s1.jsonl` on
`tilawa-student` (from `tools/student_manifest.py`). Writes `s1/<shard>.parquet` on `tilawa-student`.

    modal run modal/student_prep.py --limit 1          # one shard, in the foreground
    modal deploy modal/student_prep.py
    python -c "import modal; modal.Function.from_name('tilawa-student-prep', 'run_all').spawn()"
"""

import io

import modal

REPO = "tarteel-ai/tlog"
SR = 16_000
VERSION = "s1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("huggingface_hub", "pyarrow", "soundfile", "soxr", "numpy")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
app = modal.App("tilawa-student-prep", image=image)
tlog_vol = modal.Volume.from_name("tilawa-tlog")
cache_vol = modal.Volume.from_name("tilawa-hf-cache")
student_vol = modal.Volume.from_name("tilawa-student")


def load_manifest() -> dict[str, list[dict]]:
    import json

    by: dict[str, list[dict]] = {}
    for line in open("/student/manifest-s1.jsonl"):
        c = json.loads(line)
        by.setdefault(c["shard"], []).append(c)
    return by


@app.function(
    volumes={"/tlog": tlog_vol, "/cache": cache_vol, "/student": student_vol},
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    max_containers=60,
)
def prep_shard(shard: str, clips: list[dict]) -> dict:
    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    dest = f"/student/{VERSION}/{shard}.parquet"
    if os.path.exists(dest):
        return {"shard": shard, "skipped": True}
    want = {c["row"]: c for c in clips}
    teacher = pq.read_table(f"/tlog/pass1/{shard}.parquet", columns=["row", "hyp", "n_frames", "top5_ids", "top5_p"])
    teacher = {r["row"]: r for r in teacher.to_pylist() if r["row"] in want}
    audio = pq.read_table(hf_hub_download(REPO, f"data/{shard}.parquet", repo_type="dataset"), columns=["audio"]).column("audio")

    rows = []
    for r, c in sorted(want.items()):
        t = teacher.get(r)
        if not t or not t["hyp"]:
            continue
        wave, sr = sf.read(io.BytesIO(audio[r].as_py()["bytes"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != SR:
            wave = soxr.resample(wave, sr, SR)
        buf = io.BytesIO()
        sf.write(buf, np.clip(wave, -1, 1), SR, format="FLAC", subtype="PCM_16")
        rows.append({**c, "hyp": t["hyp"], "n_frames": t["n_frames"], "top5_ids": t["top5_ids"], "top5_p": t["top5_p"],
                     "n_samples": len(wave), "flac": buf.getvalue()})
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dest + ".tmp")
    os.replace(dest + ".tmp", dest)
    student_vol.commit()
    return {"shard": shard, "rows": len(rows), "hours": sum(x["n_samples"] for x in rows) / SR / 3600}


def pack(waves: list) -> dict:
    import numpy as np

    lens = np.array([len(w) for w in waves])
    return {"data": np.concatenate(waves).astype(np.float16), "offsets": np.concatenate([[0], np.cumsum(lens)])}


@app.function(
    volumes={"/cache": cache_vol, "/student": student_vol},
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    image=image.env({"HF_HUB_OFFLINE": "0", "HF_XET_HIGH_PERFORMANCE": "1"}),
    secrets=[modal.Secret.from_name("huggingface")],
)
def aug_bank(noise_shards: int = 3, max_noise_s: float = 10.0, n_sim_rirs: int = 3000, seed: int = 0):
    """Noise (`obadx/freesound-commercial-50k-noise-only`) and RIR (OpenSLR-28 via `EaseZh/rirs_noises`) banks at 16 kHz."""
    import glob
    import os
    import random
    import zipfile

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    def load(data: bytes):
        w, sr = sf.read(io.BytesIO(data), dtype="float32")
        if w.ndim > 1:
            w = w[:, 0]
        return soxr.resample(w, sr, SR) if sr != SR else w

    noises = []
    for i in range(noise_shards):
        f = hf_hub_download("obadx/freesound-commercial-50k-noise-only", f"data/train-{i:05d}-of-00104.parquet", repo_type="dataset")
        for a in pq.read_table(f, columns=["audio"]).column("audio").to_pylist():
            try:
                w = load(a["bytes"])[: int(max_noise_s * SR)]
            except Exception:
                continue
            if len(w) >= SR and np.sqrt(np.mean(w**2)) > 1e-4:
                noises.append(w / max(1e-3, np.abs(w).max()))

    zpath = hf_hub_download("EaseZh/rirs_noises", "rirs_noises.zip", repo_type="dataset")
    os.makedirs("/tmp/rir", exist_ok=True)
    zipfile.ZipFile(zpath).extractall("/tmp/rir")
    real = [p for p in glob.glob("/tmp/rir/**/real_rirs_isotropic_noises/*.wav", recursive=True) if "noise" not in os.path.basename(p)]
    sim = glob.glob("/tmp/rir/**/simulated_rirs/**/*.wav", recursive=True)
    random.Random(seed).shuffle(sim)
    rirs = []
    for p in real + sim[:n_sim_rirs]:
        w = load(open(p, "rb").read())[: SR]
        rirs.append(w / max(1e-6, np.abs(w).max()))

    os.makedirs("/student/aug", exist_ok=True)
    np.savez("/student/aug/noise.npz", **pack(noises))
    np.savez("/student/aug/rir.npz", **pack(rirs))
    student_vol.commit()
    return {"noises": len(noises), "noise_hours": sum(map(len, noises)) / SR / 3600, "rirs": len(rirs), "real_rirs": len(real)}


@app.function(volumes={"/student": student_vol}, timeout=12 * 60 * 60, memory=8192)
def run_all(limit: int = 0) -> list:
    by = load_manifest()
    shards = sorted(by)[: limit or None]
    out = []
    for res in prep_shard.starmap([(s, by[s]) for s in shards], return_exceptions=True, wrap_returned_exceptions=False):
        print(res)
        out.append(res if isinstance(res, dict) else {"error": repr(res)})
    return out


@app.local_entrypoint()
def main(limit: int = 1):
    for res in run_all.remote(limit):
        print(res)
