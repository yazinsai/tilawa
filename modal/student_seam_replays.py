"""Deterministic gold-excluded EveryAyah same-text seam replay corpus preparation.

Accepts text occurrence requests or the full word index; no benchmark identities are embedded.
Run on the supervisor, using only training/public augmentation sources.
"""
import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path
import random
import re
import threading
import urllib.request
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--seams")
parser.add_argument("--root", required=True)
parser.add_argument("--readers", type=int, default=3)
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--augmentation-only", action="store_true")
parser.add_argument("--all-seams", action="store_true")
parser.add_argument("--whole-surahs", action="store_true")
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--batches", type=int, default=1)
args = parser.parse_args()
root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)

def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as res:
            data = res.read()
        temp = path.with_suffix(path.suffix + f".{os.environ.get('LAB_SHARD', '0')}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return path

registry_file = download("https://everyayah.com/data/recitations.js", root / "recitations.json")
registry = json.loads(registry_file.read_text())
# Source corpus was excluded wholesale from the held-out mp3quran reader pool.
# Keep murattal Hafs Arabic tracks, one bitrate per named collection; no result-dependent selection.
collections = {}
for value in registry.values():
    if not isinstance(value, dict) or "subfolder" not in value:
        continue
    name = value["subfolder"]
    if re.search(r"warsh|shuraym.*english|multi|translation|muallim|mojaw|mujaw|nonhafs|urdu|english|parhizgar", name, re.I):
        continue
    stem = re.sub(r"[_-]?\d+kbps", "", name, flags=re.I).lower()
    if stem not in collections or name < collections[stem]:
        collections[stem] = name
readers = sorted(random.Random(0).sample(sorted(collections.values()), min(args.readers, len(collections))))
asset = json.loads(Path("packages/core/assets/ref-v2.json").read_text())
words, ayahs = [], []
def text_key(text):
    return re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]", "", text).replace("ٱ", "ا")
for ayah in asset["ayahs"]:
    ids = []
    for wi, text in enumerate(ayah["w"]):
        ids.append(len(words))
        words.append(dict(s=ayah["s"], a=ayah["a"], w=wi + 1, text=text, ayah=len(ayahs)))
    ayahs.append(dict(s=ayah["s"], a=ayah["a"], words=ids))
if args.all_seams:
    requested = list(range(len(words)))
else:
    requested = json.loads(Path(args.seams).read_text())
by_ayah = {}
for wi in requested:
    by_ayah.setdefault(words[wi]["ayah"], []).append(wi)
jobs = []
for reader in readers:
    for ai, targets in sorted(by_ayah.items()):
        context = [j for j in range(max(0, ai - 1), min(len(ayahs), ai + 2)) if ayahs[j]["s"] == ayahs[ai]["s"]]
        location = f"{ayahs[ai]['s']:03d}{ayahs[ai]['a']:03d}"
        jobs.append(dict(reader=reader, location=location, audio=str(root / "wav" / reader / f"{location}.wav"),
                         contextWords=[wi for j in context for wi in ayahs[j]["words"]], seams=targets,
                         clips=[f"{ayahs[j]['s']:03d}{ayahs[j]['a']:03d}" for j in context]))
if args.whole_surahs:
    if not args.all_seams:
        raise ValueError("--whole-surahs requires --all-seams")
    jobs = []
    by_surah = {}
    for ai, ayah in enumerate(ayahs):
        by_surah.setdefault(ayah["s"], []).append(ai)
    for reader in readers:
        for surah, context in sorted(by_surah.items()):
            location = f"{surah:03d}"
            covered = [wi for ai in context for wi in ayahs[ai]["words"]]
            jobs.append(dict(reader=reader, location=location,
                audio=str(root / "surah-wav" / reader / f"{location}.wav"),
                contextWords=covered, seams=covered,
                clips=[f"{ayahs[ai]['s']:03d}{ayahs[ai]['a']:03d}" for ai in context]))
jobs_file = "full-jobs.json" if args.all_seams else "jobs.json"
jobs_temp = root / f"{jobs_file}.{os.environ.get('LAB_SHARD', '0')}.{os.getpid()}.tmp"
jobs_temp.write_text(json.dumps(jobs, ensure_ascii=False))
jobs_temp.replace(root / jobs_file)
print(json.dumps(dict(readers=readers, requestedSeams=len(requested), ayahs=len(by_ayah), jobs=len(jobs))), flush=True)
if not args.prepare and not args.augmentation_only:
    raise SystemExit(0)

os.environ["HF_HUB_OFFLINE"] = "0"

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download
import sys
sys.path.insert(0, "tools")
from lab_sessions import degrade, phone_params

aug_path = root / "augmentation.npz"
if not aug_path.exists():
    noise_file = hf_hub_download("obadx/freesound-commercial-50k-noise-only", "data/train-00000-of-00104.parquet", repo_type="dataset")
    noises = []
    for item in pq.read_table(noise_file, columns=["audio"]).column("audio").to_pylist():
        try:
            x, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                x = soxr.resample(x, sr, 16000)
            x = x[:160000]
            if len(x) >= 16000 and np.sqrt(np.mean(x * x)) > 1e-4:
                noises.append(x / max(1e-3, np.abs(x).max()))
        except Exception:
            continue
    rir_file = hf_hub_download("EaseZh/rirs_noises", "rirs_noises.zip", repo_type="dataset")
    with zipfile.ZipFile(rir_file) as z:
        names = sorted(n for n in z.namelist() if "/simulated_rirs/" in n and n.endswith(".wav"))
        random.Random(0).shuffle(names)
        rirs = []
        for name in names[:64]:
            x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            if sr != 16000:
                x = soxr.resample(x, sr, 16000)
            rirs.append(x[:16000])
    np.savez_compressed(aug_path, **{f"noise_{i}": x for i, x in enumerate(noises)}, **{f"rir_{i}": x for i, x in enumerate(rirs)})
bank = np.load(aug_path)
noises = [bank[k] for k in sorted(bank.files) if k.startswith("noise_")]
rirs = [bank[k] for k in sorted(bank.files) if k.startswith("rir_")]
meta = dict(noises=[dict(durationS=len(x) / 16000, name=str(i)) for i, x in enumerate(noises)],
            rirs=[dict(name=str(i)) for i in range(len(rirs))])
if not noises or not rirs:
    raise ValueError("Training noise and simulated RIR bank required")
if args.augmentation_only:
    print("augmentation ready", len(noises), len(rirs), flush=True)
    raise SystemExit(0)

def prepare(job):
    dest = Path(job["audio"])
    if dest.exists():
        return True
    try:
        waves = []
        for clip in job["clips"]:
            source = download(f"https://everyayah.com/data/{job['reader']}/{clip}.mp3", root / "mp3" / job["reader"] / f"{clip}.mp3")
            wave, sr = sf.read(source, dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            if sr != 16000:
                wave = soxr.resample(wave, sr, 16000)
            waves.append(wave)
        x = np.concatenate(waves)
        ph = phone_params(f"seam-replay:{job['reader']}:{job['location']}", len(x) / 16000, meta)
        y = degrade(x, rirs[ph["rir"]], [noises[i].copy() for i in ph["noise"]], ph["snrDb"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(f".{os.environ.get('LAB_SHARD', '0')}.{os.getpid()}.tmp")
        sf.write(temp, y, 16000, format="WAV", subtype="PCM_16")
        temp.replace(dest)
        return True
    except Exception as exc:
        print("unavailable", job["reader"], job["location"], repr(exc), flush=True)
        return False

shard = int(os.environ.get("LAB_SHARD", "0"))
shards = int(os.environ.get("LAB_SHARDS", "1"))
selected = jobs[args.batch::args.batches][shard::shards]
with concurrent.futures.ThreadPoolExecutor(max_workers=1 if args.whole_surahs else 4) as executor:
    for i, ok in enumerate(executor.map(prepare, selected), 1):
        if i % 10 == 0:
            print("prepared", i, "of", len(selected), flush=True)
print("complete", shard, flush=True)
