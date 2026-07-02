"""Download MUSAN and OpenSLR RIR datasets to Modal volume for NeMo augmentation.

Downloads and extracts:
  - MUSAN corpus (~11GB) -> /training/augmentation_data/musan/
  - OpenSLR RIR (~5GB)  -> /training/augmentation_data/rir/

Builds NeMo-format manifests:
  /training/augmentation_data/noise_manifest.jsonl
  /training/augmentation_data/rir_manifest.jsonl

Usage:
  modal run scripts/download_musan_rir_modal.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("download-musan-rir")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget", "unzip")
    .pip_install("soundfile", "librosa")
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

AUG_DIR = Path("/training/augmentation_data")
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


@app.function(
    image=image,
    volumes={"/training": vol},
    cpu=4,
    memory=32768,
    timeout=6 * 3600,
)
def download_and_prepare():
    import librosa
    import soundfile as sf

    AUG_DIR.mkdir(parents=True, exist_ok=True)
    musan_dir = AUG_DIR / "musan"
    rir_dir = AUG_DIR / "rir"

    # --- Download and extract MUSAN ---
    if musan_dir.exists() and any(musan_dir.rglob("*.wav")):
        print(f"MUSAN already exists at {musan_dir}, skipping download.")
    else:
        print("Downloading MUSAN corpus...")
        musan_tar = AUG_DIR / "musan.tar.gz"
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(musan_tar), MUSAN_URL],
            check=True,
        )
        print("Extracting MUSAN...")
        subprocess.run(
            ["tar", "xzf", str(musan_tar), "-C", str(AUG_DIR)],
            check=True,
        )
        musan_tar.unlink()
        print("MUSAN extracted.")

    # --- Download and extract RIR ---
    if rir_dir.exists() and any(rir_dir.rglob("*.wav")):
        print(f"RIR already exists at {rir_dir}, skipping download.")
    else:
        print("Downloading OpenSLR RIR...")
        rir_zip = AUG_DIR / "rirs_noises.zip"
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(rir_zip), RIR_URL],
            check=True,
        )
        print("Extracting RIR...")
        # Extract to a temp dir first, then move to rir/
        rir_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["unzip", "-q", "-o", str(rir_zip), "-d", str(rir_dir)],
            check=True,
        )
        rir_zip.unlink()
        print("RIR extracted.")

    # --- Build manifests ---
    def get_duration(path: str) -> float:
        """Get audio duration in seconds."""
        try:
            info = sf.info(path)
            return info.duration
        except Exception:
            # Fallback to librosa for unusual formats
            try:
                dur = librosa.get_duration(path=path)
                return dur
            except Exception as e:
                print(f"  WARNING: could not read {path}: {e}")
                return 0.0

    def build_manifest(search_dir: Path, manifest_path: Path, label: str):
        """Walk search_dir for .wav files, write JSONL manifest."""
        wav_files = sorted(search_dir.rglob("*.wav"))
        print(f"Building {label} manifest from {len(wav_files)} files in {search_dir}...")
        count = 0
        with open(manifest_path, "w") as f:
            for i, wav in enumerate(wav_files):
                dur = get_duration(str(wav))
                if dur <= 0:
                    continue
                entry = {
                    "audio_filepath": str(wav),
                    "duration": round(dur, 4),
                    "offset": 0,
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
                if (i + 1) % 500 == 0:
                    print(f"  {label}: processed {i + 1}/{len(wav_files)} files")
        print(f"{label} manifest: {count} entries written to {manifest_path}")

    noise_manifest = AUG_DIR / "noise_manifest.jsonl"
    rir_manifest = AUG_DIR / "rir_manifest.jsonl"

    build_manifest(musan_dir, noise_manifest, "noise")

    # RIR files are typically under RIRS_NOISES/simulated_rirs/ or real_rirs/
    build_manifest(rir_dir, rir_manifest, "rir")

    # --- Commit volume ---
    print("Committing volume...")
    vol.commit()
    print("Done. Manifests written:")
    print(f"  {noise_manifest}")
    print(f"  {rir_manifest}")


@app.local_entrypoint()
def main():
    download_and_prepare.remote()
