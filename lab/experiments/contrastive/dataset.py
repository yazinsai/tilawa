"""
Dataset loader for Quran audio-text pairs.

Uses the Quran-MD dataset from HuggingFace (Buraaq/quran-md-ayahs) with streaming
to avoid downloading the full ~35GB dataset. Falls back to EveryAyah audio files
paired with quran.json text if needed.

The dataset provides multiple reciters per verse, giving natural augmentation.
Splits are by reciter (not by verse) so all verses appear in all splits.
"""

import sys
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Reuse parent project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.normalizer import normalize_arabic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QURAN_JSON = PROJECT_ROOT / "data" / "quran.json"
REFERENCE_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_audio"
ALAFASY_AUDIO_DIR = PROJECT_ROOT / "experiments" / "embedding-search" / "audio_corpus" / "alafasy"
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 15  # truncate audio longer than this (reduced from 30 for MPS memory)


def load_quran_verses() -> dict:
    """Load all verses from quran.json, keyed by (surah, ayah)."""
    with open(QURAN_JSON) as f:
        verses = json.load(f)
    return {
        (v["surah"], v["ayah"]): {
            "text_uthmani": v["text_uthmani"],
            "text_clean": v["text_clean"],
            "surah_name_en": v["surah_name_en"],
        }
        for v in verses
    }


def decode_hf_audio(audio_decoder) -> tuple[torch.Tensor, int]:
    """Decode audio from HuggingFace AudioDecoder to mono waveform tensor."""
    audio_samples = audio_decoder.get_all_samples()
    waveform = audio_samples.data  # [channels, samples]
    sr = audio_samples.sample_rate

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform.squeeze(0), TARGET_SR  # [samples]


# --- Reciter-based splits ---
# We split reciters into train/val/test so all verses appear in all splits
# but audio from a given reciter only appears in one split.
TRAIN_RECITERS = [
    "alafasy", "husary", "abdulsamad", "abdul_basit_murattal",
    "abdurrahmaan_as_sudais", "abu_bakr_ash_shaatree", "ali_jaber",
    "ghamadi", "hudhaify", "minshawy_murattal", "saood_ash_shuraym",
    "banna", "fares_abbad", "husary_mujawwad", "ibrahim_akhdar",
    "mostafa_ismail", "muhammad_jibreel", "muhsin_al_qasim",
    "nasser_alqatami", "tunaiji", "yasser_ad_dussary",
    "abdullaah_3awwaad_al_juhaynee", "abdullah_basfar",
]
VAL_RECITERS = [
    "hani_rifai", "ayman_sowaid", "minshawy_mujawwad",
]
TEST_RECITERS = [
    "hussary.teacher", "minshawy_teacher", "warsh_husary", "warsh_yassin",
]


class QuranMDStreamDataset(Dataset):
    """
    Loads a fixed number of samples from Quran-MD via streaming.
    Caches samples in memory after first load.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 500,
        max_audio_seconds: float = MAX_AUDIO_SECONDS,
        seed: int = 42,
    ):
        self.split = split
        self.max_samples = max_samples
        self.max_audio_seconds = max_audio_seconds
        self.max_audio_len = int(max_audio_seconds * TARGET_SR)

        if split == "train":
            self.reciters = set(TRAIN_RECITERS)
        elif split == "val":
            self.reciters = set(VAL_RECITERS)
        elif split == "test":
            self.reciters = set(TEST_RECITERS)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.verses = load_quran_verses()
        self.samples = []
        self._loaded = False

    def _load_samples(self):
        """Stream samples from HuggingFace and cache in memory."""
        if self._loaded:
            return

        from datasets import load_dataset

        print(f"Loading {self.split} split from Quran-MD (max {self.max_samples} samples)...")
        ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)

        count = 0
        skipped = 0
        for sample in ds:
            if count >= self.max_samples:
                break

            reciter = sample["reciter_id"]
            if reciter not in self.reciters:
                continue

            try:
                waveform, sr = decode_hf_audio(sample["audio"])

                # Truncate if too long
                if waveform.shape[0] > self.max_audio_len:
                    waveform = waveform[: self.max_audio_len]

                surah = sample["surah_id"]
                ayah = sample["ayah_id"]
                text = sample["ayah_ar"]

                self.samples.append({
                    "audio": waveform.numpy(),
                    "text": text,
                    "surah": surah,
                    "ayah": ayah,
                    "reciter": reciter,
                })
                count += 1

                if count % 50 == 0:
                    print(f"  Loaded {count}/{self.max_samples} samples...")

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Skipping sample (error: {e})")
                continue

        self._loaded = True
        print(f"Loaded {len(self.samples)} samples for {self.split} split ({skipped} skipped)")

    def __len__(self):
        self._load_samples()
        return len(self.samples)

    def __getitem__(self, idx):
        self._load_samples()
        sample = self.samples[idx]
        return {
            "audio": torch.tensor(sample["audio"], dtype=torch.float32),
            "text": sample["text"],
            "surah": sample["surah"],
            "ayah": sample["ayah"],
            "reciter": sample["reciter"],
        }


class EveryAyahDataset(Dataset):
    """
    Dataset using EveryAyah / Alafasy audio files + quran.json text.
    By default uses the full Alafasy corpus (6,236 files) if available,
    falling back to data/reference_audio/.
    """

    def __init__(
        self,
        max_audio_seconds: float = MAX_AUDIO_SECONDS,
        audio_dir: Optional[Path] = None,
        max_surah: Optional[int] = None,
    ):
        self.max_audio_len = int(max_audio_seconds * TARGET_SR)
        # Prefer the full Alafasy corpus over the 23-file reference set
        if audio_dir is not None:
            self.audio_dir = audio_dir
        elif ALAFASY_AUDIO_DIR.exists():
            self.audio_dir = ALAFASY_AUDIO_DIR
        else:
            self.audio_dir = REFERENCE_AUDIO_DIR
        self.max_surah = max_surah
        self.verses = load_quran_verses()
        self.samples = self._find_samples()

    def _find_samples(self):
        """Find all audio files and pair with verse text."""
        samples = []
        for audio_file in sorted(self.audio_dir.glob("*.mp3")):
            name = audio_file.stem
            if len(name) == 6 and name.isdigit():
                surah = int(name[:3])
                ayah = int(name[3:])
                if self.max_surah is not None and surah > self.max_surah:
                    continue
                key = (surah, ayah)
                if key in self.verses:
                    samples.append({
                        "path": str(audio_file),
                        "surah": surah,
                        "ayah": ayah,
                        "text": self.verses[key]["text_uthmani"],
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio using librosa via parent project's audio module
        from shared.audio import load_audio
        audio_np = load_audio(sample["path"], sr=TARGET_SR)

        # Truncate if needed
        if len(audio_np) > self.max_audio_len:
            audio_np = audio_np[: self.max_audio_len]

        return {
            "audio": torch.tensor(audio_np, dtype=torch.float32),
            "text": sample["text"],
            "surah": sample["surah"],
            "ayah": sample["ayah"],
            "reciter": "alafasy",
        }


def collate_fn(batch):
    """
    Custom collate function that pads audio to the same length within a batch.
    """
    # Find max audio length in this batch
    max_len = max(item["audio"].shape[0] for item in batch)

    audio_padded = []
    attention_masks = []
    texts = []
    surahs = []
    ayahs = []

    for item in batch:
        audio = item["audio"]
        pad_len = max_len - audio.shape[0]
        if pad_len > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_len))
            mask = torch.ones(max_len, dtype=torch.long)
            mask[-pad_len:] = 0
        else:
            mask = torch.ones(max_len, dtype=torch.long)

        audio_padded.append(audio)
        attention_masks.append(mask)
        texts.append(item["text"])
        surahs.append(item["surah"])
        ayahs.append(item["ayah"])

    return {
        "audio": torch.stack(audio_padded),
        "audio_attention_mask": torch.stack(attention_masks),
        "text": texts,
        "surah": torch.tensor(surahs, dtype=torch.long),
        "ayah": torch.tensor(ayahs, dtype=torch.long),
    }


def get_dataloaders(
    source: str = "everyayah",
    batch_size: int = 8,
    max_samples_per_split: int = 500,
    num_workers: int = 0,
    max_surah: Optional[int] = None,
):
    """
    Get train/val/test dataloaders.

    Args:
        source: "quran_md" for HuggingFace dataset, "everyayah" for local files
        batch_size: batch size for dataloaders
        max_samples_per_split: max samples per split (only for quran_md)
        num_workers: number of workers for dataloader
        max_surah: limit to surahs 1..max_surah (for faster experiments)
    """
    if source == "quran_md":
        train_ds = QuranMDStreamDataset("train", max_samples=max_samples_per_split)
        val_ds = QuranMDStreamDataset("val", max_samples=max_samples_per_split // 4)
        test_ds = QuranMDStreamDataset("test", max_samples=max_samples_per_split // 4)
    elif source == "everyayah":
        # Use local files -- single dataset, no proper split
        full_ds = EveryAyahDataset(max_surah=max_surah)
        n = len(full_ds)
        train_n = int(0.7 * n)
        val_n = int(0.15 * n)
        test_n = n - train_n - val_n
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [train_n, val_n, test_n],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test with EveryAyah (local, fast)
    print("=== Testing EveryAyah Dataset ===")
    ds = EveryAyahDataset()
    print(f"Found {len(ds)} audio-text pairs")
    if len(ds) > 0:
        sample = ds[0]
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Verse: {sample['surah']}:{sample['ayah']}")
        print(f"  Duration: {sample['audio'].shape[0] / TARGET_SR:.2f}s")

    # Test dataloader
    print("\n=== Testing DataLoader ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        source="everyayah", batch_size=4
    )
    for batch in train_loader:
        print(f"  Batch audio shape: {batch['audio'].shape}")
        print(f"  Batch texts: {[t[:30] for t in batch['text']]}")
        print(f"  Batch surahs: {batch['surah']}")
        break
