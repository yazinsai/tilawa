"""
Download a manageable subset of EveryAyah for local training.
Streams the dataset and saves N examples locally as a HF dataset.
"""
import sys
from pathlib import Path
from datasets import load_dataset, Audio, Dataset

TARGET = 5000  # number of examples to download
OUT_DIR = Path(__file__).parent.parent / "data" / "train_subset"

print(f"Streaming {TARGET} examples from tarteel-ai/everyayah...")
ds = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.filter(lambda x: x["duration"] <= 30.0)

examples = []
for i, example in enumerate(ds):
    if i >= TARGET:
        break
    examples.append(example)
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{TARGET} downloaded...")

print(f"Downloaded {len(examples)} examples")

# Convert to a regular HF dataset and save
local_ds = Dataset.from_list(examples)
OUT_DIR.mkdir(parents=True, exist_ok=True)
local_ds.save_to_disk(str(OUT_DIR))
print(f"Saved to {OUT_DIR}")
