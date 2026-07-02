"""Add expected_verses field to manifest entries."""
import json
from pathlib import Path

MANIFEST = Path(__file__).parent.parent / "benchmark" / "test_corpus" / "manifest.json"

with open(MANIFEST) as f:
    manifest = json.load(f)

for sample in manifest["samples"]:
    surah = sample["surah"]
    ayah_start = sample["ayah"]
    ayah_end = sample.get("ayah_end")

    if ayah_end:
        sample["expected_verses"] = [
            {"surah": surah, "ayah": a}
            for a in range(ayah_start, ayah_end + 1)
        ]
    else:
        sample["expected_verses"] = [{"surah": surah, "ayah": ayah_start}]

with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Updated {len(manifest['samples'])} samples with expected_verses")
