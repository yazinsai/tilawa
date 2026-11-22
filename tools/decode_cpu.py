"""Decode help clips with the Muaalem teacher on the local CPU, as modal/decode_audio.py does on a GPU (fp32 here).

    uv run --python 3.12 --with quran-muaalem==0.2.2 --with transformers==4.55.2 --with torch==2.8.0 \
        tools/decode_cpu.py --split train --out .scratch/help-dev

Writes `<clip>.evidence.json` into `--out`; existing files are kept. For dev-split pilots: the eval split already has
GPU teacher evidence beside its wavs.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from quran_muaalem import Muaalem

MODEL = "obadx/muaalem-model-v3_2"
SR = 16_000
TOPK = 5
WINDOW_SECONDS = 30.0
ROOT = Path(__file__).resolve().parent.parent


def decode(model, processor, wave: np.ndarray, tokens: str) -> dict:
    step = int(WINDOW_SECONDS * SR)
    windows = [w for w in (wave[i : i + step] for i in range(0, len(wave), step)) if len(w) >= SR // 10]
    ids, probs = [], []
    for w in windows:
        feats = processor([w], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            lg = model(**feats, return_dict=False)[0]["phonemes"]
        p = torch.softmax(lg.float(), dim=-1)[0]
        top_p, top_i = p.topk(TOPK, dim=-1)
        ids.append(top_i.numpy())
        probs.append(top_p.numpy())
    ids_all, probs_all = np.concatenate(ids), np.concatenate(probs)
    n = ids_all.shape[0]
    phonemes, prev = [], 0
    for f in range(n):
        i, c = int(ids_all[f, 0]), round(float(probs_all[f, 0]), 4)
        if i != 0 and i == prev:
            phonemes[-1]["end"] = f
            phonemes[-1]["conf"] = max(phonemes[-1]["conf"], c)
        elif i != 0:
            phonemes.append({"p": tokens[i - 1], "start": f, "end": f, "conf": c})
        prev = i
    return {
        "frameMs": round(1000 * len(wave) / SR / n),
        "nFrames": n,
        "phonemes": phonemes,
        "posteriors": {"k": TOPK, "ids": ids_all.flatten().tolist(), "probs": [round(float(x), 4) for x in probs_all.flatten()]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--help-dir", default=str(ROOT / ".scratch/help"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    help_dir, out = Path(a.help_dir), Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    index = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    tokens, inventory = index["inventory"]["tokens"], index["inventory"]["id"]
    labels = [json.loads(s) for s in (help_dir / "labels.jsonl").read_text().splitlines()]
    rows = [r for r in labels if r["split"] == a.split and r["scored"]]
    torch.set_num_threads(8)
    m = Muaalem(MODEL, device="cpu", dtype=torch.float32)
    model, processor = m.model.eval(), m.processor
    for k, r in enumerate(rows):
        dest = out / r["wav"].replace(".wav", ".evidence.json")
        if dest.exists():
            continue
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(help_dir / r["wav"]), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True).stdout
        body = decode(model, processor, np.frombuffer(pcm, dtype=np.float32), tokens)
        dest.write_text(json.dumps({"inventory": inventory, **body}, ensure_ascii=False, separators=(",", ":")))
        print(f"{k + 1}/{len(rows)} {r['id']}: {len(body['phonemes'])} phonemes", flush=True)


if __name__ == "__main__":
    main()
