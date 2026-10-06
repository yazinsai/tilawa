"""Decode local audio files with the Muaalem teacher into tracker evidence, as the listener would.

    modal run modal/decode_audio.py                     # every .scratch/e2e/*.wav
    modal run modal/decode_audio.py --files a.wav,b.wav

Writes `<name>.evidence.json` beside each file: decoded phonemes with frame spans and per-frame top-k.
"""

import json
from pathlib import Path

import modal

MODEL = "obadx/muaalem-model-v3_2"
SR = 16_000
TOPK = 5
WINDOW_SECONDS = 30.0
BATCH = 8
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
app = modal.App("tilawa-decode-audio", image=image)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)


@app.cls(gpu="L4", volumes={"/cache": cache_vol}, timeout=60 * 60, max_containers=4)
class Listener:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        self.torch = torch
        m = Muaalem(MODEL, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor

    def infer(self, waves):
        torch = self.torch
        feats = self.processor(waves, sampling_rate=SR, return_tensors="pt", padding=True)
        mask = feats["attention_mask"]
        feats = {k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
        with torch.no_grad():
            lg = self.model(**feats, return_dict=False)[0]["phonemes"]
        probs = torch.softmax(lg.float(), dim=-1).cpu()
        ratio = probs.shape[1] / mask.shape[1]
        return [probs[i, : max(1, round(n * ratio))] for i, n in enumerate(mask.sum(dim=1).tolist())]

    @modal.method()
    def decode(self, data: bytes, tokens: str) -> dict:
        import subprocess

        import numpy as np

        pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
            input=data,
            capture_output=True,
            check=True,
        ).stdout
        wave = np.frombuffer(pcm, dtype=np.float32)
        step = int(WINDOW_SECONDS * SR)
        windows = [w for w in (wave[i : i + step] for i in range(0, len(wave), step)) if len(w) >= SR // 10]
        ids, probs = [], []
        for b in range(0, len(windows), BATCH):
            for p in self.infer(windows[b : b + BATCH]):
                top_p, top_i = p.topk(TOPK, dim=-1)
                ids.append(top_i.numpy())
                probs.append(top_p.numpy())
        if not ids:
            return {"frameMs": 40, "nFrames": max(1, round(len(wave) / SR / 0.04)), "phonemes": [], "posteriors": {"k": TOPK, "ids": [], "probs": []}}
        ids_all, probs_all = np.concatenate(ids), np.concatenate(probs)
        n = ids_all.shape[0]
        # Greedy CTC collapse with frame spans, as tools/make_fixtures.py decodes.
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


@app.local_entrypoint()
def main(files: str = "", help_eval: bool = False):
    index = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    tokens, inventory = index["inventory"]["tokens"], index["inventory"]["id"]
    if help_eval:
        help_dir = ROOT / ".scratch/help"
        labels = [json.loads(line) for line in (help_dir / "labels.jsonl").read_text().splitlines()]
        paths = [help_dir / row["wav"] for row in labels if row["split"] == "eval"]
        paths = [p for p in paths if not p.with_suffix(".evidence.json").exists()]
        print(f"Decoding {len(paths)} remaining held-out help clips; existing evidence is kept.")
    else:
        paths = [Path(f) for f in files.split(",")] if files else sorted((ROOT / ".scratch/e2e").glob("*.wav"))
    listener = Listener()
    for path, out in zip(paths, listener.decode.starmap([(p.read_bytes(), tokens) for p in paths])):
        dest = path.with_suffix(".evidence.json")
        dest.write_text(json.dumps({"inventory": inventory, **out}, ensure_ascii=False, separators=(",", ":")))
        print(f"{path.name}: {len(out['phonemes'])} phonemes, {out['nFrames']} frames at {out['frameMs']} ms -> {dest.name}")
