"""Teacher inference pass over the QUD alignment benchmark recordings.

Recordings run for minutes, so each one is decoded in fixed windows and the
per-frame top-k is concatenated into one stream. Phonemes cut by a window
edge may decode twice or not at all; the tracker sees that as a small error.

    modal run modal/qud_pass.py
    modal volume get tilawa-bench qud .scratch/bench/
"""

import io
import json

import modal

REPO = "QUD-Technologies/quran-alignment-benchmark"
CONFIG = "v1"
MODEL = "obadx/muaalem-model-v3_2"
OUT = "/out"
SR = 16_000
TOPK = 5
WINDOW_SECONDS = 30.0
BATCH = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "quran-muaalem==0.2.2",
        "transformers==4.55.2",
        "torch==2.8.0",
        "huggingface_hub",
        "pyarrow",
        "numpy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)

app = modal.App("tilawa-qud-pass", image=image)
out_vol = modal.Volume.from_name("tilawa-bench", create_if_missing=True)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def load_wave(data: bytes):
    import subprocess

    import numpy as np

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        input=data,
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32)


@app.cls(gpu="L4", volumes={OUT: out_vol, "/cache": cache_vol}, secrets=[hf_secret], timeout=2 * 60 * 60, max_containers=16)
class QudPass:
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
    def run(self, path: str, row: int) -> dict:
        import os

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        table = pq.read_table(hf_hub_download(REPO, path, repo_type="dataset"))
        r = table.slice(row, 1).to_pylist()[0]
        dest = f"{OUT}/qud/{r['id']}.parquet"
        out_vol.reload()
        if os.path.exists(dest):
            return {"id": r["id"], "skipped": True}

        wave = load_wave(r["audio"]["bytes"])
        step = int(WINDOW_SECONDS * SR)
        windows = [wave[i : i + step] for i in range(0, len(wave), step)]
        windows = [w for w in windows if len(w) >= SR // 10]
        ids, probs, starts = [], [], []
        frame = 0
        for b in range(0, len(windows), BATCH):
            for p in self.infer(windows[b : b + BATCH]):
                top_p, top_i = p.topk(TOPK, dim=-1)
                starts.append(frame)
                frame += p.shape[0]
                ids.append(top_i.numpy().astype(np.uint8))
                probs.append(top_p.numpy().astype(np.float16))
        ids_all, probs_all = np.concatenate(ids), np.concatenate(probs)
        meta = {k: r[k] for k in ("id", "reciter", "style", "content", "noisy", "passages", "multi_surah", "duration_s", "recited_words", "repeat_events")}
        out = {
            **meta,
            "n_frames": int(ids_all.shape[0]),
            "frame_ms": 1000 * len(wave) / SR / ids_all.shape[0],
            "window_frames": starts,
            "top5_ids": ids_all.tobytes(),
            "top5_p": probs_all.tobytes(),
            "truth": json.dumps(r["truth"], ensure_ascii=False),
            "segments": json.dumps(r["segments"], ensure_ascii=False),
        }
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        pq.write_table(pa.Table.from_pylist([out]), dest + ".tmp", compression="zstd")
        os.replace(dest + ".tmp", dest)
        out_vol.commit()
        return {"id": r["id"], "frames": out["n_frames"], "frame_ms": out["frame_ms"], "seconds": len(wave) / SR}


@app.function(secrets=[hf_secret], volumes={"/cache": cache_vol}, timeout=30 * 60)
def list_rows() -> list[tuple[str, int]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(f"{CONFIG}/") and f.endswith(".parquet"))
    rows = []
    for f in files:
        n = pq.ParquetFile(hf_hub_download(REPO, f, repo_type="dataset")).metadata.num_rows
        rows += [(f, i) for i in range(n)]
    cache_vol.commit()
    return rows


@app.local_entrypoint()
def main(limit: int = 0):
    rows = list_rows.remote()
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} recordings")
    for res in QudPass().run.starmap(rows, return_exceptions=True):
        print(res)
