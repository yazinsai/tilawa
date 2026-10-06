"""Teacher pass for the E1b spliced-error probe, in the E1 row format of modal/e1_pass.py.

Reads a local parquet of clips (`audio` bytes in any ffmpeg format plus metadata columns) and writes the
same rows with the greedy decode and top-5 posteriors, minus the audio. See tools/e1b_build.py.

    modal run modal/e1b_pass.py --stage clean
    modal run modal/e1b_pass.py --stage spliced
"""

from pathlib import Path

import modal

MODEL = "obadx/muaalem-model-v3_2"
SR = 16_000
TOPK = 5
# Padded audio per batch; long murattal ayahs run out of L4 memory at a fixed batch size.
BATCH_S = 60
CHUNK = 64
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
app = modal.App("tilawa-e1b-pass", image=image)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)


@app.cls(gpu="L4", volumes={"/cache": cache_vol}, timeout=60 * 60, max_containers=4)
class E1bPass:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        self.torch = torch
        m = Muaalem(MODEL, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor
        self.vocab = m.multi_level_tokenizer.id_to_vocab["phonemes"]

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

    def greedy(self, p) -> str:
        out, prev = [], -1
        for i in p.argmax(dim=-1).tolist():
            if i != prev and i != 0:
                out.append(self.vocab[i])
            prev = i
        return "".join(out)

    @modal.method()
    def run(self, rows: list[dict]) -> list[dict]:
        import subprocess

        import numpy as np

        for r in rows:
            pcm = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                input=r.pop("audio"),
                capture_output=True,
                check=True,
            ).stdout
            r["wave"] = np.frombuffer(pcm, dtype=np.float32)
        rows.sort(key=lambda r: len(r["wave"]))
        batches, cur = [], []
        for r in rows:
            if cur and (len(cur) + 1) * len(r["wave"]) > BATCH_S * SR:
                batches.append(cur)
                cur = []
            cur.append(r)
        batches.append(cur)
        out = []
        for batch in batches:
            for r, p in zip(batch, self.infer([r["wave"] for r in batch])):
                top_p, top_i = p.topk(TOPK, dim=-1)
                wave = r.pop("wave")
                out.append(
                    {
                        **r,
                        "duration_s": len(wave) / SR,
                        "greedy": self.greedy(p),
                        "n_frames": int(p.shape[0]),
                        "top5_ids": top_i.numpy().astype(np.uint8).tobytes(),
                        "top5_p": top_p.numpy().astype(np.float16).tobytes(),
                    }
                )
        return out


@app.local_entrypoint()
def main(stage: str = "clean"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(ROOT / f".scratch/e1b/{stage}-in.parquet").to_pylist()
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    out = [r for part in E1bPass().run.map(chunks) for r in part]
    dest = ROOT / f".scratch/bench/e1/e1b-{stage}.parquet"
    pq.write_table(pa.Table.from_pylist(out), dest, compression="zstd")
    print(f"{len(out)} clips -> {dest}")
