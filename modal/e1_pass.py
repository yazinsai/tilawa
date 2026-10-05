"""Teacher inference pass over the E1 teacher-bias datasets.

Every clip is a single short ayah (or ayah fragment) with a labelled "as said"
transcript, so each one is decoded whole. Scoring happens locally in
tools/e1_report.py.

    modal run modal/e1_pass.py
    modal volume get tilawa-bench e1 .scratch/bench/
"""

import io
import json

import modal

MODEL = "obadx/muaalem-model-v3_2"
QDAT = "obadx/qdat_bench"
IKHLAS = "azizbekphd/ikhlas_recitations"
OUT = "/out"
SR = 16_000
TOPK = 5
BATCH = 16

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

app = modal.App("tilawa-e1-pass", image=image)
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


def qdat_rows() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table = pq.read_table(hf_hub_download(QDAT, "data/train-00000-of-00001.parquet", repo_type="dataset"))
    labels = ("qalo_alif_len", "qalo_waw_len", "laa_alif_len", "separate_madd", "noon_moshaddadah_len", "noon_mokhfah_len", "allam_alif_len", "madd_aared_len", "qalqalah")
    return [
        {
            "dataset": QDAT,
            "id": r["id"],
            "ayah": "5:109",
            "label": r["phonetic_transcript"],
            "meta": json.dumps({k: r[k] for k in ("original_id", "gender", "age", *labels)}),
            "audio": r["audio"]["bytes"],
        }
        for r in table.to_pylist()
    ]


def ikhlas_rows() -> list[dict]:
    import csv
    import re
    import zipfile

    from huggingface_hub import hf_hub_download

    zf = zipfile.ZipFile(hf_hub_download(IKHLAS, "data/ikhlas_recitations.zip", repo_type="dataset"))
    meta = list(csv.reader(io.TextIOWrapper(zf.open("data/metadata.csv"), encoding="utf-8"), skipinitialspace=True))[1:]
    rows = []
    for path, label in meta:
        m = re.match(r"audio/ID(\d+)V(\d)([TF])\.wav$", path)
        rows.append(
            {
                "dataset": IKHLAS,
                "id": path.removeprefix("audio/").removesuffix(".wav"),
                "ayah": f"112:{m[2]}",
                "label": label,
                "meta": json.dumps({"speaker": int(m[1]), "flag": m[3]}),
                "audio": zf.read(f"data/{path}"),
            }
        )
    return rows


@app.cls(gpu="L4", volumes={OUT: out_vol, "/cache": cache_vol}, secrets=[hf_secret], timeout=60 * 60)
class E1Pass:
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
    def run(self, name: str) -> dict:
        import os

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        dest = f"{OUT}/e1/{name}.parquet"
        rows = qdat_rows() if name == "qdat_bench" else ikhlas_rows()
        cache_vol.commit()
        for r in rows:
            r["wave"] = load_wave(r.pop("audio"))
        rows.sort(key=lambda r: len(r["wave"]))
        out = []
        for b in range(0, len(rows), BATCH):
            batch = rows[b : b + BATCH]
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
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(out), dest + ".tmp", compression="zstd")
        os.replace(dest + ".tmp", dest)
        out_vol.commit()
        return {"dataset": name, "clips": len(out)}


@app.local_entrypoint()
def main():
    for res in E1Pass().run.map(["qdat_bench", "ikhlas_recitations"], return_exceptions=True):
        print(res)
