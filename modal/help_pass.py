"""Teacher greedy decode of the help-page clips, to catch recordings of the wrong passage.

    uv run tools/help_audit.py pull
    modal run modal/help_pass.py        # writes .scratch/help/teacher.json locally
"""

import json
from pathlib import Path

import modal

MODEL = "obadx/muaalem-model-v3_2"
SR = 16_000
# Padded seconds per batch (batch size x longest clip); attention is quadratic in length.
MAX_BATCH_SECONDS = 80.0
WAV = Path(__file__).resolve().parent.parent / ".scratch/help/wav"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("quran-muaalem==0.2.2", "transformers==4.55.2", "torch==2.8.0", "soundfile", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)
app = modal.App("tilawa-help-pass", image=image)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


@app.cls(gpu="L4", volumes={"/cache": cache_vol}, secrets=[hf_secret], timeout=30 * 60, max_containers=4)
class HelpPass:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem

        self.torch = torch
        m = Muaalem(MODEL, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor
        self.vocab = m.multi_level_tokenizer.id_to_vocab["phonemes"]

    @modal.method()
    def run(self, batch: list[tuple[str, bytes]]) -> dict[str, str]:
        import io

        import numpy as np
        import soundfile as sf

        torch = self.torch
        clips = [(i, np.ascontiguousarray(sf.read(io.BytesIO(d), dtype="float32")[0])) for i, d in batch]
        groups, cur = [], []
        for c in clips:
            if cur and (len(cur) + 1) * len(c[1]) / SR > MAX_BATCH_SECONDS:
                groups.append(cur)
                cur = []
            cur.append(c)
        groups.append(cur)
        out = {}
        for group in groups:
            ids, waves = [i for i, _ in group], [w for _, w in group]
            feats = self.processor(waves, sampling_rate=SR, return_tensors="pt", padding=True)
            mask = feats["attention_mask"]
            feats = {k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
            with torch.no_grad():
                lg = self.model(**feats, return_dict=False)[0]["phonemes"]
            am = lg.argmax(dim=-1).cpu()
            ratio = am.shape[1] / mask.shape[1]
            for k, n in enumerate(mask.sum(dim=1).tolist()):
                seq, prev = [], -1
                for t in am[k, : max(1, round(n * ratio))].tolist():
                    if t != prev and t != 0:
                        seq.append(self.vocab[t])
                    prev = t
                out[ids[k]] = "".join(seq)
        return out


@app.local_entrypoint()
def main():
    files = sorted(WAV.glob("*.wav"), key=lambda p: p.stat().st_size)
    chunks = [[(p.stem, p.read_bytes()) for p in files[i : i + 64]] for i in range(0, len(files), 64)]
    hyps = {}
    for res in HelpPass().run.map(chunks):
        hyps.update(res)
    dest = WAV.parent / "teacher.json"
    dest.write_text(json.dumps(hyps, ensure_ascii=False))
    print(f"{len(hyps)} clips -> {dest}")
