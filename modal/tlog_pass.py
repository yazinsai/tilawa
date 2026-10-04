"""Teacher inference pass over tarteel-ai/tlog.

Decodes every clip with Muaalem without trusting the filename label, then
scores the decode against the filename ayah as a first-pass prior. Global
relabelling and bucketing happen downstream in the tracker.

    modal run modal/tlog_pass.py                  # 10% stratified sample
    modal run modal/tlog_pass.py --every 1        # full corpus
    modal run modal/tlog_pass.py --limit 2        # smoke test on 2 shards
"""

import io
import re

import modal

REPO = "tarteel-ai/tlog"
MODEL = "obadx/muaalem-model-v3_2"
OUT = "/out"
SR = 16_000
TOPK = 5
# Padded seconds per batch (batch size x longest clip); attention is quadratic in length.
MAX_BATCH_SECONDS = 80.0
MAX_CLIP_SECONDS = 60.0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "quran-muaalem==0.2.2",
        "quran-transcript>=0.6.4",
        "transformers==4.55.2",
        "torch==2.8.0",
        "huggingface_hub",
        "pyarrow",
        "soundfile",
        "soxr",
        "rapidfuzz",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)

app = modal.App("tilawa-tlog-pass", image=image)
out_vol = modal.Volume.from_name("tilawa-tlog", create_if_missing=True)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

FILENAME_RE = re.compile(r"(?:.*/)?(\d+)_(\d+)_([^/.]+)\.\w+$")


@app.cls(
    gpu="L4",
    volumes={OUT: out_vol, "/cache": cache_vol},
    secrets=[hf_secret],
    timeout=3 * 60 * 60,
    max_containers=20,
)
class TeacherPass:
    @modal.enter()
    def load(self):
        import torch
        from quran_muaalem import Muaalem
        from quran_transcript import MoshafAttributes

        self.torch = torch
        m = Muaalem(MODEL, device="cuda", dtype=torch.bfloat16)
        self.model = m.model.eval()
        self.processor = m.processor
        self.vocab = m.multi_level_tokenizer.id_to_vocab
        # Learner madd lengths vary; this is only the prior's reference, not a verdict.
        self.moshaf = MoshafAttributes(
            rewaya="hafs",
            madd_monfasel_len=4,
            madd_mottasel_len=4,
            madd_mottasel_waqf=4,
            madd_aared_len=2,
        )
        self.ref_cache: dict[tuple[int, int], str | None] = {}

    def reference(self, surah: int, ayah: int) -> str | None:
        key = (surah, ayah)
        if key not in self.ref_cache:
            from quran_transcript import Aya, quran_phonetizer

            try:
                uthmani = Aya(surah, ayah).get().uthmani
                self.ref_cache[key] = quran_phonetizer(
                    uthmani, self.moshaf, remove_spaces=True, sura_idx=surah
                ).phonemes
            except Exception:
                self.ref_cache[key] = None
        return self.ref_cache[key]

    def decode_audio(self, audio: dict):
        import numpy as np
        import soundfile as sf
        import soxr

        try:
            wave, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        except Exception:
            return None
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != SR:
            wave = soxr.resample(wave, sr, SR)
        return np.ascontiguousarray(wave, dtype=np.float32)

    def infer(self, waves):
        torch = self.torch
        feats = self.processor(
            waves, sampling_rate=SR, return_tensors="pt", padding=True
        )
        mask = feats["attention_mask"]
        feats = {
            k: v.to("cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in feats.items()
        }
        with torch.no_grad():
            logits = self.model(**feats, return_dict=False)[0]

        in_len = mask.shape[1]
        valid_in = mask.sum(dim=1).tolist()
        results = [dict() for _ in waves]
        for level, lg in logits.items():
            probs = torch.softmax(lg.float(), dim=-1).cpu()
            ratio = probs.shape[1] / in_len
            for i, n_in in enumerate(valid_in):
                p = probs[i, : max(1, round(n_in * ratio))]
                results[i][level] = p
        return results

    def greedy(self, p) -> tuple[str, list[float]]:
        conf, ids = p.max(dim=-1)
        text, confs, prev = [], [], -1
        for i, c in zip(ids.tolist(), conf.tolist()):
            if i != prev and i != 0:
                text.append(self.vocab["phonemes"][i])
                confs.append(c)
            prev = i
        return "".join(text), confs

    @modal.method()
    def run_shard(self, shard: str) -> dict:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from rapidfuzz.distance import Levenshtein

        name = shard.split("/")[-1].replace(".parquet", "")
        dest = f"{OUT}/pass1/{name}.parquet"
        out_vol.reload()
        import os

        if os.path.exists(dest):
            return {"shard": name, "skipped": True}

        local = hf_hub_download(REPO, shard, repo_type="dataset")
        table = pq.read_table(local)
        cols = table.column_names
        audio_col = table.column("audio").to_pylist()

        rows, pending = [], []

        def flush():
            if not pending:
                return
            outs = self.infer([w for _, w in pending])
            for (row, _), levels in zip(pending, outs):
                ph = levels["phonemes"]
                hyp, confs = self.greedy(ph)
                top_p, top_i = ph.topk(TOPK, dim=-1)
                row.update(
                    status="ok",
                    hyp=hyp,
                    hyp_conf=confs,
                    mean_conf=float(np.mean(confs)) if confs else 0.0,
                    n_frames=int(ph.shape[0]),
                    top5_ids=top_i.numpy().astype(np.uint8).tobytes(),
                    top5_p=top_p.numpy().astype(np.float16).tobytes(),
                    sifat_ids=b"".join(
                        levels[lv].argmax(dim=-1).numpy().astype(np.uint8).tobytes()
                        for lv in sorted(levels)
                        if lv != "phonemes"
                    ),
                )
                ref = row["ref"]
                if ref:
                    row["prior_per"] = Levenshtein.distance(hyp, ref) / max(1, len(ref))
                rows.append(row)
            pending.clear()

        decoded = []
        for idx, audio in enumerate(audio_col):
            path = (audio or {}).get("path") or ""
            m = FILENAME_RE.match(path)
            surah, ayah, rec_id = (int(m[1]), int(m[2]), m[3]) if m else (None, None, None)
            row = {
                "shard": name,
                "row": idx,
                "path": path,
                "surah": surah,
                "ayah": ayah,
                "rec_id": rec_id,
                "ref": self.reference(surah, ayah) if surah else None,
                "status": "decode_failed",
                "duration_s": None,
                "hyp": None,
                "hyp_conf": None,
                "mean_conf": None,
                "n_frames": None,
                "top5_ids": None,
                "top5_p": None,
                "sifat_ids": None,
                "prior_per": None,
            }
            for extra in ("is_clean",):
                if extra in cols:
                    row[extra] = table.column(extra)[idx].as_py()

            wave = self.decode_audio(audio) if audio and audio.get("bytes") else None
            if wave is None or len(wave) < SR // 4:
                rows.append(row)
                continue
            dur = len(wave) / SR
            row["duration_s"] = dur
            if dur > MAX_CLIP_SECONDS:
                row["status"] = "too_long"
                rows.append(row)
                continue
            decoded.append((row, wave))

        decoded.sort(key=lambda x: x[1].shape[0])
        for row, wave in decoded:
            dur = len(wave) / SR
            if pending and (len(pending) + 1) * dur > MAX_BATCH_SECONDS:
                flush()
            pending.append((row, wave))
        flush()

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), dest + ".tmp", compression="zstd")
        os.replace(dest + ".tmp", dest)
        out_vol.commit()
        ok = sum(r["status"] == "ok" for r in rows)
        return {"shard": name, "rows": len(rows), "ok": ok}


@app.function(secrets=[hf_secret])
def list_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


@app.local_entrypoint()
def main(every: int = 10, limit: int = 0):
    shards = list_shards.remote()
    # Stratify: every Nth shard within each split, so clean and unclean are both sampled.
    picked = []
    for split in ("clean", "unclean"):
        group = [s for s in shards if f"/{split}-" in s]
        picked += group[::every]
    if limit:
        picked = picked[:limit]
    print(f"{len(picked)} of {len(shards)} shards")
    for res in TeacherPass().run_shard.map(picked, return_exceptions=True):
        print(res)
