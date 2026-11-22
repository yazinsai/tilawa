"""Speaker embeddings and clusters for tarteel-ai/tlog, which has no speaker metadata.

WavLM-SV x-vectors per clip (first 10 s), mean-centred. Clustering is global:
k-means micro-clusters, then average linkage over their centroids. Single
linkage on a kNN graph chains 99% of clips into one cluster.

    modal deploy modal/speaker_embed.py && python -c "import modal; modal.Function.from_name('tilawa-speaker-embed', 'run_all').spawn()"
    modal run modal/speaker_embed.py --shard clean-00258-of-00411    # one shard
    modal run modal/speaker_embed.py --cluster-only --thresholds 0.3,0.4,0.5
"""

import io

import modal

REPO = "tarteel-ai/tlog"
MODEL = "microsoft/wavlm-base-plus-sv"
OUT = "/out"
SR = 16_000
MAX_SECONDS = 10.0
MIN_SECONDS = 0.5
BATCH_SECONDS = 160.0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("transformers==4.55.2", "torch==2.8.0", "huggingface_hub", "pyarrow", "soundfile", "soxr", "numpy")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)

app = modal.App("tilawa-speaker-embed", image=image)
out_vol = modal.Volume.from_name("tilawa-tlog", create_if_missing=True)
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


@app.cls(gpu="T4", volumes={OUT: out_vol, "/cache": cache_vol}, secrets=[hf_secret], timeout=2 * 60 * 60, max_containers=20)
class Embedder:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = WavLMForXVector.from_pretrained(MODEL).to("cuda").half().eval()

    def wave(self, audio):
        import numpy as np
        import soundfile as sf
        import soxr

        try:
            w, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        except Exception:
            return None
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != SR:
            w = soxr.resample(w, sr, SR)
        w = w[: int(MAX_SECONDS * SR)]
        return np.ascontiguousarray(w, dtype=np.float32) if len(w) >= MIN_SECONDS * SR else None

    def embed(self, waves):
        torch = self.torch
        f = self.fe(waves, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.no_grad():
            e = self.model(input_values=f["input_values"].to("cuda").half(), attention_mask=f["attention_mask"].to("cuda")).embeddings
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    @modal.method()
    def run_shard(self, name: str) -> dict:
        import os

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        dest = f"{OUT}/emb/{name}.parquet"
        out_vol.reload()
        if os.path.exists(dest):
            return {"shard": name, "skipped": True}
        table = pq.read_table(hf_hub_download(REPO, f"data/{name}.parquet", repo_type="dataset"), columns=["audio"])
        items = []
        for row, audio in enumerate(table.column("audio").to_pylist()):
            w = self.wave(audio) if audio and audio.get("bytes") else None
            if w is not None:
                items.append((row, (audio.get("path") or ""), w))
        items.sort(key=lambda x: len(x[2]))
        rows, paths, embs, batch = [], [], [], []

        def flush():
            if batch:
                embs.append(self.embed([w for _, _, w in batch]))
                rows.extend(r for r, _, _ in batch)
                paths.extend(p for _, p, _ in batch)
                batch.clear()

        for it in items:
            if batch and (len(batch) + 1) * len(it[2]) / SR > BATCH_SECONDS:
                flush()
            batch.append(it)
        flush()
        e = np.concatenate(embs).astype(np.float16) if embs else np.zeros((0, 512), np.float16)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        pq.write_table(
            pa.table({"shard": [name] * len(rows), "row": rows, "path": paths, "emb": [x.tobytes() for x in e]}),
            dest + ".tmp",
        )
        os.replace(dest + ".tmp", dest)
        out_vol.commit()
        return {"shard": name, "embedded": len(rows), "rows": table.num_rows}


@app.function(gpu="L4", volumes={OUT: out_vol}, timeout=6 * 60 * 60, memory=98304, image=image.pip_install("scipy"))
def cluster(thresholds: str, per_centroid: int) -> list:
    """Mean-centred cosine; k-means micro-clusters, then average linkage over centroids, cut at each threshold distance."""
    import glob
    import json
    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from scipy.cluster.hierarchy import fcluster, linkage

    out_vol.reload()
    t = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{OUT}/emb/*.parquet"))])
    n = t.num_rows
    x = torch.from_numpy(np.frombuffer(b"".join(t.column("emb").to_pylist()), dtype=np.float16).reshape(n, -1).astype(np.float32)).cuda()
    x = torch.nn.functional.normalize(x - x.mean(0), dim=-1)

    k = max(1, n // per_centroid)
    g = torch.Generator(device="cuda").manual_seed(0)
    c = x[torch.randperm(n, device="cuda", generator=g)[:k]].clone()
    assign = torch.empty(n, dtype=torch.long, device="cuda")
    for _ in range(25):
        for s in range(0, n, 16384):
            assign[s : s + 16384] = (x[s : s + 16384] @ c.T).argmax(-1)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        c = torch.where(counts[:, None] > 0, torch.nn.functional.normalize(sums, dim=-1), c)

    cn = c.cpu().numpy().astype(np.float64)
    d = 1 - cn @ cn.T
    cond = np.clip(d[np.triu_indices(k, 1)], 0, 2)
    del d
    tree = linkage(cond, "average")
    micro = assign.cpu().numpy()

    g = torch.Generator(device="cuda").manual_seed(1)
    probe = torch.randperm(n, device="cuda", generator=g)[: min(n, 50_000)]
    vs, js = [], []
    for s in range(0, len(probe), 1024):
        p = probe[s : s + 1024]
        sims = x[p] @ x.T
        sims[torch.arange(len(p)), p] = -1
        v, j = sims.max(-1)
        vs.append(v)
        js.append(j)
    v, j = torch.cat(vs).cpu().numpy(), torch.cat(js).cpu().numpy()
    probe = probe.cpu().numpy()

    os.makedirs(f"{OUT}/clusters", exist_ok=True)
    out = []
    for threshold in [float(t) for t in thresholds.split(",")]:
        cid = (fcluster(tree, threshold, "distance") - 1)[micro]
        # Leak diagnostic: a clip's nearest neighbour above a similarity, filed under another cluster.
        across = {str(th): float(((cid[j] != cid[probe]) & (v >= th)).sum() / max(1, int((v >= th).sum()))) for th in (0.9, 0.95)}
        name = f"clusters-t{threshold}-c{per_centroid}"
        pq.write_table(
            pa.table({"shard": t.column("shard"), "row": t.column("row"), "path": t.column("path"), "cluster": cid.astype(np.int32)}),
            f"{OUT}/clusters/{name}.parquet",
        )
        sizes = np.bincount(cid)
        sizes = np.sort(sizes[sizes > 0])[::-1]
        stats = {
            "file": f"clusters/{name}.parquet",
            "threshold": threshold,
            "clips": n,
            "centroids": k,
            "clusters": int(len(sizes)),
            "singletons": int((sizes == 1).sum()),
            "largest": sizes[:10].tolist(),
            "largest_share": float(sizes[0] / n),
            "nearest_neighbour_at_least": {str(th): float((v >= th).mean()) for th in (0.9, 0.95)},
            "nearest_neighbour_across_clusters": across,
        }
        print(json.dumps(stats))
        out.append(stats)
    out_vol.commit()
    return out


@app.function(secrets=[hf_secret], timeout=24 * 60 * 60)
def run_all() -> list:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    names = sorted(f[5:-8] for f in files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"{len(names)} shards")
    return list(Embedder().run_shard.map(names, return_exceptions=True, wrap_returned_exceptions=False))


@app.local_entrypoint()
def main(shard: str = "", cluster_only: bool = False, thresholds: str = "0.3", per_centroid: int = 20):
    if shard:
        print(Embedder().run_shard.remote(shard))
        return
    if not cluster_only:
        for r in run_all.remote():
            print(r)
    for stats in cluster.remote(thresholds, per_centroid):
        print(stats)
