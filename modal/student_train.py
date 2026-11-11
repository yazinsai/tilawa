"""Streaming Zipformer phoneme-CTC student distilled from Muaalem (ticket 08).

The CTC target is the teacher's greedy path from the tlog pass, or with `--target spliced` the reference inside the
teacher-anchored span and the teacher at the clip edges, runs collapsed (`student.spliced_target`). `--kd-weight` adds cross-entropy to the teacher's
renormalised top-5 posteriors per frame (E5). `--init librispeech-m` starts `full` from icefall's streaming
Zipformer-M (E2). Student and teacher both emit at 25 Hz.

    modal run modal/student_train.py --name smoke --size lite --max-hours 5 --epochs 1 --eval-max 200
    modal deploy modal/student_train.py
    python -c "import modal; modal.Function.from_name('tilawa-student-train', 'train').spawn(dict(name='e5-top1-lite', size='lite'))"

Runs live in `runs/<name>/` on volume `tilawa-student`: `ckpt.pt`, `log.jsonl`, `eval-<set>.json`.
"""

import io
import json
import os
import random
import time

import modal

ICEFALL = "3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"
K2 = (
    "https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/1.24.4.dev20260625/"
    "k2-1.24.4.dev20260625+cuda12.8.torch2.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
SR = 16_000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", K2)
    .pip_install("lhotse", "kaldialign", "sentencepiece", "tensorboard", "pypinyin==0.50.0", "typeguard", "dill",
                 "pyarrow", "soundfile", "numpy", "scipy", "huggingface_hub", "urllib3", "requests")
    .run_commands(f"git clone https://github.com/k2-fsa/icefall /icefall && cd /icefall && git checkout {ICEFALL}")
    .env({"PYTHONPATH": "/icefall:/icefall/egs/librispeech/ASR/zipformer:/root/tools", "HF_HOME": "/cache/hf"})
    .pip_install("rapidfuzz==3.14.1")
    .add_local_file("tools/student.py", "/root/tools/student.py")
    .add_local_file("tools/per_breakdown.py", "/root/tools/per_breakdown.py")
    .add_local_file("docs/benchmarks/gold-per-v1-exclude.txt", "/root/gold-per-v1-exclude.txt")
)
app = modal.App("tilawa-student-train", image=image)
vol = modal.Volume.from_name("tilawa-student")
cache_vol = modal.Volume.from_name("tilawa-hf-cache")

COMMON = dict(
    downsampling_factor="1,2,4,8,4,2",
    num_heads="4,4,4,8,4,4",
    cnn_module_kernel="31,31,15,15,15,31",
    query_head_dim="32",
    value_head_dim="12",
    pos_head_dim="4",
    pos_dim=48,
    chunk_size="16,32,64,-1",
    left_context_frames="64,128,256,-1",
)
# icefall's Zipformer-S and Zipformer-M; ticket 02 measured 20M and 70M streaming Zipformers as real time on phones.
SIZES = {
    "lite": dict(num_encoder_layers="2,2,2,2,2,2", feedforward_dim="512,768,768,768,768,768",
                 encoder_dim="192,256,256,256,256,256", encoder_unmasked_dim="192,192,192,192,192,192"),
    "full": dict(num_encoder_layers="2,2,3,4,3,2", feedforward_dim="512,768,1024,1536,1024,768",
                 encoder_dim="192,256,384,512,384,256", encoder_unmasked_dim="192,192,256,256,256,192"),
}
INITS = {"librispeech-m": ("Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17", "exp/pretrained.pt", "full")}

DEFAULTS = dict(
    name="",
    size="lite",
    init="",
    buckets="clean-match",
    target="teacher",
    max_hours=0.0,
    max_clip_s=30.0,
    epochs=6,
    max_batch_s=800.0,
    base_lr=0.045,
    lr_batches=7500,
    lr_epochs=3.5,
    ctc_weight=1.0,
    kd_weight=0.0,
    kd_delay=0,
    noise_p=0.5,
    snr_db=(5.0, 25.0),
    rir_p=0.2,
    spec_aug=True,
    eval_max=3000,
    eval_every=2000,
    ckpt_every=1000,
    workers=12,
    seed=0,
)


def ints(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split(","))


def build_model(size: str):
    import torch.nn as nn
    from scaling import ScheduledFloat
    from student import VOCAB
    from subsampling import Conv2dSubsampling
    from zipformer import Zipformer2

    p = {**COMMON, **SIZES[size]}
    dims = ints(p["encoder_dim"])

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_embed = Conv2dSubsampling(80, dims[0], dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)))
            self.encoder = Zipformer2(
                output_downsampling_factor=2,
                downsampling_factor=ints(p["downsampling_factor"]),
                num_encoder_layers=ints(p["num_encoder_layers"]),
                encoder_dim=dims,
                encoder_unmasked_dim=ints(p["encoder_unmasked_dim"]),
                query_head_dim=ints(p["query_head_dim"]),
                pos_head_dim=ints(p["pos_head_dim"]),
                value_head_dim=ints(p["value_head_dim"]),
                pos_dim=p["pos_dim"],
                num_heads=ints(p["num_heads"]),
                feedforward_dim=ints(p["feedforward_dim"]),
                cnn_module_kernel=ints(p["cnn_module_kernel"]),
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                warmup_batches=4000.0,
                causal=True,
                chunk_size=ints(p["chunk_size"]),
                left_context_frames=ints(p["left_context_frames"]),
            )
            self.ctc_output = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(max(dims), VOCAB))

        def encode(self, feats, feat_lens):
            from icefall.utils import make_pad_mask

            x, x_lens = self.encoder_embed(feats, feat_lens)
            out, out_lens = self.encoder(x.permute(1, 0, 2), x_lens, make_pad_mask(x_lens))
            return out.permute(1, 0, 2), out_lens

        def forward(self, feats, feat_lens):
            out, out_lens = self.encode(feats, feat_lens)
            return self.ctc_output(out).log_softmax(-1), out_lens

        def streaming_eval(self, chunk: int = 16, left: int = 128):
            self.encoder.chunk_size, self.encoder.left_context_frames = (chunk,), (left,)

        def training_mode(self):
            self.encoder.chunk_size = ints(p["chunk_size"])
            self.encoder.left_context_frames = ints(p["left_context_frames"])

    return Student()


def load_clips(buckets: str, split: str, max_hours: float, max_clip_s: float, seed: int, max_clips: int = 0):
    """Selected s1 clips held in memory: FLAC bytes and teacher top-5 in flat buffers with offsets."""
    import glob

    import numpy as np
    import pyarrow.parquet as pq
    from student import exclusions, held_out

    gold = exclusions(open("/root/gold-per-v1-exclude.txt").read())
    want = set(buckets.split(","))
    files = sorted(glob.glob("/student/s1/*.parquet"))
    meta = []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["shard", "row", "bucket", "split", "n_samples"]).to_pydict()
        for r, (sh, row, bk, sp, n) in enumerate(zip(t["shard"], t["row"], t["bucket"], t["split"], t["n_samples"])):
            if bk in want and sp == split and n <= max_clip_s * SR and not held_out(gold, sh, row, sp):
                meta.append((fi, r, n))
    order = np.random.default_rng(seed).permutation(len(meta))
    if max_hours:
        cum = np.cumsum([meta[i][2] for i in order]) / SR / 3600
        order = order[: int(np.searchsorted(cum, max_hours)) + 1]
    if max_clips:
        order = order[:max_clips]
    by: dict[int, list[int]] = {}
    for i in sorted(order):
        by.setdefault(meta[i][0], []).append(meta[i][1])

    out = {k: [] for k in ("key", "ref", "hyp", "n_samples", "n_frames", "flac", "ids", "probs")}
    for fi, rows in by.items():
        t = pq.read_table(files[fi], columns=["shard", "row", "ref", "hyp", "n_samples", "n_frames", "flac", "top5_ids", "top5_p"])
        t = t.take(rows).to_pydict()
        out["key"] += [f"{s}:{r}" for s, r in zip(t["shard"], t["row"])]
        for k in ("ref", "hyp", "n_samples", "n_frames"):
            out[k] += t[k]
        out["flac"] += [np.frombuffer(b, np.uint8) for b in t["flac"]]
        out["ids"] += [np.frombuffer(b, np.uint8) for b in t["top5_ids"]]
        out["probs"] += [np.frombuffer(b, np.float16) for b in t["top5_p"]]

    def flat(chunks):
        return np.concatenate(chunks), np.concatenate([[0], np.cumsum([len(c) for c in chunks])])

    flac, flac_off = flat(out.pop("flac"))
    ids, top_off = flat(out.pop("ids"))
    probs, _ = flat(out.pop("probs"))
    out["n_samples"], out["n_frames"] = np.array(out["n_samples"]), np.array(out["n_frames"])
    return dict(**out, flac=flac, flac_off=flac_off, ids=ids, probs=probs, top_off=top_off)


class ClipDataset:
    def __init__(self, clips: dict, cfg: dict, train: bool, aug: dict | None):
        self.c, self.cfg, self.train, self.aug = clips, cfg, train, aug

    def __len__(self):
        return len(self.c["ref"])

    def __getitem__(self, i):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from student import mix_noise, phoneme_ids

        c, cfg = self.c, self.cfg
        rng = np.random.default_rng()
        wave, _ = sf.read(io.BytesIO(c["flac"][c["flac_off"][i] : c["flac_off"][i + 1]].tobytes()), dtype="float32")
        if self.train and self.aug:
            if rng.random() < cfg["rir_p"]:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < cfg["noise_p"]:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0), num_mel_bins=80, sample_frequency=SR, dither=0.0,
            energy_floor=1e-10, high_freq=-400,
        )
        if self.train and cfg["spec_aug"]:
            feats = spec_augment(feats, rng)
        lo, hi = c["top_off"][i], c["top_off"][i + 1]
        return dict(
            i=i,
            feats=feats,
            target=torch.tensor(phoneme_ids(c.get("target", c["hyp"])[i]), dtype=torch.long),
            top_ids=torch.from_numpy(c["ids"][lo:hi].reshape(-1, 5).astype(np.int64)),
            top_p=torch.from_numpy(c["probs"][lo:hi].reshape(-1, 5).astype(np.float32)),
        )


def pick(bank, rng):
    k = int(rng.integers(len(bank["offsets"]) - 1))
    return bank["data"][bank["offsets"][k] : bank["offsets"][k + 1]].astype("float32")


def reverb(wave, rir):
    import numpy as np
    from scipy.signal import fftconvolve

    peak = int(np.argmax(np.abs(rir)))
    out = fftconvolve(wave, rir[peak:])[: len(wave)]
    return (out * (np.sqrt(np.mean(wave**2)) / max(1e-8, np.sqrt(np.mean(out**2))))).astype(np.float32)


def spec_augment(feats, rng, n_freq=2, max_f=27, n_time=10, max_t_frac=0.05):
    feats = feats.clone()
    T = feats.shape[0]
    for _ in range(n_freq):
        f = int(rng.integers(0, max_f + 1))
        f0 = int(rng.integers(0, 80 - f + 1))
        feats[:, f0 : f0 + f] = 0
    for _ in range(int(rng.integers(0, n_time + 1))):
        t = int(rng.integers(0, max(1, int(T * max_t_frac)) + 1))
        t0 = int(rng.integers(0, max(1, T - t)))
        feats[t0 : t0 + t] = 0
    return feats


def collate(items):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    return dict(
        i=torch.tensor([x["i"] for x in items]),
        feats=pad_sequence([x["feats"] for x in items], batch_first=True, padding_value=-23.0259),
        feat_lens=torch.tensor([x["feats"].shape[0] for x in items]),
        targets=torch.cat([x["target"] for x in items]),
        target_lens=torch.tensor([len(x["target"]) for x in items]),
        top_ids=pad_sequence([x["top_ids"] for x in items], batch_first=True),
        top_p=pad_sequence([x["top_p"] for x in items], batch_first=True),
        n_teacher=torch.tensor([x["top_ids"].shape[0] for x in items]),
    )


def batches(n_samples, max_batch_s: float, seed: int, shuffle: bool = True):
    """Duration-bucketed batches: sort within pools of 4000 random clips, cut at `max_batch_s` padded seconds."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(n_samples)) if shuffle else np.argsort(n_samples)
    out = []
    for p in range(0, len(idx), 4000):
        pool = sorted(idx[p : p + 4000], key=lambda i: n_samples[i])
        cur, longest = [], 0
        for i in pool:
            longest2 = max(longest, n_samples[i])
            if cur and longest2 * (len(cur) + 1) / SR > max_batch_s:
                out.append(cur)
                cur, longest2 = [], n_samples[i]
            cur.append(int(i))
            longest = longest2
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def losses(model, b, cfg, device):
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss

    feats, feat_lens = b["feats"].to(device), b["feat_lens"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        log_probs, out_lens = model(feats, feat_lens)
    log_probs = log_probs.float()
    frames = out_lens.sum().clamp_min(1)
    ctc = F.ctc_loss(log_probs.transpose(0, 1), b["targets"].to(device), out_lens, b["target_lens"].to(device),
                     blank=0, reduction="sum", zero_infinity=True) / frames
    out = {"ctc": ctc, "loss": cfg["ctc_weight"] * ctc}
    if cfg["kd_weight"]:
        B, T, _ = log_probs.shape
        n_t = b["n_teacher"].to(device)
        idx = (torch.arange(T, device=device)[None] - cfg["kd_delay"]).clamp(min=0)
        idx = torch.minimum(idx, (n_t - 1)[:, None])
        top_ids = b["top_ids"].to(device).gather(1, idx[..., None].expand(B, T, 5))
        top_p = b["top_p"].to(device).gather(1, idx[..., None].expand(B, T, 5))
        valid = torch.arange(T, device=device)[None] < out_lens[:, None]
        kd = soft_target_loss(log_probs, top_ids, top_p, valid)
        out["kd"] = kd
        out["loss"] = out["loss"] + cfg["kd_weight"] * kd
    out["frames"] = frames
    return out


def evaluate(model, clips, cfg, device, chunk=16, left=128):
    """Greedy streaming-equivalent decode (chunk mask) of every clip; PER against reference and teacher path."""
    import torch
    from student import corpus_per, ctc_collapse, ids_phonemes

    model.eval()
    model.streaming_eval(chunk, left)
    ds = ClipDataset(clips, cfg, train=False, aug=None)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=batches(clips["n_samples"], 600.0, 0, shuffle=False), collate_fn=collate,
        num_workers=cfg["workers"],
    )
    hyps = [None] * len(ds)
    lag = []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, lens = model(b["feats"].to(device), b["feat_lens"].to(device))
            best = lp.argmax(-1).cpu()
            for k, i in enumerate(b["i"].tolist()):
                hyps[i] = ids_phonemes(ctc_collapse(best[k, : lens[k]].tolist()))
                lag.append(int(lens[k]) - int(b["n_teacher"][k]))
    model.train()
    model.training_mode()
    refs, teacher = clips["ref"], clips["hyp"]
    return {
        "clips": len(ds),
        "chunk": chunk,
        "left": left,
        "perVsRef": corpus_per(list(zip(hyps, refs))),
        "perVsTeacher": corpus_per(list(zip(hyps, teacher))),
        "teacherPerVsRef": corpus_per(list(zip(teacher, refs))),
        "meanFrameDiff": sum(lag) / max(1, len(lag)),
        "scriptedError": None,
        "samples": [{"key": clips["key"][i], "ref": refs[i], "teacher": teacher[i], "student": hyps[i]} for i in range(min(20, len(ds)))],
    }


def init_from(model, init: str, size: str):
    import torch
    from huggingface_hub import hf_hub_download

    repo, path, need = INITS[init]
    assert size == need, f"{init} only fits size {need}"
    sd = torch.load(hf_hub_download(repo, path), map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if k.startswith(("encoder_embed.", "encoder."))}
    res = model.load_state_dict(sd, strict=False)
    return {"loaded": len(sd), "missing": res.missing_keys, "unexpected": res.unexpected_keys}


@app.function(
    gpu="H100",
    cpu=16,
    memory=131072,
    volumes={"/student": vol, "/cache": cache_vol},
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(args: dict) -> dict:
    import numpy as np
    import torch
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import Eden, ScaledAdam

    cfg = {**DEFAULTS, **args}
    assert cfg["name"], "name is required"
    run = f"/student/runs/{cfg['name']}"
    os.makedirs(run, exist_ok=True)
    log = open(f"{run}/log.jsonl", "a")

    def emit(**kw):
        kw["t"] = time.time()
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda")
    t0 = time.time()
    train_clips = load_clips(cfg["buckets"], "train", cfg["max_hours"], cfg["max_clip_s"], cfg["seed"])
    if cfg["target"] == "spliced":
        from student import spliced_target

        train_clips["target"] = [spliced_target(r, h) for r, h in zip(train_clips["ref"], train_clips["hyp"])]
    eval_clips = load_clips("clean-match", "test", 0, 60.0, 1, cfg["eval_max"])
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")} if os.path.exists("/student/aug/rir.npz") else None
    hours = train_clips["n_samples"].sum() / SR / 3600
    emit(event="data", train_clips=len(train_clips["ref"]), train_hours=float(hours), eval_clips=len(eval_clips["ref"]),
         aug=aug is not None, load_s=time.time() - t0, cfg=cfg)

    model = build_model(cfg["size"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = ScaledAdam(get_parameter_groups_with_lrs(model, lr=cfg["base_lr"], include_names=True),
                     lr=cfg["base_lr"], clipping_scale=2.0)
    sched = Eden(opt, cfg["lr_batches"], cfg["lr_epochs"])
    state = {"epoch": 0, "batch": 0, "step": 0}
    if os.path.exists(f"{run}/ckpt.pt"):
        ck = torch.load(f"{run}/ckpt.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        emit(event="resume", **state)
    elif cfg["init"]:
        emit(event="init", **init_from(model, cfg["init"], cfg["size"]))
    emit(event="model", params=n_params)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                    "cfg": cfg}, f"{run}/ckpt.pt.tmp")
        os.replace(f"{run}/ckpt.pt.tmp", f"{run}/ckpt.pt")
        vol.commit()

    ds = ClipDataset(train_clips, cfg, train=True, aug=aug)
    model.train()
    seen_s, tick = 0.0, time.time()
    while state["epoch"] < cfg["epochs"]:
        plan = batches(train_clips["n_samples"], cfg["max_batch_s"], cfg["seed"] + state["epoch"])
        loader = torch.utils.data.DataLoader(ds, batch_sampler=plan[state["batch"]:], collate_fn=collate,
                                             num_workers=cfg["workers"], prefetch_factor=4, persistent_workers=False)
        sched.step_epoch(state["epoch"])
        for b in loader:
            if state["step"] % 10 == 0:
                for m in model.modules():
                    m.batch_count = state["step"] * cfg["max_batch_s"] / 600.0
            out = losses(model, b, cfg, device)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(out["loss"]):
                state["skipped"] = state.get("skipped", 0) + 1
                emit(event="skip", **state)
                if state["skipped"] > 100:
                    raise RuntimeError("loss keeps going non-finite")
                state["batch"] += 1
                continue
            out["loss"].backward()
            opt.step()
            sched.step_batch(state["step"])
            state["step"] += 1
            state["batch"] += 1
            seen_s += b["feat_lens"].sum().item() / 100.0
            if state["step"] % 20 == 0:
                el = time.time() - tick
                emit(event="train", **state, lr=sched.get_last_lr()[0],
                     **{k: float(v) for k, v in out.items() if k != "frames"}, audio_x_realtime=seen_s / el)
                seen_s, tick = 0.0, time.time()
            if state["step"] % cfg["ckpt_every"] == 0:
                save()
            if state["step"] % cfg["eval_every"] == 0:
                emit(event="eval", set="clean-match-test", step=state["step"],
                     **{k: v for k, v in evaluate(model, eval_clips, cfg, device).items() if k != "samples"})
        state["epoch"] += 1
        state["batch"] = 0
        save()

    final = evaluate(model, eval_clips, cfg, device)
    final.update(name=cfg["name"], size=cfg["size"], params=n_params, step=state["step"], train_hours=float(hours), cfg=cfg)
    json.dump(final, open(f"{run}/eval-clean-match-test.json", "w"), ensure_ascii=False, indent=1)
    vol.commit()
    emit(event="final", **{k: v for k, v in final.items() if k not in ("samples", "cfg")})
    return {k: v for k, v in final.items() if k != "samples"}


def load_student(run: str):
    import torch

    ck = torch.load(f"/student/runs/{run}/ckpt.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]["size"])
    model.load_state_dict(ck["model"])
    return model, ck


@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/student": vol}, timeout=2 * 60 * 60)
def transcribe(run: str, src: str, dest: str, chunk: int = 16, left: int = 128) -> dict:
    """Decode `src` (parquet with an `audio` column in any ffmpeg format) with a run's student; writes `dest` in the
    teacher-pass row format (`greedy`, `n_frames`, `top5_ids`, `top5_p`), so `tools/e1_report.py` scores it as is.
    Ticket 07's scripted-error clips go through here too."""
    import subprocess

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torchaudio
    from student import ctc_collapse, ids_phonemes

    model, ck = load_student(run)
    model = model.cuda().eval()
    model.streaming_eval(chunk, left)
    rows = pq.read_table(f"/student/{src}").to_pylist()
    feats = []
    for r in rows:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                             input=r.pop("audio"), capture_output=True, check=True).stdout
        w = torch.from_numpy(np.frombuffer(pcm, dtype=np.float32).copy())
        r["duration_s"] = len(w) / SR
        feats.append(torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=SR, dither=0.0,
                                                       energy_floor=1e-10, high_freq=-400))
    order = sorted(range(len(rows)), key=lambda i: feats[i].shape[0])
    with torch.no_grad():
        for k in range(0, len(order), 16):
            idx = order[k : k + 16]
            x = torch.nn.utils.rnn.pad_sequence([feats[i] for i in idx], batch_first=True, padding_value=-23.0259).cuda()
            lens = torch.tensor([feats[i].shape[0] for i in idx]).cuda()
            lp, out_lens = model(x, lens)
            probs = lp.float().exp().cpu()
            for j, i in enumerate(idx):
                p = probs[j, : out_lens[j]]
                top_p, top_i = p.topk(5, dim=-1)
                rows[i].update(greedy=ids_phonemes(ctc_collapse(p.argmax(-1).tolist())), n_frames=int(p.shape[0]),
                               top5_ids=top_i.numpy().astype(np.uint8).tobytes(), top5_p=top_p.numpy().astype(np.float16).tobytes())
    os.makedirs(os.path.dirname(f"/student/{dest}"), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"/student/{dest}")
    vol.commit()
    return {"run": run, "rows": len(rows), "step": ck["state"]["step"], "dest": dest}


@app.local_entrypoint()
def main(name: str, size: str = "lite", init: str = "", max_hours: float = 0.0, epochs: int = 6,
         kd_weight: float = 0.0, eval_max: int = 3000, eval_every: int = 2000, max_batch_s: float = 800.0,
         base_lr: float = 0.045, target: str = "teacher"):
    print(train.remote(dict(name=name, size=size, init=init, max_hours=max_hours, epochs=epochs, kd_weight=kd_weight,
                            eval_max=eval_max, eval_every=eval_every, max_batch_s=max_batch_s, base_lr=base_lr,
                            target=target)))
