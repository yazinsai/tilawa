"""Whole-word acoustic embeddings over a frozen student's middle encoder stack.

Training reads only gold-excluded, clean-match s1 recitations and the shipped
augmentation banks. The reference supplies word boundaries, never practice labels.
The acceptance radius is the held-out real-word 99th percentile, fixed once.
Run this module directly in the training image; it does not launch Modal jobs.
"""

import argparse
import bisect
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BINS = 24
EMBED_DIM = 128


def temporal_pool(x, lengths, bins=BINS):
    """Adaptive mean pooling using cumulative sums; also exportable to ONNX.

    Unlike padding a short word to a long batch mate, each word's final bin ends
    at its own last valid frame. One-frame inputs remain well-defined.
    """
    positions = torch.arange(bins, device=x.device)[None]
    starts = torch.div(positions * lengths[:, None], bins, rounding_mode="floor")
    ends = torch.div((positions + 1) * lengths[:, None] + bins - 1, bins, rounding_mode="floor")
    integral = F.pad(x.cumsum(-1), (1, 0))
    a = integral.gather(2, starts[:, None].expand(-1, x.shape[1], -1))
    b = integral.gather(2, ends[:, None].expand(-1, x.shape[1], -1))
    return (b - a) / (ends - starts)[:, None].clamp_min(1)


def collapse_with_offsets(text):
    phones, offsets = [], []
    for i, p in enumerate(text):
        if not phones or p != phones[-1]:
            phones.append(p)
            offsets.append(i)
    return "".join(phones), offsets


class Lexicon:
    def __init__(self, asset):
        self.text, self.words, self.starts = "", [], []
        for ayah in asset["ayahs"]:
            base = len(self.text)
            for i, start in enumerate(ayah["o"]):
                end = ayah["o"][i + 1] if i + 1 < len(ayah["o"]) else len(ayah["p"])
                key = ayah["iso"][i]
                self.starts.append(base + start)
                self.words.append((base + start, base + end, key))
            self.text += ayah["p"]

    def locate(self, ref):
        """Only wholly included words of an exact canonical transcript match."""
        start = self.text.find(ref)
        if start < 0 or not ref:
            return []
        out = []
        for a, b, key in itertools.islice(self.words, bisect.bisect_left(self.starts, start), None):
            if b > start + len(ref):
                break
            if b > a and key:
                out.append((a - start, b - start, key))
        return out


def word_spans(row, lexicon):
    """CTC forced alignment on existing teacher rows; no acoustic data is added."""
    import torchaudio
    from student import VOCAB, phoneme_ids

    words = lexicon.locate(row["ref"])
    target, offsets = collapse_with_offsets(row["ref"])
    ids = np.frombuffer(row["top5_ids"], np.uint8).reshape(-1, 5)
    prob = np.frombuffer(row["top5_p"], np.float16).reshape(-1, 5).astype(np.float32)
    if not words or not target or len(target) >= len(ids):
        return []
    dense = np.full((len(ids), VOCAB), 1e-8, np.float32)
    np.put_along_axis(dense, ids.astype(np.int64), np.maximum(prob, 1e-8), axis=1)
    dense /= dense.sum(1, keepdims=True)
    try:
        path, _ = torchaudio.functional.forced_align(
            torch.from_numpy(np.log(dense))[None], torch.tensor([phoneme_ids(target)]), blank=0)
    except RuntimeError:
        return []
    # The collapsed target has no repeated adjacent labels. Each nonblank run
    # corresponds to exactly one target phone; preserve its original text offset.
    runs = []
    last = 0
    for t, p in enumerate(path[0].tolist()):
        if p and p != last:
            runs.append([t, t + 1])
        elif p:
            runs[-1][1] = t + 1
        last = p
    if len(runs) != len(target):
        return []
    out = []
    for a, b, key in words:
        lo, hi = bisect.bisect_left(offsets, a), bisect.bisect_left(offsets, b)
        if hi <= lo:
            continue
        f0, f1 = runs[lo][0], runs[hi - 1][1]
        # Include up to 80 ms of the CTC blank on either side, bounded by the
        # halfway point to the neighbouring phone, so no neighbouring word enters.
        left = max(0, f0 - 2, (runs[lo - 1][1] + f0) // 2 if lo else 0)
        right = min(len(ids), f1 + 2, (f1 + runs[hi][0]) // 2 if hi < len(runs) else len(ids))
        if not 0.16 <= (right - left) * 0.04 <= 4:
            continue
        prefixes = [(runs[k][1] - left) / (right - left) for k in range(lo, hi - 1)]
        out.append(dict(key=key, start=left * 640, end=right * 640, prefixes=prefixes))
    return out


def prepare(args):
    import pyarrow.parquet as pq
    import soundfile as sf
    from student import exclusions

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon(json.loads(Path(args.reference).read_text()))
    excluded = exclusions(Path(args.exclusions).read_text())
    rng = np.random.default_rng(0)
    files = sorted(Path(args.data).glob("*.parquet"))
    rng.shuffle(files)
    records, counts = [], {}
    audio_dir = root / "words"
    audio_dir.mkdir(exist_ok=True)
    split_n = {"train": 0, "val": 0}
    for fi, path in enumerate(files):
        meta = pq.read_table(path, columns=["shard", "row", "split", "bucket", "n_samples", "ref"]).to_pylist()
        selected = []
        for i, r in enumerate(meta):
            if (r["shard"], r["row"]) in excluded or r["bucket"] != "clean-match" or r["split"] not in split_n:
                continue
            if r["n_samples"] > 30 * 16000 or not lexicon.locate(r["ref"]):
                continue
            cap = args.train_words if r["split"] == "train" else args.val_words
            if split_n[r["split"]] < cap:
                selected.append(i)
        if not selected:
            continue
        rows = pq.read_table(path).take(selected).to_pylist()
        rng.shuffle(rows)
        for row in rows:
            split = row["split"]
            cap = args.train_words if split == "train" else args.val_words
            if split_n[split] >= cap:
                continue
            spans = word_spans(row, lexicon)
            if not spans:
                continue
            wave, sr = sf.read(io.BytesIO(row["flac"]), dtype="float32")
            assert sr == 16000 and wave.ndim == 1
            for span in spans:
                k = (split, span["key"])
                if counts.get(k, 0) >= (32 if split == "train" else 4) or split_n[split] >= cap:
                    continue
                word = wave[span["start"]:span["end"]]
                if len(word) < 2560:
                    continue
                name = f"{len(records):07d}.flac"
                sf.write(audio_dir / name, word, sr, format="FLAC", subtype="PCM_16")
                records.append(dict(**span, file=name, split=split, source=f'{row["shard"]}:{row["row"]}',
                                    cluster=row["cluster"], samples=len(word)))
                counts[k] = counts.get(k, 0) + 1
                split_n[split] += 1
        if fi % 10 == 0:
            print(json.dumps(dict(stage="prepare", files=fi + 1, words=split_n)), flush=True)
        if split_n["train"] >= args.train_words and split_n["val"] >= args.val_words:
            break
    (root / "segments.json").write_text(json.dumps(records, ensure_ascii=False))
    print(json.dumps(dict(stage="prepared", words=split_n, lexicon=len({r["key"] for r in records}))), flush=True)


def acoustic_features(wave, cfg, rng=None, aug=None):
    import torchaudio
    from student import mix_noise
    from student_train import pick, reverb, spec_augment

    if aug is not None:
        if rng.random() < cfg["rir_p"]:
            wave = reverb(wave, pick(aug["rir"], rng))
        if rng.random() < cfg["noise_p"]:
            wave = mix_noise(wave, pick(aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
    feat = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
        num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
    return spec_augment(feat, rng) if aug is not None and cfg["spec_aug"] else feat


class WordDataset:
    def __init__(self, root, records, cfg):
        self.root, self.records, self.cfg = root, records, cfg
        self.aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}

    def __len__(self):
        return len(self.records) * 3

    def __getitem__(self, item):
        import soundfile as sf
        i, view = divmod(item, 3)
        r = self.records[i]
        wave, _ = sf.read(self.root / "words" / r["file"], dtype="float32")
        rng = np.random.default_rng(item)
        return item, acoustic_features(wave, self.cfg, rng, self.aug if view else None)


def collate(items):
    return ([x[0] for x in items], nn.utils.rnn.pad_sequence([x[1] for x in items], batch_first=True,
            padding_value=-23.0259), torch.tensor([len(x[1]) for x in items]))


class FrozenMiddle(nn.Module):
    """Read the third stack; stop before the downstream encoder and CTC head."""
    def __init__(self, checkpoint):
        super().__init__()
        from student_train import build_model
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.model = build_model(ck["cfg"]["size"])
        self.model.load_state_dict(ck["model"])
        self.model.eval().requires_grad_(False)
        self.model.streaming_eval(16, 128)
        self.model.encoder.encoders[2].register_forward_hook(self._capture)

    class Captured(Exception):
        pass

    def _capture(self, module, inputs, output):
        self.captured = output
        raise self.Captured

    @torch.no_grad()
    def forward(self, feats, lengths):
        try:
            self.model(feats, lengths)
        except self.Captured:
            pass
        x = self.captured.permute(1, 2, 0).float()
        # Each stack returns at the encoder input rate (50 Hz), including the
        # downsampled stacks, which upsample internally before returning.
        lens = ((lengths - 7) // 2).clamp_min(1)
        assert x.shape[-1] == int(((feats.shape[1] - 7) // 2)), (x.shape, feats.shape)
        return temporal_pool(x, lens)


class WordEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels, 192, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(192, 128, 3, padding=1), nn.GELU())
        self.attention = nn.Conv1d(128, 1, 1)
        self.project = nn.Sequential(nn.Linear(257, 192), nn.GELU(), nn.Linear(192, EMBED_DIM))

    def forward(self, x, duration):
        x = self.conv(x)
        weight = self.attention(x).softmax(-1)
        mean = (x * weight).sum(-1)
        std = ((x - mean[:, :, None]).square() * weight).sum(-1).clamp_min(1e-6).sqrt()
        return F.normalize(self.project(torch.cat([mean, std, duration.clamp_min(.01).log()[:, None]], -1)), dim=-1)


def contrastive_loss(z, labels, source):
    """Positives must be the same word from a different real source clip."""
    logits = z @ z.T / .1
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, -1e4)
    positive = (labels[:, None] == labels[None]) & (source[:, None] != source[None]) & ~diagonal
    valid = positive.any(1)
    if not valid.any():
        return logits.sum() * 0
    logp = logits.log_softmax(-1)
    return -((logp * positive).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()


def extract(args):
    root = Path(args.out)
    records = json.loads((root / "segments.json").read_text())
    frozen = FrozenMiddle(args.checkpoint).cuda()
    ds = WordDataset(root, records, frozen.cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=96, collate_fn=collate, num_workers=6)
    features = None
    for bi, (items, x, lens) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = frozen(x.cuda(), lens.cuda()).cpu().half()
        if features is None:
            features = torch.empty((len(ds), *pooled.shape[1:]), dtype=torch.float16)
        features[items] = pooled
        if bi % 50 == 0:
            print(json.dumps(dict(stage="features", done=max(items) + 1, total=len(ds))), flush=True)
    torch.save(dict(features=features, cfg=frozen.cfg,
                    checkpoint_sha256=hashlib.file_digest(open(args.checkpoint, "rb"), "sha256").hexdigest()), root / "features.pt")


def train(args):
    root = Path(args.out)
    records = json.loads((root / "segments.json").read_text())
    train_sources = {r["source"] for r in records if r["split"] == "train"}
    val_sources = {r["source"] for r in records if r["split"] == "val"}
    if train_sources & val_sources:
        raise ValueError("Calibration recordings overlap training recordings")
    cached = torch.load(root / "features.pt", map_location="cpu", weights_only=False)
    features = cached["features"]
    keys = sorted({r["key"] for r in records if r["split"] == "train"})
    key_ids = {k: i for i, k in enumerate(keys)}
    sources = {k: i for i, k in enumerate(sorted({r["source"] for r in records}))}
    by = {}
    for i, r in enumerate(records):
        if r["split"] == "train":
            by.setdefault(key_ids[r["key"]], []).append(i)
    eligible = [k for k, ids in by.items() if len({records[i]["source"] for i in ids}) >= 2]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = WordEncoder(features.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    durations = torch.tensor([r["samples"] / 16000 for r in records])
    losses = []
    for step in range(args.steps):
        selected = rng.choice(eligible, min(64, len(eligible)), replace=False)
        segments = []
        for k in selected:
            first = int(rng.choice(by[k]))
            other = [i for i in by[k] if records[i]["source"] != records[first]["source"]]
            segments.extend([first, int(rng.choice(other))])
        views = np.array(segments) * 3 + rng.integers(0, 3, len(segments))
        z = model(features[views].float().cuda(), durations[segments].cuda())
        labels = torch.tensor([key_ids[records[i]["key"]] for i in segments], device="cuda")
        source = torch.tensor([sources[records[i]["source"]] for i in segments], device="cuda")
        loss = contrastive_loss(z, labels, source)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss))
        if (step + 1) % 250 == 0:
            print(json.dumps(dict(stage="train", step=step + 1, loss=float(np.mean(losses[-250:])))), flush=True)
    model.eval()
    with torch.no_grad():
        embeddings = torch.cat([model(features[a:a+256].float().cuda(), durations.repeat_interleave(3)[a:a+256].cuda()).cpu()
                                for a in range(0, len(features), 256)])
    bank = F.normalize(torch.stack([embeddings[[i * 3 for i in by[k]]].mean(0) for k in range(len(keys))]), dim=-1)
    distances, correct = [], []
    for i, r in enumerate(records):
        if r["split"] != "val" or r["key"] not in key_ids:
            continue
        for view in range(3):
            d = 1 - embeddings[i * 3 + view] @ bank.T
            distances.append(float(d[key_ids[r["key"]]]))
            correct.append(int(d.argmin()) == key_ids[r["key"]])
    if not distances:
        raise RuntimeError("No held-out same-word calibration segments")
    radius = float(np.quantile(distances, .99, method="higher"))
    # Prefix alternatives come from real training words only. They are never
    # positives or negatives in the contrastive objective.
    prefix_bank, prefix_keys = [], []
    with torch.no_grad():
        for k in range(len(keys)):
            candidates = []
            durations_p = []
            for i in by[k][:4]:
                for fraction in records[i]["prefixes"]:
                    n = max(1, min(BINS - 1, int(round(BINS * fraction))))
                    candidates.append(F.interpolate(features[i * 3:i * 3+1, :, :n].float(), size=BINS, mode="linear", align_corners=False)[0])
                    durations_p.append(float(durations[i]) * fraction)
            if candidates:
                e = model(torch.stack(candidates).cuda(), torch.tensor(durations_p, device="cuda")).cpu()
                prefix_bank.append(e)
                prefix_keys.extend([k] * len(e))
    report = dict(train_segments=sum(r["split"] == "train" for r in records), val_segments=sum(r["split"] == "val" for r in records),
                  train_recordings=len(train_sources), val_recordings=len(val_sources),
                  shared_clusters=len({r["cluster"] for r in records if r["split"] == "train"}
                                      & {r["cluster"] for r in records if r["split"] == "val"}),
                  segments_sha256=hashlib.sha256((root / "segments.json").read_bytes()).hexdigest(),
                  lexicon=len(keys), contrastive_words=len(eligible), calibration_examples=len(distances), radius=radius,
                  heldout_top1=float(np.mean(correct)), steps=args.steps, bins=BINS, loss=losses[-1],
                  checkpoint_sha256=cached["checkpoint_sha256"], augmentation=cached["cfg"])
    torch.save(dict(model=model.cpu().state_dict(), channels=features.shape[1], keys=keys, bank=bank,
                    prefix_bank=torch.cat(prefix_bank), prefix_keys=prefix_keys, report=report), root / "embedding.pt")
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "train", "all"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="/student/s1")
    parser.add_argument("--checkpoint", default="/student/prior-runs/spliced-clean/ckpt.pt")
    parser.add_argument("--reference", default="/app/packages/core/assets/ref-v2.json")
    parser.add_argument("--exclusions", default="/app/docs/benchmarks/gold-per-v1-exclude.txt")
    parser.add_argument("--train-words", type=int, default=60000)
    parser.add_argument("--val-words", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for name, run in [("prepare", prepare), ("extract", extract), ("train", train)]:
        if args.stage in (name, "all"):
            run(args)


if __name__ == "__main__":
    main()
