"""Train and select acoustic prototype heads on training-speaker partitions.

Run directly inside the authorized training container. Never reads practice
audio, evidence or labels. The ordinary augmentation implementation is shared
with student_train; both head families see the same batches and random draws.
"""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from student import TOKENS, VOCAB, phoneme_ids, spliced_target
from student_prototypes import ConventionalHead, PrototypeHead, ctc_occupancies
import student_train as T


def partition(speaker):
    return int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 5 == 0


def training_records(data: Path, max_clips: int, seed: int):
    import pyarrow.parquet as pq

    root = Path(__file__).resolve().parents[1]
    gold = json.loads((root / "docs/benchmarks/gold-per-v1.json").read_text())
    # Whole release-held-out speakers, not just the excluded gold clips.
    excluded = {r["speaker"] for r in gold["slices"]["gold"] + gold["slices"]["silver"]}
    candidates = []
    for path in sorted(Path("/student/s1").glob("*.parquet")):
        rows = pq.read_table(path, columns=["shard", "row", "cluster", "bucket", "split", "n_samples"]).to_pylist()
        for i, r in enumerate(rows):
            speaker = f"tlog-cluster-{r['cluster']}"
            if r["split"] == "train" and r["bucket"] == "clean-match" and speaker not in excluded and 1 <= r["n_samples"] / T.SR <= 30:
                candidates.append((path, i, speaker))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    candidates = candidates[:max_clips]
    selected = {}
    for path, i, speaker in candidates:
        selected.setdefault(path, []).append((i, speaker))
    records = []
    for path, entries in selected.items():
        rows = pq.read_table(path).take([i for i, _ in entries]).to_pylist()
        for r, (_, speaker) in zip(rows, entries):
            records.append(dict(key=f"{r['shard']}:{r['row']}", speaker=speaker,
                                target=spliced_target(r["ref"], r["hyp"]), ref=r["ref"], hyp=r["hyp"],
                                n_samples=r["n_samples"], n_frames=r["n_frames"], flac=r["flac"],
                                ids=r["top5_ids"], probs=r["top5_p"], source="original"))
    ref = json.loads((root / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {(a["s"], a["a"]): a["p"] for a in ref["ayahs"]}
    help_root = data / "help-train"
    for line in (help_root / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] != "train" or r["kind"] != "none" or r["speaker"] in excluded:
            continue
        # None-labelled clips only. Never rewrite scripts as performed speech.
        if r["events"] or r.get("non_errors"):
            continue
        import soundfile as sf
        import torchaudio

        wave, sr = sf.read(help_root / r["wav"], dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(-1)
        if sr != T.SR:
            wave = torchaudio.functional.resample(torch.tensor(wave), sr, T.SR).numpy()
        canonical = "".join(ayahs[tuple(map(int, a.split(":")))] for a in r["ayahs"])
        # The shipped target compiler collapses duration expansions.
        from per_breakdown import collapse

        target = collapse(canonical)
        buf = io.BytesIO()
        sf.write(buf, wave, T.SR, format="FLAC", subtype="PCM_16")
        records.append(dict(key=r["id"], speaker=r["speaker"], target=target, ref=canonical, hyp=target,
                            n_samples=len(wave), n_frames=max(1, len(wave) // 640), flac=buf.getvalue(),
                            ids=b"", probs=b"", source="help-none"))
    train = [r for r in records if not partition(r["speaker"])]
    valid = [r for r in records if partition(r["speaker"])]
    if not train or not valid:
        raise ValueError("speaker-disjoint training and selection sets are required")
    assert not ({r["speaker"] for r in train} & {r["speaker"] for r in valid})
    return train, valid


def packed(records):
    def flat(name, dtype):
        chunks = [np.frombuffer(r[name], dtype) for r in records]
        return np.concatenate(chunks), np.r_[0, np.cumsum([len(c) for c in chunks])]
    flac, flac_off = flat("flac", np.uint8)
    ids, top_off = flat("ids", np.uint8)
    probs, _ = flat("probs", np.float16)
    return {**{k: [r[k] for r in records] for k in ("key", "target", "ref", "hyp", "n_samples", "n_frames")},
            "n_samples": np.array([r["n_samples"] for r in records]),
            "flac": flac, "flac_off": flac_off, "ids": ids, "probs": probs, "top_off": top_off}


def encode(model, feats, lens):
    from icefall.utils import make_pad_mask

    x, n = model.encoder_embed(feats, lens)
    x, n = model.encoder(x.permute(1, 0, 2), n, make_pad_mask(n))
    return x.permute(1, 0, 2), n


def tensor_batch(ds, indices, device):
    b = T.collate([ds[i] for i in indices])
    return b, b["feats"].to(device), b["feat_lens"].to(device)


def objective(model, parent, b, feats, lens, anchor):
    lp, n = model(feats, lens)
    lp = lp.float()
    with torch.no_grad():
        old, _ = parent(feats, lens)
    mask = torch.arange(lp.shape[1], device=lp.device)[None] < n[:, None]
    ctc = F.ctc_loss(lp.transpose(0, 1), b["targets"].to(lp.device), n,
                     b["target_lens"].to(lp.device), reduction="sum", zero_infinity=True) / mask.sum()
    kl = (old.exp() * (old - lp)).sum(-1)[mask].mean()
    # Keep the shipped sparse KD objective on original data, where the
    # authorized teacher posteriors are present. None takes use CTC + parent KL.
    kd = lp.new_zeros(())
    present = b["n_teacher"] > 0
    if present.any():
        from student import soft_target_loss

        nt = b["n_teacher"][present].to(lp.device)
        idx = torch.arange(lp.shape[1], device=lp.device)[None].expand(len(nt), -1)
        idx = torch.minimum(idx, (nt - 1)[:, None])
        ids = b["top_ids"][present].to(lp.device).gather(1, idx[..., None].expand(-1, -1, 5))
        probs = b["top_p"][present].to(lp.device).gather(1, idx[..., None].expand(-1, -1, 5))
        kd = soft_target_loss(lp[present.to(lp.device)], ids, probs, mask[present.to(lp.device)])
    return ctc + anchor * kl + kd, dict(ctc=float(ctc.detach()), kl=float(kl.detach()), kd=float(kd.detach()))


def save(path, model, cfg, step):
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, cfg=cfg, state=dict(step=step)), path)


def train(args):
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    def emit(**kw):
        line = json.dumps(dict(t=time.time(), **kw))
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    train_rows, valid_rows = training_records(Path(args.data), args.clips, args.seed)
    (out / "partitions.json").write_text(json.dumps({"train": [{k: r[k] for k in ("key", "speaker", "source")} for r in train_rows],
                                                   "selection": [{k: r[k] for k in ("key", "speaker", "source")} for r in valid_rows]}))
    emit(event="data", train=len(train_rows), selection=len(valid_rows), train_speakers=len({r["speaker"] for r in train_rows}),
         selection_speakers=len({r["speaker"] for r in valid_rows}), help_none=sum(r["source"] == "help-none" for r in train_rows))
    ck = torch.load(args.parent, map_location="cpu", weights_only=False)
    parent = T.build_model(ck["cfg"]["size"])
    parent.load_state_dict(ck["model"])
    parent = parent.cuda().eval()
    parent.streaming_eval()
    for p in parent.parameters():
        p.requires_grad_(False)
    cfg = {**T.DEFAULTS, **ck["cfg"], "augmentation_seed": args.seed, "workers": 0}
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    train_clips, val_clips = packed(train_rows), packed(valid_rows)
    ds = T.ClipDataset(train_clips, cfg, True, aug)
    val_ds = T.ClipDataset(val_clips, {**cfg, "augmentation_seed": args.seed + 100000}, True, aug)
    # Selection uses fixed training-speaker augmentation draws and no test set.
    plain_ds = T.ClipDataset(train_clips, cfg, False, None)
    features, occupancy = [], []
    with torch.no_grad():
        for i in range(min(args.init_clips, len(plain_ds))):
            b, f, l = tensor_batch(plain_ds, [i], "cuda")
            x, n = encode(parent, f, l)
            lp = parent.ctc_output(x).log_softmax(-1)[0, :n[0]]
            try:
                q = ctc_occupancies(lp, b["targets"])
            except ValueError:
                continue
            # Fixed temporal subsampling bounds the bank, independent of phones.
            features.append(x[0, :n[0]:3].cpu())
            occupancy.append(q[::3].cpu())
    x, q = torch.cat(features).cuda(), torch.cat(occupancy).cuda()
    emit(event="occupancies", frames=len(x), support=q.sum(0).tolist())
    best_arms = []
    for count in (2, 4):
        for family in ("prototypes", "conventional"):
            torch.manual_seed(args.seed)
            model = copy.deepcopy(parent)
            linear = copy.deepcopy(model.ctc_output[1])
            dim = linear.in_features
            if family == "prototypes":
                head = PrototypeHead(dim, VOCAB, count).cuda()
                head.initialize(x, q, args.seed)
                head.bias.data.copy_(linear.bias)
            else:
                head = ConventionalHead(linear, count).cuda()
            model.ctc_output[1] = head
            for p in head.parameters():
                p.requires_grad_(True)
            # Last encoder stack only. The lower representation is preserved.
            upper = list(model.encoder.encoders[-1].parameters())
            for p in upper:
                p.requires_grad_(True)
            arm = f"{family}-{count}"
            arm_dir = out / arm
            arm_dir.mkdir(exist_ok=True)
            acfg = {**cfg, "phone_head": family, "prototypes": count, "anchor": args.anchor,
                    "parent_sha256": hashlib.sha256(Path(args.parent).read_bytes()).hexdigest()}
            opt = torch.optim.AdamW([{"params": head.parameters(), "lr": args.lr},
                                    {"params": upper, "lr": args.lr / 10}], weight_decay=1e-4)
            emit(event="arm", arm=arm, head_parameters=sum(p.numel() for p in head.parameters()))
            best, step, epoch = float("inf"), 0, 0
            while step < args.steps:
                cfg["augmentation_seed"] = args.seed + epoch * 1000000
                for indices in T.batches(train_clips["n_samples"], args.batch_s, args.seed + epoch):
                    model.eval()  # fixed dropout and deployed attention/cache geometry
                    b, f, l = tensor_batch(ds, indices, "cuda")
                    loss, metrics = objective(model, parent, b, f, l, args.anchor)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if step < args.head_steps:
                        for p in upper:
                            p.grad = None
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.)
                    opt.step()
                    step += 1
                    if step % 20 == 0:
                        emit(event="train", arm=arm, step=step, loss=float(loss.detach()), **metrics)
                    if step % args.select_every == 0 or step == args.steps:
                        model.eval()
                        scores = []
                        with torch.no_grad():
                            for vi in T.batches(val_clips["n_samples"], args.batch_s, args.seed, shuffle=False):
                                vb, vf, vl = tensor_batch(val_ds, vi, "cuda")
                                score, _ = objective(model, parent, vb, vf, vl, args.anchor)
                                scores.append((float(score), len(vi)))
                        score = sum(s * n for s, n in scores) / sum(n for _, n in scores)
                        emit(event="selection", arm=arm, step=step, objective=score)
                        if score < best:
                            best = score
                            save(arm_dir / "ckpt.pt", model, acfg, step)
                    if step >= args.steps:
                        break
                epoch += 1
            best_arms.append(dict(arm=arm, objective=best, path=str(arm_dir / "ckpt.pt")))
            del model, opt
            torch.cuda.empty_cache()
    # Freeze the architecture and checkpoint before any target replay.
    selected = min((r for r in best_arms if r["arm"].startswith("prototypes")), key=lambda r: r["objective"])
    result = dict(selected=selected, arms=best_arms, selection="speaker-disjoint training-only CTC + teacher KD + ordinary-reading KL")
    (out / "selection.json").write_text(json.dumps(result, indent=2))
    emit(event="complete", **result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/lab/data")
    ap.add_argument("--clips", type=int, default=3000)
    ap.add_argument("--init-clips", type=int, default=300)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--head-steps", type=int, default=150)
    ap.add_argument("--select-every", type=int, default=150)
    ap.add_argument("--batch-s", type=float, default=160.)
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--anchor", type=float, default=10.)
    ap.add_argument("--seed", type=int, default=17)
    train(ap.parse_args())
