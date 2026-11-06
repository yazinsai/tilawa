"""Full-context student ear: real 2–8 s crops, canonical CTC and teacher KD.

Run directly in the supervisor's training image. No lab labels enter training.
The offline export has dynamic time axes and no streaming cache inputs.
"""
import argparse
import io
import json
import os
from pathlib import Path
import time

import numpy as np


def canonical_alignment(ids, probs, target, vocab):
    """Viterbi CTC alignment of canonical phones on the retained teacher rows.

    Return each phone's occupied [first,last+1) frames; never substitute a
    teacher-decoded transcript for the canonical label.
    """
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    dense = np.full((len(ids), vocab), 1e-7, dtype=np.float32)
    np.put_along_axis(dense, ids, probs, axis=1)
    lp = np.log(dense[:, labels])
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    score = np.full(len(labels), -np.inf)
    score[:2] = lp[0, :2]
    back = np.zeros(lp.shape, dtype=np.uint8)
    for t in range(1, len(lp)):
        one = np.r_[-np.inf, score[:-1]]
        two = np.r_[-np.inf, -np.inf, score[:-2]]
        two[~skip] = -np.inf
        choices = np.stack([score, one, two])
        back[t] = choices.argmax(axis=0)
        score = choices.max(axis=0) + lp[t]
    state = len(labels) - 1 if score[-1] >= score[-2] else len(labels) - 2
    if not np.isfinite(score[state]):
        return None
    path = np.zeros(len(lp), dtype=np.int64)
    for t in range(len(lp) - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans = []
    for s in range(1, len(labels), 2):
        frames = np.flatnonzero(path == s)
        if not len(frames):
            return None
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


def crop_target(target, spans, start_frame, end_frame):
    """Only phones whose complete canonical alignment is inside the crop."""
    return [p for p, (a, b) in zip(target, spans) if a >= start_frame and b <= end_frame]


class Crops:
    def __init__(self, clips, alignments, cfg, aug, seed=0):
        self.clips, self.alignments, self.cfg, self.aug = clips, alignments, cfg, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.clips["ref"])

    def __getitem__(self, i):
        import torch
        import soundfile as sf
        import torchaudio
        import student_train as T
        from student import mix_noise

        c, rng = self.clips, self.rng
        wave, _ = sf.read(io.BytesIO(c["flac"][c["flac_off"][i]:c["flac_off"][i+1]].tobytes()), dtype="float32")
        target, spans, pauses = self.alignments[i]
        seconds = min(len(wave) / T.SR, rng.uniform(2, 8))
        size = int(seconds * T.SR)
        limit = len(wave) - size
        # Half the crops start in actual teacher-blank waqf gaps. The others
        # start at arbitrary sample positions, including partial word edges.
        eligible = [a for a in pauses if a * 640 <= limit]
        start = int(rng.choice(eligible)) * 640 if eligible and rng.random() < .5 else int(rng.integers(limit + 1))
        end = start + size
        lo, hi = start / 640, end / 640
        labels = crop_target(target, spans, lo, hi)
        wave = wave[start:end].copy()
        if self.aug:
            if rng.random() < self.cfg["rir_p"]:
                wave = T.reverb(wave, T.pick(self.aug["rir"], rng))
            if rng.random() < self.cfg["noise_p"]:
                wave = mix_noise(wave, T.pick(self.aug["noise"], rng), rng.uniform(*self.cfg["snr_db"]), rng)
            wave *= 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
            num_mel_bins=80, sample_frequency=T.SR, dither=0., energy_floor=1e-10, high_freq=-400)
        if self.cfg["spec_aug"]:
            feats = T.spec_augment(feats, rng)
        off, stop = c["top_off"][i:i+2]
        ids = c["ids"][off:stop].reshape(-1, 5)
        probs = c["probs"][off:stop].reshape(-1, 5)
        # Both heads run at 40 ms. Fractional random crop starts map each
        # output frame to the closest teacher frame on the source clock.
        n = max(1, (len(feats) - 7) // 4)
        at = np.clip(np.rint(lo + np.arange(n)).astype(int), 0, len(ids)-1)
        return dict(i=i, feats=feats, target=torch.tensor(labels, dtype=torch.long),
            top_ids=torch.from_numpy(ids[at].astype(np.int64)), top_p=torch.from_numpy(probs[at].astype(np.float32)))


def prepare(clips):
    from student import phoneme_ids, ctc_collapse, VOCAB
    alignments, keep = [], []
    for i, ref in enumerate(clips["ref"]):
        target = ctc_collapse(phoneme_ids(ref))
        a, b = clips["top_off"][i:i+2]
        ids = clips["ids"][a:b].reshape(-1, 5).astype(int)
        probs = clips["probs"][a:b].reshape(-1, 5).astype(np.float32)
        spans = canonical_alignment(ids, probs, target, VOCAB)
        if spans is None:
            continue
        # Natural silence between canonical phones; no splice or pause relabel.
        pauses = [right[0] for left, right in zip(spans, spans[1:]) if right[0] - left[1] >= 5]
        alignments.append((target, spans, pauses))
        keep.append(i)
    if len(keep) != len(clips["ref"]):
        raise RuntimeError(f"{len(clips['ref'])-len(keep)} impossible canonical alignments; inspect before training")
    return alignments


def build(size):
    import student_train as T
    model = T.build_model(size, causal=False)
    model.encoder.chunk_size = (-1,)
    model.encoder.left_context_frames = (-1,)
    return model


def train(args):
    import torch
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import ScaledAdam
    import student_train as T

    torch.set_num_threads(8)
    torch.manual_seed(0)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    log = open(run / "log.jsonl", "a")
    def emit(**values):
        line = json.dumps(dict(t=time.time(), **values))
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    cfg = dict(T.DEFAULTS, size=args.size, kd_weight=2., base_lr=.0015, workers=0)
    clips = T.load_clips("clean-match", "train", args.hours, 30., 0)
    emit(event="loaded", clips=len(clips["ref"]), hours=float(clips["n_samples"].sum()/16000/3600))
    alignments = prepare(clips)
    emit(event="aligned", clips=len(alignments), pause_starts=sum(len(a[2]) for a in alignments))
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    ds = Crops(clips, alignments, cfg, aug)
    model = build(args.size)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        source, own = ck["model"], model.state_dict()
        compatible = {k:v for k,v in source.items() if k in own and v.shape == own[k].shape}
        result = model.load_state_dict(compatible, strict=False)
        emit(event="init", loaded=len(compatible), missing=result.missing_keys, ignored=[k for k in source if k not in compatible])
    model = model.cuda().train()
    opt = ScaledAdam(get_parameter_groups_with_lrs(model, lr=cfg["base_lr"], include_names=True),
                    lr=cfg["base_lr"], clipping_scale=2.)
    state = dict(step=0)
    path = run / "ckpt.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        emit(event="resume", **state)
    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state,
                        cfg=cfg, causal=False), str(path)+".tmp")
        os.replace(str(path)+".tmp", path)
    rng = np.random.default_rng(0)
    started = time.monotonic()
    while state["step"] < args.steps:
        batch = T.collate([ds[int(i)] for i in rng.integers(len(ds), size=args.batch)])
        for module in model.modules():
            module.batch_count = 20000 + state["step"]
        losses = T.losses(model, batch, cfg, torch.device("cuda"))
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("nonfinite offline loss")
        opt.zero_grad(set_to_none=True); losses["loss"].backward(); opt.step()
        state["step"] += 1
        if state["step"] % 20 == 0:
            emit(event="train", **state, ctc=float(losses["ctc"]), kd=float(losses["kd"]),
                 seconds=time.monotonic()-started)
        if state["step"] % 200 == 0:
            save()
        if time.monotonic() - started > args.minutes * 60:
            save(); return
    save()


def export(run, dest):
    import torch
    import hashlib
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from scaling_converter import convert_scaled_to_non_scaled
    from student import TOKENS
    torch.set_num_threads(8)
    ck = torch.load(Path(run)/"ckpt.pt", map_location="cpu", weights_only=False)
    model = build(ck["cfg"]["size"])
    model.load_state_dict(ck["model"]); model.eval()
    convert_scaled_to_non_scaled(model, inplace=True)
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 401, 80); lens = torch.tensor([401])
    torch.onnx.export(model, (x,lens), str(dest/"fp32.onnx"), input_names=["x","x_lens"],
        output_names=["log_probs","out_lens"], dynamic_axes={"x":{1:"time"},"log_probs":{1:"frames"}},
        opset_version=17, dynamo=False)
    quantize_dynamic(str(dest/"fp32.onnx"), str(dest/"model.onnx"), weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul","Gemm"])
    sessions = [ort.InferenceSession(str(dest/f), providers=["CPUExecutionProvider"]) for f in ("fp32.onnx","model.onnx")]
    parity=[]
    # Exercise different subsampling/padding residues, not only lengths that
    # share the trace input's remainder. Seams may also exceed eight seconds.
    for n in (200,201,202,397,401,515,738,801,914):
        x = torch.randn(1,n,80); lens=torch.tensor([n])
        with torch.no_grad(): lp, lengths = model(x,lens)
        out = [s.run(None, {"x":x.numpy(),"x_lens":lens.numpy()}) for s in sessions]
        np.testing.assert_allclose(lp.numpy(), out[0][0], atol=2e-4, rtol=2e-4)
        assert int(lengths[0]) == int(out[0][1][0]) == int(out[1][1][0])
        parity.append(dict(fbank_frames=n, frames=int(lengths[0]),
            fp32_max_error=float(np.max(np.abs(lp.numpy()-out[0][0]))),
            int8_argmax_agree=float((out[0][0].argmax(-1)==out[1][0].argmax(-1)).mean())))
    raw=(dest/"model.onnx").read_bytes()
    # This is a new export descriptor, never a supplied listener manifest.
    description=dict(arch="zipformer2-offline-ctc", format="onnx-int8", inventory="muaalem-v3_2-phonemes",
        tokens=TOKENS, sampleRate=16000, frameMs=40, file="model.onnx", bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(), trainStep=ck["state"]["step"], parity=parity)
    (dest/"export.json").write_text(json.dumps(description, ensure_ascii=False, indent=2))
    print(json.dumps(description, ensure_ascii=False),flush=True)


def evaluate(run):
    """Held-out real crops; diagnostic PER, never engine wrong-call scoring."""
    import torch
    import student_train as T
    from student import ctc_collapse, edit_distance
    torch.set_num_threads(8)
    ck = torch.load(Path(run)/"ckpt.pt", map_location="cpu", weights_only=False)
    model = build(ck["cfg"]["size"]).cuda().eval()
    model.load_state_dict(ck["model"])
    clips = T.load_clips("clean-match", "test", 0, 30., 1, 512)
    ds = Crops(clips, prepare(clips), dict(ck["cfg"], spec_aug=False), None, seed=1)
    errors = phones = 0
    with torch.no_grad():
        for a in range(0,len(ds),16):
            items = [ds[i] for i in range(a,min(a+16,len(ds)))]
            b = T.collate(items)
            lp, lens = model(b["feats"].cuda(), b["feat_lens"].cuda())
            best = lp.argmax(-1).cpu()
            for j, item in enumerate(items):
                ref = item["target"].tolist()
                hyp = ctc_collapse(best[j,:lens[j]].tolist())
                errors += edit_distance(hyp,ref); phones += len(ref)
    result=dict(clips=len(ds), phone_errors=errors, reference_phones=phones, per=errors/max(1,phones),
                train_step=ck["state"]["step"])
    (Path(run)/"heldout.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=["train","export","evaluate"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--dest")
    parser.add_argument("--init", default="")
    parser.add_argument("--size", default="full")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--hours", type=float, default=30.)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--minutes", type=float, default=50.)
    args=parser.parse_args()
    if args.action == "train": train(args)
    elif args.action == "export": export(args.run,args.dest)
    else: evaluate(args.run)
