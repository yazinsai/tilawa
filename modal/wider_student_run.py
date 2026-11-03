"""Distil the 2x-width student and export it int8. Run inside an autolab train sandbox, not via `modal run`.

Same recipe as spliced-clean (clean-match, spliced CTC target, KD weight 2, shipped augmentation, 12 epochs,
40 ms frames) except the width and two things the width forces: no LibriSpeech init (those tensors do not fit;
scratch uses the recipe's 0.045, the 0.015 was only the init arm), and a KL of 1 to the shipped full student on
every clean-match frame so the decoded string stays near today's.
"""

import os

os.environ.setdefault("TILAWA_CLIP_CACHE", "/scratch/listener-wider-student-capacity/clipcache")

RUN = "wide-clean"
OUT = os.environ.get("WIDE_OUT", "/scratch/listener-wider-student-capacity/wide-clean")


def train() -> None:
    import types

    import torch

    import student_train as T

    # student_export binds this name into a Volume mount at import, so the stub stays inside train().
    T.vol = types.SimpleNamespace(commit=lambda: None)
    print("cuda", torch.cuda.get_device_name(0), flush=True)
    T.train.local(dict(
        name=RUN, size="wide", init="", buckets="clean-match", target="spliced", epochs=12,
        base_lr=0.045, kd_weight=2.0, anchor="spliced-clean", anchor_weight=1.0, workers=4,
    ))


def export() -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    import student_export as E
    import student_train as T
    from student import ctc_collapse

    E.onnx_deps()
    model, ck = T.load_student(RUN)
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / "model.onnx")
    states = E.export_onnx(model, path, 16, 128)
    manifest = E.manifest(RUN, states, 16, 128, path, ck)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    # One real clip: the streaming ONNX graph against the PyTorch streaming forward.
    import glob
    import pyarrow.parquet as pq

    raw = pq.read_table(sorted(glob.glob("/student/s1/*.parquet"))[0], columns=["flac"]).column("flac")[0].as_py()
    feats = E.fbank(raw)
    model.eval()
    model.streaming_eval(16, 128)
    with torch.no_grad():
        lp, lens = model(feats.unsqueeze(0), torch.tensor([feats.shape[0]]))
    ref = lp[0, : int(lens[0])].numpy()
    got = E.stream_onnx(path, feats, states, 16)[: ref.shape[0]]
    agree = float((ref.argmax(-1) == got.argmax(-1)).mean())
    edits = 0
    a, b = ctc_collapse(ref.argmax(-1).tolist()), ctc_collapse(got.argmax(-1).tolist())
    # edit_distance lives next to ctc_collapse
    from student import edit_distance
    edits = edit_distance(b, a)
    parity = {"top1_agree": agree, "edits": edits, "ref_len": len(a), "max_abs_prob_diff": float(np.abs(np.exp(ref) - np.exp(got)).max())}
    (out / "parity.json").write_text(json.dumps(parity))
    print("parity", parity, flush=True)
    if agree < 0.99:
        raise RuntimeError(f"ONNX parity {agree:.4f} below 0.99; not quantising")
    subprocess.run([sys.executable, "tools/quantize_onnx.py", str(out), "int8"], check=True)
    print("exported", out.parent / f"{out.name}-int8", flush=True)


if __name__ == "__main__":
    if os.environ.get("EXPORT_ONLY"):
        export()
    else:
        train()
        export()
