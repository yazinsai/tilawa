"""Streaming ONNX export of a student run (ticket 09), with a PyTorch parity check.

    modal run modal/student_export.py --run spliced-clean            # export + parity on help eval clips
    modal run modal/student_export.py::fixture                       # tiny random model for SDK CI tests

Writes `runs/<run>/onnx/{model.onnx,manifest.json,parity.json}` on `tilawa-student`, and copies them to
`.scratch/models/<run>/` locally. The graph is icefall's streaming CTC Zipformer2 (`export-onnx-streaming-ctc.py`):
one call takes `T = 2 * chunk + 13` fbank frames, advances by `2 * chunk`, and emits `chunk / 2` CTC frames at 25 Hz.
"""

import json
from pathlib import Path

import modal
from student_train import SR, image, load_student, vol

app = modal.App("tilawa-student-export", image=image.add_local_python_source("student_train", "student_specialists"))


def onnx_deps():
    """The shared training image ends in local files, so it can't take another build step."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx==1.19.0", "onnxruntime==1.22.1"], check=True)
PAD = 7 + 2 * 3
LOG_EPS = -23.0259
FBANK = dict(num_mel_bins=80, sample_frequency=SR, dither=0.0, energy_floor=1e-10, high_freq=-400)


def fbank(data: bytes):
    import subprocess

    import numpy as np
    import torch
    import torchaudio

    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                         input=data, capture_output=True, check=True).stdout
    w = torch.from_numpy(np.frombuffer(pcm, dtype=np.float32).copy())
    return torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), **FBANK)


def wrap(model, chunk: int, left: int):
    import torch
    import torch.nn as nn

    model.eval()
    model.streaming_eval(chunk, left)

    class Streaming(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.encoder_embed, self.ctc_output = model.encoder, model.encoder_embed, model.ctc_output

        def forward(self, x, states):
            T = chunk * 2 + PAD
            x_lens = torch.tensor([T] * x.size(0))
            x, x_lens, new_pad = self.encoder_embed.streaming_forward(x=x, x_lens=x_lens, cached_left_pad=states[-2])
            processed = states[-1]
            mask = (processed.unsqueeze(1) <= torch.arange(left).expand(x.size(0), left)).to(torch.int32).flip(1)
            mask = torch.cat([mask, torch.zeros(x.size(0), chunk, dtype=torch.int32)], dim=1)
            out, _, new_states = self.encoder.streaming_forward(x=x.permute(1, 0, 2), x_lens=x_lens, states=states[:-2],
                                                                src_key_padding_mask=mask)
            return self.ctc_output(out.permute(1, 0, 2)).log_softmax(-1), new_states + [new_pad, processed + x_lens]

        def init_states(self):
            return self.encoder.get_init_states(1, torch.device("cpu")) + [self.encoder_embed.get_init_states(1, torch.device("cpu")),
                                                                          torch.zeros(1, dtype=torch.int64)]

    return Streaming()


def export_onnx(model, path: str, chunk: int, left: int) -> list[dict]:
    import torch
    from icefall.utils import get_onnx_export_kwargs

    s = wrap(model, chunk, left)
    cls = s.encoder.__class__
    plain, cls.forward = cls.forward, cls.streaming_forward
    init = s.init_states()
    kinds = ["cached_key", "cached_nonlin_attn", "cached_val1", "cached_val2", "cached_conv1", "cached_conv2"]
    names = [f"{kinds[i % 6]}_{i // 6}" for i in range(len(init) - 2)] + ["embed_states", "processed_lens"]
    x = torch.rand(1, chunk * 2 + PAD, 80)
    with torch.no_grad():
        torch.onnx.export(s, (x, init), path, opset_version=17, input_names=["x", *names],
                          output_names=["log_probs", *[f"new_{n}" for n in names]], **get_onnx_export_kwargs())
    cls.forward = plain
    return [{"name": n, "shape": list(t.shape), "dtype": "int64" if t.dtype == torch.int64 else "float32"} for n, t in zip(names, init)]


def stream_onnx(path: str, feats, states: list[dict], chunk: int):
    """Streams fbank frames through the ONNX graph exactly as the SDK does; returns (frames, vocab) log-probs."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    st = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in states}
    T, shift = chunk * 2 + PAD, chunk * 2
    f = feats.numpy()
    n = f.shape[0]
    need = ((n - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (chunk // 2)))
    total = T + (windows - 1) * shift
    f = np.concatenate([f, np.full((max(0, total - n), 80), LOG_EPS, dtype=np.float32)])
    outs = []
    for a in range(0, total - T + 1, shift):
        res = sess.run(None, {"x": f[None, a : a + T], **st})
        outs.append(res[0][0])
        st = {s["name"]: r for s, r in zip(states, res[1:])}
    return np.concatenate(outs)


def manifest(run: str, states: list[dict], chunk: int, left: int, path: str, ck: dict | None) -> dict:
    import hashlib

    from student import TOKENS

    data = Path(path).read_bytes()
    return {
        "schema": 1, "id": run, "format": "onnx-fp32", "arch": "zipformer2-streaming-ctc",
        "inventory": "muaalem-v3_2-phonemes", "tokens": TOKENS, "blank": 0,
        "sampleRate": SR, "frameMs": 40, "fbank": {"mels": 80, "frameShiftMs": 10, "frameLengthMs": 25, "highFreq": -400,
                                                    "snipEdges": True, "padValue": LOG_EPS},
        "window": chunk * 2 + PAD, "shift": chunk * 2, "chunk": chunk, "leftContext": left, "states": states,
        "file": "model.onnx", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "trainStep": ck["state"]["step"] if ck else None,
        # v1 checker classes this evidence is scored on; every pronunciation class ships `unsupported` (Decision 4).
        "supported": ["wrong_word", "omitted_word", "added_word", "skipped_passage"],
    }


@app.function(cpu=8, memory=32768, volumes={"/student": vol}, timeout=60 * 60)
def export(run: str, clips: list[tuple[str, bytes]], chunk: int = 16, left: int = 128) -> dict:
    import os

    onnx_deps()
    import numpy as np
    import torch
    from student import ctc_collapse, edit_distance, ids_phonemes

    torch.set_num_threads(8)
    model, ck = load_student(run)
    out = f"/student/runs/{run}/onnx"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/model.onnx"
    states = export_onnx(model, path, chunk, left)
    model, _ = load_student(run)
    model.eval()
    model.streaming_eval(chunk, left)
    rows = []
    for name, data in clips:
        f = fbank(data)
        with torch.no_grad():
            lp, lens = model(f.unsqueeze(0), torch.tensor([f.shape[0]]))
        ref = lp[0, : lens[0]].numpy()
        got = stream_onnx(path, f, states, chunk)[: ref.shape[0]]
        a, b = ctc_collapse(ref.argmax(-1).tolist()), ctc_collapse(got.argmax(-1).tolist())
        rows.append(dict(clip=name, frames=int(ref.shape[0]), onnx_frames=int(got.shape[0]),
                         top1_agree=float((ref.argmax(-1) == got.argmax(-1)).mean()),
                         max_abs_prob_diff=float(np.abs(np.exp(ref) - np.exp(got)).max()),
                         edits=edit_distance(b, a), ref_len=len(a), torch=ids_phonemes(a), onnx=ids_phonemes(b)))
    parity = {"run": run, "chunk": chunk, "left": left, "clips": rows,
              "per_onnx_vs_torch": sum(r["edits"] for r in rows) / max(1, sum(r["ref_len"] for r in rows))}
    m = manifest(run, states, chunk, left, path, ck)
    Path(f"{out}/manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    Path(f"{out}/parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=1))
    vol.commit()
    return {"manifest": m, "parity": parity}


@app.function(cpu=4, memory=8192, timeout=30 * 60)
def tiny_fixture(wav: bytes, chunk: int = 16, left: int = 64) -> dict:
    """A random-weight Zipformer2 with the student's streaming interface, small enough to commit, plus reference
    outputs, so the SDK's fbank and ONNX streaming loop are tested in CI. `logProbs` come from the Python streaming loop
    over the same graph: random weights make the chunk-masked PyTorch forward a poor oracle (it diverges from its own
    streaming graph), and `export` already checks ONNX against PyTorch on the real model."""
    onnx_deps()
    import student_train
    import torch

    torch.manual_seed(0)
    student_train.SIZES["tiny"] = dict(num_encoder_layers="1,1,1,1,1,1", feedforward_dim="32,32,32,32,32,32",
                                       encoder_dim="16,16,16,16,16,16", encoder_unmasked_dim="16,16,16,16,16,16")
    student_train.COMMON.update(num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4", pos_head_dim="2", pos_dim=8,
                                cnn_module_kernel="3,3,3,3,3,3")
    model = student_train.build_model("tiny")
    model.eval()
    path = "/tmp/model.onnx"
    states = export_onnx(model, path, chunk, left)
    model.streaming_eval(chunk, left)
    f = fbank(wav)
    with torch.no_grad():
        lp, lens = model(f.unsqueeze(0), torch.tensor([f.shape[0]]))
    m = manifest("tiny-random", states, chunk, left, path, None)
    n = int(lens[0])
    streamed = stream_onnx(path, f, states, chunk)[:n]
    return {"onnx": Path(path).read_bytes(), "manifest": m, "fbank": f[:40].flatten().tolist(), "fbankFrames": int(f.shape[0]),
            "logProbs": streamed.flatten().tolist(), "torchLogProbs": lp[0, :n].flatten().tolist(), "frames": n}


@app.local_entrypoint()
def main(run: str = "spliced-clean", n: int = 6, main_checkout: str = "../tilawa", dest: str = ".scratch/models"):
    import subprocess

    help_dir = Path(main_checkout).resolve() / ".scratch/help"
    labels = [json.loads(s) for s in (help_dir / "eval-labels.jsonl").read_text().splitlines()]
    picks = [l for l in labels if not l["exclude"]][:: max(1, len(labels) // n)][:n]
    res = export.remote(run, [(l["id"], (help_dir / l["wav"]).read_bytes()) for l in picks])
    print(json.dumps(res["parity"], ensure_ascii=False, indent=1)[:4000])
    local = Path(main_checkout).resolve() / dest / run
    local.mkdir(parents=True, exist_ok=True)
    for f in ["model.onnx", "manifest.json", "parity.json"]:
        subprocess.run(["modal", "volume", "get", "--force", "tilawa-student", f"runs/{run}/onnx/{f}", str(local / f)], check=True)
    print(local)


@app.local_entrypoint()
def fixture(wav: str = "packages/sdk/test/fixtures/tlog-clean-00010-630.wav", dest: str = "packages/sdk/test/fixtures/tiny"):
    res = tiny_fixture.remote(Path(wav).read_bytes())
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(res.pop("onnx"))
    (d / "manifest.json").write_text(json.dumps(res.pop("manifest"), ensure_ascii=False, indent=1))
    (d / "reference.json").write_text(json.dumps(res))
    print(d, res["frames"], res["fbankFrames"])
