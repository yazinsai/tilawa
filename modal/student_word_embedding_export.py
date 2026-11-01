"""Export the frozen middle encoder and learned word head as a standalone ONNX.

The shipped CTC graph is unchanged. The caller supplies one cropped word's
fbank, its valid length, and its audio duration. No lexicon or practice data is
needed for graph export or numerical parity checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from student_word_embedding import FrozenMiddle, WordEncoder


class AcousticWord(nn.Module):
    def __init__(self, checkpoint, artifact):
        super().__init__()
        self.middle = FrozenMiddle(checkpoint)
        self.head = WordEncoder(artifact["channels"])
        self.head.load_state_dict(artifact["model"])

    def forward(self, x, lengths, duration):
        return self.head(self.middle(x, lengths), duration)


def main():
    import onnxruntime as ort
    from icefall.utils import get_onnx_export_kwargs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--checkpoint", default="/student/prior-runs/spliced-clean/ckpt.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    artifact = torch.load(args.embedding, map_location="cpu", weights_only=False)
    model = AcousticWord(args.checkpoint, artifact).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "word-encoder.onnx"
    x = torch.randn(1, 123, 80)
    lengths = torch.tensor([123])
    duration = torch.tensor([1.25])
    with torch.no_grad():
        torch.onnx.export(model, (x, lengths, duration), str(path), opset_version=17,
            input_names=["x", "lengths", "duration"], output_names=["embedding"],
            dynamic_axes={"x": {1: "frames"}}, **get_onnx_export_kwargs())
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    rows = []
    for n in (14, 31, 50, 123, 203, 397):
        x = torch.randn(1, n, 80)
        lengths, duration = torch.tensor([n]), torch.tensor([(n * 160 + 240) / 16000])
        with torch.no_grad():
            expected = model(x, lengths, duration).numpy()
        actual = session.run(None, dict(x=x.numpy(), lengths=lengths.numpy(), duration=duration.numpy()))[0]
        error = float(np.max(np.abs(expected - actual)))
        rows.append(dict(frames=n, max_abs_error=error, cosine_distance=float(1 - (expected * actual).sum())))
        if not np.isfinite(actual).all() or error > 1e-3:
            raise RuntimeError(f"Word encoder ONNX parity failed: {rows[-1]}")
    fingerprint = hashlib.sha256()
    for key, value in sorted(artifact["model"].items()):
        fingerprint.update(key.encode())
        fingerprint.update(value.numpy().tobytes())
    report = dict(onnx_sha256=hashlib.file_digest(open(path, "rb"), "sha256").hexdigest(),
                  head_sha256=fingerprint.hexdigest(), bytes=path.stat().st_size, parity=rows)
    (out / "word-encoder-parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
