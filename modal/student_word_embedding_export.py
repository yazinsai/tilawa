"""Export the frozen middle encoder and learned word head as a standalone ONNX.

The shipped CTC graph is unchanged. The caller supplies one cropped word's
fbank, its valid length, and its audio duration. No lexicon or practice data is
needed for graph export or numerical parity checks.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from student_word_embedding import FrozenMiddle, WordEncoder, temporal_pool


class AcousticWord(nn.Module):
    def __init__(self, checkpoint, artifact):
        super().__init__()
        self.middle = FrozenMiddle(checkpoint)
        self.head = WordEncoder(artifact["channels"])
        self.head.load_state_dict(artifact["model"])

    def forward(self, x, lengths, duration):
        return self.head(self.middle(x, lengths), duration)


class PadDownsample(nn.Module):
    """Make icefall's eager edge padding explicit when tracing variable lengths.

    Its streaming export assumes a whole encoder chunk, divisible by every
    downsampling factor. A standalone word can end between those boundaries.
    Repeating the last frame matches the original eager downsampler exactly.
    """
    def __init__(self, inner, factor):
        super().__init__()
        self.inner, self.factor = inner, factor

    def forward(self, x):
        n = x.shape[0]
        end = ((n + self.factor - 1) // self.factor) * self.factor
        indices = torch.arange(end, device=x.device).clamp(max=n - 1)
        return self.inner(x.index_select(0, indices))


class DynamicChunkConv(nn.Module):
    """The original chunk convolution with tensor-shaped short-chunk edges."""
    def __init__(self, inner, chunk_size):
        super().__init__()
        self.inner = inner
        self.chunk_size = chunk_size

    def forward(self, x, chunk_size=-1):
        # ConvolutionModule omits this argument during tracing because its
        # usual exporter uses stateful streaming. Preserve cold-start chunks.
        if chunk_size < 0:
            chunk_size = self.chunk_size
        batch, channels, n = x.shape
        size = torch._shape_as_tensor(x)[2]
        if chunk_size > 0:
            size = size.clamp(max=chunk_size)
        pad = self.inner.kernel_size // 2
        causal = self.inner.causal_conv(torch.nn.functional.pad(x, (pad, 0)))
        chunks = torch.nn.functional.pad(x, (0, -n % size))
        count = chunks.shape[2] // size
        chunks = chunks.reshape(batch, channels, count, size).permute(0, 2, 1, 3).reshape(batch * count, channels, size)
        chunks = self.inner.chunkwise_conv(chunks)
        positions = torch.arange(size, device=x.device)
        kernel = self.inner.kernel_size
        left = self.inner.chunkwise_conv_scale[0].index_select(1, positions.clamp(max=kernel - 1)) * (positions < kernel)
        right_indices = positions + kernel - size
        right = self.inner.chunkwise_conv_scale[1].index_select(1, right_indices.clamp(min=0, max=kernel - 1)) * (right_indices >= 0)
        chunks = chunks * (1 + left + right)
        chunks = chunks.reshape(batch, count, channels, size).permute(0, 2, 1, 3).reshape(batch, channels, count * size)
        return chunks[..., :n] + causal


class ExportMiddle(nn.Module):
    """Explicit first-three-stack forward retaining the streaming attention mask.

    Icefall's default ONNX branch removes this mask for its stateful exporter.
    Word crops use the ordinary cold-start streaming-simulation forward instead.
    """
    def __init__(self, middle):
        super().__init__()
        self.embed = middle.model.encoder_embed
        encoder = middle.model.encoder
        self.stacks = nn.ModuleList(list(encoder.encoders[:3]))
        self.dimensions = encoder.encoder_dim[:3]
        self.factors = encoder.downsampling_factor[:3]
        for stack, factor in zip(self.stacks, self.factors):
            stack._forward_hooks.clear()
            if factor > 1:
                stack.downsample = PadDownsample(stack.downsample, factor)
        def replace(module, chunk_size):
            for name, child in list(module.named_children()):
                if child.__class__.__name__ == "ChunkCausalDepthwiseConv1d":
                    setattr(module, name, DynamicChunkConv(child, chunk_size))
                else:
                    replace(child, chunk_size)
        for stack, factor in zip(self.stacks, self.factors):
            replace(stack, 16 // factor)

    def forward(self, feats, lengths):
        from zipformer import convert_num_channels
        x, lens = self.embed(feats, lengths)
        x = x.permute(1, 0, 2)
        positions = torch.arange(x.shape[0], device=x.device)
        chunks = positions // 16
        delta = chunks[:, None] - chunks[None, :]
        attention_mask = (delta < 0) | (delta > 8)
        padding_mask = positions[None, :] >= lens[:, None]
        for stack, dim, factor in zip(self.stacks, self.dimensions, self.factors):
            x = convert_num_channels(x, dim)
            x = stack(x, chunk_size=16, feature_mask=1.0, attn_mask=attention_mask,
                      src_key_padding_mask=padding_mask[..., ::factor])
        return temporal_pool(x.permute(1, 2, 0).float(), lens)


def bundle(artifact, encoder, listener, out):
    """Package an optional sidecar alongside a byte-identical shipped CTC graph."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((listener / "manifest.json").read_text())
    shutil.copyfile(listener / manifest["file"], out / manifest["file"])
    shutil.copyfile(encoder, out / "word-encoder.onnx")
    artifact["bank"].numpy().astype("<f4").tofile(out / "word-prototypes.f32")
    artifact["prefix_bank"].numpy().astype("<f4").tofile(out / "word-prefixes.f32")
    np.asarray(artifact["prefix_keys"], dtype="<u4").tofile(out / "word-prefix-keys.u32")
    def spec(name):
        path = out / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return dict(file=name, sha256=digest, bytes=path.stat().st_size)
    if spec(manifest["file"])["sha256"] != manifest["sha256"]:
        raise RuntimeError("Base CTC graph hash differs from its manifest")
    asset = dict(schema=1, sampleRate=16000, dimensions=artifact["bank"].shape[1], keys=artifact["keys"],
        radius=artifact["report"]["radius"], encoder=spec("word-encoder.onnx"),
        words=spec("word-prototypes.f32"), prefixes=spec("word-prefixes.f32"), prefixKeys=spec("word-prefix-keys.u32"))
    (out / "word-embedding.json").write_text(json.dumps(asset, ensure_ascii=False))
    manifest["wordEmbedding"] = spec("word-embedding.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(bundle=str(out), bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))), flush=True)


def main():
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--checkpoint", default="/student/prior-runs/spliced-clean/ckpt.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--listener", help="Optional shipped listener directory to package without changing its CTC graph")
    parser.add_argument("--bundle", help="Destination model directory, with --listener")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    artifact = torch.load(args.embedding, map_location="cpu", weights_only=False)
    with open(args.checkpoint, "rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha256 != artifact["report"]["checkpoint_sha256"]:
        raise ValueError("Export checkpoint differs from the calibrated frozen encoder")
    model = AcousticWord(args.checkpoint, artifact).eval()
    # build_model installs the vendored icefall paths used in the training image.
    from icefall.utils import get_onnx_export_kwargs
    cases = []
    for n in (14, 31, 50, 123, 203, 397):
        x = torch.randn(1, n, 80)
        lengths, duration = torch.tensor([n]), torch.tensor([(n * 160 + 240) / 16000])
        with torch.no_grad():
            expected = model(x, lengths, duration).numpy()
        cases.append((n, x, lengths, duration, expected))
    model.middle = ExportMiddle(model.middle)
    for n, x, lengths, duration, expected in cases:
        with torch.no_grad():
            actual = model(x, lengths, duration).numpy()
        error = float(np.max(np.abs(expected - actual)))
        if not np.isfinite(actual).all() or error > 1e-3:
            raise RuntimeError(f"Export wrapper parity failed at {n} frames: {error}")
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
    for n, x, lengths, duration, expected in cases:
        actual = session.run(None, dict(x=x.numpy(), lengths=lengths.numpy(), duration=duration.numpy()))[0]
        error = float(np.max(np.abs(expected - actual)))
        rows.append(dict(frames=n, max_abs_error=error, cosine_distance=float(1 - (expected * actual).sum())))
    print(json.dumps(dict(parity=rows)), flush=True)
    if any(not np.isfinite(row["max_abs_error"]) or row["max_abs_error"] > 1e-3 for row in rows):
        raise RuntimeError("Word encoder ONNX parity failed")
    fingerprint = hashlib.sha256()
    for key, value in sorted(artifact["model"].items()):
        fingerprint.update(key.encode())
        fingerprint.update(value.numpy().tobytes())
    report = dict(onnx_sha256=hashlib.file_digest(open(path, "rb"), "sha256").hexdigest(),
                  head_sha256=fingerprint.hexdigest(), bytes=path.stat().st_size, parity=rows)
    (out / "word-encoder-parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if args.bundle:
        if not args.listener:
            parser.error("--bundle requires --listener")
        bundle(artifact, path, Path(args.listener), Path(args.bundle))


if __name__ == "__main__":
    main()
