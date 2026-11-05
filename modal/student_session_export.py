"""Export the paired frozen/conditioned streaming encoders with the usual int8 MatMul quantizer."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from student_export import wrap, PAD, manifest, onnx_deps
from student_session import SessionStudent
import student_train as T


class Streaming(nn.Module):
    def __init__(self, model, chunk=16, left=128):
        super().__init__()
        self.model = model
        self.frozen = wrap(model.frozen, chunk, left)
        self.acoustic = wrap(model.acoustic, chunk, left)
        self.n = len(self.acoustic.init_states())

    def forward(self, x, embedding, states):
        _, frozen_states = self.frozen(x, states[self.n:])
        features = self.model.frame_encoder(self.model.mid.permute(1, 0, 2))
        self.model.embedding = embedding
        lp, acoustic_states = self.acoustic(x, states[:self.n])
        return lp, features, acoustic_states + frozen_states

    def init_states(self):
        return self.acoustic.init_states() + self.frozen.init_states()


def load(source, checkpoint):
    ck = torch.load(source, map_location='cpu', weights_only=False)
    parent = T.build_model(ck['cfg']['size'])
    parent.load_state_dict(ck['model'])
    model = SessionStudent(parent)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    return model.eval(), ck


def export(source, checkpoint, out):
    from icefall.utils import get_onnx_export_kwargs
    from onnxruntime.quantization import quantize_dynamic, QuantType
    torch.set_num_threads(4)
    model, ck = load(source, checkpoint)
    stream = Streaming(model)
    states = stream.init_states()
    kinds = ['cached_key', 'cached_nonlin_attn', 'cached_val1', 'cached_val2', 'cached_conv1', 'cached_conv2']
    names = [f'{kinds[i % 6]}_{i // 6}' for i in range(stream.n - 2)] + ['embed_states', 'processed_lens']
    names += ['frozen_' + n for n in names]
    specs = [dict(name=n, shape=list(t.shape), dtype='int64' if t.dtype == torch.int64 else 'float32') for n,t in zip(names, states)]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fp = str(out / 'float.onnx')
    cls = stream.acoustic.encoder.__class__
    plain = cls.forward
    try:
        cls.forward = cls.streaming_forward
        with torch.no_grad():
            torch.onnx.export(stream, (torch.rand(1, 45, 80), torch.zeros(1, 64), states), fp, opset_version=17,
                input_names=['x', 'session_embedding', *names],
                output_names=['log_probs', 'session_features', *['new_' + n for n in names]], **get_onnx_export_kwargs())
    finally:
        cls.forward = plain
    path = str(out / 'model.onnx')
    quantize_dynamic(fp, path, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
    m = manifest('session-conditioned-int8', specs, 16, 128, path, ck)
    m.update(format='onnx-int8', sessionConditioning=dict(dimensions=64, frameMs=20, startMs=1000))
    (out / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(json.dumps(dict(event='exported', sha256=m['sha256'], bytes=m['bytes'])), flush=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/app/tools')
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    onnx_deps()
    export(a.source, a.checkpoint, a.out)
