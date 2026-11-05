"""Export the paired frozen/conditioned streaming encoders with the usual int8 MatMul quantizer."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from student_export import wrap, manifest, onnx_deps
from student_session import SessionStudent
import student_train as T


class Streaming(nn.Module):
    def __init__(self, model, chunk=16, left=128):
        super().__init__()
        self.model = model
        self.frozen = wrap(model.frozen, chunk, left)
        self.acoustic = wrap(model.acoustic, chunk, left)
        self.n = len(self.acoustic.init_states())
        # ONNX restores the outer module's training flag after tracing. Keep
        # this inference wrapper in eval mode so it cannot re-enable dropout.
        self.eval()

    def forward(self, x, embedding, states):
        _, frozen_states = self.frozen(x, states[self.n:])
        features = self.model.frame_encoder(self.model.mid.permute(1, 0, 2))
        self.model.embedding = embedding
        lp, acoustic_states = self.acoustic(x, states[:self.n])
        return lp, features, acoustic_states + frozen_states

    def init_states(self):
        return self.acoustic.init_states() + self.frozen.init_states()


def load(source, checkpoint, require_complete=False):
    ck = torch.load(source, map_location='cpu', weights_only=False)
    parent = T.build_model(ck['cfg']['size'])
    parent.load_state_dict(ck['model'])
    model = SessionStudent(parent)
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if require_complete and saved['step'] != saved['totalSteps']:
            raise ValueError('Final export requires the complete fixed training run')
        model.load_state_dict(saved['model'])
        ck['sessionStep'] = saved['step']
    return model.eval(), ck


def export(source, checkpoint, out, require_complete=False):
    from icefall.utils import get_onnx_export_kwargs
    from onnxruntime.quantization import quantize_dynamic, QuantType
    torch.set_num_threads(4)
    model, ck = load(source, checkpoint, require_complete)
    print(json.dumps(dict(event='export-checkpoint', sessionStep=ck.get('sessionStep'))), flush=True)
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
    m.update(format='onnx-int8', sessionTrainingSteps=ck.get('sessionStep'),
        sessionConditioning=dict(dimensions=64, frameMs=20, startMs=1000))
    (out / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(json.dumps(dict(event='exported', sha256=m['sha256'], bytes=m['bytes'])), flush=True)
    # Parity uses training-pool clips, never practice. The float graph
    # must preserve both acoustic outputs and the feedback feature stream.
    import io
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import torchaudio
    ort_session = ort.InferenceSession(fp, sess_options=_ort_options(), providers=['CPUExecutionProvider'])
    int8_session = ort.InferenceSession(path, sess_options=_ort_options(), providers=['CPUExecutionProvider'])
    pool_path = Path(checkpoint).parent / 'pool.pt'
    rows = []
    for clip in torch.load(pool_path, weights_only=False)['clips'][:3]:
        wave, _ = sf.read(io.BytesIO(clip['flac']), dtype='float32')
        f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave)[None], **__import__('student_export').FBANK)
        need = ((len(f)-7)//2+1)//2
        windows = max(1, (need+7)//8)
        f = torch.nn.functional.pad(f, (0,0,0,max(0,45+(windows-1)*32-len(f))), value=-23.0259)
        st = stream.init_states()
        ort_st = {n: t.numpy().copy() for n,t in zip(names,st)}
        q_st = {n: t.numpy().copy() for n,t in zip(names,st)}
        sums = np.zeros(64, dtype=np.float64)
        q_sum = sums.copy(); count = 0
        worst_lp = 0.; worst_features = 0.; agree = 0; frames = 0
        with torch.no_grad():
            for i in range(windows):
                embedding = (sums/count if count>=50 else np.zeros(64)).astype(np.float32)[None]
                q_embedding = (q_sum/count if count>=50 else np.zeros(64)).astype(np.float32)[None]
                x = f[None,i*32:i*32+45]
                lp, features, st = stream(x,torch.from_numpy(embedding),st)
                got = dict(zip([v.name for v in ort_session.get_outputs()], ort_session.run(None, dict(x=x.numpy(), session_embedding=embedding, **ort_st))))
                quant = dict(zip([v.name for v in int8_session.get_outputs()], int8_session.run(None, dict(x=x.numpy(), session_embedding=q_embedding, **q_st))))
                ort_st = {n: got['new_'+n] for n in names}
                q_st = {n: quant['new_'+n] for n in names}
                keep = min(8, need-i*8)
                worst_lp = max(worst_lp,float(np.abs(lp.numpy()-got['log_probs']).max()))
                worst_features = max(worst_features,float(np.abs(features.numpy()-got['session_features']).max()))
                agree += int((got['log_probs'][0,:keep].argmax(-1)==quant['log_probs'][0,:keep].argmax(-1)).sum())
                frames += keep
                sums += got['session_features'][0,:keep*2].sum(0,dtype=np.float64)
                q_sum += quant['session_features'][0,:keep*2].sum(0,dtype=np.float64)
                count += keep*2
        rows.append(dict(key=f"{clip['shard']}:{clip['row']}", floatLogProbMaxDiff=worst_lp,
            floatFeatureMaxDiff=worst_features, int8Top1Agreement=agree/frames, frames=frames))
    (out/'parity.json').write_text(json.dumps(rows,indent=1))
    print(json.dumps(dict(event='parity',clips=rows)),flush=True)
    if any(r['floatLogProbMaxDiff']>0.003 or r['floatFeatureMaxDiff']>0.003 for r in rows):
        raise ValueError('Float streaming export parity failed')


def _ort_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return options


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/app/tools')
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()
    onnx_deps()
    export(a.source, a.checkpoint, a.out, a.require_complete)
