"""Frozen ONNX feature extraction for whole-recording weak-label training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NODE = '/out_combiner_1/Add_output_0'


def eligible(row):
    if row.get('split') != 'train' or not row.get('scored') or row.get('excluded') or row.get('exclude'):
        return False
    if len(row.get('ayahs', [])) != 1 or row.get('non_errors') or row.get('unsupported'):
        return False
    events = row.get('events', [])
    return (row.get('kind') == 'none' and not events) or (
        row.get('kind') == 'skip_word' and len(events) == 1 and
        events[0].get('errorClass') == 'omitted_word' and events[0].get('words', 1) == 1)


def feature_model(model_dir, output):
    import onnx
    m = json.loads((model_dir/'manifest.json').read_text())
    graph = onnx.load(model_dir/m['file'])
    if FEATURE_NODE not in {o for n in graph.graph.node for o in n.output}:
        raise ValueError('Frozen encoder tensor unavailable')
    # Expose an existing activation; retain every original output and weight.
    graph.graph.output.append(onnx.helper.make_tensor_value_info(FEATURE_NODE, onnx.TensorProto.FLOAT, None))
    onnx.save(graph, str(output))
    return m


class FrozenEncoder:
    def __init__(self, path, manifest):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
        self.manifest = manifest

    def __call__(self, wave):
        import torch
        import torchaudio
        m = self.manifest
        f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
            num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()
        n = max(0, ((len(f)-7)//2+1)//2)
        if not n:
            raise ValueError('Recording has no acoustic frames')
        f = np.pad(f, ((0,m['window']), (0,0)), constant_values=m['fbank']['padValue'])
        states = {s['name']: np.zeros(s['shape'], np.int64 if s['dtype']=='int64' else np.float32) for s in m['states']}
        names = [o.name for o in self.session.get_outputs()]
        features = []
        for at in range(0,n*4,m['shift']):
            out = dict(zip(names,self.session.run(None,{'x':f[None,at:at+m['window']], **states})))
            states = {s['name']:out['new_'+s['name']] for s in m['states']}
            z, count = out[FEATURE_NODE][:,0], out['log_probs'].shape[1]
            if len(z) % count:
                raise ValueError('Encoder activation cannot be pooled to native output clock')
            z = z.reshape(count,len(z)//count,z.shape[-1]).mean(1)
            features.append(z)
        return np.concatenate(features)[:n]


def collect(args):
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    torch.set_num_threads(1)
    source, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = feature_model(Path(args.model), out/'segment-encoder.onnx')
    encoder = FrozenEncoder(out/'segment-encoder.onnx', m)
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a for a in ref['ayahs']}
    labels = [json.loads(line) for line in (source/'labels.jsonl').read_text().splitlines()]
    # A speaker marked evaluation anywhere is rejected wholesale, including mixed splits.
    excluded = {r['speaker'] for r in labels if r.get('split') != 'train'}
    rows = [r for r in labels if eligible(r) and r['speaker'] not in excluded]
    records = []
    for i, row in enumerate(rows):
        a = ayahs[row['ayahs'][0]]
        words, word_numbers = [], []
        for j, start in enumerate(a['o']):
            end = a['o'][j+1] if j+1<len(a['o']) else len(a['p'])
            if end>start:
                words.append(a['p'][start:end]); word_numbers.append(j+1)
        omitted = None
        if row['events']:
            loc = row['events'][0]['location']
            if f"{loc['surah']}:{loc['ayah']}" != row['ayahs'][0] or loc['word'] not in word_numbers:
                continue
            omitted = word_numbers.index(loc['word'])
        wave, sr = sf.read(source/row['wav'], dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            wave = resample_poly(wave,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000)).astype(np.float32)
        filename = f'features-{i}.npy'
        x = encoder(wave)
        if len(x)<len(words):
            continue
        np.save(out/filename, x)
        records.append(dict(speaker=row['speaker'],recording=row['id'],features=filename,words=words,omitted=omitted,
                            audioSha256=hashlib.sha256((source/row['wav']).read_bytes()).hexdigest()))
        if i%25==0:
            print('features',i,len(rows),x.shape,flush=True)
    meta = dict(examples=records,tokens=ref['inventory']['tokens'],inventory=ref['inventory']['id'],
                featureNode=FEATURE_NODE,encoderSha256=hashlib.sha256((out/'segment-encoder.onnx').read_bytes()).hexdigest(),
                excludedSpeakers=sorted(excluded),source='help-train training speakers, whole-recording labels')
    (out/'examples.json').write_text(json.dumps(meta))
    (out/'primary-manifest.json').write_text(json.dumps(m))
    print('collected',len(records),'none',sum(r['omitted'] is None for r in records),
          'omissions',sum(r['omitted'] is not None for r in records),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    collect(p.parse_args())
