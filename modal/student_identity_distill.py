"""Matched ordinary and conditional identity distillation on a frozen streaming encoder.

Only authorized training speakers enter prepare/fit. No practice data is read.
Export adds one identity output; original logits and every cache stay unchanged.
The SDK distributes the original speech probability over that conditional head.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RECIPE = dict(seed=831, original_limit=256, views=3, epochs=12, batch=8,
              lr=.0001, anchor=.1, max_seconds=30.)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False))
    temp.replace(path)

def reserved_speakers(rows):
    speakers = sorted({r['speaker'] for r in rows}, key=stable)
    if len(speakers) < 2:
        raise ValueError('At least two training speakers required per source')
    return set(speakers[:max(1, math.ceil(len(speakers)/5))])

def authorized_records():
    """Keep source provenance; remove held-out/gold speaker clusters in full."""
    import pyarrow.parquet as pq
    import soundfile as sf

    files = sorted(Path('/student/s1').glob('*.parquet'))
    if not files:
        raise FileNotFoundError('Authorized original training recordings are unavailable')
    excluded = {tuple(line.rsplit(':', 1)) for line in
                (ROOT/'docs/benchmarks/gold-per-v1-exclude.txt').read_text().split()}
    meta = []
    for path in files:
        cols = ['shard', 'row', 'cluster', 'bucket', 'split', 'n_samples']
        if not set(cols).issubset(pq.read_schema(path).names):
            raise ValueError('Original recordings lack training-speaker provenance')
        for index, r in enumerate(pq.read_table(path, columns=cols).to_pylist()):
            meta.append(dict(r, file=str(path), index=index, speaker=f"original/{r['cluster']}",
                             key=f"original/{r['shard']}:{r['row']}", source='original'))
    banned = {r['speaker'] for r in meta if r['split'] != 'train' or
              (r['shard'], str(r['row'])) in excluded}
    ordinary = [r for r in meta if r['split'] == 'train' and r['speaker'] not in banned
                and r['cluster'] is not None and str(r['cluster']) != '-1'
                and r['bucket'] == 'clean-match' and 0 < r['n_samples'] <= RECIPE['max_seconds']*16000]
    reserved_original = reserved_speakers(ordinary)
    # Sampling is frozen before any acoustic comparisons, with no duplication
    # enrichment. Both arms see exactly this same finite training population.
    ordinary = sorted(ordinary, key=lambda r: stable(r['key']))[:RECIPE['original_limit']]

    help_root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (help_root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned_help = {str(l['speaker']) for l in labels if l.get('split') != 'train' or
                   l.get('exclude') or l.get('excluded')}
    help_rows = []
    for label in labels:
        if label.get('split') != 'train' or str(label['speaker']) in banned_help or label.get('kind') != 'none':
            continue
        if label.get('scored') is False or any(label.get(k) for k in ('events', 'non_errors', 'unsupported')):
            continue
        path = (help_root/label['wav']).resolve()
        if not path.is_relative_to(help_root.resolve()) or not path.is_file():
            raise FileNotFoundError('Authorized help waveform is missing or escapes its source')
        if not 0 < sf.info(path).duration <= RECIPE['max_seconds']:
            continue
        help_rows.append(dict(key=f"help/{label['id']}", source='help', file=str(path),
                              speaker=f"help/{label['speaker']}", ayahs=label['ayahs']))
    reserved = reserved_original | reserved_speakers(help_rows)
    rows = ordinary + help_rows
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Authorized recording keys are not unique')
    for r in rows:
        r['partition'] = 'validation' if r['speaker'] in reserved else 'train'
    for source in ('original', 'help'):
        for partition in ('train', 'validation'):
            if not any(r['source'] == source and r['partition'] == partition for r in rows):
                raise ValueError('Required training-speaker partition has no clean recordings')
    return rows, sorted(banned | {f'help/{s}' for s in banned_help}), digest(help_root/'labels.jsonl')

def training_helpers():
    """Reuse the shipped augmentation functions without initializing a Modal app."""
    tree = ast.parse((ROOT/'modal/student_train.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                n.name in ('pick', 'reverb', 'spec_augment')]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'student_train.py', 'exec'), namespace)
    return namespace

def linear_head(graph, vocab):
    """Identify the shipped classifier, requiring an unambiguous known layout."""
    import numpy as np
    from onnx import numpy_helper

    nodes = [n for n in graph.graph.node if 'ctc_output' in n.name and n.op_type in ('MatMul', 'MatMulInteger')]
    if len(nodes) != 1:
        raise ValueError('Cannot identify exactly one shipped linear phone head')
    node = nodes[0]
    initial = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    weight_name = node.input[1]
    if weight_name not in initial:
        raise ValueError('Classifier weight is not a constant tensor')
    weight = initial[weight_name]
    quantized = node.op_type == 'MatMulInteger'
    scale_name = None
    zero_name = None
    if quantized:
        base = weight_name.removesuffix('_quantized')
        scale_name, zero_name = base+'_scale', base+'_zero_point'
        if scale_name not in initial or zero_name not in initial or node.input[3] != zero_name:
            raise ValueError('Unrecognized classifier quantization layout')
        weight = (weight.astype(np.float32)-initial[zero_name])*initial[scale_name]
    biases = [(k, v) for k, v in initial.items() if 'ctc_output' in k and k.endswith('bias')]
    if weight.ndim != 2 or weight.shape[1] != vocab or len(biases) != 1 or biases[0][1].shape != (vocab,):
        raise ValueError('Classifier geometry differs from the phone inventory')
    feature = node.input[0]
    if quantized:
        producer = next((n for n in graph.graph.node if feature in n.output), None)
        if producer is None or producer.op_type != 'DynamicQuantizeLinear':
            raise ValueError('Unrecognized classifier activation quantization')
        feature = producer.input[0]
    return dict(weight=weight, bias=biases[0][1], weight_name=weight_name,
                bias_name=biases[0][0], scale_name=scale_name, zero_name=zero_name,
                quantized=quantized, feature=feature,
                quantized_input=node.input[0] if quantized else None,
                activation_scale=producer.output[1] if quantized else None,
                activation_zero=node.input[2] if quantized else None)

def feature_runtime(model, out):
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    manifest = json.loads((model/'manifest.json').read_text())
    path = model/manifest['file']
    if digest(path) != manifest['sha256']:
        raise ValueError('Shipped primary checksum mismatch')
    graph = onnx.load(str(path))
    head = linear_head(graph, len(manifest['tokens'])+1)
    if head['quantized']:
        # Cache the actual frozen activations consumed by the integer head, not
        # the unrounded pre-quantizer tensor. This isolates weight refitting from
        # an additional change to activation quantization in the two controls.
        name = 'phone_identity_features'
        if any(name in n.output for n in graph.graph.node):
            raise ValueError('Feature debug output already exists')
        graph.graph.node.append(helper.make_node('DequantizeLinear',
            [head['quantized_input'], head['activation_scale'], head['activation_zero']], [name], name=name))
        head['feature'] = name
    graph.graph.output.append(helper.make_tensor_value_info(head['feature'], TensorProto.FLOAT,
                                                          [1, None, head['weight'].shape[0]]))
    onnx.checker.check_model(graph)
    feature_path = out/'encoder-features.onnx'
    onnx.save(graph, str(feature_path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    original = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    extractor = ort.InferenceSession(str(feature_path), opts, providers=['CPUExecutionProvider'])
    st = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
          for s in manifest['states']}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(RECIPE['seed'])
    for _ in range(8):
        feed = dict(st, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
        a, b = original.run(names, feed), extractor.run(names, feed)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg='Feature capture changes '+name)
        st = {s['name']: b[names.index('new_'+s['name'])] for s in manifest['states']}
    return manifest, head, extractor

def stream_features(runtime, manifest, feats, feature):
    import numpy as np

    state = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
             for s in manifest['states']}
    outputs = ['log_probs', *['new_'+s['name'] for s in manifest['states']], feature]
    window, shift = manifest['window'], manifest['shift']
    n = len(feats)
    need = ((n-7)//2+1)//2
    if need <= 0:
        raise ValueError('Recording is too short for complete streaming output')
    windows = max(1, math.ceil(need/(manifest['shift']//4)))
    total = window+(windows-1)*shift
    padded = np.concatenate([feats, np.full((max(0, total-n), 80), manifest['fbank']['padValue'], dtype='float32')])
    features, log_probs = [], []
    for at in range(0, total-window+1, shift):
        found = runtime.run(outputs, dict(state, x=padded[None, at:at+window]))
        log_probs.append(found[0][0])
        features.append(found[-1][0])
        state = {s['name']: value for s, value in zip(manifest['states'], found[1:-1])}
    # The SDK pads input for the last call, but emits only outputsFor(n) rows.
    # Training on the extra rows would teach from audio absent in that stream.
    return np.concatenate(features)[:need], np.concatenate(log_probs)[:need]

def clean_target(pronunciation, tokens):
    """Collapse held-phone runs, as the shipped spliced CTC recipe does."""
    ids = {p: i+1 for i, p in enumerate(tokens)}
    return [ids[p] for i, p in enumerate(pronunciation) if i == 0 or p != pronunciation[i-1]]


def softmax(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)


def reconstruct(base, identity):
    """Python counterpart of SDK conditionalPhoneRows, retaining blank exactly."""
    import numpy as np
    if base.ndim != 2 or identity.shape != (len(base), base.shape[1]-1):
        raise ValueError('Conditional identity geometry differs from primary')
    if not np.isfinite(identity).all() or not np.isfinite(base[:, 0]).all() or (base[:, 0] > 0).any():
        raise ValueError('Invalid conditional identity or primary blank')
    q = softmax(identity)
    result = np.empty_like(base)
    result[:, 0] = base[:, 0]
    with np.errstate(divide='ignore'):
        result[:, 1:] = np.log(-np.expm1(base[:, 0].astype('float64')))[:, None] + np.log(q)
    return result


def ctc_occupancy(log_probs, target):
    """Full-transcript forward/backward; return phone-occurrence state occupancy.

    One odd state per occurrence, including distinct repeated articulations.
    Blank states remain in the alignment, but receive no identity supervision.
    Impossible complete transcripts fail rather than supervising silence.
    """
    import numpy as np
    lp = np.asarray(log_probs, dtype='float64')
    target = np.asarray(target, dtype='int64')
    if lp.ndim != 2 or not len(lp) or not len(target) or (target <= 0).any() or (target >= lp.shape[1]).any():
        raise ValueError('Invalid transcript or CTC geometry')
    labels = np.zeros(2*len(target)+1, dtype='int64')
    labels[1::2] = target
    count = len(labels)
    jump = np.zeros(count, dtype=bool)
    jump[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = np.full((len(lp), count), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for t in range(1, len(lp)):
        previous = alpha[t-1]
        one = np.r_[-np.inf, previous[:-1]]
        two = np.r_[-np.inf, -np.inf, previous[:-2]]
        alpha[t] = np.logaddexp(np.logaddexp(previous, one), np.where(jump, two, -np.inf)) + lp[t, labels]
    partition = np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if not np.isfinite(partition):
        raise ValueError('Transcript has no complete acoustic alignment')
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0
    for t in range(len(lp)-2, -1, -1):
        following = beta[t+1] + lp[t+1, labels]
        one = np.r_[following[1:], -np.inf]
        two = np.r_[following[2:], -np.inf, -np.inf]
        can_jump = np.r_[jump[2:], False, False]
        beta[t] = np.logaddexp(np.logaddexp(following, one), np.where(can_jump, two, -np.inf))
    states = np.exp(alpha + beta - partition)
    if not np.allclose(states.sum(1), 1, atol=1e-7):
        raise ValueError('CTC occupancy is not normalized')
    return states[:, 1::2]


def teacher_runtime(tokens):
    import torch
    from quran_muaalem import Muaalem
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = Muaalem('obadx/muaalem-model-v3_2', device=device, dtype=torch.float32)
    teacher.model.eval().float()
    vocab = teacher.multi_level_tokenizer.id_to_vocab['phonemes']
    if [vocab[i] for i in range(1, len(tokens)+1)] != list(tokens):
        raise ValueError('Teacher inventory differs from the frozen student')

    def infer(wave):
        inputs = teacher.processor([wave], sampling_rate=16000, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = teacher.model(**inputs, return_dict=False)[0]['phonemes']
        return logits[0].float().log_softmax(-1).cpu().numpy()
    return infer, dict(model='obadx/muaalem-model-v3_2', precision='float32',
                       revision=getattr(teacher.model.config, '_commit_hash', None), device=device)


def inherited_assets(model):
    """Fail before expensive training if the folded teacher bundle is incomplete."""
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    if not assets:
        raise ValueError('Cannot establish inherited auxiliary asset declarations')
    for file, size, sha in assets:
        path = model/file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError('Inherited SDK asset unavailable or corrupt: '+file)
    return {file: sha for file, _, sha in assets}


def prepare(model, out):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    import sys
    sys.path.insert(0, str(ROOT/'tools'))
    from student import TOKENS, phoneme_ids, spliced_target, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Frozen training plan exists; use a new output directory')
    assets = inherited_assets(model)
    records, excluded, labels_sha = authorized_records()
    manifest, head, runtime = feature_runtime(model, out)
    if manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['sampleRate'] != 16000:
        raise ValueError('Student inventory or frontend differs from training recipe')
    np.savez(out/'initial-head.npz', weight=head['weight'][:, 1:], bias=head['bias'][1:])
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    teacher, teacher_info = teacher_runtime(TOKENS)
    helpers = training_helpers()
    aug = {k: np.load(Path('/student/aug')/(k+'.npz')) for k in ('noise', 'rir')}
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    views = []
    torch.set_num_threads(2)
    # Read one source table at a time; no practice audio or evidence enters here.
    waves = {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        rows = pq.read_table(path, columns=['flac', 'ref', 'hyp']).take([r['index'] for r in selected]).to_pylist()
        for r, value in zip(selected, rows):
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Prepared training waveform is not mono 16 kHz')
            waves[r['key']] = wave
            r['target'] = phoneme_ids(spliced_target(value['ref'], value['hyp']))
    for r in records:
        if r['source'] == 'help':
            wave, sr = sf.read(r['file'], dtype='float32')
            if wave.ndim == 2:
                wave = wave.mean(1)
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                wave = resample_poly(wave, 16000//divisor, sr//divisor).astype('float32')
            waves[r['key']] = wave
            r['target'] = clean_target(''.join(ayahs[a] for a in r['ayahs']), TOKENS)
    for r in records:
        r['waveform_sha256'] = hashlib.sha256(waves[r['key']].tobytes()).hexdigest()
        for view in range(RECIPE['views']):
            rng = np.random.default_rng(int(stable(f"{r['key']}:{view}:{RECIPE['seed']}")[:16], 16))
            wave = waves[r['key']].copy()
            if view:
                if rng.random() < .2:
                    wave = helpers['reverb'](wave, helpers['pick'](aug['rir'], rng))
                if rng.random() < .5:
                    wave = mix_noise(wave, helpers['pick'](aug['noise'], rng), rng.uniform(5., 25.), rng)
                wave *= 10**(rng.uniform(-10, 6)/20)
            feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            # Teacher hears the identical augmented waveform; feature masking is
            # applied only to the student, as in the shipped KD recipe.
            tlp = teacher(wave)
            occurrence = ctc_occupancy(tlp, r['target']).sum(1)
            if view:
                feats = helpers['spec_augment'](feats, rng)
            features, baseline = stream_features(runtime, manifest, feats.numpy(), head['feature'])
            if abs(len(tlp)-len(baseline)) > 2:
                raise ValueError('Teacher and streaming student clocks differ by more than frontend rounding')
            # Both clocks are 40 ms: clamp only the final frontend edge, never
            # stretch teacher timing to the student's utterance length.
            at = np.minimum(np.arange(len(baseline)), len(tlp)-1)
            target = softmax(tlp[at, 1:])
            speech = -np.expm1(tlp[at, 0].astype('float64'))
            file = cache/(stable(f"{r['key']}:{view}")+'.npz')
            np.savez_compressed(file, features=features, baseline=baseline, teacher_identity=target.astype('float32'),
                                teacher_speech=speech.astype('float32'), occupancy=occurrence[at].astype('float32'))
            views.append(dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'],
                              file=str(file), sha256=digest(file)))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    plan = dict(schema=1, recipe=RECIPE, records=records, views=views, excluded_speakers=excluded,
                help_labels_sha256=labels_sha, teacher=teacher_info, primary_sha256=manifest['sha256'],
                inherited_assets=assets, initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx'))
    write_json(out/'plan.json', plan)


def objective(weight, bias, data, conditional, gradient=False):
    """Same head and alignment; ordinary KD scales identity by teacher speech.

    The denominator is identical in both arms. Renormalizing ordinary KD by its
    speech-weighted mass would erase the tested attenuation mechanism.
    """
    import numpy as np
    features = data['features'].astype('float64')
    q = softmax(features@weight+bias)
    wanted = data['teacher_identity'].astype('float64')
    baseline = softmax(data['baseline'][:, 1:])
    occupancy = data['occupancy'].astype('float64')
    mass = occupancy if conditional else occupancy*data['teacher_speech']
    denominator = max(float(occupancy.sum()), 1e-12)
    anchor = RECIPE['anchor']/len(features)
    lp = np.log(np.maximum(q, np.finfo(float).tiny))
    loss = -float((mass[:, None]*wanted*lp).sum())/denominator
    loss += anchor*float((baseline*(np.log(np.maximum(baseline, np.finfo(float).tiny))-lp)).sum())
    if not gradient:
        return loss
    residual = mass[:, None]*(q-wanted)/denominator + anchor*(q-baseline)
    return loss, features.T@residual, residual.sum(0)


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training recipe changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen training inputs changed')
    for v in plan['views']:
        if digest(v['file']) != v['sha256']:
            raise ValueError('Cached training features changed')
    training = {v['speaker'] for v in plan['views'] if v['partition'] == 'train'}
    validation = {v['speaker'] for v in plan['views'] if v['partition'] == 'validation'}
    if not training or not validation or training & validation or (training | validation) & set(plan['excluded_speakers']):
        raise ValueError('Training-speaker partition is empty, overlapping or excluded')
    return plan


def fit(out):
    import numpy as np
    from functools import lru_cache
    plan = read_plan(out)
    initial = np.load(out/'initial-head.npz')
    arms = {k: [initial['weight'].astype('float64').copy(), initial['bias'].astype('float64').copy()]
            for k in ('ordinary', 'conditional')}
    train = [v for v in plan['views'] if v['partition'] == 'train']
    validation = [v for v in plan['views'] if v['partition'] == 'validation']

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as d:
            return {k: d[k].copy() for k in d.files}
    history = []
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch']):
            batch = order[start:start+RECIPE['batch']]
            for name, (weight, bias) in arms.items():
                gw, gb = np.zeros_like(weight), np.zeros_like(bias)
                for i in batch:
                    loss, dw, db = objective(weight, bias, cached(train[i]['file']), name == 'conditional', True)
                    gw += dw/len(batch)
                    gb += db/len(batch)
                if not np.isfinite(gw).all() or not np.isfinite(gb).all():
                    raise ValueError('Non-finite identity-distillation gradient')
                weight -= RECIPE['lr']*gw
                bias -= RECIPE['lr']*gb
        losses = {}
        for name, (weight, bias) in arms.items():
            speakers = {}
            for v in validation:
                speakers.setdefault(v['speaker'], []).append(objective(weight, bias, cached(v['file']), True))
            losses[name] = float(np.mean([np.mean(values) for values in speakers.values()]))
            np.savez(out/f'{name}-{epoch+1}.npz', weight=weight.astype('float32'), bias=bias.astype('float32'))
        history.append(dict(epoch=epoch+1, updates=(epoch+1)*math.ceil(len(train)/RECIPE['batch']), validation=losses))
        print(json.dumps(history[-1]), flush=True)
    # The ordinary arm is an update-matched control at the conditional arm's
    # training-selected step, so target comparisons cannot choose either head.
    selected = min(history, key=lambda r: (r['validation']['conditional'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'), history=history,
        selected=selected, identical_minibatches=True, identical_updates=True,
        arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()}))


def add_identity_head(graph, weight, bias):
    """Append a float identity projection; never change shipped nodes/tensors."""
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto
    head = linear_head(graph, len(bias)+1)
    if weight.shape != (head['weight'].shape[0], len(bias)) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Invalid identity head geometry or parameters')
    names = ['phone_identity_weight', 'phone_identity_bias', 'phone_identity_linear', 'phone_identity_logits']
    occupied = {x for n in graph.graph.node for x in [*n.input, *n.output]} | {t.name for t in graph.graph.initializer}
    if occupied & set(names):
        raise ValueError('Conditional identity output already installed')
    # Use the same dequantized activation as prepare's frozen feature cache.
    feature = head['feature']
    if head['quantized']:
        feature = 'phone_identity_features'
        graph.graph.node.append(helper.make_node('DequantizeLinear',
            [head['quantized_input'], head['activation_scale'], head['activation_zero']], [feature]))
    graph.graph.initializer.extend([numpy_helper.from_array(weight.astype('float32'), names[0]),
                                   numpy_helper.from_array(bias.astype('float32'), names[1])])
    graph.graph.node.extend([helper.make_node('MatMul', [feature, names[0]], [names[2]]),
                            helper.make_node('Add', [names[2], names[1]], [names[3]])])
    graph.graph.output.append(helper.make_tensor_value_info(names[3], TensorProto.FLOAT, [1, None, len(bias)]))


def export(model, out):
    import numpy as np
    import onnx
    import onnxruntime as ort
    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection differs from the frozen plan')
    manifest = json.loads((model/'manifest.json').read_text())
    path = model/manifest['file']
    if manifest['sha256'] != plan['primary_sha256'] or digest(path) != manifest['sha256']:
        raise ValueError('Installed primary differs from the frozen encoder')
    assets = inherited_assets(model)
    if assets != plan['inherited_assets']:
        raise ValueError('Inherited SDK assets changed since preparation')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    # Match the SDK's default ONNX optimization when verifying preserved blank
    # and streaming state, including any fusion affected by the added output.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    original = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    names = [v.name for v in original.get_outputs()]
    reports = {}
    for name, checkpoint in selection['arms'].items():
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Training-selected head changed')
        graph = onnx.load(str(path))
        nodes = [n.SerializeToString() for n in graph.graph.node]
        tensors = [t.SerializeToString() for t in graph.graph.initializer]
        ck = np.load(checkpoint['path'])
        add_identity_head(graph, ck['weight'], ck['bias'])
        if nodes != [n.SerializeToString() for n in list(graph.graph.node)[:len(nodes)]] or \
                tensors != [t.SerializeToString() for t in list(graph.graph.initializer)[:len(tensors)]]:
            raise ValueError('Identity export changed shipped encoder or blank head')
        onnx.checker.check_model(graph)
        directory = out/(name+'-bundle')
        directory.mkdir(exist_ok=True)
        output = directory/manifest['file']
        onnx.save(graph, str(output))
        candidate = ort.InferenceSession(str(output), opts, providers=['CPUExecutionProvider'])
        state = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
                 for s in manifest['states']}
        rng = np.random.default_rng(RECIPE['seed'])
        for _ in range(8):
            feed = dict(state, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
            a = original.run(names, feed)
            b = candidate.run(names+['phone_identity_logits'], feed)
            for field, x, y in zip(names, a, b):
                np.testing.assert_array_equal(x, y, err_msg='Identity export changes '+field)
            rebuilt = reconstruct(b[0][0], b[-1][0])
            np.testing.assert_array_equal(rebuilt[:, 0], a[0][0, :, 0])
            np.testing.assert_allclose(np.exp(rebuilt[:, 1:]).sum(1), -np.expm1(a[0][0, :, 0]), atol=1e-6, rtol=1e-5)
            state = {s['name']: b[names.index('new_'+s['name'])] for s in manifest['states']}
        for file in assets:
            shutil.copyfile(model/file, directory/file)
        for file, sha in assets.items():
            if digest(directory/file) != sha:
                raise ValueError('Copied inherited asset differs: '+file)
        write_json(directory/'manifest.json', dict(manifest, id='identity-distill-'+name,
            bytes=output.stat().st_size, sha256=digest(output)))
        reports[name] = dict(path=str(directory), sha256=digest(output), parity_windows=8,
            unchanged_encoder=True, unchanged_blank=True, inherited_assets=assets,
            identity_output='phone_identity_logits', precision='float32-head-over-frozen-int8-encoder')
    write_json(out/'export-report.json', reports)
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'export'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command != 'fit' and args.model is None:
        parser.error('prepare/export require a complete --model bundle')
    {'prepare': lambda: prepare(args.model, args.out), 'fit': lambda: fit(args.out),
     'export': lambda: export(args.model, args.out)}[args.command]()
