"""Source-family versus original-weight continuations of the shipped phone head.

Run in the supervisor training image. `prepare` admits authorized clean training
recordings, reserves whole training speakers, groups performances within each
partition, and caches encoder features in the deployed streaming geometry.
`fit` uses identical minibatches and update counts in both arms, selecting their
common checkpoint step solely on reserved training speakers. `export` changes
only the selected head tensors and preserves the complete installed SDK bundle.

    python modal/student_source_head.py prepare --model /lab/data/models/spliced-clean-int8 --out /scratch/.../head
    python modal/student_source_head.py fit --out /scratch/.../head
    python modal/student_source_head.py export --model /lab/data/models/<complete-installed-bundle> --out /scratch/.../head

This utility never reads practice sessions, incident keys or benchmark scores.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

from student_source_families import CopyPolicy, partition_aliases, source_families, view_weights, waveform_hash

ROOT = Path(__file__).resolve().parents[1]
RECIPE = dict(seed=831, original_limit=4096, views=3, epochs=12, batch=8,
              lr=.0001, anchor=2., max_seconds=30.)


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
        name = 'source_family_head_features'
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
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
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
    windows = max(1, math.ceil(need/(manifest['chunk']//2)))
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


def prepare(model, out):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    from student import TOKENS, phoneme_ids, spliced_target, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Frozen plan already exists; use a new output directory')
    records, excluded, labels_sha = authorized_records()
    manifest, head, runtime = feature_runtime(model, out)
    if (manifest['tokens'] != TOKENS or manifest['blank'] != 0 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Training recipe does not match the shipped phone inventory or frontend')
    np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    waves = {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        table = pq.read_table(path, columns=['flac', 'ref', 'hyp']).take([r['index'] for r in selected]).to_pylist()
        for r, value in zip(selected, table):
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Prepared original training audio is not mono 16 kHz')
            waves[r['key']] = wave
            r['target'] = phoneme_ids(spliced_target(value['ref'], value['hyp']))
    for r in records:
        if r['source'] != 'help':
            continue
        wave, sr = sf.read(r['file'], dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000//divisor, sr//divisor).astype('float32')
        waves[r['key']] = wave
        r['target'] = clean_target(''.join(ayahs[a] for a in r['ayahs']), manifest['tokens'])
    if any(not r['target'] for r in records):
        raise ValueError('Clean recording has no authorized pronunciation')
    for r in records:
        r['waveform_sha256'] = waveform_hash(waves[r['key']], manifest['sampleRate'])
    # Settings are fixed in code, before all incident queries. Apply grouping
    # separately to training-speaker partitions; no held-out waveform calibrates it.
    policy = CopyPolicy()
    mapping, certificates = {}, []
    for partition in ('train', 'validation'):
        group, proof = source_families([(r['key'], waves[r['key']]) for r in records if r['partition'] == partition], policy)
        mapping.update(group)
        certificates.extend(proof)
    # Audit cross-partition aliases with the same already-frozen acoustic policy.
    # Both exact and codec copies would leak a held-out performance. Abort rather
    # than moving speakers after seeing validation or silently training on it.
    aliases = partition_aliases([(r['key'], waves[r['key']]) for r in records],
                                {r['key']: r['partition'] for r in records}, policy)
    if aliases:
        write_json(out/'partition-aliases.json', aliases)
        raise ValueError('Acoustically verified source alias crosses the training-speaker partition')
    helpers = training_helpers()
    aug = {k: np.load(Path('/student/aug')/(k+'.npz')) for k in ('noise', 'rir')}
    cache = out/'features'
    cache.mkdir()
    views = []
    torch.set_num_threads(2)
    for r in records:
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
            if view:
                feats = helpers['spec_augment'](feats, rng)
            x, lp = stream_features(runtime, manifest, feats.numpy(), head['feature'])
            logits = x@head['weight']+head['bias']
            peak = logits.max(1, keepdims=True)
            reconstructed = logits-peak-np.log(np.exp(logits-peak).sum(1, keepdims=True))
            if not np.allclose(reconstructed, lp, atol=.002, rtol=0):
                raise ValueError('Cached head activations do not reproduce shipped logits within float rounding')
            file = cache/(stable(f"{r['key']}:{view}")+'.npz')
            np.savez_compressed(file, features=x, baseline=lp, target=np.asarray(r['target'], dtype=np.int64))
            views.append(dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'],
                              original_weight=1/RECIPE['views'], file=str(file), sha256=digest(file)))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    plan = dict(schema=1, recipe=RECIPE, grouping_policy=asdict(policy), records=records,
                family_by_recording=mapping, certificates=certificates, views=views,
                excluded_speakers=excluded, help_labels_sha256=labels_sha,
                primary_sha256=manifest['sha256'], feature_graph_sha256=digest(out/'encoder-features.onnx'),
                initial_head_sha256=digest(out/'initial-head.npz'))
    write_json(out/'plan.json', plan)


def item_loss(head, data):
    import torch
    import torch.nn.functional as F

    lp = head(data['features']).log_softmax(-1)
    n = len(lp)
    ctc = F.ctc_loss(lp[:, None, :], data['target'], torch.tensor([n]),
                     torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)/n
    anchor = F.kl_div(lp, data['baseline'].exp(), reduction='sum')/n
    return ctc+RECIPE['anchor']*anchor


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE or plan['grouping_policy'] != asdict(CopyPolicy()):
        raise ValueError('Frozen training or grouping settings changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen initial head or encoder feature graph changed')
    for view in plan['views']:
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen training features changed')
    return plan


def fit(out):
    import copy
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    initial = np.load(out/'initial-head.npz')
    model = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(initial['weight']))
        model.bias.copy_(torch.from_numpy(initial['bias']))
    arms = {'original': model, 'balanced': copy.deepcopy(model)}
    optimizers = {k: torch.optim.SGD(h.parameters(), lr=RECIPE['lr']) for k, h in arms.items()}
    train = [v for v in plan['views'] if v['partition'] == 'train']
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    ordinary, balanced = view_weights(train, plan['family_by_recording'])
    weights = {'original': ordinary, 'balanced': balanced}
    from functools import lru_cache

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}
    history, initial_gradients = [], {}
    # Exact full-population initial gradients establish the mechanism separately
    # from minibatch stochasticity. They may later be compared with frozen target
    # margins by a diagnostic; target data never enters optimization/selection.
    for name, head in arms.items():
        head.zero_grad()
        for i, view in enumerate(train):
            (item_loss(head, cached(view['file']))*weights[name][i]).backward()
        initial_gradients[name+'_weight'] = head.weight.grad.numpy().copy()
        initial_gradients[name+'_bias'] = head.bias.grad.numpy().copy()
    np.savez(out/'initial-gradients.npz', **initial_gradients)
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch']):
            batch = order[start:start+RECIPE['batch']]
            for name, head in arms.items():
                opt = optimizers[name]
                opt.zero_grad()
                # Fixed global importance weights, never batch-renormalized:
                # renormalization would erase balancing on singleton batches.
                for i in batch:
                    loss = item_loss(head, cached(train[i]['file']))
                    (loss*weights[name][i]*len(train)/len(batch)).backward()
                if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError('Non-finite continuation gradient')
                opt.step()
        losses = {}
        with torch.no_grad():
            for name, head in arms.items():
                by_speaker = {}
                for v in validation:
                    family = plan['family_by_recording'][v['recording']]
                    by_speaker.setdefault(v['speaker'], {}).setdefault(family, []).append(float(item_loss(head, cached(v['file']))))
                losses[name] = float(np.mean([np.mean([np.mean(values) for values in families.values()])
                                             for families in by_speaker.values()]))
                np.savez(out/f'{name}-{epoch+1}.npz', weight=head.weight.detach().numpy(), bias=head.bias.detach().numpy())
        history.append(dict(epoch=epoch+1, validation=losses,
                            updates=(epoch+1)*math.ceil(len(train)/RECIPE['batch'])))
        print(json.dumps(history[-1]), flush=True)
    selected = min(history, key=lambda r: (r['validation']['balanced'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    report = dict(schema=1, plan_sha256=digest(out/'plan.json'), selected=selected, history=history,
                  arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
                  gradients_sha256=digest(out/'initial-gradients.npz'),
                  original_weights=ordinary, balanced_weights=balanced,
                  identical_updates=True, identical_minibatches=True,
                  difference_norm=float(np.linalg.norm(initial_gradients['balanced_weight']-initial_gradients['original_weight'])))
    write_json(out/'selection.json', report)


def gradient_effect(out, features, written, rival):
    """Diagnostic signed corrective effect; it never chooses a checkpoint.

    Caller supplies a frozen comparison after preparing training partitions.
    The card's absent-in-control premise is balanced > 0 and original <= 0.
    Values describe an infinitesimal SGD step, not a won warning or practice gain.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    initial = np.load(out/'initial-head.npz')
    gradients = np.load(out/'initial-gradients.npz')
    weight = torch.tensor(initial['weight'], requires_grad=True)
    bias = torch.tensor(initial['bias'], requires_grad=True)
    lp = F.linear(torch.as_tensor(features, dtype=torch.float32), weight, bias).log_softmax(-1)
    lengths = torch.tensor([len(lp)])

    def nll(ids):
        return F.ctc_loss(lp[:, None, :], torch.tensor(ids, dtype=torch.long), lengths,
                          torch.tensor([len(ids)]), reduction='sum', zero_infinity=False)

    margin = nll(rival)-nll(written)
    gw, gb = torch.autograd.grad(margin, (weight, bias))
    effects = {name: -float(np.sum(gw.numpy()*gradients[name+'_weight'])+
                           np.sum(gb.numpy()*gradients[name+'_bias'])) for name in ('original', 'balanced')}
    if not all(math.isfinite(v) for v in effects.values()):
        raise ValueError('Unavailable complete-pronunciation gradient')
    return dict(effects, balancing_only_correction=effects['balanced'] > 0 and effects['original'] <= 0)


def replace_head_parameters(graph, weight, bias):
    """Replace classifier constants; all encoder nodes/tensors stay identical."""
    import numpy as np
    from onnx import numpy_helper

    head = linear_head(graph, len(bias))
    weight, bias = np.asarray(weight, dtype='float32'), np.asarray(bias, dtype='float32')
    if weight.shape != head['weight'].shape or bias.shape != head['bias'].shape or \
            not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Refitted head has invalid geometry or parameters')
    changed = {head['bias_name']: bias}
    if head['quantized']:
        scale = np.asarray(max(float(np.max(np.abs(weight)))/127, 1e-8), dtype='float32')
        changed.update({head['weight_name']: np.clip(np.round(weight/scale), -127, 127).astype('int8'),
                        head['scale_name']: scale, head['zero_name']: np.asarray(0, dtype='int8')})
    else:
        changed[head['weight_name']] = weight
    before = {t.name: t.SerializeToString() for t in graph.graph.initializer}
    for i, tensor in enumerate(graph.graph.initializer):
        if tensor.name in changed:
            graph.graph.initializer[i].CopyFrom(numpy_helper.from_array(changed[tensor.name], tensor.name))
    for tensor in graph.graph.initializer:
        if tensor.name not in changed and tensor.SerializeToString() != before[tensor.name]:
            raise ValueError('Export changed an encoder tensor')
    return sorted(changed)


def export(model, out):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import re

    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection does not match frozen training plan')
    manifest = json.loads((model/'manifest.json').read_text())
    if manifest['sha256'] != plan['primary_sha256'] or digest(model/manifest['file']) != manifest['sha256']:
        raise ValueError('Installed primary differs from the frozen encoder')
    # All inherited declared assets must exist and match before exporting either
    # head. Missing files must not silently remove the folded insertion teacher.
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    if not assets:
        raise ValueError('Cannot establish the inherited SDK auxiliary asset declarations')
    for file, size, sha in assets:
        path = model/file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError('Inherited SDK asset unavailable or corrupt: '+file)
    reports = {}
    for name, checkpoint in selection['arms'].items():
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen selected head changed')
        graph = onnx.load(str(model/manifest['file']))
        ck = np.load(checkpoint['path'])
        weight, bias = ck['weight'].T.astype('float32'), ck['bias'].astype('float32')
        changed = replace_head_parameters(graph, weight, bias)
        directory = out/(name+'-bundle')
        directory.mkdir(exist_ok=True)
        path = directory/manifest['file']
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        a = ort.InferenceSession(str(model/manifest['file']), options, providers=['CPUExecutionProvider'])
        b = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        states = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
                  for s in manifest['states']}
        outputs = [v.name for v in a.get_outputs()]
        rng = np.random.default_rng(RECIPE['seed'])
        for _ in range(8):
            feeds = dict(states, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
            original, candidate = a.run(outputs, feeds), b.run(outputs, feeds)
            if original[0].shape != candidate[0].shape or not np.isfinite(candidate[0]).all():
                raise ValueError('Export changed the phone output geometry or produced invalid rows')
            for output, x, y in zip(outputs[1:], original[1:], candidate[1:]):
                np.testing.assert_array_equal(x, y, err_msg='Head export changed '+output)
            states = {s['name']: value for s, value in zip(manifest['states'], candidate[1:])}
        for file, _, _ in assets:
            shutil.copyfile(model/file, directory/file)
        exported = dict(manifest, id='source-family-'+name, file=manifest['file'], bytes=path.stat().st_size,
                        sha256=digest(path))
        write_json(directory/'manifest.json', exported)
        reports[name] = dict(path=str(directory), model_sha256=digest(path), changed_tensors=sorted(changed),
                             cache_parity_windows=8, inherited_assets={f: digest(directory/f) for f, _, _ in assets})
    write_json(out/'export-report.json', reports)
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export'))
    p.add_argument('--model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command != 'fit' and args.model is None:
        p.error('prepare/export require the complete --model bundle')
    if args.command == 'prepare':
        prepare(args.model, args.out)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out)
