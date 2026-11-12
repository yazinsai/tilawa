"""Bounded CPU matched phone-head refits; execute via the supplied supervisor.

Prepare admits complete none-labelled takes from help-train training speakers.
Optional additional performed transcripts must explicitly attest authorization,
verification and training provenance and bind the waveform by SHA256. No
mistake scripts, practice recordings or practice labels enter this program.
Only unmodified full recordings are cached: real speech and real pauses retain
their transcript supervision. Both arms consume the exact same cached data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance
from student_pareto import (admitted_none, speaker_partition, log_softmax,
                           ctc_occupancies, matched_refit)
from student_pareto_export import (digest, load_source, expose, session, stream,
                                  insertion_assets, graft, head_parameters)

SEED, EPOCHS, LR = 1729, 12, .02


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Audio/cache leaves its enrolled corpus or is missing')
    return path


def recordings(help_labels, transcribed, excluded, reference):
    labels = [json.loads(s) for s in Path(help_labels).read_text().splitlines() if s.strip()]
    additional = ([json.loads(s) for s in Path(transcribed).read_text().splitlines() if s.strip()]
                  if transcribed else [])
    # Discover all exclusions before accepting any recording from either corpus.
    excluded = set(excluded) | {r.get('speaker') for r in labels + additional if r.get('split') != 'train'}
    ayahs = {f'{a["s"]}:{a["a"]}': a['p'] for a in reference['ayahs']}
    rows = [dict(speaker=r['speaker'], source='help-none', phones=''.join(ayahs[a] for a in r['ayahs']),
                 wav=contained(Path(help_labels).parent, r['wav']))
            for r in admitted_none(labels, excluded)]
    for r in additional:
        if r.get('speaker') in excluded or r.get('split') != 'train':
            continue
        if (not r.get('speaker') or r.get('authorized') is not True or r.get('verified') is not True or
                not r.get('phones') or not r.get('wav')):
            raise ValueError('Additional recording is not an authorized performed transcript')
        path = contained(Path(transcribed).parent, r['wav'])
        if not r.get('sha256') or digest(path) != r['sha256']:
            raise ValueError('Audio identity differs from its performed transcript')
        rows.append(dict(speaker=r['speaker'], source='transcribed', phones=r['phones'], wav=path))
    if not any(r['source'] == 'help-none' for r in rows):
        raise ValueError('Need training-speaker none-labelled takes')
    seen = set()
    for r in rows:
        r['audioSha256'] = digest(r['wav'])
        if r['audioSha256'] in seen:
            raise ValueError('Duplicate audio would leak partitions or duplicate gradients')
        seen.add(r['audioSha256'])
        if not phoneme_ids(r['phones']):
            raise ValueError('None take needs its complete speech transcript')
    return rows, sorted(s for s in excluded if s)


def fbank(wave):
    import torch
    import torchaudio
    # Same raw Kaldi frontend as the shipped primary student.
    return torchaudio.compliance.kaldi.fbank(torch.tensor(wave)[None], num_mel_bins=80,
        sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()


def prepare(args):
    import soundfile as sf
    import torch
    import torchaudio
    torch.set_num_threads(1)
    source, out = Path(args.model).resolve(), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation directory must be empty')
    manifest, graph, _, _ = load_source(source)
    head_parameters(graph)
    if manifest['tokens'] != TOKENS:
        raise ValueError('Ordered alphabet differs from the installed listener')
    auxiliary = insertion_assets(source)
    excluded = json.loads(Path(args.exclusions).read_text()) if args.exclusions else []
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError('Invalid excluded-speaker enrollment')
    reference = json.loads((Path(__file__).resolve().parents[1] / 'packages/core/assets/ref-v2.json').read_text())
    rows, excluded = recordings(args.help, args.transcribed, excluded, reference)
    groups = speaker_partition([r['speaker'] for r in rows])
    manifest, feature, width, path = expose(source, out)
    original, frozen = session(source / manifest['file']), session(path)
    feeds = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
             for s in manifest['states']}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(SEED)
    for _ in range(8):
        feeds['x'] = rng.normal(-10., 2., (1, manifest['window'], 80)).astype('float32')
        a, b = original.run(names, feeds), frozen.run(names, feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg='Feature extraction changed ' + name)
        feeds.update({s['name']: b[names.index('new_' + s['name'])] for s in manifest['states']})
    cached = []
    for i, r in enumerate(rows):
        wave, sr = sf.read(r['wav'], dtype='float32', always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Empty or nonfinite enrolled audio')
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        features = fbank(wave.numpy())
        lp, x = stream(frozen, manifest, features, feature)
        target = np.asarray(phoneme_ids(r['phones']), dtype=np.int64)
        ctc_occupancies(log_softmax(lp), target)  # reject unalignable full transcripts
        file = f'recording-{i:05d}.npz'
        np.savez_compressed(out / file, features=x, shipped=lp, target=target, fbank=features)
        cached.append(dict(speaker=r['speaker'], source=r['source'], audioSha256=r['audioSha256'],
            transcriptSha256=hashlib.sha256(r['phones'].encode()).hexdigest(),
            views=[file], viewHashes=[digest(out / file)]))
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / 'prepared.json', dict(schema=1, source=str(source), sourceSha256=manifest['sha256'],
        tokens=TOKENS, width=width, featureOutput=feature, rows=cached, groups=groups,
        excludedSpeakers=excluded, auxiliaryHashes=auxiliary,
        dataPolicy='authorized-full-none-takes-and-optional-verified-transcripts-no-augmentation'))


def load_prepared(path):
    info = json.loads((path / 'prepared.json').read_text())
    manifest, _, _, width = load_source(info['source'])
    if (manifest['sha256'] != info['sourceSha256'] or width != info['width'] or
            info['tokens'] != TOKENS or insertion_assets(info['source']) != info['auxiliaryHashes']):
        raise ValueError('Installed model/teacher changed since preparation')
    groups = speaker_partition([r['speaker'] for r in info['rows']])
    if groups != info['groups'] or set(groups) & set(info['excludedSpeakers']):
        raise ValueError('Speaker provenance or partition changed')
    rows = []
    for r in info['rows']:
        if len(r['views']) != 1 or len(r['viewHashes']) != 1:
            raise ValueError('Missing matched complete recording')
        views = []
        for file, checksum in zip(r['views'], r['viewHashes']):
            file = contained(path, file)
            if digest(file) != checksum:
                raise ValueError('Cached training features changed')
            with np.load(file, allow_pickle=False) as data:
                views.append({k: data[k].copy() for k in ('features', 'target', 'shipped')})
            if views[-1]['features'].shape != (len(views[-1]['shipped']), width):
                raise ValueError('Cached feature clock differs')
        rows.append(dict(r, cacheFiles=r['views'], views=views))
    return info, rows


def validate(head, rows):
    speakers = {}
    for r in rows:
        values = []
        for v in r['views']:
            lp = log_softmax(v['features'] @ head[0].T + head[1])
            loss, _ = ctc_occupancies(lp, v['target'])
            hyp = ctc_collapse(lp.argmax(1).tolist())
            values.append([loss / len(lp), edit_distance(hyp, v['target'].tolist()) / len(v['target'])])
        speakers.setdefault(r['speaker'], []).append(np.mean(values, axis=0))
    result = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(result[0]), per=float(result[1]), speakers=len(speakers))


def fit(args):
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Fit directory must be empty')
    info, rows = load_prepared(prepared)
    _, graph, _, _ = load_source(info['source'])
    heads, report = matched_refit(head_parameters(graph), rows, info['groups'], TOKENS,
        validate, EPOCHS, LR, SEED, emit=lambda r: print(json.dumps(r), flush=True))
    report.update(seed=SEED, epochs=EPOCHS, learningRate=LR,
        objective='equal-recording-mean-CTC-normalized-by-valid-frames',
        head='existing-affine-head-dequantized-identically-in-both-arms',
        groups=info['groups'], excludedSpeakers=info['excludedSpeakers'],
        preparedSha256=digest(prepared / 'prepared.json'), arms={})
    out.mkdir(parents=True, exist_ok=True)
    selection = [r for r in rows if info['groups'][r['speaker']] == 'selection']
    for arm, head in heads.items():
        np.savez(out / (arm + '.npz'), weight=head[0], bias=head[1])
        model_path = graft(info['source'], head, out / arm, arm)
        runtime = session(model_path, optimize=True)
        manifest = json.loads((out / arm / 'manifest.json').read_text())
        worst = 0.
        for r in selection:
            for file, v in zip(r['cacheFiles'], r['views']):
                with np.load(contained(prepared, file), allow_pickle=False) as data:
                    got, _ = stream(runtime, manifest, data['fbank'])
                want = log_softmax(v['features'] @ head[0].T + head[1])
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report['arms'][arm] = dict(sha256=digest(model_path), parityMaxProbabilityError=worst,
            parityRecordings=len(selection), selectedUpdates=report['selectedUpdates'],
            trainingUpdates=report['trainingUpdates'], auxiliaryHashes=insertion_assets(out / arm))
    write_json(out / 'training-report.json', report)  # only after BOTH complete bundles verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ('model', 'help', 'out'):
        p.add_argument('--' + name, required=True)
    for name in ('transcribed', 'exclusions'):
        p.add_argument('--' + name)
    p = commands.add_parser('fit')
    for name in ('prepared', 'out'):
        p.add_argument('--' + name, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else fit)(args)
