"""Training-only control and export-effective discrete phone-head refits.

Remote CLI (no Modal credentials needed in this module):
  prepare --model DIR --recordings eligible.json --excluded speakers.json --out CACHE
  control --cache CACHE --out RUN
  discrete --cache CACHE --out RUN

The recording list carries speaker, source, held_out=false, split=train,
transcript_kind=performed, phones, audio and sha256. help-train-none additionally
requires label=none, events=[], non_errors=[] and unsupported=[]. Exclusions must
be resolved from the enrolled corpus before preparing this list. This runner
does not infer transcripts from mistake-labelled or repeat takes.

Run control first, then audit its floating and actual int8 exports on the
registered targets without selecting on them. Run discrete only if that audit
establishes export loss. No practice data enter caching, updates or selection.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.nn import functional as F

from student_quantized_head import QuantizedHead, bundle, digest, predict, quantize_control, runtime, stream_chunks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def speaker_partition(rows, seed=0):
    speakers = {r['speaker'] for r in rows}
    if len(speakers) < 5 or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError('Need at least five identified eligible training speakers')
    ordered = sorted(speakers, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).digest())
    return {s: 'selection' if i % 5 == 0 else 'fit' for i, s in enumerate(ordered)}


def eligible(rows, excluded):
    # Any nontraining entry excludes that entire speaker before reading audio.
    excluded = set(excluded) | {r.get('speaker') for r in rows
                               if r.get('split') != 'train' or r.get('held_out') is not False}
    admitted = []
    for r in rows:
        if r.get('speaker') in excluded:
            continue
        if (not r.get('speaker') or r.get('source') not in ('authorized-transcribed', 'help-train-none')
                or r.get('transcript_kind') != 'performed' or not r.get('phones')
                or not r.get('audio') or not r.get('sha256')):
            raise ValueError('Recording lacks eligible performed transcript and audio identity')
        if r['source'] == 'help-train-none' and (r.get('label') != 'none' or
                any(r.get(k) != [] for k in ('events', 'non_errors', 'unsupported'))):
            raise ValueError('Help training requires none-labelled ordinary takes without repeats')
        admitted.append(copy.deepcopy(r))
    if not admitted:
        raise ValueError('No eligible recordings')
    return admitted


def prepare(args):
    import soundfile as sf
    import torchaudio
    torch.set_num_threads(1)
    started = time.monotonic()
    out, source, listing = Path(args.out), Path(args.model).resolve(), Path(args.recordings).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Need an empty cache directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    model = source / manifest['file']
    if digest(model) != manifest['sha256']:
        raise ValueError('Source model checksum mismatch')
    if manifest['blank'] != 0 or manifest['frameMs'] != 40 or manifest['sampleRate'] != 16000:
        raise ValueError('This refit requires the unchanged deployed alphabet and clock')
    head = QuantizedHead(onnx.load(model))
    excluded = json.loads(Path(args.excluded).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError('Need the enrolled release-speaker exclusion list')
    rows = eligible(json.loads(listing.read_text()), excluded)
    partitions = speaker_partition(rows)
    tokens = manifest['tokens']
    mapping = {p: i + 1 for i, p in enumerate(tokens)}
    if len(mapping) != len(tokens) or head.bias.shape != (len(tokens) + 1,):
        raise ValueError('Ordered deployed inventory differs from the phone head')
    exposed, primary, isolated = head.exposed(), runtime(head.source), runtime(head.graph)
    out.mkdir(parents=True, exist_ok=True)
    cached, seen = [], set()
    for r in rows:
        if time.monotonic() - started >= 24 * 3600:
            raise TimeoutError('One-day experiment budget exhausted')
        audio = (listing.parent / r['audio']).resolve()
        if not audio.is_relative_to(listing.parent) or digest(audio) != r['sha256']:
            raise ValueError('Enrolled audio path or identity mismatch')
        if r['sha256'] in seen:
            raise ValueError('Duplicate audio could leak between speakers')
        seen.add(r['sha256'])
        target = np.asarray([mapping[p] for p in r['phones']], dtype=np.int64)
        wave, sr = sf.read(audio, dtype='float32', always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Invalid enrolled waveform')
        wave = torch.from_numpy(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        fbank = torchaudio.compliance.kaldi.fbank(wave[None], num_mel_bins=80,
            sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()
        chunks, baseline = stream_chunks(exposed, manifest, fbank, head.feature)
        # Exposing features must not alter any primary output or carried state.
        states = {s['name']: np.zeros(s['shape'], dtype=s['dtype']) for s in manifest['states']}
        exposed_session = runtime(exposed)
        names = ['log_probs'] + ['new_' + s['name'] for s in manifest['states']]
        windows = []
        for j in range(len(chunks)):
            window = np.full((1, manifest['window'], 80), manifest['fbank']['padValue'], dtype=np.float32)
            view = fbank[j * manifest['shift']:j * manifest['shift'] + manifest['window']]
            window[0, :len(view)] = view
            windows.append(torch.from_numpy(window.copy()))
            a, b = primary.run(names, {'x': window, **states}), exposed_session.run(names, {'x': window, **states})
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=0)
            states = {s['name']: v for s, v in zip(manifest['states'], a[1:])}
        got = predict(isolated, head.feature, chunks)
        np.testing.assert_allclose(got, baseline, atol=1e-5, rtol=0)
        np.testing.assert_array_equal(got.argmax(-1), baseline.argmax(-1))
        if len(target) + int((target[1:] == target[:-1]).sum()) > len(baseline):
            raise ValueError('Performed transcript has no complete CTC path')
        cached.append(dict(speaker=r['speaker'], partition=partitions[r['speaker']],
                           source=r['source'], audio_sha256=r['sha256'], target=torch.from_numpy(target),
                           chunks=[(torch.from_numpy(x), keep) for x, keep in chunks],
                           windows=windows, anchor=torch.from_numpy(baseline)))
    onnx.save(head.source, out / 'source.onnx')
    torch.save(cached, out / 'recordings.pt')
    write_json(out / 'cache.json', dict(source=str(source), manifest=manifest, partitions=partitions,
        list_sha256=digest(listing), exclusions_sha256=digest(args.excluded),
        cache_sha256=digest(out / 'recordings.pt'), source_onnx_sha256=digest(out / 'source.onnx'),
        recordings=len(cached),
        started_unix=time.time() - (time.monotonic() - started), elapsed_s=time.monotonic() - started))


def losses(lp, row):
    target = row['target']
    if (lp.ndim != 2 or lp.shape != row['anchor'].shape or not len(target)
            or (target <= 0).any() or (target >= lp.shape[1]).any()):
        raise ValueError('Invalid complete-transcript loss input')
    ctc = F.ctc_loss(lp[:, None], target, [len(lp)], [len(target)],
                     blank=0, reduction='sum', zero_infinity=False) / len(target)
    preservation = F.kl_div(lp, row['anchor'], log_target=True, reduction='sum') / len(lp)
    if not torch.isfinite(ctc + preservation):
        raise ValueError('Nonfinite transcript or preservation loss')
    return ctc, preservation


def actual_loss(head, graph, rows, preserve=True):
    session = runtime(graph)
    terms = []
    for row in rows:
        chunks = [(x.numpy(), keep) for x, keep in row['chunks']]
        lp = torch.from_numpy(predict(session, head.feature, chunks))
        ctc, kl = losses(lp, row)
        terms.append(float(ctc + kl if preserve else ctc))
    if not terms:
        raise ValueError('Empty speaker partition')
    return sum(terms) / len(terms)


def proposals(codes, origin, gradient, scale, count, bound):
    """Gradient ranks bounded one-code moves; it never decides acceptance."""
    if (codes.dtype != np.int8 or origin.shape != codes.shape or gradient.shape != codes.shape
            or not np.isfinite(gradient).all() or count < 1 or bound < 1 or bound > 127):
        raise ValueError('Invalid bounded coordinate proposal')
    ranked = []
    for flat in np.argsort(-np.abs(gradient * float(np.asarray(scale).item())), axis=None, kind='stable'):
        index = np.unravel_index(flat, codes.shape)
        g = gradient[index]
        if g == 0:
            continue
        value = int(codes[index]) + (-1 if g > 0 else 1)
        if -128 <= value <= 127 and abs(value - int(origin[index])) <= bound:
            ranked.append((index, value))
        if len(ranked) >= count:
            break
    return ranked


def code_gradient(head, codes, rows):
    """Actual CTC output derivatives suggest moves via a float affine tangent.

    The tangent is only a proposal heuristic. Activation rounding and integer
    accumulation are evaluated by ONNX for every accepted move.
    """
    session = runtime(head.with_codes(codes))
    gradient = np.zeros(codes.shape, dtype=np.float64)
    for row in rows:
        chunks = [(x.numpy(), keep) for x, keep in row['chunks']]
        lp = torch.tensor(predict(session, head.feature, chunks), requires_grad=True)
        ctc, kl = losses(lp, row)
        (ctc + kl).backward()
        g = lp.grad - lp.detach().exp() * lp.grad.sum(-1, keepdim=True)
        x = torch.cat([x[0, :keep] for x, keep in row['chunks']])
        gradient += (x.T @ g).numpy()
    return gradient / len(rows)


def discrete_step(head, codes, rows, *, count=8, bound=2):
    baseline = actual_loss(head, head.with_codes(codes), rows)
    gradient = code_gradient(head, codes, rows)
    for index, value in proposals(codes, head.initial_codes, gradient, head.scale, count, bound):
        candidate = codes.copy()
        candidate[index] = value
        score = actual_loss(head, head.with_codes(candidate), rows)
        if score < baseline:
            return candidate, score, True
    return codes.copy(), baseline, False


def verify_product(head, graph, manifest, rows):
    """Replay real cached frontend windows; check primary and state parity."""
    product, original, isolated = runtime(head.graft(graph)), runtime(head.source), runtime(graph)
    names = ['log_probs'] + ['new_' + s['name'] for s in manifest['states']]
    for row in rows:
        states = {s['name']: np.zeros(s['shape'], dtype=s['dtype']) for s in manifest['states']}
        for window, (x, _) in zip(row['windows'], row['chunks']):
            feeds = {'x': window.numpy(), **states}
            got, before = product.run(names, feeds), original.run(names, feeds)
            expected = isolated.run(['log_probs'], {head.feature: x.numpy()})[0]
            np.testing.assert_allclose(got[0], expected, atol=1e-5, rtol=0)
            np.testing.assert_array_equal(got[0].argmax(-1), expected.argmax(-1))
            for a, b in zip(got[1:], before[1:]):
                np.testing.assert_array_equal(a, b)
            states = {s['name']: value for s, value in zip(manifest['states'], got[1:])}


def fit(args, discrete=False):
    torch.set_num_threads(1)
    started = time.monotonic()
    cache, out = Path(args.cache), Path(args.out)
    meta = json.loads((cache / 'cache.json').read_text())
    if (digest(cache / 'source.onnx') != meta['source_onnx_sha256']
            or digest(cache / 'recordings.pt') != meta['cache_sha256']):
        raise ValueError('Training cache identity changed')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Need a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    head = QuantizedHead(onnx.load(cache / 'source.onnx'))
    rows = torch.load(cache / 'recordings.pt', map_location='cpu', weights_only=True)
    groups = speaker_partition(rows)
    if groups != meta['partitions'] or any(r['partition'] != groups[r['speaker']] for r in rows):
        raise ValueError('Training-speaker partition changed')
    training = [r for r in rows if r['partition'] == 'fit']
    selection = [r for r in rows if r['partition'] == 'selection']
    history = []
    if discrete:
        codes = head.initial_codes.copy()
        best = codes.copy()
        best_loss = actual_loss(head, head.graph, selection, preserve=False)
        accepted_loss = actual_loss(head, head.graph, training)
        for step in range(args.steps):
            if time.time() - meta['started_unix'] >= 24 * 3600:
                raise TimeoutError('One-day experiment budget exhausted')
            codes, accepted_loss, accepted = discrete_step(head, codes, training,
                                                           count=args.proposals, bound=args.bound)
            selected_loss = actual_loss(head, head.with_codes(codes), selection, preserve=False)
            history.append(dict(step=step + 1, accepted=accepted, training_loss=accepted_loss,
                                selection_ctc=selected_loss))
            if selected_loss < best_loss:
                best, best_loss = codes.copy(), selected_loss
            if not accepted:
                break
        graph = head.with_codes(best)
        np.save(out / 'selected-codes.npy', best)
        name = 'discrete'
    else:
        # Bias stays fixed in both arms: the contrast is quantized weight updates.
        weights = torch.tensor(head.weights(), requires_grad=True)
        bias = torch.from_numpy(head.bias)
        optimizer = torch.optim.Adam([weights], lr=args.lr)
        best, best_loss = weights.detach().clone(), float('inf')
        for step in range(args.steps + 1):
            if time.time() - meta['started_unix'] >= 24 * 3600:
                raise TimeoutError('One-day experiment budget exhausted')
            with torch.no_grad():
                scores = [float(losses(torch.cat([(x[0] @ weights + bias).log_softmax(-1)[:keep]
                    for x, keep in r['chunks']]), r)[0]) for r in selection]
                score = sum(scores) / len(scores)
            history.append(dict(step=step, selection_ctc=score))
            if score < best_loss:
                best, best_loss = weights.detach().clone(), score
            if step == args.steps:
                break
            optimizer.zero_grad()
            for r in training:
                lp = torch.cat([(x[0] @ weights + bias).log_softmax(-1)[:keep] for x, keep in r['chunks']])
                ctc, kl = losses(lp, r)
                ((ctc + kl) / len(training)).backward()
            torch.nn.utils.clip_grad_norm_([weights], 2.)
            optimizer.step()
        floating = head.floating(best.numpy())
        graph = quantize_control(floating, out / 'export')
        verify_product(head, floating, meta['manifest'], rows)
        bundle(meta['source'], meta['manifest'], head.graft(floating), out / 'control-fp32', 'control-fp32')
        np.save(out / 'selected-weights.npy', best.numpy())
        name = 'control-int8'
    product = head.graft(graph)
    verify_product(head, graph, meta['manifest'], rows)
    bundle(meta['source'], meta['manifest'], product, out / name, name)
    write_json(out / 'refit.json', dict(arm=name, best_selection_ctc=best_loss, history=history,
        recordings=meta['recordings'], partitions=meta['partitions'], bias_frozen=True,
        source_sha256=meta['manifest']['sha256'], cache_sha256=meta['cache_sha256'],
        selection_uses_practice=False, elapsed_s=time.monotonic() - started,
        steps=args.steps, bound=args.bound if discrete else None,
        actual_export_training_loss=actual_loss(head, graph, training),
        actual_export_selection_ctc=actual_loss(head, graph, selection, preserve=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ('model', 'recordings', 'excluded', 'out'):
        prep.add_argument('--' + name, required=True)
    for name in ('control', 'discrete'):
        command = commands.add_parser(name)
        command.add_argument('--cache', required=True)
        command.add_argument('--out', required=True)
        command.add_argument('--steps', type=int, default=96)
        command.add_argument('--lr', type=float, default=.001)
        command.add_argument('--proposals', type=int, default=8)
        command.add_argument('--bound', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.steps < 1 or not np.isfinite(args.lr) or args.lr <= 0 or args.proposals < 1 or not 1 <= args.bound <= 127:
            raise ValueError('Invalid bounded training configuration')
        fit(args, args.command == 'discrete')


if __name__ == '__main__':
    main()
