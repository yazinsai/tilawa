"""Frozen-encoder temporal CTC readouts with matched channel-consistency controls.

Run through the supervisor, never train on practice sessions. Geometry and
selection are fixed here; labels and speaker folds enter only training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

SEED, DIMENSIONS, HIDDEN, EPOCHS = 1729, 32, 48, 36
CHECKPOINTS = (12, 24, 36)
CONSISTENCY = 1.


def training_take(r, excluded):
    if r.get('split') != 'train' or not r.get('speaker') or r['speaker'] in excluded:
        return False
    if r.get('excluded') or r.get('scored') is False or r.get('kind') != 'none':
        return False
    if r.get('events') or r.get('non_errors'):
        raise ValueError('None-labelled transcript includes a deviation')
    return True


def speaker_folds(rows):
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three eligible training speakers required')
    return {s: i % 3 for i, s in enumerate(speakers)}


def channel_views(audio, rng, sample_rate=16000):
    """Filter full recordings before crossfading; no seam/reset or sample delay."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or len(x) < sample_rate or not np.isfinite(x).all():
        raise ValueError('A finite mono recording of at least one second is required')
    filters = [(float(rng.uniform(80, 300)), float(rng.uniform(3000, 7600))) for _ in range(2)]
    a, b = [sosfiltfilt(butter(4, band, btype='bandpass', fs=sample_rate, output='sos'), x) for band in filters]
    duration = len(x)/sample_rate
    width = min(float(rng.uniform(.4, 1.2)), duration/2)
    center = float(rng.uniform(.25, .75))*duration
    phase = np.clip((np.arange(len(x))/sample_rate - center + width/2)/width, 0, 1)
    blend = .5-.5*np.cos(np.pi*phase)
    transition = a*(1-blend)+b*blend
    # A single common scale handles all views; no view-specific normalization
    # could silently undo the filter or introduce a level transition.
    scale = min(1., .89/max(float(np.max(np.abs(v))) for v in (a, b, transition)))
    return tuple((v*scale).astype('float32') for v in (a, transition, b)), dict(
        filters=filters, centerSeconds=center, widthSeconds=width, commonScale=scale)


def prepare(labels_path, out):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    path = Path(labels_path)
    labels = [json.loads(s) for s in path.read_text().splitlines() if s]
    excluded = {r['speaker'] for r in labels if r.get('split') != 'train'}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recordings, skipped = {}, []
    for r in labels:
        if not training_take(r, excluded):
            continue
        if not r.get('id') or not r.get('wav') or not r.get('ayahs'):
            raise ValueError('Missing authorized recording provenance')
        if r['id'] in recordings:
            raise ValueError('Duplicate recording identity')
        audio, sr = sf.read(path.parent/r['wav'], dtype='float32', always_2d=True)
        audio = audio.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000//divisor, sr//divisor).astype('float32')
        if len(audio) < 16000 or not np.isfinite(audio).all():
            skipped.append(r['id'])
            continue
        key = hashlib.sha256(r['id'].encode()).hexdigest()
        clean = out/f'{key}-clean.wav'
        sf.write(clean, audio, 16000, subtype='FLOAT')
        item = dict(clean=str(clean), train=[], validation=[], geometry=[])
        for partition, salt in [('train', 0), ('validation', 100000)]:
            rng = np.random.default_rng(SEED+salt+int(key[:8], 16))
            for replica in range(2):
                views, geometry = channel_views(audio, rng)
                files = []
                for name, wave in zip(('fixedA', 'transition', 'fixedB'), views):
                    file = out/f'{key}-{partition}-{replica}-{name}.wav'
                    sf.write(file, wave, 16000, subtype='FLOAT')
                    files.append(str(file))
                item[partition].append(files)
                item['geometry'].append(dict(partition=partition, replica=replica, **geometry))
        recordings[r['id']] = item
    if not recordings:
        raise ValueError('No eligible none-labelled training recordings')
    (out/'views.json').write_text(json.dumps(recordings))
    (out/'excluded-speakers.json').write_text(json.dumps(sorted(excluded)))
    (out/'views-report.json').write_text(json.dumps(dict(recordings=len(recordings), skipped=skipped,
        labelsSha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed=SEED), indent=2))


def occupancies(log_probs, ids):
    """Canonical CTC soft supervision for the shared three-epoch warm-up."""
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    labels = np.array([0]+[p for i in ids for p in (i, 0)])
    if not ids or len(lp) < len(ids)+sum(a == b for a, b in zip(ids, ids[1:])):
        return None
    incoming = [[s]+([s-1] if s else [])+([s-2] if s>1 and labels[s] and labels[s]!=labels[s-2] else []) for s in range(len(labels))]
    outgoing = [[] for _ in labels]
    for s, edges in enumerate(incoming):
        for before in edges:
            outgoing[before].append(s)
    alpha = np.full((len(lp), len(labels)), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for t in range(1, len(lp)):
        for s, before in enumerate(incoming):
            alpha[t, s] = np.logaddexp.reduce(alpha[t-1, before])+lp[t, labels[s]]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0
    for t in range(len(lp)-2, -1, -1):
        for s, after in enumerate(outgoing):
            beta[t, s] = np.logaddexp.reduce(beta[t+1, after]+lp[t+1, labels[after]])
    targets = np.zeros_like(lp)
    states = np.exp(alpha+beta-z)
    for s, phone in enumerate(labels):
        targets[:, phone] += states[:, s]
    return targets.astype('float32')


def alternatives(ids, vocab, rng):
    """Complete transcript candidates, generated only from training transcripts."""
    result = [list(ids)]
    for _ in range(3):
        rival = list(ids)
        slot = int(rng.integers(len(ids)))
        choices = [p for p in range(1, vocab) if p != rival[slot]]
        rival[slot] = int(rng.choice(choices))
        result.append(rival)
    return result


def temporal_features(x, center, scale):
    import numpy as np
    a = (np.asarray(x, dtype='float32')-center)/scale
    return np.concatenate([np.concatenate([a[:1], a[:-1]]), a,
                           np.concatenate([a[1:], a[-1:]])], axis=1)


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(3*DIMENSIONS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def ctc_nll(lp, sequences):
    import torch
    n = len(sequences)
    target = torch.tensor([p for seq in sequences for p in seq], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return torch.nn.functional.ctc_loss(lp[:, None, :].expand(-1, n, -1), target,
        torch.full((n,), len(lp), dtype=torch.long), lengths, reduction='none', zero_infinity=False)


def pronunciation_consistency(a, b, candidates):
    """JS divergence of complete-pronunciation CTC distributions; no frame alignment assumed."""
    import torch
    scores = [-ctc_nll(x, candidates) for x in (a, b)]
    if not all(torch.isfinite(s).all() for s in scores):
        raise ValueError('Infeasible training pronunciation comparison')
    qa, qb = [s.softmax(0) for s in scores]
    mixture = (qa+qb)/2
    return sum((q*(q.clamp_min(1e-12).log()-mixture.clamp_min(1e-12).log())).sum() for q in (qa, qb))/2


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
        output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def fit(rows, vocab, checkpoints=CHECKPOINTS):
    import numpy as np
    import torch
    raw = np.concatenate([v for r in rows for v in [r['clean']['features']]+[
        view for triplet in r['train'] for view in triplet]]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    data = []
    for r in rows:
        rng = np.random.default_rng(SEED+int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16))
        y = torch.tensor(r['targets'])
        data.append((r, [torch.tensor(temporal_features(r['clean']['features'], center, scale))]+[
            [torch.tensor(temporal_features(x, center, scale)) for x in triplet] for triplet in r['train']],
            y, alternatives(r['pronunciation'], vocab, rng)))
    torch.manual_seed(SEED)
    models = [head(vocab)]
    models.extend(copy.deepcopy(models[0]) for _ in range(2))
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for m in models]
    rng = np.random.default_rng(SEED)
    saved = {}
    for epoch in range(EPOCHS):
        for i in rng.permutation(len(data)):
            r, views, targets, candidates = data[i]
            triplet = views[1+int(rng.integers(len(views)-1))]
            batch = [views[0]]+triplet
            for arm, (model, opt) in enumerate(zip(models, optimizers)):
                opt.zero_grad()
                lp = [model(x).log_softmax(-1) for x in batch]
                if epoch < 3:
                    loss = sum(-(targets*x).sum(-1).mean() for x in lp)/4
                else:
                    loss = sum(ctc_nll(x, [r['pronunciation']])[0]/len(r['pronunciation']) for x in lp)/4
                    if arm != 1:
                        loss = loss+CONSISTENCY*pronunciation_consistency(lp[1], lp[2 if arm == 0 else 3], candidates)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
        if epoch+1 in checkpoints:
            saved[epoch+1] = [copy.deepcopy(m).eval() for m in models]
    return saved, center, scale


def validation_nll(model, r, center, scale):
    import numpy as np
    import torch
    views = [r['clean']['features']]+[v for triplet in r['validation'] for v in triplet]
    with torch.no_grad():
        losses = [ctc_nll(model(torch.tensor(temporal_features(x, center, scale))).log_softmax(-1),
                         [r['pronunciation']])[0].item()/len(r['pronunciation']) for x in views]
    return float(np.mean(losses))


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    vocab = len(metadata['tokens'])+1
    rows, skipped = [], []
    for r in data:
        if r.get('source') != 'help-train' or r.get('excluded') is not False or not training_take(r, excluded):
            raise ValueError('Missing eligible none-labelled training provenance')
        ids = r['pronunciation']
        if not ids or any(p < 1 or p >= vocab for p in ids):
            raise ValueError('Invalid canonical phone sequence')
        views = [r['clean']['features']]+[v for group in ('train', 'validation') for triplet in r[group] for v in triplet]
        if len(r['train']) != 2 or len(r['validation']) != 2 or any(len(t) != 3 for g in ('train','validation') for t in r[g]):
            raise ValueError('Missing matched triplets')
        if any(np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != DIMENSIONS or not np.isfinite(x).all() for x in views):
            raise ValueError('Invalid frozen feature observations')
        if len({len(x) for x in views}) != 1:
            raise ValueError('Channel views changed the sample clock')
        targets = occupancies(r['clean']['logProbs'], ids)
        # All rival candidates must also complete; no per-arm/data-dependent omissions.
        if targets is None or len(targets) != len(views[0]) or len(targets) < len(ids)+sum(a == b for a,b in zip(ids,ids[1:]))+2:
            skipped.append(r['id'])
            continue
        rows.append(dict(r, targets=targets))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training identities')
    folds = speaker_folds(rows)
    risks = [{e: {} for e in CHECKPOINTS} for _ in range(3)]
    for fold in range(3):
        saved, center, scale = fit([r for r in rows if folds[r['speaker']] != fold], vocab)
        for epoch, models in saved.items():
            for arm, model in enumerate(models):
                for r in rows:
                    if folds[r['speaker']] == fold:
                        risks[arm][epoch].setdefault(r['speaker'], []).append(validation_nll(model, r, center, scale))
        print(json.dumps(dict(completedFold=fold)), flush=True)
    mean_risks = [{e:float(np.mean([np.mean(v) for v in speakers.values()])) for e,speakers in arm.items()} for arm in risks]
    if any(not np.isfinite(list(r.values())).all() for r in mean_risks):
        raise ValueError('Nonfinite training-validation loss')
    shared_epoch = min(CHECKPOINTS, key=lambda e: (sum(r[e] for r in mean_risks)/3, e))
    selected = [shared_epoch]*3
    saved, center, scale = fit(rows, vocab)
    metadata.update(center=center.tolist(), scale=scale.tolist(),
        **{name:serialize(saved[e][a]) for a,(name,e) in enumerate(zip(('transition','unpaired','fixed'),selected))})
    out = Path(args.out)
    asset = out/'channel-transition-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=sorted(folds), folds=folds, skipped=skipped,
        excludedSpeakers=sorted(excluded), seed=SEED, epochs=EPOCHS, consistency=CONSISTENCY,
        validation=mean_risks, selectedEpochs=dict(zip(('transition','unpaired','fixed'),selected)),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'channel-transition-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'channel-transition.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Missing frozen feature graph')
    (out/'channel-transition-asset.json').write_text(json.dumps(dict(schema=1, files=[dict(file=p.name,
        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-labels')
    p.add_argument('--input')
    p.add_argument('--metadata')
    p.add_argument('--excluded-speakers')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    prepare(args.prepare_labels, args.out) if args.prepare_labels else train(args)
