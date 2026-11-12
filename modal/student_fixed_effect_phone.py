"""Frozen acoustic readouts with training-only, centered recording effects.

Both arms see identical recordings, initialization, update order and exact
transcript CTC. The treatment additionally fits regularized recording offsets;
held-out scoring and the exported model use the shared readout alone. Scripted
substitution labels locate preservation controls, never performed transcripts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN = 32, 48
OFFSET_PENALTIES = (.01, .1, 1.)
CHECKPOINTS = (8, 16, 24)
WARMUP_EPOCHS = 4


def recording_offsets(raw):
    """Zero blank bias, zero-sum phones per take, zero recording mean per phone.

    Double centering fixes both softmax and shared-intercept identifiability.
    This operation stays in the gradient graph; no held-out recording is added.
    """
    import torch
    phones = raw - raw.mean(dim=1, keepdim=True)
    phones = phones - phones.mean(dim=0, keepdim=True)
    return torch.cat((phones.new_zeros((len(phones), 1)), phones), dim=1)


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(),
                               torch.nn.Linear(HIDDEN, vocab))


def transcript_loss(logits, ids):
    """Marginalize latent CTC frame assignments, including separating blanks.

    Normalize by transcript length so every training recording gets equal mass.
    Impossible alignments are errors rather than silently zeroed losses.
    """
    import torch
    return torch.nn.functional.ctc_loss(logits.log_softmax(-1).unsqueeze(1),
        torch.tensor(ids, dtype=torch.long), [len(logits)], [len(ids)],
        blank=0, reduction='sum', zero_infinity=False) / len(ids)


def speaker_folds(rows, speakers):
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    ordered = anchors + sorted(set(speakers) - set(anchors))
    return {speaker: i % 3 for i, speaker in enumerate(ordered)}


def prepare(data, excluded, vocab):
    """Reject mixed provenance and script supervision before fitting anything."""
    import numpy as np
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be present and unique')
    rows, controls, skipped = [], [], []
    for r in data:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or
            r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded):
            raise ValueError('Missing or ineligible training-speaker provenance')
        features = np.asarray(r['features'], dtype='float32')
        if features.ndim != 2 or features.shape[1] != DIMENSIONS or not np.isfinite(features).all():
            raise ValueError('Invalid frozen acoustic features')
        if r['kind'] == 'none':
            ids = r['pronunciation']
            if not ids or any(not isinstance(p, int) or p < 1 or p >= vocab for p in ids):
                raise ValueError('Invalid authorized transcript')
            lp = np.asarray(r['logProbs'], dtype='float32')
            if lp.shape != (len(features), vocab) or not np.isfinite(lp).all():
                raise ValueError('Invalid shipped initialization posteriors')
            if len(features) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
                skipped.append(r['id'])
            else:
                rows.append(dict(r, features=features, logProbs=lp))
        elif r['kind'] == 'substituted_word':
            for e in r['controls']:
                x = np.asarray(e['features'], dtype='float32')
                if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all():
                    raise ValueError('Invalid preservation features')
                if any(not s or any(not isinstance(p, int) or p < 1 or p >= vocab for p in s)
                       for s in (e['written'], e['rival'])):
                    raise ValueError('Invalid preservation pronunciation')
                controls.append(dict(e, speaker=r['speaker']))
        else:
            raise ValueError('Scripts cannot provide performed transcript supervision')
    return rows, controls, skipped


def fit(rows, vocab, penalty):
    """Fit paired arms; snapshots contain shared parameters and offset diagnostics."""
    import numpy as np
    import torch
    if not rows:
        raise ValueError('No eligible recordings in training fold')
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    xs = [torch.tensor((r['features']-center)/scale) for r in rows]
    initial = head(vocab)
    # Common initialization only: approximate the shipped readout on these same
    # training recordings. Subsequent supervision is exact transcript CTC.
    warm = torch.optim.AdamW(initial.parameters(), lr=.002, weight_decay=.0001)
    rng = np.random.default_rng(1729)
    for _ in range(WARMUP_EPOCHS):
        for i in rng.permutation(len(rows)):
            warm.zero_grad()
            target = torch.tensor(rows[i]['logProbs']).softmax(-1)
            loss = -(target * initial(xs[i]).log_softmax(-1)).sum(-1).mean()
            loss.backward()
            warm.step()
    control, treatment = copy.deepcopy(initial), copy.deepcopy(initial)
    offsets = torch.nn.Parameter(torch.zeros(len(rows), vocab-1))
    a = torch.optim.AdamW(control.parameters(), lr=.001, weight_decay=.0001)
    b = torch.optim.AdamW([*treatment.parameters(), offsets], lr=.001, weight_decay=.0001)
    snapshots = {}
    for epoch in range(1, max(CHECKPOINTS)+1):
        for i in rng.permutation(len(rows)):
            a.zero_grad()
            ordinary = transcript_loss(control(xs[i]), rows[i]['pronunciation'])
            b.zero_grad()
            centered = recording_offsets(offsets)
            fixed = transcript_loss(treatment(xs[i])+centered[i], rows[i]['pronunciation'])
            joint = fixed + penalty * centered.square().mean()
            if not torch.isfinite(ordinary) or not torch.isfinite(joint):
                raise ValueError('Nonfinite latent-assignment CTC objective')
            ordinary.backward()
            joint.backward()
            torch.nn.utils.clip_grad_norm_(control.parameters(), 5.)
            torch.nn.utils.clip_grad_norm_([*treatment.parameters(), offsets], 5.)
            a.step()
            b.step()
        if epoch in CHECKPOINTS:
            effect = recording_offsets(offsets).detach()
            snapshots[epoch] = (copy.deepcopy(control).eval(), copy.deepcopy(treatment).eval(),
                dict(recordings=len(rows), rms=float(effect.square().mean().sqrt()),
                     maxRecordingMean=float(effect.mean(0).abs().max()),
                     maxPhoneSum=float(effect[:, 1:].sum(1).abs().max())))
    return snapshots, center, scale


def outputs(model, features, center, scale):
    import numpy as np
    import torch
    # There is deliberately no recording identity or offset argument here.
    with torch.no_grad():
        return model(torch.tensor((np.asarray(features, dtype='float32')-center)/scale)).log_softmax(-1)


def sequence_score(log_probs, ids):
    """Exact complete CTC log likelihood; matches the SDK forward algorithm."""
    import torch
    if not ids or len(log_probs) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return float('-inf')
    lp = torch.as_tensor(log_probs, dtype=torch.float32)
    return -float(torch.nn.functional.ctc_loss(lp.unsqueeze(1), torch.tensor(ids),
        [len(lp)], [len(ids)], reduction='sum', zero_infinity=False))


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    rows, controls, skipped = prepare(data, excluded, len(metadata['tokens'])+1)
    folds = speaker_folds(rows, {r['speaker'] for r in data})
    vocab = len(metadata['tokens'])+1
    validation = []
    for penalty in OFFSET_PENALTIES:
        collected = {epoch: dict(errors={}, controlErrors={}, preservation=[]) for epoch in CHECKPOINTS}
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            snapshots, center, scale = fit(training, vocab, penalty)
            for epoch, (control, treatment, _) in snapshots.items():
                v = collected[epoch]
                for r in rows:
                    if folds[r['speaker']] != fold:
                        continue
                    for h, field in ((treatment, 'errors'), (control, 'controlErrors')):
                        loss = -sequence_score(outputs(h, r['features'], center, scale), r['pronunciation']) / len(r['pronunciation'])
                        v[field].setdefault(r['speaker'], []).append(loss)
                for e in controls:
                    if folds[e['speaker']] != fold:
                        continue
                    margins = []
                    for h in (control, treatment):
                        lp = outputs(h, e['features'], center, scale)
                        margins.append(sequence_score(lp, e['written'])-sequence_score(lp, e['rival']))
                    if all(np.isfinite(margins)):
                        v['preservation'].append(margins[0] < 0 and margins[1] > 0)
        for epoch, v in collected.items():
            result = dict(offsetPenalty=penalty, epochs=epoch,
                sharedSpeakerCtcNll=float(np.mean([np.mean(x) for x in v['errors'].values()])),
                pooledSpeakerCtcNll=float(np.mean([np.mean(x) for x in v['controlErrors'].values()])),
                preservationComparisons=len(v['preservation']), preservationReversals=sum(v['preservation']))
            validation.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No real training substitution preservation comparisons available')
    selected = min(validation, key=lambda v: (
        v['preservationReversals']/max(1, v['preservationComparisons']),
        v['sharedSpeakerCtcNll'], v['epochs'], v['offsetPenalty']))
    snapshots, center, scale = fit(rows, vocab, selected['offsetPenalty'])
    control, treatment, diagnostics = snapshots[selected['epochs']]
    metadata.update(center=center.tolist(), scale=scale.tolist(),
                    fixedEffect=serialize(treatment), control=serialize(control))
    out = Path(args.out)
    asset = out/'fixed-effect-phone-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), trainingRecordings=[r['id'] for r in rows], folds=folds,
        controls=len(controls), excludedSpeakers=sorted(excluded), impossibleAlignments=skipped,
        validation=validation, selected=selected, offsetDiagnostics=diagnostics,
        offsetParametersExported=False, warmupEpochs=WARMUP_EPOCHS,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'fixed-effect-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'fixed-effect-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'fixed-effect-phone-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
