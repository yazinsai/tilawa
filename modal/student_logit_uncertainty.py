"""Transcript-marginalized logistic-normal readout on a frozen encoder.

No practice audio, labels or incident identities enter training or selection.
An independent latent is drawn per absolute frame block and shared within it.
The deterministic control has identical parameters and uses every output channel.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

DIMENSIONS, HIDDEN, WARMUP, EPOCHS = 32, 48, 4, 6
RANKS, EXTENTS, REGULARIZATION = (1, 2), (4, 8), (.001, .01)
SEED, ORDER = 1729, 5


def quadrature(rank, order=ORDER):
    import numpy as np
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    points = list(itertools.product(range(order), repeat=rank))
    return (np.asarray([[nodes[i]*np.sqrt(2) for i in p] for p in points]),
            np.asarray([np.prod([weights[i]/np.sqrt(np.pi) for i in p]) for p in points]))


def possible(frames, ids):
    return bool(ids) and frames >= len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def sequence_likelihood(mean, loadings, ids, extent, from_frame=0, order=ORDER):
    """Differentiable complete CTC forward likelihood with blockwise integration.

    Finite unreachable sentinels avoid undefined gradients of logsumexp(-inf).
    Actual impossible sequences are rejected before training or comparison.
    Integrating boundary alphas sums all independent block-latent combinations.
    """
    import torch
    labels = torch.tensor([0]+[v for p in ids for v in (p, 0)], device=mean.device)
    n = len(labels)
    neg = -1e30
    if loadings is None:
        z, weight = mean.new_zeros((1, 0)), mean.new_ones(1)
    else:
        nodes, weights = quadrature(loadings.shape[1], order)
        z, weight = mean.new_tensor(nodes), mean.new_tensor(weights)
    alpha = mean.new_full((n,), neg)
    alpha[0] = 0
    skip = torch.zeros(n, dtype=torch.bool, device=mean.device)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    start = 0
    while start < len(mean):
        stop = min(len(mean), start+extent-(from_frame+start) % extent)
        logits = mean[start:stop].unsqueeze(0).expand(len(z), -1, -1)
        if loadings is not None:
            logits = logits + torch.einsum('qd,tdv->qtv', z, loadings[start:stop])
        emissions = logits.log_softmax(-1)
        local = alpha.unsqueeze(0).expand(len(z), -1)
        for t in range(stop-start):
            one = torch.cat((local.new_full((len(z), 1), neg), local[:, :-1]), 1)
            two = torch.cat((local.new_full((len(z), 2), neg), local[:, :-2]), 1)
            two = torch.where(skip, two, neg)
            local = torch.logsumexp(torch.stack((local, one, two)), 0) + emissions[:, t, labels]
        alpha = torch.logsumexp(local+weight.log().unsqueeze(1), 0)
        start = stop
    return torch.logsumexp(alpha[-2:], 0)


def head(vocab, rank):
    import torch
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(),
                                torch.nn.Linear(HIDDEN, (rank+1)*vocab))
    with torch.no_grad():
        model[2].weight[vocab:].normal_(0, .01)
        model[2].bias[vocab:].zero_()
    return model


def outputs(model, x, vocab, rank, deterministic=False):
    import torch
    raw = model(x)
    mean = raw[:, :vocab]
    loading = raw[:, vocab:].reshape(-1, rank, vocab)
    if deterministic:
        # Equal capacity: all rank*V extra channels participate in the control.
        return mean + torch.tanh(loading).sum(1)/(rank**.5), None
    return mean, loading


def folds_for(rows, speakers):
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('At least three eligible none-labelled training speakers required')
    ordered = anchors + sorted(set(speakers)-set(anchors))
    return {s: i % 3 for i, s in enumerate(ordered)}


def fit(rows, vocab, rank, extent, regularization):
    import numpy as np
    import torch
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    inputs = [torch.tensor((np.asarray(r['features'], dtype='float32')-center)/scale) for r in rows]
    initial = head(vocab, rank)
    opt = torch.optim.AdamW(initial.parameters(), lr=.002, weight_decay=regularization)
    rng = np.random.default_rng(SEED)
    # Initialization only: the same parent distributions and updates for both arms.
    # Final fitting/selection use canonical full-transcript likelihood, not forced labels.
    for _ in range(WARMUP):
        for i in rng.permutation(len(rows)):
            lp = torch.tensor(rows[i]['logProbs'], dtype=torch.float32)
            y = lp.softmax(-1)
            opt.zero_grad()
            loss = -(y*initial(inputs[i])[:, :vocab].log_softmax(-1)).sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(initial.parameters(), 5.)
            opt.step()
    control, treatment = copy.deepcopy(initial), copy.deepcopy(initial)
    opts = [torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=regularization) for m in (control, treatment)]
    for _ in range(EPOCHS):
        for i in rng.permutation(len(rows)):
            r, x = rows[i], inputs[i]
            for model, optimizer in zip((control, treatment), opts):
                optimizer.zero_grad()
                mean, loading = outputs(model, x, vocab, rank, model is control)
                nll = -sequence_likelihood(mean, loading, r['pronunciation'], extent)/len(r['pronunciation'])
                residual = model(x)[:, vocab:] if loading is None else loading
                penalty = regularization*residual.square().mean()
                loss = nll + penalty
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite transcript-marginalized training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
    return control.eval(), treatment.eval(), center, scale


def score(model, raw, ids, center, scale, vocab, rank, extent, deterministic=False, order=ORDER, mean_only=False, from_frame=0):
    import numpy as np
    import torch
    if not possible(len(raw), ids):
        return None
    with torch.no_grad():
        x = torch.tensor((np.asarray(raw, dtype='float32')-center)/scale)
        mean, loading = outputs(model, x, vocab, rank, deterministic)
        return float(sequence_likelihood(mean, None if mean_only else loading, ids, extent, from_frame=from_frame, order=order))


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def validate_rows(data, metadata, excluded):
    import numpy as np
    vocab = len(metadata['tokens'])+1
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be present and unique')
    for r in data:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False
                or not r.get('speaker') or r['speaker'] in excluded):
            raise ValueError('Missing or ineligible training-speaker provenance')
        if r.get('kind') not in ('none', 'substituted_word'):
            raise ValueError('Only clean transcripts and weak substitution controls are authorized')
        f = np.asarray(r['features'])
        if f.ndim != 2 or not len(f) or f.shape[1] != DIMENSIONS or not np.isfinite(f).all():
            raise ValueError('Invalid native frozen features')
        if r['kind'] == 'none':
            lp, ids = np.asarray(r['logProbs']), r['pronunciation']
            if lp.shape != (len(f), vocab) or not np.isfinite(lp).all():
                raise ValueError('Parent posterior and feature clocks differ')
            if not ids or any(not isinstance(p, int) or p < 1 or p >= vocab for p in ids):
                raise ValueError('Invalid canonical complete pronunciation')
        else:
            for c in r['controls']:
                cf = np.asarray(c['features'])
                if (cf.ndim != 2 or cf.shape[1] != DIMENSIONS or not np.isfinite(cf).all()
                        or not isinstance(c.get('fromFrame'), int) or c['fromFrame'] < 0):
                    raise ValueError('Invalid preservation features')
                for ids in (c['written'], c['rival']):
                    if not ids or any(not isinstance(p, int) or p < 1 or p >= vocab for p in ids):
                        raise ValueError('Invalid complete preservation pronunciation')


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    validate_rows(data, metadata, excluded)
    anchors = [r for r in data if r['kind'] == 'none']
    rows = [r for r in anchors if possible(len(r['features']), r['pronunciation'])]
    skipped = [r['id'] for r in anchors if r not in rows]
    speakers = sorted({r['speaker'] for r in data})
    folds = folds_for(rows, speakers)
    controls = [dict(c, speaker=r['speaker']) for r in data if r['kind'] == 'substituted_word' for c in r['controls']]
    vocab = len(metadata['tokens'])+1
    validation = []
    for rank, extent, reg in itertools.product(RANKS, EXTENTS, REGULARIZATION):
        errors, reversals, stability = {}, [], []
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            control, treatment, center, scale = fit(training, vocab, rank, extent, reg)
            for r in rows:
                if folds[r['speaker']] != fold:
                    continue
                args_score = (r['features'], r['pronunciation'], center, scale, vocab, rank, extent)
                a, b = score(control, *args_score, deterministic=True), score(treatment, *args_score)
                dense = score(treatment, *args_score, order=7)
                errors.setdefault(r['speaker'], []).append([-a/len(r['pronunciation']), -b/len(r['pronunciation'])])
                stability.append(abs(b-dense)/len(r['pronunciation']))
            for c in controls:
                if folds[c['speaker']] != fold:
                    continue
                values = [score(treatment, c['features'], ids, center, scale, vocab, rank, extent,
                                mean_only=mean_only, from_frame=c['fromFrame'])
                          for mean_only in (True, False) for ids in (c['written'], c['rival'])]
                if all(v is not None for v in values):
                    reversals.append(values[0] < values[1] and values[2] > values[3])
        speaker_nll = np.asarray([np.mean(v, axis=0) for v in errors.values()]).mean(0)
        row = dict(rank=rank, extent=extent, regularization=reg,
                   controlNll=float(speaker_nll[0]), integratedNll=float(speaker_nll[1]),
                   pairedNllChange=float(speaker_nll[1]-speaker_nll[0]),
                   preservationComparisons=len(reversals), preservationReversals=sum(reversals),
                   quadratureMaxNllError=float(max(stability)))
        validation.append(row)
        print(json.dumps(row), flush=True)
    # All choices use held-out training speakers only. No practice comparison may tune them.
    # Prefer preservation, then improvement over the equally parameterized control.
    selected = min(validation, key=lambda r: (r['preservationReversals']/max(1, r['preservationComparisons']),
                                               r['pairedNllChange'], r['integratedNll'], r['rank'], r['extent'], r['regularization']))
    # Five points per axis are fixed before selection; seven provide a diagnostic.
    order = ORDER
    control, treatment, center, scale = fit(rows, vocab, selected['rank'], selected['extent'], selected['regularization'])
    metadata.update(center=center.tolist(), scale=scale.tolist(), rank=selected['rank'], extent=selected['extent'],
                    quadrature=order, uncertainty=serialize(treatment), control=serialize(control))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'logit-uncertainty-heads.json'
    report = dict(recordings=len(rows), speakers=speakers, excludedSpeakers=sorted(excluded), folds=folds,
                  impossibleTranscripts=skipped, preservationControls=len(controls), validation=validation,
                  selected=selected, quadrature=order, warmupEpochs=WARMUP, epochs=EPOCHS,
                  inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    asset.write_text(json.dumps(metadata, allow_nan=False))
    (out/'logit-uncertainty-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'logit-uncertainty.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Bundle lacks frozen feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'logit-uncertainty-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
