"""Frozen temporal readouts with correctly labelled crossed acoustic intervals."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN, RADIUS, EPOCHS = 32, 48, 3, 18

def occupancies(log_probs, ids):
    """Exact forward/backward soft CTC state and phone occupancies.

    Context keys describe surrounding transcript phones of the most probable
    state (including its central identity), while supervision retains the
    entire posterior distribution.
    """
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or not np.isfinite(lp).all() or any(p <= 0 or p >= lp.shape[1] for p in ids):
        raise ValueError('Invalid canonical transcript or posterior rows')
    labels = np.array([0] + [p for i in ids for p in (i, 0)])
    if not ids or len(lp) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
        return None
    n, t = len(labels), len(lp)
    incoming = [[s] + ([s-1] if s else []) +
                ([s-2] if s > 1 and labels[s] and labels[s] != labels[s-2] else []) for s in range(n)]
    outgoing = [[] for _ in labels]
    for s, edges in enumerate(incoming):
        for before in edges:
            outgoing[before].append(s)
    alpha = np.full((t, n), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for f in range(1, t):
        for s in range(n):
            alpha[f, s] = np.logaddexp.reduce(alpha[f-1, incoming[s]]) + lp[f, labels[s]]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full((t, n), -np.inf)
    beta[-1, -2:] = 0
    for f in range(t-2, -1, -1):
        for s in range(n):
            nxt = outgoing[s]
            beta[f, s] = np.logaddexp.reduce(beta[f+1, nxt] + lp[f+1, labels[nxt]])
    states = np.exp(alpha + beta - z)
    states /= states.sum(1, keepdims=True)
    targets = np.zeros_like(lp)
    for s, phone in enumerate(labels):
        targets[:, phone] += states[:, s]
    context = []
    for s in states.argmax(1):
        # Blank intervals use their immediately adjacent phones. Articulations
        # use the previous/next articulation, including genuine doubled phones.
        pos = (int(s)-1)//2
        if s % 2:
            context.append((ids[pos-1] if pos else 0, ids[pos], ids[pos+1] if pos+1 < len(ids) else 0))
        else:
            context.append((ids[s//2-1] if s else 0, 0, ids[s//2] if s//2 < len(ids) else 0))
    return targets.astype('float32'), context


def sequence_score(lp, ids):
    """Same complete CTC marginal and normalized timing prior as the SDK."""
    import numpy as np
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
             ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else []) for s, p in enumerate(labels)]
    a = np.full(len(labels), -np.inf)
    prior = a.copy()
    a[:2] = lp[0, labels[:2]]
    prior[:2] = 0
    for frame in lp[1:]:
        b, q = np.full_like(a, -np.inf), np.full_like(a, -np.inf)
        for s, nxt in enumerate(edges):
            cost = np.log(len(nxt))
            for j in nxt:
                b[j] = np.logaddexp(b[j], a[s]-cost)
                q[j] = np.logaddexp(q[j], prior[s]-cost)
        a, prior = b + frame[labels], q
    likelihood, timing = np.logaddexp.reduce(a[-2:]), np.logaddexp.reduce(prior[-2:])
    return float(likelihood-timing) if np.isfinite(likelihood) and np.isfinite(timing) else -np.inf


def outputs(model, features, center, scale):
    import numpy as np
    import torch
    with torch.no_grad():
        return model(torch.tensor(temporal((np.asarray(features, dtype='float32')-center)/scale))).log_softmax(-1).numpy()


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def alignment(log_probs, ids):
    """Forward table for exact posterior path sampling, including repeated phones."""
    import numpy as np
    lp = np.asarray(log_probs, dtype=np.float64)
    if not ids or lp.ndim != 2 or not np.isfinite(lp).all() or any(p <= 0 or p >= lp.shape[1] for p in ids):
        raise ValueError('Invalid transcript or posterior rows')
    labels = np.array([0] + [v for p in ids for v in (p, 0)])
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = np.full((len(lp), len(labels)), -np.inf)
    if not len(lp):
        return None
    alpha[0, :2] = lp[0, labels[:2]]
    for t in range(1, len(lp)):
        previous = alpha[t-1]
        incoming = previous.copy()
        incoming[1:] = np.logaddexp(incoming[1:], previous[:-1])
        incoming[2:] = np.logaddexp(incoming[2:], np.where(skip, previous[:-2], -np.inf))
        alpha[t] = incoming + lp[t, labels]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        return None
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0
    for t in range(len(lp)-2, -1, -1):
        following = beta[t+1] + lp[t+1, labels]
        beta[t] = following
        beta[t, :-1] = np.logaddexp(beta[t, :-1], following[1:])
        beta[t, :-2] = np.logaddexp(beta[t, :-2], np.where(skip, following[2:], -np.inf))
    states = np.exp(alpha + beta - z)
    targets = np.zeros_like(lp)
    for s, p in enumerate(labels):
        targets[:, p] += states[:, s]
    return dict(alpha=alpha, labels=labels, targets=targets.astype('float32'))


def sample_path(aligned, rng):
    """Backward sampling from the complete transcript-conditioned CTC posterior.

    No Viterbi or argmax phone boundaries: new paths are drawn each epoch.
    """
    import numpy as np
    alpha, labels = aligned['alpha'], aligned['labels']

    def draw(candidates, scores):
        probability = np.exp(scores - np.logaddexp.reduce(scores))
        return int(rng.choice(candidates, p=probability))

    state = draw([len(labels)-2, len(labels)-1], alpha[-1, -2:])
    path = np.empty(len(alpha), dtype=np.int32)
    for t in range(len(alpha)-1, -1, -1):
        path[t] = state
        if t:
            incoming = [state] + ([state-1] if state else [])
            if state > 1 and labels[state] and labels[state] != labels[state-2]:
                incoming.append(state-2)
            state = draw(incoming, alpha[t-1, incoming])
    return path


def temporal(features):
    """Same seven-frame readout input and zero padding as SDK inference."""
    import numpy as np
    x = np.asarray(features, dtype='float32')
    padded = np.pad(x, ((RADIUS, RADIUS), (0, 0)))
    return np.concatenate([padded[i:i+len(x)] for i in range(2*RADIUS+1)], axis=1)


def occurrences(rows, tokens, rng):
    """Two-phone flanks around each consonant, keeping intervening blank states."""
    import numpy as np
    # Quran inventory letters are consonants; harakat, madd and pause tokens are not.
    consonants = {i+1 for i, p in enumerate(tokens) if p in 'ءبتثجحخدذرزسشصضطظعغفقكلمنهويں۾'}
    result = []
    for r in rows:
        path = sample_path(r['aligned'], rng)
        labels = r['aligned']['labels'][path]
        for pos, phone in enumerate(r['pronunciation']):
            if phone not in consonants:
                continue
            at = np.flatnonzero(path == 2*pos+1)
            if not len(at):
                raise ValueError('Sampled path skipped a required articulation')
            a, b = int(at[0]), int(at[-1])+1
            region = np.flatnonzero((path >= max(0, 2*(pos-2))) &
                                    (path <= min(len(r['aligned']['labels'])-1, 2*(pos+3))))
            start, end = int(region[0]), int(region[-1])+1
            left = r['pronunciation'][pos-1] if pos else 0
            right = r['pronunciation'][pos+1] if pos+1 < len(r['pronunciation']) else 0
            result.append(dict(recording=r['id'], speaker=r['speaker'], phone=phone, context=(left, right),
                parts=[(r['x'][start:a], labels[start:a]), (r['x'][a:b], labels[a:b]),
                       (r['x'][b:end], labels[b:end])]))
    return result


def donors(items, origin, mode):
    """Cross-recording only; flanks match central phone, cores match vowel/phone context."""
    return [j for j, other in enumerate(items) if other['recording'] != origin['recording'] and
            ((mode == 'context' and other['phone'] == origin['phone'] and other['context'] != origin['context']) or
             (mode == 'central' and other['phone'] != origin['phone'] and other['context'] == origin['context']) or
             mode == 'unstructured')]


def compose(origin, donor, mode):
    """Supervise the material actually used, never the unchanged source transcript."""
    import numpy as np
    if mode == 'context':
        parts = [donor['parts'][0], origin['parts'][1], donor['parts'][2]]
    elif mode == 'central':
        parts = [origin['parts'][0], donor['parts'][1], origin['parts'][2]]
    else:
        raise ValueError('Unknown crossed operation')
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    if not len(x) or len(x) != len(y):
        raise ValueError('Empty or mismatched feature composition')
    return temporal(x), y


def donor_index(items):
    groups, phones, contexts = {}, {}, {}
    for i, item in enumerate(items):
        groups.setdefault((item['phone'], item['context']), {}).setdefault(item['recording'], []).append(i)
    for k in groups:
        phones.setdefault(k[0], []).append(k)
        contexts.setdefault(k[1], []).append(k)
    return dict(groups=groups, phones=phones, contexts=contexts)


def draw_donor(index, origin, mode, rng):
    """Choose a eligible group then recording then interval without an all-pairs table."""
    groups = index['groups']
    candidates = (index['phones'].get(origin['phone'], []) if mode == 'context' else
                  index['contexts'].get(origin['context'], []) if mode == 'central' else list(groups))
    keys = [k for k in candidates if any(r != origin['recording'] for r in groups[k]) and
            (mode == 'unstructured' or
             (mode == 'context' and k[0] == origin['phone'] and k[1] != origin['context']) or
             (mode == 'central' and k[0] != origin['phone'] and k[1] == origin['context']))]
    if not keys:
        return None
    key = keys[int(rng.integers(len(keys)))]
    records = sorted(r for r in groups[key] if r != origin['recording'])
    record = records[int(rng.integers(len(records)))]
    return int(rng.choice(groups[key][record]))


def head(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS*(2*RADIUS+1), HIDDEN),
                               torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def speaker_folds(rows, speakers):
    anchors = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(anchors) < 3:
        raise ValueError('At least three none-labelled training speakers required')
    ordered = anchors + sorted(set(speakers)-set(anchors))
    return {s: i % 3 for i, s in enumerate(ordered)}


def fit(rows, vocab, tokens, epochs):
    """Matched initialization, updates, original frames and augmented frame count."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    rows = [dict(r, x=(np.asarray(r['features'], dtype='float32')-center)/scale) for r in rows]
    x = np.concatenate([temporal(r['x']) for r in rows])
    y = np.concatenate([r['aligned']['targets'] for r in rows])
    weights = np.concatenate([np.full(len(r['x']), 1/len(r['x'])) for r in rows])
    weights /= weights.sum()
    ordinary = head(vocab)
    models = [ordinary, copy.deepcopy(ordinary), copy.deepcopy(ordinary)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01) for m in models]
    rng = np.random.default_rng(1729)
    count = {'context': 0, 'central': 0}
    for epoch in range(epochs):
        items = occurrences(rows, tokens, rng)
        index = donor_index(items)
        original_labels = np.concatenate([r['aligned']['labels'][sample_path(r['aligned'], rng)] for r in rows])
        pairs = [(i, j, mode) for i, origin in enumerate(items) for mode in ('context', 'central')
                 for j in [draw_donor(index, origin, mode, rng)] if j is not None]
        if not any(p[2] == 'central' for p in pairs) or not any(p[2] == 'context' for p in pairs):
            raise ValueError('Both same-phone context exchanges and different-phone context-matched core exchanges required')
        crossed, mixing = [], []
        for i, j, mode in pairs:
            crossed.append(compose(items[i], items[j], mode))
            mixing.append(compose(items[i], items[draw_donor(index, items[i], 'unstructured', rng)], mode))
            count[mode] += 1
        # Equal recording mass for pairs, independent of take length.
        from collections import Counter
        mass = Counter(items[i]['recording'] for i, _, _ in pairs)
        pair_weights = np.array([1/mass[items[i]['recording']] for i, _, _ in pairs])
        pair_weights /= pair_weights.sum()
        for _ in range(max(1, len(x)//256)):
            original = rng.choice(len(x), 256, p=weights)
            chosen = rng.choice(len(pairs), 256, p=pair_weights)
            augmented = []
            # Equal 256 classified frames, even when joins have different durations.
            for examples in (crossed, mixing):
                frames = [int(rng.integers(len(examples[k][0]))) for k in chosen]
                augmented.append((np.array([examples[k][0][t] for k, t in zip(chosen, frames)]),
                                  np.array([examples[k][1][t] for k, t in zip(chosen, frames)])))
            endpoint = rng.choice(len(x), 256, p=weights)
            for arm, (m, opt) in enumerate(zip(models, optimizers)):
                opt.zero_grad()
                main = -(torch.tensor(y[original])*m(torch.tensor(x[original])).log_softmax(-1)).sum(-1).mean()
                if arm == 0:
                    extra = torch.nn.functional.cross_entropy(m(torch.tensor(x[endpoint])),
                        torch.tensor(original_labels[endpoint], dtype=torch.long))
                else:
                    features, labels = augmented[arm-1]
                    extra = torch.nn.functional.cross_entropy(m(torch.tensor(features)), torch.tensor(labels, dtype=torch.long))
                ((main+extra)/2).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
                opt.step()
        print(json.dumps(dict(epoch=epoch+1, pairs=len(pairs), operations=count)), flush=True)
    return [m.eval() for m in models], center, scale, count


def train(args):
    import numpy as np
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    source = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if any(r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or
           not r.get('speaker') or r['speaker'] in excluded for r in source):
        raise ValueError('Only authorized training-speaker recordings may fit or select heads')
    rows, impossible = [], []
    for r in source:
        if r['kind'] != 'none':
            continue
        found = alignment(r['logProbs'], r['pronunciation'])
        if found is None:
            impossible.append(r['id'])
        else:
            rows.append(dict(r, aligned=found))
    speakers = sorted({r['speaker'] for r in source})
    folds = speaker_folds(rows, speakers)
    controls = [dict(c, speaker=r['speaker']) for r in source for c in r.get('controls', [])]
    vocab = len(metadata['tokens'])+1
    for c in controls:
        if any(not s or any(p < 1 or p >= vocab for p in s) for s in (c['written'], c['rival'])):
            raise ValueError('Invalid complete preservation pronunciation')
    validation = []
    # Fixed capacity and regularization; checkpoint selection uses training speakers only.
    for epochs in (6, 12, 18):
        errors, reversals = {}, []
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            models, center, scale, _ = fit(training, vocab, metadata['tokens'], epochs)
            for r in rows:
                if folds[r['speaker']] == fold:
                    lp = outputs(models[1], r['features'], center, scale)
                    errors.setdefault(r['speaker'], []).append(float(-(r['aligned']['targets']*lp).sum(1).mean()))
            for c in controls:
                if folds[c['speaker']] != fold:
                    continue
                lp = [outputs(m, c['features'], center, scale) for m in models]
                margins = [sequence_score(p, c['written'])-sequence_score(p, c['rival']) for p in lp]
                if all(np.isfinite(margins)):
                    reversals.append(margins[0] < 0 and margins[1] > 0 and margins[2] < 0)
        validation.append(dict(epochs=epochs, meanSpeakerSoftNll=float(np.mean([np.mean(v) for v in errors.values()])),
            preservationComparisons=len(reversals), preservationReversals=sum(reversals)))
        print(json.dumps(validation[-1]), flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No actual scripted-substitution engine comparisons for preservation selection')
    selected = min(validation, key=lambda v: (v['preservationReversals']/max(1, v['preservationComparisons']),
                                               v['meanSpeakerSoftNll'], v['epochs']))
    models, center, scale, count = fit(rows, vocab, metadata['tokens'], selected['epochs'])
    metadata.update(radius=RADIUS, center=center.tolist(), scale=scale.tolist(),
                    control=serialize(models[0]), crossed=serialize(models[1]), mixing=serialize(models[2]))
    out = Path(args.out)
    asset = out/'crossed-context-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=speakers, folds=folds, excludedSpeakers=sorted(excluded),
        impossibleAlignments=impossible, validation=validation, selected=selected, operations=count,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'crossed-context-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'crossed-context.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Export lacks frozen feature graph')
    descriptor = dict(schema=1, files=[dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'crossed-context-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
