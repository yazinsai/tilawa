"""Frozen acoustic-head gradient boosting. No practice inputs enter training.

Preparation replays exact deployment features. A shared shallow regression tree
fits occupancy - posterior at every round, then adds its vector output to frozen
logits. Speaker-fold CTC plus shipped-output preservation selects depth, rate,
and stopping; mistake labels are never transcribed as canonical speech.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def log_probs(logits):
    import numpy as np
    z = np.asarray(logits, dtype=np.float64)
    maximum = z.max(axis=1, keepdims=True)
    return z - maximum - np.log(np.exp(z-maximum).sum(axis=1, keepdims=True))


def ctc(logits, ids):
    """Log forward/backward sum and negative NLL gradient, including repeated labels."""
    import numpy as np
    lp = log_probs(logits)
    if not ids or min(ids) < 1 or max(ids) >= lp.shape[1]:
        raise ValueError('Invalid transcript')
    labels = np.array([0] + [v for p in ids for v in (p, 0)], dtype=int)
    t, s = len(lp), len(labels)
    if t == 0:
        raise ValueError('Empty observation')
    alpha = np.full((t, s), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    skip = np.array([False, False] + [labels[i] != 0 and labels[i] != labels[i-2] for i in range(2, s)])
    for i in range(1, t):
        a = alpha[i-1]
        v = np.logaddexp(a, np.r_[-np.inf, a[:-1]])
        v = np.logaddexp(v, np.where(skip, np.r_[-np.inf, -np.inf, a[:-2]], -np.inf))
        alpha[i] = v + lp[i, labels]
    likelihood = np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if not np.isfinite(likelihood):
        raise ValueError('Transcript has no complete CTC path')
    beta = np.full((t, s), -np.inf)
    beta[-1, -2:] = 0
    for i in range(t-2, -1, -1):
        b = beta[i+1] + lp[i+1, labels]
        v = np.logaddexp(b, np.r_[b[1:], -np.inf])
        beta[i] = np.logaddexp(v, np.r_[np.where(skip[2:], b[2:], -np.inf), -np.inf, -np.inf])
    occupancy = np.zeros_like(lp)
    state = np.exp(alpha + beta - likelihood)
    for j, p in enumerate(labels):
        occupancy[:, p] += state[:, j]
    np.testing.assert_allclose(occupancy.sum(axis=1), 1, atol=1e-7)
    return float(likelihood), occupancy - np.exp(lp)


def tree_json(estimator):
    t = estimator.tree_
    return dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(),
        left=t.children_left.tolist(), right=t.children_right.tolist(), value=t.value[:, :, 0].tolist())


def predict_tree(tree, x):
    import numpy as np
    # sklearn converts features to float32; preparation preserves that representation.
    x = np.asarray(x, dtype=np.float32)
    values = []
    for row in x:
        node = 0
        while tree['left'][node] != -1:
            node = tree['left'][node] if row[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
        values.append(tree['value'][node])
    return np.asarray(values)


def speaker_folds(rows, count=3):
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(speakers) < count:
        raise ValueError('Insufficient training-speaker groups')
    return [{s for i, s in enumerate(speakers) if i % count == f} for f in range(count)]


def observations(path, metadata, excluded):
    import numpy as np
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('split') != 'train' or not r.get('speaker') or r['speaker'] in excluded or r.get('excluded'):
            raise ValueError('Unauthorized training recording')
        if r['role'] == 'transcript':
            if r.get('kind') != 'none' or r.get('events') or r.get('non_errors'):
                raise ValueError('Mistake or repeat cannot supply canonical supervision')
            r['ids'] = [metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        elif r['role'] != 'preservation' or r.get('kind') == 'none':
            raise ValueError('Invalid preservation role')
        r['x'] = np.asarray(r['features'], dtype=np.float32)
        r['z'] = np.asarray(r['logits'], dtype=np.float64)
        if r['x'].ndim != 2 or r['z'].ndim != 2 or len(r['x']) != len(r['z']) or not len(r['x']) or r['x'].shape[1] != metadata['width'] or r['z'].shape[1] != metadata['dimensions'] or not np.isfinite(r['x']).all() or not np.isfinite(r['z']).all():
            raise ValueError('Invalid frozen observation geometry')
        if r['role'] == 'transcript':
            ctc(r['z'], r['ids'])
        rows.append(r)
    if not any(r['role'] == 'preservation' for r in rows):
        raise ValueError('Missing mistake-labelled shipped-output preservation controls')
    return rows


def metrics(rows, z):
    import numpy as np
    speakers = sorted({r['speaker'] for r in rows})
    losses, kls, drifts = [], [], []
    for speaker in speakers:
        supervised, control, drift = [], [], []
        for r, current in zip(rows, z):
            if r['speaker'] != speaker:
                continue
            if r['role'] == 'transcript':
                now, _ = ctc(current, r['ids'])
                baseline, _ = ctc(r['z'], r['ids'])
                supervised.append((baseline-now)/len(current))
            else:
                base = log_probs(r['z'])
                control.append(float(np.mean(np.sum(np.exp(base)*(base-log_probs(current)), axis=1))))
                drift.append(float(np.mean(current.argmax(axis=1) != r['z'].argmax(axis=1))))
        if supervised:
            losses.append(np.mean(supervised))
        if control:
            kls.append(np.mean(control))
            drifts.append(np.mean(drift))
    if not losses or not kls:
        raise ValueError('Each held speaker partition needs transcript and mistake controls')
    # Fixed training objective; neither score nor targets choose this tradeoff.
    return dict(objective=float(np.mean(losses)+5*np.mean(kls)), ctcChange=float(np.mean(losses)),
        preservationKL=float(np.mean(kls)), preservationArgmaxDrift=float(np.mean(drifts)))


def boost(train, validation, depth, rate, rounds):
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    supervised = [r for r in train if r['role'] == 'transcript']
    if not supervised:
        raise ValueError('Empty fitting partition')
    x = np.concatenate([r['x'] for r in supervised])
    counts = {s: sum(r['speaker'] == s for r in supervised) for s in {r['speaker'] for r in supervised}}
    weights = np.concatenate([np.full(len(r['x']), 1/(len(r['x'])*counts[r['speaker']])) for r in supervised])
    z = [r['z'].copy() for r in supervised]
    held = [r['z'].copy() for r in validation]
    trees, history = [], [metrics(validation, held)] if validation else []
    for iteration in range(rounds):
        # Recomputed from the ensemble's current transcript-conditioned distribution.
        gradient = np.concatenate([ctc(v, r['ids'])[1] for r, v in zip(supervised, z)])
        estimator = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(16, int(len(x)*.005)), random_state=1729)
        estimator.fit(x, gradient, sample_weight=weights)
        tree = tree_json(estimator)
        np.testing.assert_allclose(predict_tree(tree, x), estimator.predict(x), atol=1e-12)
        trees.append(tree)
        z = [v+rate*predict_tree(tree, r['x']) for r, v in zip(supervised, z)]
        held = [v+rate*predict_tree(tree, r['x']) for r, v in zip(validation, held)]
        if validation:
            history.append(metrics(validation, held))
        print(json.dumps(dict(depth=depth, rate=rate, round=iteration+1,
            **(history[-1] if validation else {}))), flush=True)
    return trees, history


def train(args):
    metadata = json.loads((Path(args.extractor)/'boosted-residuals-metadata.json').read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = observations(args.observations, metadata, excluded)
    folds = speaker_folds(rows)
    trials = []
    for depth in (1, 2, 3):
        for rate in (.1, .3):
            histories = []
            for fold in folds:
                fitting = [r for r in rows if r['speaker'] not in fold]
                held = [r for r in rows if r['speaker'] in fold]
                _, history = boost(fitting, held, depth, rate, args.rounds)
                histories.append(history)
            for stop in range(args.rounds+1):
                objective = sum(h[stop]['objective'] for h in histories)/len(histories)
                trials.append(dict(depth=depth, rate=rate, rounds=stop, objective=objective,
                    folds=[h[stop] for h in histories]))
    selected = min(trials, key=lambda r: (r['objective'], r['rounds'], r['depth'], r['rate']))
    trees, _ = boost(rows, [], selected['depth'], selected['rate'], selected['rounds'])
    bundle = Path(args.out)
    bundle.mkdir(parents=True, exist_ok=True)
    # Copy complete primary bundle including any insertion side-ear; never mutate shipped files.
    for path in Path(args.model).iterdir():
        if path.is_file() and not path.name.startswith('boosted-residuals'):
            shutil.copy2(path, bundle/path.name)
    shutil.copy2(Path(args.extractor)/'boosted-residuals.onnx', bundle/'boosted-residuals.onnx')
    ensemble = dict(**metadata, rate=selected['rate'], trees=trees)
    (bundle/'boosted-residuals-ensemble.json').write_text(json.dumps(ensemble, allow_nan=False))
    files = []
    for name in ('boosted-residuals-ensemble.json', 'boosted-residuals.onnx'):
        data = (bundle/name).read_bytes()
        files.append(dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (bundle/'boosted-residuals-asset.json').write_text(json.dumps(dict(schema=1, files=files)))
    (bundle/'boosted-residuals-selection.json').write_text(json.dumps(dict(selected=selected, trials=trials,
        recordings=len(rows), transcripts=sum(r['role']=='transcript' for r in rows),
        controls=sum(r['role']=='preservation' for r in rows),
        speakerFolds=[sorted(f) for f in folds], excludedSpeakers=sorted(excluded)), allow_nan=False))
    print(json.dumps(dict(selected=selected, bundle=str(bundle))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--extractor', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--excluded-speakers', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=32)
    options = parser.parse_args()
    if not 1 <= options.rounds <= 128:
        parser.error('rounds must be in [1, 128]')
    train(options)
