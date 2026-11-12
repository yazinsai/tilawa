"""CPU blank-cause HMM. SDK observations only, none-labelled training speakers only.

Both quiet/speech boundaries are marginalized in EM. Hyperparameter selection
uses disjoint speaker groups entirely within help-train; no practice inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def stationary(transition):
    import numpy as np
    a, b = transition[0, 1], transition[1, 0]
    return np.array([b, a]) / (a + b)


def emissions(model, x):
    import numpy as np
    mu, var = np.array(model['means']), np.array(model['variances'])
    return -.5 * (np.log(2*np.pi*var)[None] + (x[:, None]-mu[None])**2/var[None]).sum(2)


def chain(e, transition, posterior=False):
    """Normalized two-state forward/backward; no fixed quiet/speech boundaries."""
    import numpy as np
    loga = np.log(transition)
    alpha = np.zeros_like(e)
    alpha[0] = np.log(stationary(transition)) + e[0]
    for t in range(1, len(e)):
        alpha[t] = np.logaddexp.reduce(alpha[t-1, :, None]+loga, axis=0)+e[t]
    likelihood = np.logaddexp.reduce(alpha[-1])
    if not posterior:
        return float(likelihood)
    beta = np.zeros_like(e)
    for t in range(len(e)-2, -1, -1):
        beta[t] = np.logaddexp.reduce(loga+e[t+1][None]+beta[t+1][None], axis=1)
    gamma = np.exp(alpha+beta-likelihood)
    xi = np.exp(alpha[:-1, :, None]+loga[None]+e[1:, None]+beta[1:, None]-likelihood).sum(0)
    return float(likelihood), gamma, xi


def fit(rows, floor, epochs=24):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    xs = [(r['x']-center)/scale for r in rows]
    xall = np.concatenate(xs)
    # Only initialization uses an energy partition. EM integrates every state placement.
    quiet = xall[:, 0] < np.quantile(xall[:, 0], .3)
    means = np.array([xall[quiet].mean(0), xall[~quiet].mean(0)])
    variances = np.array([np.maximum(xall[quiet].var(0), floor), np.maximum(xall[~quiet].var(0), floor)])
    transition = np.array([[.95, .05], [.05, .95]])
    model = dict(center=center.tolist(), scale=scale.tolist(), means=means.tolist(), variances=variances.tolist())
    history = []
    for _ in range(epochs):
        counts, sums, seconds, transitions = np.zeros(2), np.zeros((2, 4)), np.zeros((2, 4)), np.ones((2, 2))
        values = []
        for x in xs:
            value, gamma, xi = chain(emissions(model, x), transition, True)
            counts += gamma.sum(0)
            sums += gamma.T @ x
            seconds += gamma.T @ (x**2)
            transitions += xi
            values.append(value/len(x))
        means = sums / counts[:, None]
        variances = np.maximum(seconds/counts[:, None]-means**2, floor)
        transition = transitions/transitions.sum(1, keepdims=True)
        # Stable semantics: state 0 is always the lower-energy quiet cause.
        order = np.argsort(means[:, 0])
        means, variances = means[order], variances[order]
        transition = transition[order][:, order]
        model.update(means=means.tolist(), variances=variances.tolist())
        history.append(float(np.mean(values)))
    model.update(transition=transition.tolist(), initial=stationary(transition).tolist())
    return model, history


def validation(model, rows):
    import numpy as np
    by_speaker = {}
    for r in rows:
        x = (r['x']-model['center'])/model['scale']
        value = chain(emissions(model, x), np.array(model['transition']))/len(x)
        by_speaker.setdefault(r['speaker'], []).append(value)
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    for key in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for r in map(json.loads, Path(args.input).read_text().splitlines()):
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Only non-excluded, none-labelled help-train training speakers may be fitted')
        x = np.array(r['features'], dtype=float)
        if x.ndim != 2 or x.shape[1] != 4 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('Invalid SDK blank-cause observations')
        rows.append(dict(x=x, speaker=r['speaker']))
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(speakers) < 3:
        raise ValueError('At least three training speakers required for grouped validation')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for floor in [.05, .2, .5]:
        values = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], floor)
            values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
        results.append(dict(varianceFloor=floor, foldLogP=values, meanLogP=sum(values)/3))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['varianceFloor']))
    model, history = fit(rows, chosen['varianceFloor'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, dimensions=4, maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'blank-cause-model.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'blank-cause-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history, asset=descriptor)
    (out/'blank-cause-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
