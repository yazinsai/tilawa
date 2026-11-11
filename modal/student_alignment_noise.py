"""CPU pronunciation emissions with latent alignment-label contamination.

Two independently initialized intact-transcript aligners supply the SAME fixed
occupancies to both heads. Disagreement estimates phone-level contamination;
the noise head integrates nominal versus neighbouring phone assignments in EM.
Inference reads the resulting clean emissions, not the training corruption law.
Only none-labelled help-train training speakers are accepted. No practice inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 8
ALIGNER_SEEDS = (181, 733)
ALIGNER_EPOCHS = 8
EM_EPOCHS = 12
STRENGTHS = (2, 8, 32)


def chain(e, posterior=False):
    import numpy as np
    tmax, states = e.shape
    if not states or tmax < states:
        raise ValueError('Every complete phone needs an observation')
    # Completion conditioning cancels the constant geometric stay/step factors.
    a = np.full_like(e, -np.inf)
    a[0, 0] = e[0, 0]
    for t in range(1, tmax):
        a[t] = np.logaddexp(a[t-1], np.r_[-np.inf, a[t-1, :-1]])+e[t]
    from scipy.special import gammaln
    log_paths = gammaln(tmax)-gammaln(states)-gammaln(tmax-states+1)
    value = float(a[-1, -1]-log_paths)
    if not posterior:
        return value
    b = np.full_like(e, -np.inf)
    b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        b[t] = np.logaddexp(e[t+1]+b[t+1], np.r_[e[t+1, 1:]+b[t+1, 1:], -np.inf])
    return value, np.exp(a+b-a[-1, -1])


def empty(vocab):
    import numpy as np
    return [np.zeros(vocab), np.zeros((vocab, DIMENSIONS)), np.zeros((vocab, DIMENSIONS))]


def accumulate(stats, x, weights):
    stats[0][:] += weights.sum(0)
    stats[1][:] += weights.T@x
    stats[2][:] += weights.T@(x*x)


def update(stats, strength):
    import numpy as np
    count, sums, squares = stats
    n = max(count.sum(), 1)
    mu = sums.sum(0)/n
    second = squares.sum(0)/n
    means = (sums+strength*mu)/(count[:, None]+strength)
    variances = np.maximum((squares+strength*second)/(count[:, None]+strength)-means*means, .05)
    return dict(mean=means, variance=variances, count=count.copy())


def emission(head, x):
    import numpy as np
    return -.5*(DIMENSIONS*np.log(2*np.pi)+np.log(head['variance']).sum(1)[None, :]+
        ((x[:, None, :]-head['mean'][None, :, :])**2/head['variance'][None, :, :]).sum(2))


def phone_weights(gamma, ids, vocab):
    import numpy as np
    weights = np.zeros((len(gamma), vocab))
    for s, p in enumerate(ids):
        weights[:, p] += gamma[:, s]
    return weights


def fit_aligner(rows, vocab, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    stats = empty(vocab)
    for r in rows:
        lengths = rng.gamma(3, 1, len(r['ids']))
        edges = np.r_[0, np.cumsum(lengths)/lengths.sum()*len(r['x'])]
        assigned = np.searchsorted(edges[1:], np.arange(len(r['x']))+.5)
        assigned = np.minimum(assigned, len(r['ids'])-1)
        weights = np.eye(vocab)[np.asarray(r['ids'])[assigned]]
        accumulate(stats, r['x'], weights)
    head = update(stats, 8)
    for _ in range(ALIGNER_EPOCHS):
        stats = empty(vocab)
        for r in rows:
            _, gamma = chain(emission(head, r['x'])[:, r['ids']], True)
            accumulate(stats, r['x'], phone_weights(gamma, r['ids'], vocab))
        head = update(stats, 8)
    return head


def assignments(rows, vocab):
    import numpy as np
    aligners = [fit_aligner(rows, vocab, seed) for seed in ALIGNER_SEEDS]
    out, totals, disagreements = [], np.zeros(vocab), np.zeros(vocab)
    for r in rows:
        gammas = [chain(emission(h, r['x'])[:, r['ids']], True)[1] for h in aligners]
        labels = [np.asarray(r['ids'])[g.argmax(1)] for g in gammas]
        nominal = np.eye(vocab)[labels[0]]
        # A contaminated state can draw only from its actual neighbouring transcript
        # labels; no alternate word or assumed performed lexical edit is introduced.
        neighbours = np.zeros_like(nominal)
        for s, p in enumerate(r['ids']):
            adjacent = [r['ids'][q] for q in (s-1,s+1) if 0 <= q < len(r['ids']) and r['ids'][q] != p]
            if not adjacent:
                adjacent = [p]
            for q in adjacent:
                neighbours[:, q] += (gammas[0].argmax(1) == s)/len(adjacent)
        totals += nominal.sum(0)
        disagreements += (nominal*(labels[0] != labels[1])[:, None]).sum(0)
        out.append(dict(r, nominal=nominal, neighbours=neighbours))
    pooled = disagreements.sum()/max(totals.sum(), 1)
    rho = (disagreements+8*pooled)/(totals+8)
    return out, rho, dict(alignerSeeds=list(ALIGNER_SEEDS),frames=int(totals.sum()),
        disagreementRate=float(pooled),phoneOccupancy=totals.tolist(),phoneDisagreement=disagreements.tolist())


def latent_weights(log_emission, nominal, neighbours, rho):
    """Posterior over correct versus contaminated assignment, summed not selected."""
    import numpy as np
    # Conditional contamination distribution retains each nominal state's probability.
    rate = nominal@rho
    prior = nominal*(1-rho)[None, :]+neighbours*rate[:, None]
    prior /= prior.sum(1)[:, None]
    log_prior = np.full_like(prior, -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    logits = log_prior+log_emission
    logits -= logits.max(1)[:, None]
    weights = np.exp(logits)
    return weights/weights.sum(1)[:, None]


def fit_heads(rows, vocab, rho, strength):
    stats = empty(vocab)
    for r in rows:
        accumulate(stats, r['x'], r['nominal'])
    control = update(stats, strength)
    noise = dict(mean=control['mean'].copy(), variance=control['variance'].copy(), count=control['count'].copy())
    for _ in range(EM_EPOCHS):
        stats = empty(vocab)
        for r in rows:
            weights = latent_weights(emission(noise, r['x']), r['nominal'], r['neighbours'], rho)
            accumulate(stats, r['x'], weights)
        noise = update(stats, strength)
    return control, noise


def prepare(raw, tokens, excluded):
    import numpy as np
    rows = []
    for r in raw:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r.get('speaker') in excluded:
            raise ValueError('Only none-labelled non-excluded help-train training speakers may train')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or not ids or len(x) < len(ids):
            raise ValueError('Invalid intact-transcript observations')
        rows.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three training speakers required')
    return rows


def normalized(rows, center, scale):
    return [dict(r, x=(r['x']-center)/scale) for r in rows]


def validation(head, rows):
    import numpy as np
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r['speaker'], []).append(chain(emission(head, r['x'])[:, r['ids']])/len(r['x']))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def export_head(head):
    return [dict(mean=mu.tolist(), variance=var.tolist(), count=float(n)) for mu, var, n in zip(head['mean'], head['variance'], head['count'])]


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    for key in ('input','out','tokens','inventory','excluded-speakers'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    vocab = len(args.tokens)+1
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s:i%3 for i,s in enumerate(speakers)}
    scores = {strength:[] for strength in STRENGTHS}
    fold_reports = []
    for fold in range(3):
        train = [r for r in rows if folds[r['speaker']] != fold]
        raw = np.concatenate([r['x'] for r in train])
        center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
        aligned, rho, report = assignments(normalized(train, center, scale), vocab)
        test = normalized([r for r in rows if folds[r['speaker']] == fold], center, scale)
        for strength in STRENGTHS:
            control, noise = fit_heads(aligned, vocab, rho, strength)
            scores[strength].append(dict(control=validation(control, test), noise=validation(noise, test)))
        fold_reports.append(report)
        print(json.dumps(dict(fold=fold,disagreement=report['disagreementRate'],scores=scores)), flush=True)
    selected = max(STRENGTHS, key=lambda s:(np.mean([v['noise'] for v in scores[s]]), s))
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    aligned, rho, alignment = assignments(normalized(rows, center, scale), vocab)
    control, noise = fit_heads(aligned, vocab, rho, selected)
    model = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, dimensions=8,
        center=center.tolist(), scale=scale.tolist(), control=export_head(control), noise=export_head(noise),
        contamination=rho.tolist(), stay=.8, maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'alignment-noise-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'alignment-noise-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows),speakers=len(speakers),speakerFolds=folds,excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),validation=scores,selected=selected,
        alignment=alignment,foldAlignments=fold_reports,parametersPerHead=2*vocab*DIMENSIONS,asset=descriptor)
    (out/'alignment-noise-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
