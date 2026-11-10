"""CPU spectral EM with an exact four-frame causal phone-state memory.

Only explicitly authorized, correctly transcribed training recordings may fit
emissions. help-train accepts none-labelled training speakers only. No practice
audio, incident identifiers or scripted performed transcripts are accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

HISTORY = 4
DIMENSIONS = 20


def kernel(mix, decay):
    tail = decay ** np.arange(HISTORY)
    return np.r_[1-mix, mix*tail/tail.sum()]


def topology(ids, initial):
    """After four frames all incoming memory states merge exactly."""
    nodes = [(0, 0, p) for p, prob in enumerate(initial) if prob > 0]
    stages = [dict(nodes=nodes, weights=np.log(initial[[p for _, _, p in nodes]]))]
    stable = [(i, mask, 0) for i in range(len(ids)) for mask in range(1 << HISTORY) if mask.bit_count() <= i]
    mapping = {node: j for j, node in enumerate(stable)}
    for t in range(1, HISTORY+1):
        destinations = mapping.copy() if t == HISTORY else {}
        edges = []
        for source, (i, mask, seed) in enumerate(nodes):
            for advance in (0, 1):
                if i+advance >= len(ids):
                    continue
                dest = (i+advance, ((mask << 1) | advance) & ((1 << HISTORY)-1), 0 if t >= HISTORY else seed)
                if dest not in destinations:
                    destinations[dest] = len(destinations)
                edges.append((source, destinations[dest], ids[i], advance))
        nodes = list(destinations)
        stages.append(dict(nodes=nodes, edges=np.asarray(edges, dtype=int)))
    # Fully reachable steady-state transitions; no beam or Viterbi approximation.
    edges = []
    for source, (i, mask, _) in enumerate(stable):
        for advance in (0, 1):
            if i+advance < len(ids):
                dest = (i+advance, ((mask << 1) | advance) & ((1 << HISTORY)-1), 0)
                edges.append((source, mapping[dest], ids[i], advance))
    stages.append(dict(nodes=stable, edges=np.asarray(edges, dtype=int)))
    return stages


def node_phones(nodes, ids, t):
    return np.array([[ids[i]] + [seed if t < lag else ids[i-(mask & ((1 << lag)-1)).bit_count()]
        for lag in range(1, HISTORY+1)] for i, mask, seed in nodes], dtype=int)


def evaluate(model, x, ids, posterior=False, memory=True):
    """Forward/backward marginalizes boundaries and incoming/recent phone states."""
    if len(x) < len(ids):
        raise ValueError('Impossible complete pronunciation')
    stages = topology(ids, np.asarray(model['initial']))
    spectra, variance = np.asarray(model['spectra']), np.asarray(model['variances'])
    weights = kernel(model['mix'], model['decay']) if memory else np.r_[1., np.zeros(HISTORY)]
    a, es, ps, edges_by_t, phones_by_t = [], [], [], [], []
    for t, y in enumerate(x):
        stage = stages[min(t, HISTORY+1)]
        phones = node_phones(stage['nodes'], ids, t)
        mean = np.log((spectra[phones]*weights[:, None]).sum(1))
        var = variance[phones[:, 0]]
        e = -.5*(np.log(2*np.pi*var)+(y-mean)**2/var).sum(1)
        if t == 0:
            prior = stage['weights'].copy()
            alpha = prior+e
            edge = None
        else:
            edge = stage['edges']
            source, dest, phone, advance = edge.T
            stay = np.asarray(model['stay'])[phone]
            lp = np.where(advance, np.log1p(-stay), np.log(stay))
            prior = np.full(len(phones), -np.inf)
            alpha = prior.copy()
            np.logaddexp.at(prior, dest, ps[-1][source]+lp)
            np.logaddexp.at(alpha, dest, a[-1][source]+lp)
            alpha += e
        a.append(alpha); es.append(e); ps.append(prior); edges_by_t.append(edge); phones_by_t.append(phones)
    final = np.array([i == len(ids)-1 for i, _, _ in stages[min(len(x)-1, HISTORY+1)]['nodes']])
    likelihood, timing = logsumexp(a[-1][final]), logsumexp(ps[-1][final])
    if not posterior:
        return float(likelihood-timing)
    b = [np.full_like(alpha, -np.inf) for alpha in a]
    b[-1][final] = 0
    counts = np.zeros((len(spectra), 2))
    for t in range(len(x)-2, -1, -1):
        source, dest, phone, advance = edges_by_t[t+1].T
        stay = np.asarray(model['stay'])[phone]
        lp = np.where(advance, np.log1p(-stay), np.log(stay))
        continuation = lp+es[t+1][dest]+b[t+1][dest]
        np.logaddexp.at(b[t], source, continuation)
        xi = np.exp(a[t][source]+continuation-likelihood)
        np.add.at(counts, (phone, advance), xi)
    gamma = [np.exp(alpha+beta-likelihood) for alpha, beta in zip(a, b)]
    return float(likelihood-timing), gamma, phones_by_t, counts


def attributes(p):
    manner = next((name for chars, name in [('َُِاۦۥ', 'vowel'), ('منں۾', 'nasal'),
        ('بتدطكقءڇ', 'stop'), ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant')]
        if p in chars), 'other')
    place = next((name for chars, name in [('بمفو', 'labial'), ('ثذظ', 'dental'),
        ('تدطسصزشنرل', 'coronal'), ('يكقخغ', 'dorsal'), ('ءهحع', 'guttural')] if p in chars), 'other')
    return manner, place


def emission_objective(params, phones, x, weights, variance, target, shrinkage):
    """Expected log-spectrum error; analytic gradients include normalized decay."""
    vocab = len(target)
    spectra = softmax(params[:-2].reshape(vocab, DIMENSIONS), axis=1)
    mix, decay = expit(params[-2:])
    w = kernel(mix, decay)
    components = spectra[phones]
    prediction = (components*w[:, None]).sum(1)
    residual = np.log(prediction)-x
    var = variance[phones[:, 0]]
    total = .5*np.sum(weights[:, None]*(residual**2/var+np.log(2*np.pi*var)))
    g_prediction = weights[:, None]*residual/var/prediction
    g_spectra = np.zeros_like(spectra)
    for l in range(HISTORY+1):
        np.add.at(g_spectra, phones[:, l], g_prediction*w[l])
    g_logits = spectra*(g_spectra-(g_spectra*spectra).sum(1, keepdims=True))
    # A fixed phonetic-sharing prior on normalized phone spectra.
    total -= shrinkage*np.sum(target*np.log(spectra))
    g_logits += shrinkage*(spectra-target)
    tail = w[1:]/mix
    lag = np.arange(HISTORY)
    d_tail = tail*(lag/decay-np.sum(tail*lag/decay))
    d_mix = (components[:, 1:]*tail[:, None]).sum(1)-components[:, 0]
    d_decay = mix*(components[:, 1:]*d_tail[:, None]).sum(1)
    grad_mix = np.sum(g_prediction*d_mix)*mix*(1-mix)
    grad_decay = np.sum(g_prediction*d_decay)*decay*(1-decay)
    # Mild, fixed Beta(2,2) regularization prevents a degenerate all-memory solution.
    total -= np.log(mix*(1-mix))+np.log(decay*(1-decay))
    return float(total), np.r_[g_logits.ravel(), grad_mix+2*mix-1, grad_decay+2*decay-1]


def fit(rows, tokens, shrinkage, epochs=8):
    vocab = len(tokens)+1
    sums, counts = np.zeros((vocab, DIMENSIONS)), np.zeros(vocab)
    for r in rows:
        placement = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        for i, p in enumerate(r['ids']):
            part = np.exp(r['x'][placement == i])
            sums[p] += part.sum(0); counts[p] += len(part)
    global_spectrum = sums.sum(0)/max(counts.sum(), 1)
    attrs = [attributes(p) for p in '_' + tokens]
    target = []
    for p in range(vocab):
        shared = []
        for f in range(2):
            pool = [q for q in range(1, vocab) if attrs[p][f] == attrs[q][f]]
            shared.append((sums[pool].sum(0)+shrinkage*global_spectrum)/(counts[pool].sum()+shrinkage))
        target.append(np.mean(shared, axis=0))
    target = np.maximum(target, 1e-9); target /= target.sum(1, keepdims=True)
    spectra = (sums+shrinkage*target)/(counts[:, None]+shrinkage)
    spectra = np.maximum(spectra, 1e-9); spectra /= spectra.sum(1, keepdims=True)
    model = dict(spectra=spectra.tolist(), variances=np.ones((vocab, DIMENSIONS)).tolist(),
        counts=counts.tolist(), initial=((counts+1)/(counts.sum()+vocab)).tolist(),
        stay=np.full(vocab, .8).tolist(), mix=.25, decay=.5)
    history = []
    for epoch in range(epochs):
        phones, observations, masses, values = [], [], [], []
        transitions = np.zeros((vocab, 2))
        for r in rows:
            value, gamma, identities, tc = evaluate(model, r['x'], r['ids'], True)
            transitions += tc
            values.append(value/len(r['x']))
            for y, g, p in zip(r['x'], gamma, identities):
                # Training-only numerical truncation, never inference beam pruning.
                keep = g > 1e-9
                phones.append(p[keep]); observations.append(np.repeat(y[None], keep.sum(), axis=0)); masses.append(g[keep])
        phones, x, masses = np.concatenate(phones), np.concatenate(observations), np.concatenate(masses)
        params = np.r_[np.log(model['spectra']).ravel(), np.log(model['mix']/(1-model['mix'])),
            np.log(model['decay']/(1-model['decay']))]
        var = np.asarray(model['variances'])
        optimized = minimize(emission_objective, params, args=(phones, x, masses, var, target, shrinkage),
            method='L-BFGS-B', jac=True, options={'maxiter': 60, 'ftol': 1e-8},
            bounds=[(-25, 5)]*(vocab*DIMENSIONS)+[(-6, 6), (-6, 6)])
        if not np.isfinite(optimized.fun):
            raise ValueError('Non-finite training objective')
        spectra = softmax(optimized.x[:-2].reshape(vocab, DIMENSIONS), axis=1)
        mix, decay = expit(optimized.x[-2:])
        residual = x-np.log((spectra[phones]*kernel(mix, decay)[:, None]).sum(1))
        seconds, counts = np.zeros_like(spectra), np.zeros(vocab)
        np.add.at(seconds, phones[:, 0], masses[:, None]*residual**2)
        np.add.at(counts, phones[:, 0], masses)
        global_var = seconds.sum(0)/counts.sum()
        variances = np.maximum((seconds+shrinkage*global_var)/(counts[:, None]+shrinkage), .05)
        stay = (transitions[:, 0]+1)/(transitions.sum(1)+2)
        model.update(spectra=spectra.tolist(), variances=variances.tolist(), counts=counts.tolist(),
            stay=stay.tolist(), mix=float(mix), decay=float(decay))
        history.append(dict(epoch=epoch, meanLogP=float(np.mean(values)), mix=float(mix), decay=float(decay),
            optimizerSuccess=bool(optimized.success), objective=float(optimized.fun)))
    return model, history


def prepare(raw, tokens, excluded):
    out = []
    for r in raw:
        authorized = (r.get('source') == 'help-train' and r.get('kind') == 'none') or (
            r.get('source') == 'authorized-quran' and r.get('verifiedTranscript') is True)
        if not authorized or r.get('split') != 'train' or r.get('excluded') is not False or r.get('speaker') in excluded:
            raise ValueError('Only authorized, non-excluded training speakers with correct transcripts may be fitted')
        x = np.array(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if not r.get('speaker') or x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or (
            not ids or len(x) < len(ids) or len(x) > 2048 or not np.allclose(np.exp(x).sum(1), 1, atol=1e-5)):
            raise ValueError('Invalid normalized SDK observations or impossible pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers are required')
    return out


def validation(model, rows):
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r['speaker'], []).append(evaluate(model, r['x'], r['ids'])/len(r['x']))
    return float(np.mean([np.mean(values) for values in by_speaker.values()]))


def main():
    parser = argparse.ArgumentParser()
    for name in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    validation_results = []
    for shrinkage in [8, 32, 128]:
        values = []
        for fold in range(3):
            fitted, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, shrinkage)
            values.append(validation(fitted, [r for r in rows if folds[r['speaker']] == fold]))
        validation_results.append(dict(shrinkage=shrinkage, foldLogP=values, meanLogP=sum(values)/3))
        print(json.dumps(validation_results[-1]), flush=True)
    selected = max(validation_results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    model, history = fit(rows, args.tokens, selected['shrinkage'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40,
        dimensions=DIMENSIONS, history=HISTORY, maxFrames=2048)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    asset = out/'reverberant-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'reverberant-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=validation_results,
        selected=selected, history=history, asset=descriptor)
    (out/'reverberant-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
