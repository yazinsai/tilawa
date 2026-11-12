"""Class-preserving optimal transport of frozen acoustic features.

Canonical none-labelled training takes supply soft CTC occupancies. Scripts
select preservation comparisons only. Neither pronunciation nor phone class
enters the residual adapter at inference. All selection is speaker grouped.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

DIMENSIONS, HIDDEN, ADAPTER_HIDDEN = 32, 48, 16
STEPS, HEAD_STEPS, SINKHORN_STEPS = 400, 600, 60
REGULARIZATION = (.1, 1.)

def occupancies(log_probs, ids):
    """Exact forward/backward soft CTC state and phone occupancies.

    Retain the entire posterior distribution over transcript states so context
    conditioning never depends on a hard alignment or an engine word visit.
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
    return targets.astype('float32'), states, labels


def context_weights(lp, pronunciation, tokens):
    """Soft mass for (phone, adjacent vowel qualities); never use an engine visit.

    Different occurrences and states of one key contribute their full posterior
    mass, including ambiguous boundaries. Blank has its own context-free key.
    """
    import numpy as np
    found = occupancies(lp, pronunciation)
    if found is None:
        return None
    targets, states, _ = found
    quality = {'َ': 'a', 'ا': 'a', 'ِ': 'i', 'ۦ': 'i', 'ُ': 'u', 'ۥ': 'u'}
    vowels = [quality.get(tokens[p-1], '-') for p in pronunciation]
    groups = {(0, '-', '-'): targets[:, 0].astype('float64')}
    for i, phone in enumerate(pronunciation):
        left = next((v for v in reversed(vowels[:i]) if v != '-'), '-')
        right = next((v for v in vowels[i+1:] if v != '-'), '-')
        key = (phone, left, right)
        groups.setdefault(key, np.zeros(len(lp)))
        groups[key] += states[:, 2*i+1]
    return targets, groups


def pools(rows):
    """Per-recording distributions prevent long recordings dominating a phone."""
    result = {}
    for r in rows:
        for key, weights in r['groups'].items():
            mass = weights.sum()
            if mass > 1e-8:  # Numerical underflow guard, not an occupancy cutoff.
                result.setdefault(key, []).append((r['id'], r['speaker'], r['x'], weights/mass))
    return result


def draw(pool, key, n, rng):
    import numpy as np
    records = pool[key]
    picked = []
    for _ in range(n):
        _, _, x, weights = records[int(rng.integers(len(records)))]
        picked.append(x[int(rng.choice(len(x), p=weights))])
    return np.asarray(picked, dtype='float32')


def sinkhorn(cost, epsilon):
    """Balanced entropic OT with log-domain updates and uniform marginals."""
    import torch
    if cost.ndim != 2 or not torch.isfinite(cost).all() or epsilon <= 0:
        raise ValueError('Invalid transport cost')
    kernel = -cost.detach()/epsilon
    u = torch.zeros(cost.shape[0], dtype=cost.dtype)
    v = torch.zeros(cost.shape[1], dtype=cost.dtype)
    a, b = -torch.log(torch.tensor(float(len(u)))), -torch.log(torch.tensor(float(len(v))))
    for _ in range(SINKHORN_STEPS):
        u = a - torch.logsumexp(kernel + v[None, :], 1)
        v = b - torch.logsumexp(kernel + u[:, None], 0)
    return torch.exp(kernel + u[:, None] + v[None, :])


def transport_loss(source, target, keys, conditional, epsilon):
    """Conditional couplings cannot cross a phone/vowel key.

    The unconditional arm pools the exact same balanced samples. Both losses
    have unit total mass, the same distance, entropy scale and update count.
    """
    import torch
    cost = torch.cdist(source, target).square()/source.shape[1]
    if not conditional:
        return (sinkhorn(cost, epsilon)*cost).sum()
    losses = []
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        block = cost[idx][:, idx]
        losses.append((sinkhorn(block, epsilon)*block).sum())
    return torch.stack(losses).mean()


def scorer(vocab):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, vocab))


def adapter():
    import torch
    a = torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, ADAPTER_HIDDEN), torch.nn.Tanh(),
                            torch.nn.Linear(ADAPTER_HIDDEN, DIMENSIONS))
    torch.nn.init.zeros_(a[2].weight)
    torch.nn.init.zeros_(a[2].bias)
    return a


def equal_phone_weights(rows):
    """Each phone gets equal mass, split equally among recordings carrying it."""
    import numpy as np
    masses = [np.asarray(r['targets']).sum(0) for r in rows]
    counts = sum(m > 0 for m in masses)
    weights = np.concatenate([np.divide(r['targets'], m, out=np.zeros_like(r['targets']), where=m > 0)
                              / np.maximum(counts, 1) for r, m in zip(rows, masses)])
    weights /= weights.sum()
    return weights


def fit(rows, vocab, regularization):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    studio = [r for r in rows if r['domain'] == 'studio']
    source = [r for r in rows if r['domain'] == 'phone']
    raw = np.concatenate([r['features'] for r in studio]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    rows = [dict(r, x=(r['features']-center)/scale) for r in rows]
    studio = [r for r in rows if r['domain'] == 'studio']
    source = [r for r in rows if r['domain'] == 'phone']
    x = torch.tensor(np.concatenate([r['x'] for r in studio]))
    weights = equal_phone_weights(studio)
    sampling = weights.sum(1)
    y = torch.tensor(weights/sampling[:, None], dtype=torch.float32)
    readout = scorer(vocab)
    opt = torch.optim.AdamW(readout.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(1729)
    for _ in range(HEAD_STEPS):
        batch = rng.choice(len(x), 256, p=sampling)
        opt.zero_grad()
        loss = -(y[batch]*readout(x[batch]).log_softmax(-1)).sum(-1).mean()
        loss.backward()
        opt.step()
    readout.eval().requires_grad_(False)
    a, b = pools(source), pools(studio)
    common = sorted(k for k in set(a)&set(b) if k[0] and
                    len({r[1] for r in a[k]}) >= 2 and len({r[1] for r in b[k]}) >= 2)
    if len({k[0] for k in common}) < 8:
        raise ValueError('Insufficient independently recorded shared phone/vowel coverage')
    phones = sorted({k[0] for k in common})
    phone_keys = {p: [k for k in common if k[0] == p] for p in phones}
    conditional = adapter()
    unconditional = copy.deepcopy(conditional)
    opts = [torch.optim.AdamW(h.parameters(), lr=.002, weight_decay=.01) for h in (conditional, unconditional)]
    # Entropy scale comes from unadapted training features, and is frozen across arms.
    first = common[0]
    xx = torch.tensor(draw(a, first, 32, rng))
    yy = torch.tensor(draw(b, first, 32, rng))
    epsilon = max(float(torch.cdist(xx, yy).square().median()/DIMENSIONS)*.1, 1e-4)
    for _ in range(STEPS):
        selected = rng.choice(phones, min(4, len(phones)), replace=False)
        chosen = [phone_keys[p][int(rng.integers(len(phone_keys[p])))] for p in selected]
        sx = torch.tensor(np.concatenate([draw(a, k, 16, rng) for k in chosen]))
        tx = torch.tensor(np.concatenate([draw(b, k, 16, rng) for k in chosen]))
        keys = [k for k in chosen for _ in range(16)]
        for h, opt, restricted in zip((conditional, unconditional), opts, (True, False)):
            opt.zero_grad()
            delta = h(sx)
            loss = transport_loss(sx+delta, tx, keys, restricted, epsilon)
            # Same identity penalty in both arms, also on studio inputs.
            loss += regularization*(delta.square().mean()+h(tx).square().mean())/2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 5.)
            opt.step()
    return readout, conditional.eval(), unconditional.eval(), center, scale, common, epsilon


def output(readout, adapter, features, center, scale, temperature=1.):
    import numpy as np
    import torch
    with torch.no_grad():
        x = torch.tensor((np.asarray(features, dtype='float32')-center)/scale)
        return (readout(x+adapter(x))/temperature).log_softmax(-1).numpy()


def serialize(model):
    return dict(hidden=model[0].weight.detach().tolist(), hiddenBias=model[0].bias.detach().tolist(),
                output=model[2].weight.detach().tolist(), outputBias=model[2].bias.detach().tolist())


def speaker_folds(rows, controls):
    """Balance domains without splitting any speaker, including preservation-only speakers."""
    domains = {s: {r['domain'] for r in rows if r['speaker'] == s}
               for s in {r['speaker'] for r in rows}}
    counts = {d: [0, 0, 0] for d in ('phone', 'studio')}
    folds = {}
    for speaker in sorted(domains, key=lambda s: (-len(domains[s]), s)):
        ds = domains[speaker]
        fold = min(range(3), key=lambda f: (sum(counts[d][f] for d in ds), f))
        folds[speaker] = fold
        for d in ds:
            counts[d][fold] += 1
    for speaker in sorted({c['speaker'] for c in controls} - folds.keys()):
        fold = min(range(3), key=lambda f: (sum(v == f for v in folds.values()), f))
        folds[speaker] = fold
    return folds


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    if len({r.get('id') for r in data}) != len(data) or any(not r.get('id') for r in data):
        raise ValueError('Recording identities must be unique')
    rows, controls, skipped = [], [], []
    for r in data:
        if r.get('source') not in ('help-train', 'authorized-studio') or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Ineligible or missing training-speaker provenance')
        if (r.get('domain'), r['source']) not in (('phone', 'help-train'), ('studio', 'authorized-studio')):
            raise ValueError('A learner laptop is not an authorized studio reference')
        features = np.asarray(r['features'], dtype='float32')
        if features.ndim != 2 or features.shape[1] != DIMENSIONS or not np.isfinite(features).all():
            raise ValueError('Invalid frozen acoustic features')
        if r['kind'] == 'substituted_word':
            controls.extend(dict(c, speaker=r['speaker']) for c in r['controls'])
            continue
        if r['kind'] != 'none':
            raise ValueError('Scripts must not masquerade as canonical transcripts')
        found = context_weights(r['logProbs'], r['pronunciation'], metadata['tokens'])
        if found is None:
            skipped.append(r['id'])
            continue
        targets, groups = found
        if len(targets) != len(features):
            raise ValueError('Feature and occupancy clocks differ')
        rows.append(dict(r, features=features, targets=targets, groups=groups))
    # A speaker remains in the same fold across both domains and all recordings.
    speakers = sorted({r['speaker'] for r in rows})
    folds = speaker_folds(rows, controls)
    if any(len({r['speaker'] for r in rows if r['domain'] == domain}) < 6 for domain in ('phone', 'studio')):
        raise ValueError('Need six training speakers per domain for three-fold selection')
    vocab = len(metadata['tokens'])+1
    validation = []
    for reg in REGULARIZATION:
        soft, preservation = [], []
        for fold in range(3):
            fitted = fit([r for r in rows if folds[r['speaker']] != fold], vocab, reg)
            head, conditional, unconditional, center, scale, _, _ = fitted
            held = [r for r in rows if folds[r['speaker']] == fold]
            # One scorer temperature shared by both arms, fitted from training-only OOF rows.
            temperatures = []
            for temp in (.5, 1., 2.):
                errors = [float(-(r['targets']*output(head, conditional, r['features'], center, scale, temp)).sum(1).mean()) for r in held]
                temperatures.append((float(np.mean(errors)), temp))
            nll, temp = min(temperatures)
            soft.append((nll, temp))
            for c in controls:
                if folds.get(c['speaker']) != fold:
                    continue
                from_score = sequence_score(output(head, unconditional, c['features'], center, scale, temp), c['written']) - sequence_score(output(head, unconditional, c['features'], center, scale, temp), c['rival'])
                to_score = sequence_score(output(head, conditional, c['features'], center, scale, temp), c['written']) - sequence_score(output(head, conditional, c['features'], center, scale, temp), c['rival'])
                if np.isfinite(from_score) and np.isfinite(to_score):
                    preservation.append(from_score < 0 < to_score)
        validation.append(dict(regularization=reg, meanSoftNll=float(np.mean([n for n, _ in soft])),
            temperature=float(np.median([t for _, t in soft])), preservationComparisons=len(preservation),
            preservationReversals=sum(preservation)))
        print(json.dumps(validation[-1]), flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No training-speaker substitution preservation comparisons')
    selected = min(validation, key=lambda v: (v['preservationReversals']/v['preservationComparisons'], v['meanSoftNll']))
    head, conditional, unconditional, center, scale, common, epsilon = fit(rows, vocab, selected['regularization'])
    metadata.update(center=center.tolist(), scale=scale.tolist(), readout=serialize(head),
        conditional=serialize(conditional), control=serialize(unconditional), temperature=selected['temperature'])
    out = Path(args.out)
    asset = out/'transport-phone-heads.json'
    asset.write_text(json.dumps(metadata, allow_nan=False))
    report = dict(recordings=len(rows), speakers=speakers, folds=folds, excludedSpeakers=sorted(excluded),
        validation=validation, selected=selected, sharedKeys=common, epsilon=epsilon, impossibleAlignments=skipped,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), steps=STEPS, headSteps=HEAD_STEPS)
    (out/'transport-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    files = [asset, out/'transport-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Incomplete feature bundle')
    (out/'transport-phone-asset.json').write_text(json.dumps(dict(schema=1, files=[dict(file=p.name,
        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    print(json.dumps(report), flush=True)


def sequence_score(lp, ids):
    """Normalized complete CTC timing model, mirrored in the SDK."""
    import numpy as np
    labels = [0]+[v for p in ids for v in (p, 0)]
    edges = [[s]+([s+1] if s+1 < len(labels) else [])+([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else []) for s, p in enumerate(labels)]
    a = np.full(len(labels), -np.inf)
    prior = a.copy()
    a[:2] = lp[0, labels[:2]]
    prior[:2] = 0
    for row in lp[1:]:
        b, q = np.full_like(a, -np.inf), np.full_like(a, -np.inf)
        for s, nxt in enumerate(edges):
            for j in nxt:
                b[j] = np.logaddexp(b[j], a[s]-np.log(len(nxt)))
                q[j] = np.logaddexp(q[j], prior[s]-np.log(len(nxt)))
        a, prior = b+row[labels], q
    likelihood, timing = np.logaddexp.reduce(a[-2:]), np.logaddexp.reduce(prior[-2:])
    return float(likelihood-timing) if np.isfinite(likelihood) and np.isfinite(timing) else -np.inf


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())
