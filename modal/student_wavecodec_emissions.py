"""Equal-capacity complete-pronunciation EM with latent boundaries, no recognition targets."""
DIMENSIONS = 16
STRENGTHS = (4, 16, 64)
EPOCHS = 5


def topology(ids):
    labels = [0] + [v for p in ids for v in (p, 0)]
    edges = [[s] + ([s+1] if s+1 < len(labels) else []) +
        ([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else [])
        for s, p in enumerate(labels)]
    return labels, edges


def boundary(values, ids, posterior=False):
    import numpy as np
    labels, edges = topology(ids)
    labels = np.asarray(labels)
    tmax, n = len(values), len(labels)
    if not tmax or not ids:
        if posterior:
            raise ValueError('Recording cannot carry the complete canonical pronunciation')
        return -np.inf
    cost = np.log([len(targets) for targets in edges])
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    def forward(previous):
        outgoing = previous-cost
        current = outgoing.copy()
        current[1:] = np.logaddexp(current[1:], outgoing[:-1])
        current[2:] = np.logaddexp(current[2:], np.where(skip, outgoing[:-2], -np.inf))
        return current
    a = np.full((tmax, n), -np.inf)
    prior = np.full(n, -np.inf)
    a[0, :2] = values[0, labels[:2]]
    prior[:2] = 0
    for t in range(1, tmax):
        a[t] = forward(a[t-1])+values[t, labels]
        prior = forward(prior)
    z = np.logaddexp(*a[-1, -2:])
    timing = np.logaddexp(*prior[-2:])
    score = z-timing if np.isfinite(z) and np.isfinite(timing) else -np.inf
    if not posterior:
        return float(score)
    if not np.isfinite(z):
        raise ValueError('Recording cannot carry the complete canonical pronunciation')
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        incoming = b[t+1]+values[t+1, labels]
        current = incoming.copy()
        current[:-1] = np.logaddexp(current[:-1], incoming[1:])
        current[:-2] = np.logaddexp(current[:-2], np.where(skip, incoming[2:], -np.inf))
        b[t] = current-cost
    occupancy = np.zeros_like(values)
    for s, p in enumerate(labels):
        occupancy[:, p] += np.exp(a[:, s]+b[:, s]-z)
    return float(score), occupancy


def emissions(arm, x):
    import numpy as np
    columns = []
    for e in arm['phones']:
        z = (x-e['mean']) @ np.array(e['precision']).T
        columns.append(-.5*(DIMENSIONS*np.log(2*np.pi)+e['logDet']+(z*z).sum(1)))
    return np.array(columns).T


def update(count, sums, cross, strength):
    import numpy as np
    total = count.sum()
    mu = sums.sum(0)/total
    cov = cross.sum(0)/total-np.outer(mu, mu)
    cov = (cov+cov.T)/2 + .01*np.eye(DIMENSIONS)
    phones = []
    for p, n in enumerate(count):
        mean = (sums[p]+strength*mu)/(n+strength)
        scatter = cross[p]-np.outer(sums[p], mean)-np.outer(mean, sums[p])+n*np.outer(mean, mean)
        d = mu-mean
        sigma = (scatter+strength*(cov+np.outer(d, d)))/(n+strength)
        sigma = (sigma+sigma.T)/2 + .01*np.eye(DIMENSIONS)
        l = np.linalg.cholesky(sigma)
        phones.append(dict(mean=mean.tolist(), precision=np.linalg.inv(l).tolist(),
            logDet=float(2*np.log(l.diagonal()).sum()), count=float(n)))
    return phones


def fit(rows, key, vocab, strength):
    import numpy as np
    raw = np.concatenate([r[key] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    def statistics(posts):
        count = np.zeros(vocab)
        sums = np.zeros((vocab, DIMENSIONS))
        cross = np.zeros((vocab, DIMENSIONS, DIMENSIONS))
        for r, w in zip(rows, posts):
            x = (r[key]-center)/scale
            count += w.sum(0)
            sums += w.T @ x
            cross += np.einsum('tp,ti,tj->pij', w, x, x)
        return update(count, sums, cross, strength)
    # Uniform transcript occupancy initializes EM without recognition targets.
    # Subsequent iterations marginalize boundaries under the fitted likelihood.
    posts = []
    for r in rows:
        labels, _ = topology(r['ids'])
        w = np.zeros((len(r[key]), vocab))
        for t in range(len(w)):
            w[t, labels[min(len(labels)-1, t*len(labels)//len(w))]] = 1
        posts.append(w)
    arm = dict(center=center.tolist(), scale=scale.tolist(), phones=statistics(posts))
    for _ in range(EPOCHS):
        posts = [boundary(emissions(arm, (r[key]-center)/scale), r['ids'], True)[1] for r in rows]
        arm['phones'] = statistics(posts)
    return arm
