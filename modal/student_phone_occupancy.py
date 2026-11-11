"""Transcript-conditioned soft CTC occupancies and complete sequence scores."""
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
