"""Training-only phone/letter CTC with latent, synchronous word boundaries.

Every nonempty word interval has its own CTC alignment in each alphabet. The
shared lattice advances to the next word only when BOTH alignments have finished.
The control marginalizes each alphabet's word boundaries separately. A uniform
prior over the same nonempty partitions is used in both models. Neither lattice
requires frame labels or teacher-selected boundaries.
"""

import hashlib
import math
import unicodedata


def spelling(word):
    """Undiacritized Arabic spelling; preserve hamza and compositional letters."""
    word = unicodedata.normalize("NFC", word).replace("ٱ", "ا")
    # Quranic small waw/ya and tatweel are modifier letters (Lm), not spelling.
    return "".join(c for c in word if unicodedata.category(c) == "Lo")


def authorized_none(row):
    return (row.get("split") == "train" and row.get("kind") == "none"
            and not row.get("excluded") and not row.get("events")
            and not row.get("non_errors") and not row.get("unsupported")
            and bool(row.get("speaker")) and bool(row.get("ayahs")))


def speaker_partition(speakers, seed=0):
    ordered = sorted(set(speakers), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if len(ordered) < 2:
        raise ValueError("Need at least two training speakers")
    validation = set(ordered[:min(len(ordered) - 1, max(1, math.ceil(len(ordered) / 5)))])
    return {s: "validation" if s in validation else "train" for s in ordered}


def safe_logsumexp(values, dim):
    """Unreachable states stay -inf and have zero, rather than NaN, gradients."""
    import torch
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    result = torch.logsumexp(torch.where(reachable, values, torch.zeros_like(values)), dim=dim)
    return result.masked_fill(~reachable.squeeze(dim), -torch.inf)


class WordCTC:
    def __init__(self, words, vocabulary, device):
        import torch
        if not words or any(not w for w in words):
            raise ValueError("Each word must contain nonblank transcript occurrences")
        if any(not isinstance(i, int) or i <= 0 or i >= vocabulary for w in words for i in w):
            raise ValueError("Transcript outside CTC inventory")
        self.count = len(words)
        self.width = max(2 * len(w) + 1 for w in words)
        self.labels = torch.zeros((self.count, self.width), dtype=torch.long, device=device)
        self.valid = torch.zeros_like(self.labels, dtype=torch.bool)
        for j, word in enumerate(words):
            self.labels[j, 1:2 * len(word):2] = torch.tensor(word, device=device)
            self.valid[j, :2 * len(word) + 1] = True
        self.skip = (self.labels != 0) & (self.labels != self.labels.roll(2, dims=1))
        self.skip[:, :2] = False
        self.last = torch.tensor([2 * len(w) for w in words], device=device)
        self.rows = torch.arange(self.count, device=device)
        # Across a boundary the exported decoder still needs a blank between
        # adjacent equal labels. Local CTC reset must not invent an articulation.
        previous = torch.tensor([0] + [w[-1] for w in words[:-1]], device=device)
        first = torch.tensor([w[0] for w in words], device=device)
        self.repeat = previous == first

    def advance(self, alpha, axis):
        import torch
        one = alpha.roll(1, dims=axis)
        two = alpha.roll(2, dims=axis)
        slices = [slice(None)] * alpha.ndim
        slices[axis] = 0
        one[tuple(slices)] = -torch.inf
        shape = [self.count] + [1] * (alpha.ndim - 1)
        shape[axis] = self.width
        two = two.masked_fill(~self.skip.reshape(shape), -torch.inf)
        return safe_logsumexp(torch.stack((alpha, one, two)), 0)

    def partition(self, log_probs):
        import torch
        emit = log_probs[:, self.labels]
        alpha = emit[0].new_full(self.labels.shape, -torch.inf)
        alpha[0, :2] = emit[0, 0, :2]
        for t in range(1, len(log_probs)):
            end = torch.stack((alpha[self.rows, self.last], alpha[self.rows, self.last - 1]), -1)
            # destination blank accepts either previous endpoint; destination
            # first label excludes previous identical nonblank endpoint.
            sources = end[:, None, :].expand(-1, 2, -1)
            allow = torch.ones_like(sources, dtype=torch.bool)
            allow[:-1, 1, 1] = ~self.repeat[1:]
            portals = safe_logsumexp(sources.masked_fill(~allow, -torch.inf), -1)
            boundary = torch.cat((portals.new_full((1, 2), -torch.inf), portals[:-1]), 0)
            boundary = torch.nn.functional.pad(boundary, (0, self.width - 2), value=-torch.inf)
            alpha = safe_logsumexp(torch.stack((self.advance(alpha, 1), boundary)), 0) + emit[t]
            alpha = alpha.masked_fill(~self.valid, -torch.inf)
        return safe_logsumexp(alpha[-1, self.last[-1] - 1:self.last[-1] + 1], 0)


class BoundaryCTC:
    def __init__(self, phone_words, letter_words, phone_vocabulary, letter_vocabulary, device="cpu"):
        if len(phone_words) != len(letter_words):
            raise ValueError("Alphabets must describe the same ordered words")
        self.phone = WordCTC(phone_words, phone_vocabulary, device)
        self.letter = WordCTC(letter_words, letter_vocabulary, device)

    def partitions(self, phone, letter):
        """Return shared and independent log likelihoods, including boundary priors."""
        import torch
        p, l = self.phone, self.letter
        if (phone.ndim != 2 or letter.ndim != 2 or len(phone) != len(letter)
                or len(phone) < p.count):
            raise ValueError("Heads need the same clock and at least one frame per word")
        pe, le = phone[:, p.labels], letter[:, l.labels]
        valid = p.valid[:, :, None] & l.valid[:, None, :]
        alpha = phone.new_full(valid.shape, -torch.inf)
        alpha[0, :2, :2] = pe[0, 0, :2, None] + le[0, 0, None, :2]
        # Four terminal endpoint pairs feed four initial endpoint pairs. Repeated
        # labels independently require a blank in their own alphabet.
        allow = torch.ones((p.count, 2, 2, 2, 2), dtype=torch.bool, device=phone.device)
        allow[:-1, 1, :, 1, :] &= ~p.repeat[1:, None, None]
        allow[:-1, :, 1, :, 1] &= ~l.repeat[1:, None, None]
        for t in range(1, len(phone)):
            ends = torch.stack([alpha[p.rows, p.last - i, l.last - j]
                                for i in range(2) for j in range(2)], -1).reshape(p.count, 2, 2)
            sources = ends[:, None, None, :, :].expand(-1, 2, 2, -1, -1)
            portals = safe_logsumexp(sources.masked_fill(~allow, -torch.inf).flatten(-2), -1)
            boundary = torch.cat((portals.new_full((1, 2, 2), -torch.inf), portals[:-1]), 0)
            boundary = torch.nn.functional.pad(boundary, (0, l.width - 2, 0, p.width - 2), value=-torch.inf)
            within = l.advance(p.advance(alpha, 1), 2)
            alpha = safe_logsumexp(torch.stack((within, boundary)), 0)
            alpha = (alpha + pe[t, :, :, None] + le[t, :, None, :]).masked_fill(~valid, -torch.inf)
        ends = alpha[-1, p.last[-1] - 1:p.last[-1] + 1, l.last[-1] - 1:l.last[-1] + 1]
        shared = safe_logsumexp(ends.flatten(), 0)
        independent = p.partition(phone) + l.partition(letter)
        # Uniform nonempty partitions: choose W-1 cuts among T-1 frame edges.
        prior = math.lgamma(len(phone)) - math.lgamma(p.count) - math.lgamma(len(phone) - p.count + 1)
        values = {"shared": shared - prior, "independent": independent - 2 * prior}
        if any(not bool(torch.isfinite(value)) for value in values.values()):
            raise ValueError("Transcript has no legal word-boundary/CTC path")
        return values


def objectives(heads, features, shipped, lattice, occurrences):
    """Same preservation and transcript normalization for both experimental arms."""
    import torch
    phone = heads["phone"](features).log_softmax(-1)
    letter = heads["letter"](features).log_softmax(-1)
    likelihoods = lattice.partitions(phone, letter)
    preserve = torch.nn.functional.kl_div(phone, shipped.exp(), reduction="batchmean")
    return {arm: -value / occurrences + preserve for arm, value in likelihoods.items()}


def recording_update(heads, optimizer, arm, features, shipped, lattice, occurrences):
    """Matched forward/backward compute, applying only the arm's own gradient."""
    import torch
    if arm not in ("shared", "independent"):
        raise ValueError("Unknown boundary arm")
    optimizer.zero_grad()
    values = objectives(heads, features, shipped, lattice, occurrences)
    params = tuple(heads.parameters())
    unused = "independent" if arm == "shared" else "shared"
    torch.autograd.grad(values[unused], params, retain_graph=True)
    gradients = torch.autograd.grad(values[arm], params)
    if not bool(torch.isfinite(values[arm])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Nonfinite objective or gradient; abort both arms")
    for param, gradient in zip(params, gradients):
        param.grad = gradient
    torch.nn.utils.clip_grad_norm_(params, 5.)
    optimizer.step()
    return values[arm].detach()
