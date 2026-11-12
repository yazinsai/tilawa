"""Training-only recycling of Zipformer feed-forward expansion channels.

No hooks or extra modules survive export. Selection uses every calibration
recording, not a pooled statistic which can hide an intermittently active unit.
SwooshL is smooth: negative preactivations are not evidence of dormancy.
"""

import hashlib

import numpy as np


def swoosh_l(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64) - 4.0) - .08 * x - .035


def channel_statistics(preactivation, outgoing):
    """Actual post-activation RMS and vector outgoing contributions; no padding."""
    x, w = np.asarray(preactivation), np.asarray(outgoing)
    if (x.ndim != 2 or w.ndim != 2 or not len(x) or x.shape[1] != w.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(w).all()):
        raise ValueError("Invalid valid-frame channel observations")
    a = swoosh_l(x)
    activity = np.sqrt(np.mean(a * a, axis=0))
    contribution = activity * np.linalg.norm(w, axis=0)
    output_rms = float(np.sqrt(np.mean(np.sum((a @ w.T) ** 2, axis=1))))
    return dict(activity=activity, contribution=contribution, output_rms=output_rms)


def dormant_channels(observations, activity_ratio=.1, contribution_ratio=1e-6):
    """Conservative candidates; numerical thresholds are fixed before decoding.

    This is an admission test, not a proof of function preservation. The runner
    independently checks the changed probabilities and greedy paths.
    """
    if len(observations) < 2 or not 0 < activity_ratio < 1 or not 0 < contribution_ratio < 1:
        raise ValueError("Need multiple recordings and bounded admission ratios")
    size = len(observations[0]["activity"])
    keep = np.ones(size, dtype=bool)
    for r in observations:
        a, c = np.asarray(r["activity"]), np.asarray(r["contribution"])
        if (a.shape != (size,) or c.shape != (size,) or not np.isfinite(a).all()
                or not np.isfinite(c).all() or np.any(a < 0) or np.any(c < 0)
                or not np.isfinite(r["output_rms"]) or r["output_rms"] < 0):
            raise ValueError("Invalid per-recording channel statistics")
        keep &= (a <= activity_ratio * np.median(a)) & (c <= contribution_ratio * r["output_rms"])
    return np.flatnonzero(keep).tolist()


def speaker_partition(speakers, seed=1729):
    speakers = set(speakers)
    if len(speakers) < 3 or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("At least three identified training speakers are required")
    order = sorted(speakers, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    selection = set(order[:max(1, len(order) // 5)])
    return {s: "selection" if s in selection else "fit" for s in sorted(speakers)}


def admitted_none(labels, excluded):
    """A script is never a performed transcript; only audited clean takes enter."""
    forbidden = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    return [r for r in labels if r.get("speaker") and r["speaker"] not in forbidden
            and r.get("split") == "train" and r.get("kind") == "none"
            and r.get("scored") is True and not r.get("excluded") and not r.get("exclude")
            and r.get("events") == [] and r.get("non_errors") == []
            and r.get("unsupported", []) == [] and r.get("ayahs") and r.get("wav")]


def transcript_path(log_probs, target):
    """CTC Viterbi alignment, including the repeated-label blank constraint.

    It is used only to locate poorly represented training consonants, never to
    label practice audio or select a checkpoint.
    """
    lp, target = np.asarray(log_probs), np.asarray(target, dtype=np.int64)
    if (lp.ndim != 2 or not len(lp) or target.ndim != 1 or not len(target) or np.any(target <= 0)
            or np.any(target >= lp.shape[1]) or not np.isfinite(lp).all()):
        raise ValueError("Invalid CTC alignment inputs")
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    previous = np.full(len(labels), -np.inf)
    previous[:2] = lp[0, labels[:2]]
    back = np.zeros((len(lp), len(labels)), dtype=np.int8)
    skip = (labels != 0) & (labels != np.roll(labels, 2))
    skip[:2] = False
    for t in range(1, len(lp)):
        one = np.r_[-np.inf, previous[:-1]]
        two = np.r_[[-np.inf, -np.inf], previous[:-2]]
        two[~skip] = -np.inf
        choices = np.stack([previous, one, two])
        back[t] = choices.argmax(0)
        previous = choices.max(0) + lp[t, labels]
    end = len(labels) - 2 + int(previous[-1] > previous[-2])
    if not np.isfinite(previous[end]):
        raise ValueError("Complete transcript does not fit the acoustic clock")
    path = np.empty(len(lp), dtype=np.int64)
    for t in range(len(lp) - 1, -1, -1):
        path[t] = labels[end]
        end -= int(back[t, end]) if t else 0
    return path


def upper_feedforwards(model):
    """Last two encoder stacks; residual/cache channel geometry stays intact."""
    out = {}
    stacks = model.encoder.encoders
    for stage in range(max(0, len(stacks) - 2), len(stacks)):
        for local, module in stacks[stage].named_modules():
            if module.__class__.__name__ != "FeedforwardModule":
                continue
            if module.out_proj.activation != "SwooshL":
                raise ValueError("Unsupported expansion activation")
            out[f"encoder.encoders.{stage}.{local}"] = (module, model.encoder.downsampling_factor[stage])
    if not out:
        raise ValueError("No supported upper expansion channels")
    return out


def recycle(module, channels, examples):
    """Initialize template detectors; zero outgoing columns, preserve geometry.

    This removes the *old* contributions. Nonzero contributions must pass the
    runner's independent initial-function check, even if they looked dormant.
    Random controls use this identical operation and record their initial drift.
    """
    import torch

    w, b, outgoing = module.in_proj.weight, module.in_proj.bias, module.out_proj.weight
    channels = list(channels)
    examples = np.asarray(examples)
    if (not channels or len(set(channels)) != len(channels) or min(channels) < 0
            or max(channels) >= w.shape[0] or outgoing.shape[1] != w.shape[0]
            or examples.ndim != 2 or not len(examples) or examples.shape[1] != w.shape[1]
            or not np.isfinite(examples).all() or np.any(np.linalg.norm(examples, axis=1) == 0)):
        raise ValueError("Invalid recycling plan or weak-acoustic templates")
    with torch.no_grad():
        scale = w.norm(dim=1).median()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Degenerate incoming-weight scale")
        for i, channel in enumerate(channels):
            x = torch.as_tensor(examples[i % len(examples)], dtype=w.dtype, device=w.device)
            direction = x / x.norm()
            w[channel].copy_(direction * scale)
            # The template lies at the existing SwooshL activation shoulder.
            b[channel].copy_(4.0 - torch.dot(w[channel], x))
        outgoing[:, channels] = 0


def freeze_stem(model):
    model.requires_grad_(False)
    for stack in list(model.encoder.encoders)[-2:]:
        stack.requires_grad_(True)
    model.encoder.downsample_output.requires_grad_(True)
    model.ctc_output.requires_grad_(True)


def preservation(reference, changed, tolerance=1e-6):
    """Empirical probability/greedy parity, explicitly not an algebraic proof."""
    a, b = np.asarray(reference), np.asarray(changed)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Initial acoustic clock or probability shape changed")
    drift = float(np.max(np.abs(np.exp(a) - np.exp(b))))
    agree = bool(np.array_equal(a.argmax(-1), b.argmax(-1)))
    return dict(max_probability_drift=drift, frame_argmax_equal=agree,
                passes=drift <= tolerance and agree)
