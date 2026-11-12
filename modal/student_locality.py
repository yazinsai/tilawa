"""Training-only acoustic grounding of transcript-conditioned CTC occurrences.

No alignment is hardened: CTC state occupancies supply soft own/neighbour masks.
Antithetic probes avoid second derivatives through Icefall's custom autograd.
Only the last two Zipformer stacks and existing phone head are refitted.
"""

import copy
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def occurrence_posteriors(log_probs, target):
    """Return log P(transcript) and distinct extended-state occupancies [T,2U+1].

    Forward includes the current emission; backward includes only future
    emissions. Repeated occurrences remain separate, and equal consecutive
    labels require a blank between them. This calculation has no training graph.
    """
    lp = log_probs.detach().double().cpu().numpy()
    y = np.asarray(target.detach().cpu(), dtype=np.int64)
    if (lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all() or y.ndim != 1
            or not len(y) or np.any(y <= 0) or np.any(y >= lp.shape[1])):
        raise ValueError("Need finite CTC rows and a nonempty nonblank transcript")
    if len(lp) < len(y) + int(np.sum(y[1:] == y[:-1])):
        raise ValueError("Transcript has no feasible CTC alignment")
    labels = np.zeros(2 * len(y) + 1, dtype=np.int64)
    labels[1::2] = y
    states, frames = len(labels), len(lp)
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = np.full((frames, states), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for t in range(1, frames):
        previous = alpha[t - 1]
        advance = np.concatenate(([-np.inf], previous[:-1]))
        jump = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        alpha[t] = np.logaddexp(np.logaddexp(previous, advance),
                               np.where(skip, jump, -np.inf)) + lp[t, labels]
    log_z = float(np.logaddexp(alpha[-1, -1], alpha[-1, -2]))
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0.
    for t in range(frames - 2, -1, -1):
        following = beta[t + 1] + lp[t + 1, labels]
        advance = np.concatenate((following[1:], [-np.inf]))
        jump = np.concatenate((following[2:], [-np.inf, -np.inf]))
        can_jump = np.concatenate((skip[2:], [False, False]))
        beta[t] = np.logaddexp(np.logaddexp(following, advance),
                              np.where(can_jump, jump, -np.inf))
    gamma = np.exp(alpha + beta - log_z)
    if not math.isfinite(log_z) or not np.allclose(gamma.sum(1), 1., atol=1e-7):
        raise ValueError("Invalid transcript-conditioned occurrence distribution")
    return log_z, torch.as_tensor(gamma, device=log_probs.device, dtype=log_probs.dtype)


def occurrence_masks(gamma, occurrence, lower_frames):
    """Probabilities of own emission and occurrence plus immediate neighbours.

    Interior blanks belong to the local neighbourhood; leading/trailing silence
    does not. Zipformer's final factor-two pooling fixes the lower-frame clock.
    """
    count = (gamma.shape[1] - 1) // 2
    if not 0 <= occurrence < count or lower_frames not in (2 * len(gamma), 2 * len(gamma) - 1):
        raise ValueError("Occurrence or lower/output frame clock differs")
    state = 2 * occurrence + 1
    lo, hi = max(1, state - 2), min(2 * count - 1, state + 2)
    own = gamma[:, state]
    local = gamma[:, lo:hi + 1].sum(1).clamp(0., 1.)
    return (own.repeat_interleave(2)[:lower_frames].detach(),
            local.repeat_interleave(2)[:lower_frames].detach())


def transcript_score(log_probs, target):
    """Differentiable CTC forward sum, including all occurrence boundaries."""
    if (log_probs.ndim != 2 or target.ndim != 1 or not len(log_probs)
            or torch.any(target <= 0) or torch.any(target >= log_probs.shape[1])):
        raise ValueError("Invalid CTC sequence")
    required = len(target) + int((target[1:] == target[:-1]).sum())
    if len(log_probs) < required:
        raise ValueError("Transcript has no feasible CTC alignment")
    score = -F.ctc_loss(log_probs[:, None], target,
                        torch.tensor([len(log_probs)]), torch.tensor([len(target)]),
                        blank=0, reduction="sum", zero_infinity=False)
    if not torch.isfinite(score):
        raise ValueError("Nonfinite CTC score")
    return score


def occurrence_score(log_probs, target, occurrence):
    if not 0 <= occurrence < len(target):
        raise ValueError("Occurrence outside transcript")
    without = torch.cat((target[:occurrence], target[occurrence + 1:]))
    # Antithetic differences subtract four utterance-level forward sums. Retain
    # their precision even when upper activations and parameter gradients are fp32.
    lp = log_probs.double()
    return transcript_score(lp, target) - transcript_score(lp, without)


@dataclass
class LowerCache:
    frames: torch.Tensor
    attention_mask: object
    chunk: int


class _CacheReady(Exception):
    pass


class UpperRefit(nn.Module):
    """Cache the frozen prefix; run the unchanged final stacks on that cache.

    Evaluation semantics make both probes deterministic. Gradients remain on
    upper parameters; augmentation supplies the refit's stochastic input views.
    """

    def __init__(self, model, chunk=16, left=128):
        super().__init__()
        self.model = model.eval()
        self.model.streaming_eval(chunk, left)
        enc = model.encoder
        if (len(enc.encoders) < 3 or enc.output_downsampling_factor != 2
                or enc.encoder_dim[-1] != max(enc.encoder_dim)):
            raise ValueError("Unsupported Zipformer stack/clock/full-dimension merge")
        self.cut = len(enc.encoders) - 2
        for p in model.parameters():
            p.requires_grad_(False)
        for module in list(enc.encoders)[self.cut:] + [model.ctc_output]:
            for p in module.parameters():
                p.requires_grad_(True)
        self.chunk = chunk

    def cache(self, features):
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("Need one complete finite recording")
        captured = []

        def stop(module, args, kwargs):
            if kwargs.get("feature_mask", 1.) != 1.:
                raise ValueError("Frozen prefix must use deterministic feature masks")
            captured.append(LowerCache(args[0].detach().clone(),
                                       kwargs.get("attn_mask"), kwargs["chunk_size"]))
            raise _CacheReady()

        hook = self.model.encoder.encoders[self.cut].register_forward_pre_hook(stop, with_kwargs=True)
        try:
            with torch.no_grad():
                self.model(features[None], torch.tensor([len(features)], device=features.device))
        except _CacheReady:
            pass
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("Failed to capture exactly one frozen lower-layer sequence")
        return captured[0]

    def forward(self, cache, frames=None):
        x = cache.frames if frames is None else frames
        if x.shape != cache.frames.shape:
            raise ValueError("A probe changed the acoustic frame clock")
        enc = self.model.encoder
        # Native convert_num_channels pads/truncates channels; no learned mapping.
        for i in range(self.cut, len(enc.encoders)):
            dim = enc.encoder_dim[i]
            if x.shape[-1] < dim:
                x = F.pad(x, (0, dim - x.shape[-1]))
            else:
                x = x[..., :dim]
            x = enc.encoders[i](x, chunk_size=cache.chunk, feature_mask=1.,
                                attn_mask=cache.attention_mask, src_key_padding_mask=None)
        x = enc.downsample_output(x)
        return self.model.ctc_output(x[:, 0]).log_softmax(-1)


def probe_directions(cache, own, local, seed):
    """Hutchinson directions with soft masks, scaled in acoustic feature units."""
    x = cache.frames
    generator = torch.Generator(device=x.device).manual_seed(seed)
    signs = torch.randint(0, 2, x.shape, generator=generator, device=x.device).to(x.dtype) * 2 - 1
    scale = x.detach().square().mean(0, keepdim=True).sqrt().clamp_min(1e-4)
    # No normalization by remote length: this estimates total squared sensitivity.
    return (signs * scale * own.sqrt()[:, None, None],
            signs * scale * (1. - local).sqrt()[:, None, None])


def directional_response(model, cache, target, occurrence, direction, epsilon=.01):
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("Probe radius must be positive and finite")
    a = occurrence_score(model(cache, cache.frames + epsilon * direction), target, occurrence)
    b = occurrence_score(model(cache, cache.frames - epsilon * direction), target, occurrence)
    return (a - b) / (2. * epsilon)


def grounding_loss(own_response, remote_response, initial_own_response):
    """Reduce remote response without obtaining zero loss by flattening all scores.

    The initial student's own response supplies only a magnitude floor, never a
    teacher sensitivity direction or a remote-dependence target. The same fixed
    scale is used throughout the matched experiment, not refitted on practice.
    """
    anchor = initial_own_response.detach().square()
    scale = anchor.clamp_min(1e-4)
    return (remote_response.square() + F.relu(anchor - own_response.square())) / scale


def speaker_partition(speakers, excluded=()):
    unique = set(speakers)
    if len(unique) < 5 or any(not isinstance(s, str) or not s for s in unique):
        raise ValueError("Need at least five identified training speakers")
    if unique.intersection(excluded):
        raise ValueError("Excluded speaker enrolled in refit")
    order = sorted(unique, key=lambda s: hashlib.sha256(s.encode()).digest())
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(order)}


def paired_refit(initial, rows, groups, epochs=4, lr=1e-5, seed=1729,
                 max_seconds=45 * 60, emit=lambda value: None):
    """Identical order, views, probes and updates; one shared training-only epoch.

    Rows contain cached lower frames, transcript, initial full-frame posteriors,
    and fixed occurrence probes/anchors. Selection uses ordinary transcript plus
    preservation loss, equal weight per recording and then per speaker.
    """
    import time

    if epochs < 1 or lr <= 0 or max_seconds <= 0:
        raise ValueError("Invalid bounded refit budget")
    if any(r["speaker"] not in groups for r in rows) or not rows:
        raise ValueError("Missing training speaker partition")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need disjoint fit and selection speakers")
    arms = {name: copy.deepcopy(initial) for name in ("ordinary", "grounded")}
    optimizers = {name: torch.optim.Adam([p for p in arm.parameters() if p.requires_grad], lr=lr)
                  for name, arm in arms.items()}

    def ordinary(arm, view):
        lp = arm(view["cache"])
        ctc = -transcript_score(lp, view["target"]) / len(lp)
        preserve = F.kl_div(lp, view["shipped"].exp(), reduction="batchmean")
        return ctc + preserve

    def validate(arm):
        by_speaker = {}
        with torch.no_grad():
            for row in selection:
                values = [float(ordinary(arm, v)) for v in row["views"]]
                by_speaker.setdefault(row["speaker"], []).append(float(np.mean(values)))
        return float(np.mean([np.mean(v) for v in by_speaker.values()]))

    def snapshot():
        return {name: copy.deepcopy(arm.state_dict()) for name, arm in arms.items()}

    started, updates = time.monotonic(), 0
    scores = {name: validate(arm) for name, arm in arms.items()}
    best, best_epoch, best_states = np.mean(list(scores.values())), 0, snapshot()
    history = [dict(epoch=0, updates=0, selection=scores)]
    for epoch in range(1, epochs + 1):
        for index in np.random.default_rng(seed + epoch).permutation(len(fit)):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Refit budget exhausted before a complete matched epoch")
            row = fit[int(index)]
            for name, arm in arms.items():
                optimizers[name].zero_grad(set_to_none=True)
                losses = []
                for view in row["views"]:
                    loss = ordinary(arm, view)
                    probe = view["probes"][(epoch - 1) % len(view["probes"])]
                    own = directional_response(arm, view["cache"], view["target"],
                                               probe["occurrence"], probe["own"])
                    remote = directional_response(arm, view["cache"], view["target"],
                                                  probe["occurrence"], probe["remote"])
                    penalty = grounding_loss(own, remote, probe["anchor"])
                    # The control executes identical probes with zero objective weight.
                    losses.append(loss + (.01 if name == "grounded" else 0.) * penalty)
                total = torch.stack(losses).mean()
                if not torch.isfinite(total):
                    raise ValueError("Nonfinite matched refit objective")
                total.backward()
                params = [p for p in arm.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                optimizers[name].step()
            updates += 1
        scores = {name: validate(arm) for name, arm in arms.items()}
        record = dict(epoch=epoch, updates=updates, selection=scores)
        history.append(record)
        emit(record)
        value = np.mean(list(scores.values()))
        if value < best:
            best, best_epoch, best_states = value, epoch, snapshot()
    for name, arm in arms.items():
        arm.load_state_dict(best_states[name])
    return arms, dict(selectedEpoch=best_epoch, selectedUpdates=best_epoch * len(fit),
                      trainingUpdates=updates, history=history,
                      selectionRule="minimum pair-average ordinary objective, equal training speakers")
