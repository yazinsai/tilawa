"""Frozen-encoder phone-head refitting with transcript-constrained prefix CTC.

This is a training objective only. The deployed head, CTC inventory and streaming
state layout are unchanged. Alignment targets are detached and fixed before either
refit arm runs; validation never supplies prefixes to training.
"""

from dataclasses import dataclass
import copy
import hashlib
import math
import time

import torch


@dataclass(frozen=True)
class Prefix:
    frames: int
    # Number of required occurrences, and uncertainty over that number. The next
    # occurrence may be unentered or still active at this boundary.
    counts: tuple[int, ...]
    weights: tuple[float, ...]


def _sum(values, dim):
    """logsumexp with zero gradients for unreachable CTC states."""
    valid = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(valid, values, torch.zeros_like(values))
    return torch.where(valid.squeeze(dim), torch.logsumexp(safe, dim), -torch.inf)


def _graph(log_probs, target):
    if log_probs.ndim != 2 or not len(log_probs) or not len(target):
        raise ValueError("CTC needs nonempty frames and a nonempty transcript")
    target = torch.as_tensor(target, device=log_probs.device, dtype=torch.long)
    if target.ndim != 1 or (target <= 0).any() or (target >= log_probs.shape[1]).any():
        raise ValueError("Transcript must contain only nonblank inventory IDs")
    labels = torch.zeros(2 * len(target) + 1, device=log_probs.device, dtype=torch.long)
    labels[1::2] = target
    skip = torch.zeros_like(labels, dtype=torch.bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    return labels, skip


def _forward(log_probs, labels, skip):
    state = log_probs.new_full((len(labels),), -torch.inf)
    state[:2] = log_probs[0, labels[:2]]
    rows = [state]
    for emission in log_probs[1:]:
        one = torch.cat((state.new_full((1,), -torch.inf), state[:-1]))
        two = torch.cat((state.new_full((2,), -torch.inf), state[:-2]))
        state = _sum(torch.stack((state, one, two.masked_fill(~skip, -torch.inf))), 0) + emission[labels]
        rows.append(state)
    return torch.stack(rows)


def forward_states(log_probs, target):
    labels, skip = _graph(log_probs, target)
    return _forward(log_probs, labels, skip)


@torch.no_grad()
def alignment_prefixes(log_probs, target, boundaries, confidence=0.95):
    """Use the full-transcript posterior to bound progress at internal boundaries.

    At most two adjacent progress counts may hold the requested posterior mass.
    Diffuse boundaries are omitted, rather than admitting an unrestricted suffix.
    A count is complete at its trailing blank; its successor can be partly heard.
    These confidence and boundary choices belong to training, never practice.
    """
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    labels, skip = _graph(log_probs, target)
    alpha = _forward(log_probs, labels, skip)
    z = torch.logsumexp(alpha[-1, -2:], 0)
    if not torch.isfinite(z):
        raise ValueError("Transcript has no feasible CTC alignment")
    beta = log_probs.new_full(alpha.shape, -torch.inf)
    beta[-1, -2:] = 0
    for t in range(len(log_probs) - 2, -1, -1):
        nxt = log_probs[t + 1, labels] + beta[t + 1]
        one = torch.cat((nxt[1:], nxt.new_full((1,), -torch.inf)))
        two = torch.cat((nxt[2:], nxt.new_full((2,), -torch.inf)))
        allowed = torch.cat((skip[2:], skip.new_zeros(2)))
        beta[t] = _sum(torch.stack((nxt, one, two.masked_fill(~allowed, -torch.inf))), 0)
    posterior = (alpha + beta - z).exp()
    result = []
    for frames in sorted(set(boundaries)):
        if not 0 < frames < len(log_probs):
            raise ValueError("Prefix boundaries must be internal output-frame counts")
        # Blank 2k and active label 2k+1 both represent k completed phones.
        mass = posterior[frames - 1, 0::2].clone()
        mass[:-1] += posterior[frames - 1, 1::2]
        candidates = []
        for width in (1, 2):
            for k in range(1, len(target) - width + 1):
                weight = mass[k:k + width].sum().item()
                if weight >= confidence:
                    candidates.append((width, -weight, k))
            if candidates:
                break
        if not candidates:
            continue
        width, _, k = min(candidates)
        weights = mass[k:k + width]
        weights = weights / weights.sum()
        result.append(Prefix(frames, tuple(range(k, k + width)), tuple(weights.tolist())))
    return tuple(result)


def prefix_ctc_loss(log_probs, target, prefixes):
    """Marginalize the partially observed last occurrence, with bounded progress.

    Each count k accepts the last required label, its trailing blank, or the
    active optional label k+1. It cannot enter the blank after that optional
    label, nor any later phone. Repeated IDs still require the ordinary blank
    transition. Uncertain counts are a normalized mixture, not extra targets.
    """
    if not prefixes:
        return log_probs.exp().sum() * 0
    labels, skip = _graph(log_probs, target)
    losses = []
    for prefix in prefixes:
        if (not 0 < prefix.frames < len(log_probs) or not prefix.counts
                or len(prefix.counts) != len(prefix.weights)
                or any(k < 1 or k >= len(target) for k in prefix.counts)
                or tuple(sorted(set(prefix.counts))) != prefix.counts
                or prefix.counts[-1] - prefix.counts[0] > 1
                or any(not math.isfinite(w) or w <= 0 for w in prefix.weights)
                or not math.isclose(sum(prefix.weights), 1, abs_tol=1e-6)):
            raise ValueError("Invalid bounded prefix target")
        # Never compute a suffix or allow its labels to affect this loss.
        end = 2 * prefix.counts[-1] + 2
        alpha = _forward(log_probs[:prefix.frames], labels[:end], skip[:end])[-1]
        terms = [_sum(alpha[2*k-1:2*k+2], 0) + math.log(w)
                 for k, w in zip(prefix.counts, prefix.weights)]
        loss = -_sum(torch.stack(terms), 0) / prefix.frames
        if not torch.isfinite(loss):
            raise ValueError("Prefix target has no feasible CTC path")
        losses.append(loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def streaming_features(model, feats, chunk=16, left=128):
    """Cache frozen features using the export/SDK state loop, not a chunk mask.

    Preserve the original recording, warm context, lookahead and final padding.
    The output count is SDK outputsFor(fbankFrames), at the existing 40 ms clock.
    """
    if chunk <= 0 or chunk % 2 or left <= 0 or feats.ndim != 2 or feats.shape[1] != 80:
        raise ValueError("Invalid streaming feature geometry")
    model.eval()
    model.streaming_eval(chunk, left)
    device = next(model.parameters()).device
    feats = feats.to(device)
    total = max(0, ((len(feats) - 7) // 2 + 1) // 2)
    if not total:
        raise ValueError("Recording is too short for the deployed encoder")
    window, shift = 2 * chunk + 13, 2 * chunk
    states = model.encoder.get_init_states(1, device)
    pad = model.encoder_embed.get_init_states(1, device)
    processed = torch.zeros(1, dtype=torch.int64, device=device)
    outputs = []
    for start in range(0, math.ceil(total / (chunk // 2)) * shift, shift):
        x = feats[start:start + window]
        if len(x) < window:
            x = torch.cat((x, x.new_full((window - len(x), 80), -23.0259)))
        x_lens = torch.tensor([window], device=device)
        x, x_lens, pad = model.encoder_embed.streaming_forward(x=x[None], x_lens=x_lens, cached_left_pad=pad)
        mask = (processed[:, None] <= torch.arange(left, device=device)[None]).to(torch.int32).flip(1)
        mask = torch.cat((mask, mask.new_zeros((1, chunk))), 1)
        out, _, states = model.encoder.streaming_forward(x=x.permute(1, 0, 2), x_lens=x_lens, states=states,
                                                        src_key_padding_mask=mask)
        processed = processed + x_lens
        outputs.append(out[:, 0].detach())
    return torch.cat(outputs)[:total].cpu()


def speaker_partitions(speakers, seed=0):
    """Stable speaker-grouped train/validation partitions from eligible speakers."""
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if len(unique) < 2 or any(not s for s in unique):
        raise ValueError("At least two identified eligible speakers are required")
    n = max(1, len(unique) // 5)
    return set(unique[n:]), set(unique[:n])


def refit_pair(head, recordings, *, epochs=1, lr=0.001, seed=0, prefix_weight=1.0,
               preservation_weight=1.0, confidence=0.95, prefixes_per_recording=4, chunk=16,
               budget_seconds=24 * 3600):
    """Matched-data/update/compute ordinary and prefix-supervised head refits.

    Recordings hold fixed streaming features, authorized phone transcripts and
    eligible speaker IDs. The encoder is never in either optimizer. Both arms
    compute the prefix loss; the control discards it. Checkpoints are chosen
    independently by validation-speaker transcript CTC, never target recovery.
    Full-recording KL retains ordinary speech and pauses in both arms.
    """
    if (epochs < 1 or lr <= 0 or prefixes_per_recording < 1 or chunk <= 0 or chunk % 2
            or prefix_weight <= 0 or preservation_weight < 0 or not 0 < budget_seconds <= 24 * 3600):
        raise ValueError("Invalid head-refit configuration")
    started = time.monotonic()
    train_speakers, validation_speakers = speaker_partitions([r['speaker'] for r in recordings], seed)
    device = next(head.parameters()).device
    # Dropout is disabled for fixed encoder features and both matched head arms.
    anchor = copy.deepcopy(head).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    rng = torch.Generator().manual_seed(seed)
    prepared = []
    with torch.no_grad():
        for r in recordings:
            if time.monotonic() - started >= budget_seconds:
                raise TimeoutError('One-day head-refit budget exhausted')
            if (r.get('source') not in ('authorized-transcribed', 'help-train-none')
                    or r.get('held_out') is not False
                    or r.get('transcript_kind') != 'performed'
                    or (r['source'] == 'help-train-none' and r.get('label') != 'none')):
                raise ValueError("Refit accepts only explicitly eligible transcribed training recordings")
            x = r['features'].to(device)
            target = torch.as_tensor(r['target'], dtype=torch.long, device=device)
            lp = anchor(x).log_softmax(-1)
            _graph(lp, target)
            boundaries = list(range(chunk // 2, len(lp), chunk // 2))
            candidates = (alignment_prefixes(lp, target, boundaries, confidence)
                          if r['speaker'] in train_speakers else ())
            if candidates:
                order = torch.randperm(len(candidates), generator=rng).tolist()[:prefixes_per_recording]
                prefixes = tuple(candidates[i] for i in sorted(order))
            else:
                prefixes = ()
            prepared.append(dict(x=x, target=target, anchor=lp, prefixes=prefixes, speaker=r['speaker']))
    training = [r for r in prepared if r['speaker'] in train_speakers]
    validation = [r for r in prepared if r['speaker'] in validation_speakers]
    if not any(r['prefixes'] for r in training):
        raise ValueError("No confident internal training prefixes; refit would not test the mechanism")
    arms = {name: copy.deepcopy(head).eval() for name in ('control', 'prefix')}
    opts = {name: torch.optim.Adam(m.parameters(), lr=lr) for name, m in arms.items()}
    best, history = {}, []

    def full_loss(lp, r):
        return torch.nn.functional.ctc_loss(lp[:, None], r['target'], [len(lp)], [len(r['target'])],
                                            blank=0, reduction='sum', zero_infinity=False) / len(lp)

    for epoch in range(epochs):
        for i in torch.randperm(len(training), generator=rng).tolist():
            if time.monotonic() - started >= budget_seconds:
                raise TimeoutError('One-day head-refit budget exhausted')
            r = training[i]
            for name, m in arms.items():
                lp = m(r['x']).log_softmax(-1)
                ordinary = full_loss(lp, r)
                prefix = prefix_ctc_loss(lp, r['target'], r['prefixes'])
                preserve = torch.nn.functional.kl_div(lp, r['anchor'], log_target=True, reduction='sum') / len(lp)
                loss = ordinary + preservation_weight * preserve + (prefix_weight if name == 'prefix' else 0) * prefix
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite head-refit loss")
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                opts[name].step()
        with torch.no_grad():
            scores = {name: sum(full_loss(m(r['x']).log_softmax(-1), r).item() for r in validation) / len(validation)
                      for name, m in arms.items()}
        history.append(dict(epoch=epoch + 1, validation_ctc=scores))
        for name, score in scores.items():
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation CTC")
            if name not in best or score < best[name]['validation_ctc']:
                best[name] = dict(epoch=epoch + 1, validation_ctc=score,
                                  state_dict=copy.deepcopy(arms[name].state_dict()))
    return best, dict(history=history, train_speakers=sorted(train_speakers),
                      validation_speakers=sorted(validation_speakers),
                      training_recordings=len(training), validation_recordings=len(validation),
                      training_prefixes=sum(len(r['prefixes']) for r in training),
                      updates_per_arm=epochs * len(training), seed=seed)
