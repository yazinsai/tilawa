"""Training-only cross-recording phone-occurrence distillation.

Boundaries are marginalized by exact transcript-conditioned CTC forward-backward,
then detached: the trainable encoder must change its features, rather than move
boundaries to reduce the relationship loss. Blank states never become phones.
Both arms pool the same occurrences; no expected text reaches inference.
"""
import math
from contextlib import contextmanager


def occurrence_weights(log_probs, labels, blank=0):
    """Return (occurrences, frames) pooling weights over all valid CTC paths.

    Repeated labels have separate odd states, with a required intervening blank.
    Normalize each occurrence by its expected occupied duration, so a held vowel
    contributes one occurrence rather than many independent feature targets.
    """
    import torch

    lp = log_probs.detach().double()
    labels = torch.as_tensor(labels, device=lp.device, dtype=torch.long)
    if (lp.ndim != 2 or lp.shape[0] == 0 or labels.ndim != 1 or labels.numel() == 0
            or (labels == blank).any() or (labels < 0).any() or (labels >= lp.shape[1]).any()
            or torch.isnan(lp).any() or torch.isposinf(lp).any()):
        raise ValueError("Require nonempty valid phone transcript and CTC rows")
    states = torch.full((2 * len(labels) + 1,), blank, device=lp.device, dtype=torch.long)
    states[1::2] = labels
    skip = torch.zeros_like(states, dtype=torch.bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    emission = lp[:, states]
    initial = lp.new_full((len(states),), -torch.inf)
    initial[:2] = emission[0, :2]
    forward = [initial]
    for row in emission[1:]:
        prev = forward[-1]
        one = torch.cat((lp.new_full((1,), -torch.inf), prev[:-1]))
        two = torch.cat((lp.new_full((2,), -torch.inf), prev[:-2])).masked_fill(~skip, -torch.inf)
        forward.append(torch.logsumexp(torch.stack((prev, one, two)), 0) + row)
    log_z = torch.logsumexp(forward[-1][-2:], 0)
    if not torch.isfinite(log_z):
        raise ValueError("Transcript has no finite complete CTC alignment")
    final = lp.new_full((len(states),), -torch.inf)
    final[-2:] = 0
    backward = [final]
    for row in reversed(emission[1:]):
        nxt = backward[-1] + row
        one = torch.cat((nxt[1:], lp.new_full((1,), -torch.inf)))
        two = torch.cat((nxt[2:].masked_fill(~skip[2:], -torch.inf), lp.new_full((2,), -torch.inf)))
        backward.append(torch.logsumexp(torch.stack((nxt, one, two)), 0))
    gamma = (torch.stack(forward) + torch.stack(list(reversed(backward))) - log_z).exp()[:, 1::2].T
    duration = gamma.sum(1, keepdim=True)
    if (duration <= 0).any() or not torch.isfinite(gamma).all():
        raise ValueError("Transcript occurrence has no acoustic occupancy")
    return (gamma / duration).to(log_probs.dtype)


def pool_occurrences(features, log_probs, labels):
    if features.ndim != 2 or features.shape[0] != log_probs.shape[0]:
        raise ValueError("Features and CTC rows must share their model's valid clock")
    return occurrence_weights(log_probs, labels).to(features) @ features


def comparison_mask(phones, recordings, device):
    """All cross-recording rivals, including every confusable-phone negative.

    Train anchors only when another recording supplies both a same-phone example
    and a different phone. Sparse batches must be reported, never used as a
    pointwise or same-recording fallback. Candidate columns remain cross-recording
    even when they are not themselves eligible anchors.
    """
    import torch

    if len(phones) != len(recordings) or not phones:
        raise ValueError("Require phone and recording identity for every occurrence")
    ids = {key: i for i, key in enumerate(dict.fromkeys(recordings))}
    records = torch.tensor([ids[r] for r in recordings], device=device)
    phones = torch.as_tensor(phones, device=device)
    cross = records[:, None] != records[None, :]
    same = phones[:, None] == phones[None, :]
    anchors = (cross & same).any(1) & (cross & ~same).any(1)
    return cross, anchors


def normalized_features(features):
    import torch
    import torch.nn.functional as F

    if (features.ndim != 2 or not torch.isfinite(features).all()
            or (features.norm(dim=1) == 0).any()):
        raise ValueError("Require finite, nonzero occurrence features")
    return F.normalize(features, dim=1)


def distillation_loss(student, teacher, phones, recordings, mode, temperature, projection=None):
    """Teacher KL over normalized similarity distributions, or matched hint MSE.

    Relationship transfer needs no common embedding coordinates or equal feature
    dimensions. Every retained anchor has equal weight. Both modes use exactly
    the same eligibility, occurrences and frozen teacher features.
    """
    import torch
    import torch.nn.functional as F

    if (mode not in ("relational", "hint") or not math.isfinite(temperature) or temperature <= 0
            or student.shape[0] != teacher.shape[0] or student.shape[0] != len(phones)):
        raise ValueError("Invalid matched occurrence distillation inputs")
    cross, anchors = comparison_mask(phones, recordings, student.device)
    count = int(anchors.sum())
    if not count:
        raise ValueError("Batch has no cross-recording anchors with positive and negative phones")
    teacher = normalized_features(teacher.detach().to(student.device))
    if mode == "hint":
        if projection is None:
            raise ValueError("Matched hint arm requires a training-only projection")
        predicted = normalized_features(projection(student))
        if predicted.shape != teacher.shape:
            raise ValueError("Hint projection differs from teacher feature shape")
        loss = (predicted[anchors] - teacher[anchors]).square().sum(1).mean()
    else:
        student = normalized_features(student)
        ts = (teacher @ teacher.T / temperature)[anchors].masked_fill(~cross[anchors], -torch.inf)
        ss = (student @ student.T / temperature)[anchors].masked_fill(~cross[anchors], -torch.inf)
        target = ts.softmax(1)
        # Select legal columns before multiplication: 0 * -inf is NaN.
        log_target = ts.log_softmax(1).masked_fill(~cross[anchors], 0)
        log_student = ss.log_softmax(1).masked_fill(~cross[anchors], 0)
        loss = (target * (log_target - log_student)).sum(1).mean()
    return loss, {"occurrences": len(phones), "anchors": count,
                  "comparisons": int(cross[anchors].sum()), "recordings": len(set(recordings))}


@contextmanager
def deterministic_features(model):
    """Disable training noise in feature, acoustic-KD and preservation passes.

    Do not change the deployed chunk/left-context geometry. Restoring individual
    module flags also preserves any intentionally frozen submodules.
    """
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in flags:
            module.training = training


def batch_objective(model, anchor, teacher_rows, batch, mode, temperature, weights, projection=None):
    """Joint encoder objective on matched augmented audio and real pauses.

    teacher_rows contains fresh features and full posteriors for each exact PCM
    item. Each model pools on its own valid grid; no frame-boundary interpolation
    is needed. Output KD retains the existing synchronous timing correspondence.
    """
    import torch
    import torch.nn.functional as F

    if (set(weights) != {"ctc", "kd", "preservation", "distillation"}
            or any(not math.isfinite(w) or w <= 0 for w in weights.values())):
        raise ValueError("Retain positive acoustic, preservation and distillation losses")
    if (len(teacher_rows) != len(batch["keys"]) or len(set(batch["keys"])) != len(batch["keys"])
            or len(batch["recordings"]) != len(batch["keys"])):
        raise ValueError("Matched batch needs unique item keys and parent recording identities")
    device = next(model.parameters()).device
    feats, lens = batch["feats"].to(device), batch["feat_lens"].to(device)
    with deterministic_features(model):
        features, out_lens = model.forward_features(feats, lens)
        lp = model.ctc_output(features).float().log_softmax(-1)
    with torch.no_grad(), deterministic_features(anchor):
        anchored, anchor_lens = anchor(feats, lens)
    if not torch.equal(out_lens, anchor_lens):
        raise ValueError("Ordinary-reading preservation changed the primary clock")
    targets = batch["targets"].to(device)
    target_lens = batch["target_lens"].to(device)
    n_frames = out_lens.sum()
    ctc = F.ctc_loss(lp.transpose(0, 1), targets, out_lens, target_lens, blank=0,
                     reduction="sum", zero_infinity=False) / n_frames
    valid = torch.arange(lp.shape[1], device=device)[None] < out_lens[:, None]
    preservation = F.kl_div(lp[valid], anchored.detach().float().exp()[valid], reduction="sum") / n_frames
    students, teachers, phones, recordings = [], [], [], []
    kd_sum, kd_frames, offset = lp.new_zeros(()), 0, 0
    for i, (key, rows) in enumerate(zip(batch["keys"], teacher_rows)):
        if rows["key"] != key:
            raise ValueError("Teacher features belong to another recording")
        n, length = int(out_lens[i]), int(target_lens[i])
        labels = targets[offset:offset + length]
        offset += length
        tlp = rows["log_probs"].detach().to(lp)
        tf = rows["features"].detach().to(features)
        if tlp.shape[1] != lp.shape[2]:
            raise ValueError("Teacher alphabet differs from primary listener")
        students.append(pool_occurrences(features[i, :n], lp[i, :n], labels))
        teachers.append(pool_occurrences(tf, tlp, labels))
        phones.extend(labels.tolist())
        recordings.extend([batch["recordings"][i]] * length)
        # Same edge-clamped clock and renormalized top-five CE as output KD.
        # Refresh targets on the augmented PCM rather than reuse clean targets.
        index = torch.arange(n, device=device).clamp(max=len(tlp) - 1)
        top_p, top_ids = tlp[index].exp().topk(min(5, tlp.shape[1]), dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True).clamp_min(1e-6)
        kd_sum = kd_sum - (top_p * lp[i, :n].gather(-1, top_ids)).sum()
        kd_frames += n
    distilled, counts = distillation_loss(torch.cat(students), torch.cat(teachers), phones, recordings,
                                          mode, temperature, projection)
    terms = {"ctc": ctc, "kd": kd_sum / kd_frames, "preservation": preservation, "distillation": distilled}
    return sum(weights[k] * v for k, v in terms.items()), terms, counts
