"""Matched frozen-head refits with acoustic-regime optimizer histories.

The objective is the shipped CTC + 2 * top-five KD, per valid frame. CTC's
posterior occupancy splits its *gradient*, not its transcripts or objective.
Conditional regime gradients are recombined with their original occupancy
mass, preventing three adaptive histories from tripling the update scale.
"""
import copy
import hashlib
import math
import random
import time

import torch
from torch.nn import functional as F

REGIMES = ("consonant", "duration", "blank")
# Same ordinary consonants as core/gap-decode.ts. Nasals (including the explicit
# ghunna symbols) join vowels, madd and pronunciation marks in duration; their
# inventory identities remain unchanged in CTC and the exported head.
CONSONANTS = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
NASALS = frozenset("منں۾")
ARMS = ("shared", "separated", "memoryless")


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five identified training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for row in labels:
        if row.get("split") != "train" or row.get("kind") != "none":
            continue
        if not row.get("speaker") or row["speaker"] in excluded:
            continue
        if (row.get("scored") is not True or any(row.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or row.get("excluded") or row.get("exclude")):
            continue
        if not row.get("id") or not row.get("wav") or not row.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if row["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(row["id"])
        selected.append(row)
    return selected


def phone_head(weight, bias):
    w, b = torch.as_tensor(weight), torch.as_tensor(bias)
    if w.ndim != 2 or b.shape != (len(w),) or not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ValueError("Invalid existing affine head")
    head = torch.nn.Linear(w.shape[1], w.shape[0], dtype=w.dtype)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    return head


def log_probs(head, features):
    return head(features.detach()).log_softmax(-1)


def ctc_loss(lp, target):
    if (lp.ndim != 2 or not len(lp) or target.ndim != 1 or target.dtype != torch.int64 or
            (target <= 0).any() or (target >= lp.shape[1]).any() or
            len(lp) < len(target) + int((target[1:] == target[:-1]).sum()) or
            not torch.isfinite(lp).all()):
        raise ValueError("Impossible or invalid complete transcript alignment")
    return F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
        torch.tensor([len(target)]), blank=0, reduction="sum", zero_infinity=False) / len(lp)


def occupancy(lp, target):
    """Exact transcript-conditioned label occupancy, including repeated phones.

    PyTorch CTC's log-probability backward returns exp(lp) - occupancy. Extract
    the latter in double precision on a detached copy. Exhaustive path tests
    check this identity independently; encoder and teacher receive no gradient.
    """
    with torch.enable_grad():
        detached = lp.detach().double().requires_grad_(True)
        loss = ctc_loss(detached, target) * len(lp)
        grad, = torch.autograd.grad(loss, detached)
    q = detached.detach().exp() - grad
    if (not torch.isfinite(q).all() or q.min() < -1e-8 or
            not torch.allclose(q.sum(-1), torch.ones(len(lp), dtype=q.dtype), atol=1e-7, rtol=1e-7)):
        raise ValueError("CTC occupancy is not a conditional probability distribution")
    return q.clamp_min(0).to(lp.dtype).detach()


def teacher_distribution(teacher, frames):
    if teacher.ndim != 2 or not len(teacher) or not torch.isfinite(teacher).all():
        raise ValueError("Need finite teacher evidence")
    # The original losses() clamps its frame mapping at the teacher boundary.
    index = torch.arange(frames).clamp_max(len(teacher) - 1)
    values, ids = teacher.detach().log_softmax(-1).exp()[index].topk(min(5, teacher.shape[1]), -1)
    values = values / values.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.zeros((frames, teacher.shape[1]), dtype=teacher.dtype).scatter_(-1, ids, values)


def objectives(lp, target, teacher, shipped=None):
    if teacher.shape[1] != lp.shape[1]:
        raise ValueError("Teacher alphabet differs")
    ctc = ctc_loss(lp, target)
    kd = -(teacher_distribution(teacher, len(lp)) * lp).sum(-1).mean()
    return ctc, kd


def gradient_parts(head, views, tokens):
    if not views or len(tokens) + 1 != head.out_features:
        raise ValueError("Need matched views and the unchanged ordered alphabet")
    ids = [2] + [0 if ph in CONSONANTS and ph not in NASALS else 1 for ph in tokens]
    parts = [torch.zeros_like(torch.cat([head.weight, head.bias[:, None]], 1)) for _ in REGIMES]
    masses = torch.zeros(3, dtype=head.weight.dtype)
    values = []
    for view in views:
        lp = log_probs(head, view["features"])
        q = occupancy(lp, view["target"])
        ctc, kd = objectives(lp, view["target"], view["teacher"])
        ce = -(teacher_distribution(view["teacher"], len(lp)) * lp).sum(-1)
        for r in range(3):
            selected = q * torch.tensor([i == r for i in ids], dtype=q.dtype)[None]
            frame_mass = selected.sum(-1)
            # Only a gradient surrogate: q is detached. Its derivative is the
            # exact CTC contribution plus the same occupancy share of KD.
            surrogate = -(selected * lp).sum() / len(lp) + 2 * (frame_mass * ce).mean()
            w, b = torch.autograd.grad(surrogate, (head.weight, head.bias), retain_graph=True)
            parts[r] += torch.cat([w, b[:, None]], 1).detach() / len(views)
            masses[r] += frame_mass.mean() / len(views)
        # Check the actual loss gradient rather than trusting the surrogate.
        w, b = torch.autograd.grad(ctc + 2 * kd, (head.weight, head.bias))
        expected = torch.cat([w, b[:, None]], 1).detach()
        # Accumulate the independent oracle over equal recording views.
        values.append((float(ctc.detach()), float(kd.detach()), expected))
    expected = torch.stack([v[2] for v in values]).mean(0)
    error = float((sum(parts) - expected).abs().max())
    if not torch.allclose(sum(parts), expected, atol=5e-6, rtol=5e-5):
        raise ValueError("Regime gradients changed the complete acoustic objective")
    return parts, masses, dict(ctc=sum(v[0] for v in values) / len(values),
        kd=sum(v[1] for v in values) / len(values), decompositionMaxError=error)


class AcousticMoments:
    """Adam head refit, with occupancy-weighted conditional regime updates.

    Shared uses the complete gradient. Separated keeps one first/second moment
    pair per regime; absent regimes do not advance their clocks. Memoryless
    uses precisely the separated algebra with beta1=beta2=0, retaining no
    gradient history. All receive the same already-clipped loss contributions.
    """
    def __init__(self, arm, betas=(.9, .98), eps=1e-8):
        if arm not in ARMS or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or eps <= 0:
            raise ValueError("Invalid optimizer memory")
        self.arm, self.betas, self.eps, self.state = arm, betas, eps, {}

    def _update(self, key, gradient):
        b1, b2 = (0., 0.) if self.arm == "memoryless" else self.betas
        m, v, step = self.state.get(key, (torch.zeros_like(gradient), torch.zeros_like(gradient), 0))
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * gradient.square()
        step += 1
        if self.arm != "memoryless":
            self.state[key] = (m, v, step)
        return (m / (1 - b1 ** step)) / ((v / (1 - b2 ** step)).sqrt() + self.eps)

    def direction(self, parts, masses):
        if (len(parts) != 3 or masses.shape != (3,) or (masses < 0).any() or
                not torch.isfinite(masses).all() or not torch.allclose(masses.sum(), masses.new_tensor(1.), atol=1e-6) or
                any(p.shape != parts[0].shape or not torch.isfinite(p).all() for p in parts) or
                any(m == 0 and torch.count_nonzero(p) for m, p in zip(masses, parts))):
            raise ValueError("Invalid additive acoustic gradients")
        if self.arm == "shared":
            return self._update("shared", sum(parts))
        result = torch.zeros_like(parts[0])
        for regime, part, mass in zip(REGIMES, parts, masses):
            if mass > 0:
                result += mass * self._update(regime, part / mass)
            elif torch.count_nonzero(part):
                raise ValueError("Zero-occupancy regime has a gradient")
        return result


class WholeGradientClip:
    """Shipped ScaledAdam median-norm clipping, applied once before splitting.

    Non-scalar phone weight/bias only. The pinned icefall optimizer measures
    gradients in parameter-RMS units, refreshes RMS every four steps, and sets
    the threshold to clipping_scale=2 times its 100-update median (with the
    same initial 10/20/40-update estimates). This is not a norm bound of 2.

    Adapted from icefall/egs/librispeech/ASR/zipformer/optim.py at
    3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1, copyright 2022 Xiaomi Corp.,
    author Daniel Povey, licensed under Apache-2.0:
    https://www.apache.org/licenses/LICENSE-2.0
    """
    def __init__(self):
        self.step, self.norms, self.threshold, self.rms = 0, torch.zeros(100), None, None

    def factor(self, head, total):
        parameters = (head.weight.detach(), head.bias.detach())
        gradients = (total[:, :-1], total[:, -1])
        if not torch.isfinite(total).all():
            raise ValueError("Non-finite complete gradient")
        if self.rms is None:
            self.rms = [p.square().mean().sqrt() for p in parameters]
        factor = 1.
        if self.step:
            norm = sum((g * rms).square().sum() for g, rms in zip(gradients, self.rms)).sqrt()
            self.norms[self.step % 100] = norm
            if self.step % 100 == 0 or self.step in (10, 20, 40):
                norms = self.norms.sort().values
                if self.step in (10, 20, 40):
                    norms = norms[-self.step:]
                median = norms[(len(norms) // 4) * 2]
                self.threshold = 2 * median * (2 if self.step in (10, 20, 40) else 1)
            if self.threshold is not None:
                factor = min(1., float(self.threshold / (norm + 1e-20)))
        # Original clipping runs before scaling_step's periodic RMS refresh.
        if self.step % 4 == 3:
            self.rms = [p.square().mean().sqrt() for p in parameters]
        self.step += 1
        return factor


def matched_refit(initial, rows, groups, validate, tokens, epochs, lr, seed, emit=None, deadline=None):
    if epochs < 1 or lr <= 0 or not math.isfinite(lr):
        raise ValueError("Invalid fixed refit schedule")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection or {r["speaker"] for r in fit} & {r["speaker"] for r in selection}:
        raise ValueError("Need disjoint fit and selection speakers")
    heads = {arm: copy.deepcopy(initial) for arm in ARMS}
    optimizers = {arm: AcousticMoments(arm) for arm in ARMS}
    clippers = {arm: WholeGradientClip() for arm in ARMS}
    history = {arm: [dict(epoch=0, **validate(head, selection))] for arm, head in heads.items()}
    # Freeze ONE shared checkpoint count by shared-control selection objective.
    # Separated performance, memoryless performance and practice cannot pick it.
    chosen_epoch, best_score = 0, history["shared"][0]["objective"]
    best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            diagnostics = []
            for i in order:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("The fixed one-day experiment budget expired")
                parts, masses, diag = gradient_parts(head, fit[i]["views"], tokens)
                clip = clippers[arm].factor(head, sum(parts))
                direction = optimizers[arm].direction([p * clip for p in parts], masses)
                if not torch.isfinite(direction).all():
                    raise ValueError("Non-finite applied head update")
                with torch.no_grad():
                    head.weight.sub_(lr * direction[:, :-1])
                    head.bias.sub_(lr * direction[:, -1])
                diagnostics.append(dict(diag, clippingFactor=clip))
            metrics = dict(epoch=epoch, **validate(head, selection), recordings=len(order),
                decompositionMaxError=max(d["decompositionMaxError"] for d in diagnostics),
                clippedUpdates=sum(d["clippingFactor"] < 1 for d in diagnostics))
            history[arm].append(metrics)
            if emit:
                emit(dict(arm=arm, **metrics))
        score = history["shared"][-1]["objective"]
        if score < best_score:
            chosen_epoch, best_score = epoch, score
            best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
        head.eval()
    return heads, dict(history=history, selectedEpoch=chosen_epoch, selectionObjective=best_score,
        selectedUpdates=chosen_epoch * len(fit), trainingUpdates=epochs * len(fit),
        selectionRule="minimum-shared-control-objective-at-one-shared-epoch-on-selection-speakers")
