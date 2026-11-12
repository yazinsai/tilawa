"""Training-only temporal attention supervision for the streaming student.

Maps *probability mass*, not attention logits or hidden vectors. Source and
query positions use physical frontend grids; no sequence-length rescaling or
edge repetition is allowed. Nothing in this module is attached to an exported
student or used by the listener at inference.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import inspect

import torch
from torch import nn


@dataclass(frozen=True)
class AcousticGrid:
    """Centres of encoder observations, in 16 kHz sample coordinates.

The pinned Kaldi frontend has a 400-sample window and 160-sample hop.
Student valid convolutions cover fbank rows 2j..2j+8: centre 840+320j.
Teacher stacks rows 2j,2j+1: centre 280+320j, before its CTC adapter.
Pooling groups adjacent observations, rather than stretching either clock.
"""

    origin: int
    step: int
    pool: int = 1

    def __post_init__(self):
        if any(type(x) is not int for x in (self.origin, self.step, self.pool)):
            raise ValueError("Acoustic grid must use integer sample coordinates")
        if self.step <= 0 or self.pool <= 0:
            raise ValueError("Grid stride and pooling must be positive")

    def bins(self, positions):
        # Left edge of observation zero's pooled group. Half-open bins avoid
        # double consumption at boundaries and preserve each teacher key once.
        return torch.floor((positions - self.origin + self.step / 2)
                           / (self.step * self.pool)).long()

    def centres(self, indices):
        return self.origin + self.step * (indices * self.pool + (self.pool - 1) / 2)


TEACHER_GRID = AcousticGrid(280, 320)
STUDENT_GRID = AcousticGrid(840, 320)


def normalized_rows(probabilities, allowed):
    """Exclude unavailable keys before normalization; reject empty valid rows.

`allowed` broadcasts to heads/batch/query/key. NaN sentinels outside support
are ignored completely. Their gradient is exactly zero.
"""
    if allowed.dtype != torch.bool:
        raise ValueError("Attention support must be boolean")
    allowed = allowed.expand_as(probabilities)
    p = torch.where(allowed, probabilities.float(), 0.0)
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid attention on available acoustic positions")
    total = p.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("Attention row has no mass in the streaming receptive field")
    return p / total


def project_teacher(attention, teacher_grid, student_grid, queries, keys, allowed):
    """Rebin a singleton teacher map onto exact streaming query/key indices.

Teacher: (heads, queries, keys); student positions: absolute pooled indices.
`allowed` is the student's actual key-padding/attention support (Q,K).
Average teacher query rows in each query bin, sum source probability mass in
each key bin, and renormalize separately for every teacher head. This conditions
each teacher head on available acoustic positions *before* head correspondence.
Teacher rows with no paired query are excluded, never clamped to an edge row.
"""
    if (attention.ndim != 3 or attention.shape[-1] != attention.shape[-2]
            or queries.ndim != 1 or keys.ndim != 1
            or allowed.shape != (len(queries), len(keys)) or allowed.dtype != torch.bool):
        raise ValueError("Invalid teacher map or explicit streaming support")
    attention = attention.detach().float()
    positions = teacher_grid.centres(torch.arange(attention.shape[-1], device=attention.device))
    bins = student_grid.bins(positions)
    q_members = bins[None] == queries[:, None]
    k_members = bins[:, None] == keys[None]
    paired = q_members.any(-1) & allowed.any(-1)
    if not paired.any():
        raise ValueError("Teacher/student acoustic grids have no valid query overlap")
    rows = []
    for members, support in zip(q_members[paired], allowed[paired]):
        source_valid = (k_members & support[None]).any(-1)
        # Select query rows, then mask source positions before arithmetic.
        # Padded/unavailable entries can carry NaN sentinels in diagnostics.
        safe = torch.where(source_valid[None, None], attention[:, members], 0.0)
        rows.append(safe.mean(1) @ k_members.float())
    rebinned = torch.stack(rows, dim=1)
    support = allowed[paired][None]
    target = normalized_rows(rebinned, support)
    return target, paired


class HeadCorrespondence(nn.Module):
    """A convex teacher-head mixture for each student head, trained then frozen.

Head width never enters the loss. Neither a projection of values nor a hidden
feature match can stand in for this correspondence over acoustic retrieval.
"""

    def __init__(self, student_heads, teacher_heads):
        super().__init__()
        if student_heads < 1 or teacher_heads < 1:
            raise ValueError("Attention must have at least one head")
        self.logits = nn.Parameter(torch.zeros(student_heads, teacher_heads))

    def forward(self, teacher):
        if teacher.ndim != 3 or teacher.shape[0] != self.logits.shape[1]:
            raise ValueError("Teacher head count differs from the calibrated correspondence")
        return torch.einsum("sh,hqk->sqk", self.logits.softmax(-1), teacher.detach())

    def freeze(self):
        self.requires_grad_(False)
        return self


def routing_loss(student, teacher, correspondence, allowed):
    """Mean KL on available keys; gradient flows only into student/map weights."""
    if student.ndim != 3 or allowed.shape != student.shape[-2:]:
        raise ValueError("Student routing map must be heads/query/key")
    target = correspondence(teacher)
    if target.shape != student.shape:
        raise ValueError("Unpaired attention positions or head counts")
    prediction = normalized_rows(student, allowed[None])
    # xlogy gives exact zero for zero target probability. Finite clamp only
    # protects a student softmax underflow; masked keys carry no target mass.
    return (torch.special.xlogy(target, target)
            - target * prediction.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1).mean()


@dataclass
class StreamingAttention:
    probabilities: torch.Tensor  # heads, batch, query, key
    features: torch.Tensor       # query, batch, channels, input to attention
    queries: torch.Tensor       # absolute indices at this layer's resolution
    keys: torch.Tensor
    allowed: torch.Tensor       # batch, query, key
    grid: AcousticGrid


@contextmanager
def capture_streaming_attention(modules, processed_frames):
    """Observe direct `streaming_forward` calls without changing any result.

`modules` maps each chosen module to its encoder-stack downsampling factor.
The wrapper uses the actual invocation's cache length and padding mask, so
startup cache, shortened left context and current-chunk support cannot leak.
Instance methods are restored even when a forward or capture fails. Ordinary
nn.Module hooks do not observe these direct method calls in pinned icefall.
"""
    captures, originals = {}, []
    try:
        for module, factor in modules.items():
            if processed_frames % factor:
                raise ValueError("Streaming chunk does not align to the encoder stack")
            original = module.streaming_forward
            existed = "streaming_forward" in module.__dict__
            originals.append((module, existed, module.__dict__.get("streaming_forward")))
            signature = inspect.signature(original)

            def observe(*args, _module=module, _factor=factor, _fn=original,
                        _signature=signature, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                bound.apply_defaults()
                result = _fn(*args, **kwargs)
                weights = result[0]
                x = bound.arguments["x"]
                left = bound.arguments["left_context_len"]
                mask = bound.arguments["key_padding_mask"]
                start = processed_frames // _factor
                q = torch.arange(start, start + x.shape[0], device=x.device)
                k = torch.arange(start - left, start + x.shape[0], device=x.device)
                if weights.shape != (_module.num_heads, x.shape[1], len(q), len(k)):
                    raise ValueError("Pinned streaming attention layout changed")
                if mask is None or mask.shape != (x.shape[1], len(k)):
                    raise ValueError("Streaming attention requires its actual key padding mask")
                support = (~mask.bool())[:, None].expand(-1, len(q), -1) & (k >= 0)[None, None]
                captures[_module] = StreamingAttention(weights, x, q, k, support,
                                                       AcousticGrid(STUDENT_GRID.origin, STUDENT_GRID.step, _factor))
                return result

            module.streaming_forward = observe
        yield captures
    finally:
        for module, existed, old in reversed(originals):
            if existed:
                module.streaming_forward = old
            else:
                del module.streaming_forward


@contextmanager
def capture_teacher_attention(modules):
    """Capture the pinned Muaalem attention operator's returned probabilities.

Both model mode and output flags remain the caller's responsibility. Requiring
eval prevents stochastic targets. Only selected layers are retained; no teacher
graph is kept, and no request for all 24 quadratic attention maps is needed.
"""
    captures, handles = {}, []
    try:
        for module in modules:
            if module.training:
                raise ValueError("Teacher routing targets require inference mode")

            def observe(_module, inputs, kwargs, output):
                hidden, attention = output
                if attention.ndim != 4:
                    raise ValueError("Pinned teacher attention layout changed")
                bound = inspect.signature(_module.forward).bind(*inputs, **kwargs)
                captures[_module] = (attention.detach(), bound.arguments["hidden_states"].detach())

            handles.append(module.register_forward_hook(observe, with_kwargs=True))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def deterministic_streaming(model, chunk, left):
    """Use deployed recurrent execution with gradients and no stochastic masks."""
    flags = [(m, m.training) for m in model.modules()]
    context = model.encoder.chunk_size, model.encoder.left_context_frames
    try:
        model.eval()
        model.streaming_eval(chunk, left)
        yield
    finally:
        model.encoder.chunk_size, model.encoder.left_context_frames = context
        for module, flag in flags:
            module.training = flag


def feature_hint_loss(student, teacher, projection, valid):
    """Matched control: normalize vectors, without supervising routing."""
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("Feature hints require nonempty paired rows")
    predicted = nn.functional.normalize(projection(student[valid].float()), dim=-1)
    target = nn.functional.normalize(teacher.detach()[valid].float(), dim=-1)
    if predicted.shape != target.shape:
        raise ValueError("Feature hint correspondence has the wrong width")
    return (predicted - target).square().sum(-1).mean()
