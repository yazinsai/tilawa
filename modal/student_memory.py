"""Duration-neutral hierarchical attention for the upper streaming Zipformer.

Keep frame values and positions; change the measure over acoustic memory. For a
group G the score is logmeanexp(frame scores), followed by ordinary conditional
attention inside G. Expanding that hierarchy back to frames is exactly ordinary
attention divided by |G| and renormalized. This avoids ragged ONNX tensors and
keeps every existing value/cache on its original clock.

Adjacent projected keys determine learned contiguous groups. Hard groups are
used in both training and export, with a soft connectivity straight-through
gradient for learning the boundaries. No transcript or reference enters memory.
"""

import torch
from torch import nn
from torch.nn import functional as F


class AcousticMemory(nn.Module):
    def __init__(self, heads: int, key_dim: int, frame_ms: float):
        super().__init__()
        self.frame_ms = frame_ms
        # A learned diagonal metric on adjacent key changes, per attention head.
        self.log_metric = nn.Parameter(torch.zeros(heads, key_dim))
        self.join_bias = nn.Parameter(torch.zeros(heads))
        # Duration is available separately from repeated key multiplicity.
        self.duration_query = nn.Parameter(torch.zeros(heads, key_dim))

    def connectivity(self, keys):
        """keys: (head,batch,time,dim); return contiguous group membership.

        The hard decision uses a unit mean-square change in learned key units.
        The sigmoid relaxation affects gradients only, never inference scores.
        """
        # Keep cumulative group IDs exact during bfloat16 acoustic autocast.
        keys = keys.float()
        delta = keys[:, :, 1:] - keys[:, :, :-1]
        metric = self.log_metric.exp()[:, None, None, :]
        change = (delta.square() * metric).mean(-1)
        logits = self.join_bias[:, None, None] + 1.0 - change
        breaks = (logits < 0).to(keys.dtype)
        group = F.pad(breaks.cumsum(-1), (1, 0))
        hard = (group.unsqueeze(-1) == group.unsqueeze(-2)).to(keys.dtype)
        if self.training:
            # Products of adjacency probabilities, evaluated in log space.
            prefix = F.pad(F.logsigmoid(logits).cumsum(-1), (1, 0))
            soft = (-(prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).abs()).exp()
            return hard + (soft - soft.detach())
        return hard

    def forward(self, weights, query, keys, allowed):
        """Return expanded hierarchical weights on the original frame clock.

        allowed has shape (batch,query,key), independent of posterior underflow.
        Counts include only keys visible to that query, so padding, future chunks
        and evicted history have no vote. Zipformer uses contiguous visible
        intervals; the adapter refuses a noncontiguous visibility mask.
        """
        dtype = weights.dtype
        weights, query = weights.float(), query.float()
        members = self.connectivity(keys)
        visible = allowed.unsqueeze(0).to(weights.dtype)
        # A query sees one interval. Prefix sums count membership in that
        # interval in O(key^2 + query*key), avoiding a cubic training matmul.
        prefix = F.pad(members.cumsum(-2), (0, 0, 1, 0))
        start = allowed.to(torch.int64).argmax(-1)
        stop = start + allowed.sum(-1)
        shape = (members.shape[0], *allowed.shape)
        lo = start[None, ..., None].expand(shape)
        hi = stop[None, ..., None].expand(shape)
        count = (prefix.gather(-2, hi) - prefix.gather(-2, lo)).clamp_min(1)
        # Keep exact relative positions in the base scores and annotate elapsed
        # duration explicitly. log1p(seconds) is a unit-defined feature, not a
        # recognition cut. Its query-dependent coefficient starts at zero.
        duration = torch.log1p(count * (self.frame_ms / 1000.0))
        coefficient = (query * self.duration_query[:, None, None, :]).sum(-1)
        annotation = coefficient.unsqueeze(-1) * duration
        # Stable normalized product; all-hidden queries stay zero rather than
        # making an invalid memory group from masked padding.
        annotation = annotation.masked_fill(~allowed.unsqueeze(0), -1000)
        annotation = annotation - annotation.amax(-1, keepdim=True)
        mass = weights * visible / count * annotation.exp()
        return (mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)).to(dtype)


def visibility(x, key_padding_mask=None, attn_mask=None, key_len=None):
    length, batch = x.shape[:2]
    key_len = length if key_len is None else key_len
    allowed = torch.ones(batch, length, key_len, dtype=torch.bool, device=x.device)
    if key_padding_mask is not None:
        allowed = allowed & ~key_padding_mask.to(torch.bool).unsqueeze(1)
    if attn_mask is not None:
        allowed = allowed & ~attn_mask
    if not torch.jit.is_tracing() and not torch.jit.is_scripting():
        # A hidden hole could otherwise join groups across unavailable audio.
        # Production chunk masks and streaming valid-prefix masks are contiguous.
        transitions = allowed[..., 1:].to(torch.int32) - allowed[..., :-1].to(torch.int32)
        starts = allowed[..., 0].to(torch.int32) + (transitions == 1).sum(-1)
        if (starts > 1).any():
            raise ValueError("acoustic memory requires contiguous visible key intervals")
    return allowed


class DurationNeutralAttention(nn.Module):
    """Adapter retaining icefall's attention projections and cache semantics."""
    def __init__(self, base, frame_ms):
        super().__init__()
        if base.dropout != 0:
            raise ValueError("hierarchical weights must precede attention dropout")
        self.base = base
        self.memory = AcousticMemory(base.num_heads, base.query_head_dim, frame_ms)

    def projected(self, x):
        # Base key balancing/whitening change backward behavior only. Their
        # forward values are these exact projected keys, also stored in cache.
        projected = self.base.in_proj(x)
        dim = self.base.num_heads * self.base.query_head_dim
        q, k = projected[..., :dim], projected[..., dim:2 * dim]
        return q, k

    def headed(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.base.num_heads,
                         self.base.query_head_dim).permute(2, 1, 0, 3)

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        weights = self.base(x, pos_emb, key_padding_mask, attn_mask)
        q, k = self.projected(x)
        allowed = visibility(x, key_padding_mask, attn_mask)
        return self.memory(weights, self.headed(q), self.headed(k), allowed)

    def streaming_forward(self, x, pos_emb, cached_key, left_context_len, key_padding_mask):
        weights, new_key = self.base.streaming_forward(
            x, pos_emb, cached_key, left_context_len, key_padding_mask)
        q, k = self.projected(x)
        k = torch.cat([cached_key, k], dim=0)
        allowed = visibility(x, key_padding_mask, key_len=k.shape[0])
        return self.memory(weights, self.headed(q), self.headed(k), allowed), new_key


def install_memory(model):
    """Use the last two encoder stages; leave subsampling and output unchanged."""
    stages = model.encoder.encoders
    for stage, downsample in zip(stages[-2:], model.encoder.downsampling_factor[-2:]):
        # Conv2dSubsampling emits one frame per 20 ms, before the stage factor.
        inner = getattr(stage, "encoder", stage)
        for layer in inner.layers:
            if isinstance(layer.self_attn_weights, DurationNeutralAttention):
                raise ValueError("acoustic memory already installed")
            layer.self_attn_weights = DurationNeutralAttention(layer.self_attn_weights, 20.0 * downsample)
