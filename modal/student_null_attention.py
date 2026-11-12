"""Query-conditioned abstention in the last acoustic block of the Zipformer.

The new entry has no key/value cache and an exactly zero value. Returning only
its real-key weights is equivalent to multiplying the extended distribution by
[real values; 0] in every consumer of these shared weights. Residuals, value
projections, positions and all real context remain the existing block's.

Only the per-head query projection is trainable. The matched temperature arm
uses the same projection geometry, scaling the original logits without adding
an entry. Existing acoustic modules remain in evaluation mode during fitting.
"""

import torch
from torch import nn


class QueryAttention(nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        if mode not in ("null", "temperature"):
            raise ValueError("attention mode must be null or temperature")
        if base.dropout != 0:
            raise ValueError("null attention requires the shipped zero weight dropout")
        self.base = base.requires_grad_(False)
        self.mode = mode
        self.score_weight = nn.Parameter(torch.zeros(base.num_heads, base.query_head_dim))
        # Near-ordinary initialization, fixed before any acoustic inspection.
        # Temperature starts at exactly one. This bias is fitted, not a gate.
        self.score_bias = nn.Parameter(torch.full((base.num_heads,), -10.0 if mode == "null" else 0.0))

    def distribute(self, scores, query, hidden):
        """scores (H,B,Q,K), query (H,B,Q,D), hidden (B,Q,K)."""
        score = (query * self.score_weight[:, None, None, :]).sum(-1)
        score = score + self.score_bias[:, None, None]
        if self.mode == "temperature":
            # Bounded log temperature prevents exponential overflow; it does
            # not limit the real attention's available positions or values.
            scores = scores * torch.exp(-score.clamp(-10.0, 10.0)).unsqueeze(-1)
        scores = scores.masked_fill(hidden.unsqueeze(0), float("-inf"))
        if self.mode == "null":
            return torch.cat([scores, score.unsqueeze(-1)], dim=-1).softmax(-1)[..., :-1]
        # Padded queries can have no real keys. No padded value is retrieved.
        visible = (~hidden).any(-1, keepdim=True).unsqueeze(0)
        safe = torch.where(visible, scores, torch.zeros_like(scores))
        return safe.softmax(-1) * visible.to(scores.dtype)

    def logits(self, x, pos_emb, cached_key=None):
        """Exact evaluation-mode content and relative-position scores.

        Follows pinned icefall 3f848bb6's attention. ScaledLinear already
        incorporates the query/key scaling; no additional sqrt(dim) is used.
        Gather is the upstream export equivalent of its eager as_strided.
        """
        b = self.base
        projected = b.in_proj(x)
        length, batch = x.shape[:2]
        dim = b.num_heads * b.query_head_dim
        q, k, p = projected[..., :dim], projected[..., dim:2 * dim], projected[..., 2 * dim:]
        if cached_key is not None:
            k = torch.cat([cached_key, k], dim=0)
        key_len = k.shape[0]
        query = q.reshape(length, batch, b.num_heads, b.query_head_dim).permute(2, 1, 0, 3)
        keys = k.reshape(key_len, batch, b.num_heads, b.query_head_dim).permute(2, 1, 3, 0)
        positional_query = p.reshape(length, batch, b.num_heads, b.pos_head_dim).permute(2, 1, 0, 3)
        pos = b.linear_pos(pos_emb).reshape(-1, key_len + length - 1, b.num_heads, b.pos_head_dim).permute(2, 0, 3, 1)
        relative = torch.matmul(positional_query, pos)
        indexes = (torch.arange(length - 1, -1, -1, device=x.device)[:, None]
                   + torch.arange(key_len, device=x.device)[None, :])
        indexes = indexes[None, None].expand(b.num_heads, batch, length, key_len)
        positional_scores = relative.gather(-1, indexes)
        return torch.matmul(query, keys) + positional_scores, query, k

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        if self.training:
            raise ValueError("frozen acoustic refits must keep the model in eval mode")
        scores, query, _ = self.logits(x, pos_emb)
        hidden = torch.zeros(x.shape[1], x.shape[0], x.shape[0], dtype=torch.bool, device=x.device)
        if key_padding_mask is not None:
            hidden = hidden | key_padding_mask.to(torch.bool).unsqueeze(1)
        if attn_mask is not None:
            hidden = hidden | attn_mask.to(torch.bool)
        return self.distribute(scores, query, hidden)

    def streaming_forward(self, x, pos_emb, cached_key, left_context_len, key_padding_mask):
        if self.training:
            raise ValueError("frozen acoustic refits must keep the model in eval mode")
        scores, query, keys = self.logits(x, pos_emb, cached_key)
        hidden = torch.zeros(x.shape[1], x.shape[0], keys.shape[0], dtype=torch.bool, device=x.device)
        if key_padding_mask is not None:
            hidden = hidden | key_padding_mask.to(torch.bool).unsqueeze(1)
        # Avoid the -0 slice returning all keys when context is empty.
        new_key = keys[keys.shape[0] - left_context_len:]
        return self.distribute(scores, query, hidden), new_key


def upper_block(model):
    stage = model.encoder.encoders[-1]
    return getattr(stage, "encoder", stage).layers[-1]


def install_query_attention(model, mode):
    """One uppermost block, with every pre-existing parameter frozen."""
    layer = upper_block(model)
    if isinstance(layer.self_attn_weights, QueryAttention):
        raise ValueError("query attention already installed")
    model.requires_grad_(False)
    layer.self_attn_weights = QueryAttention(layer.self_attn_weights, mode)
    model.eval()
    return layer.self_attn_weights
