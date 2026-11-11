"""Reversible upper acoustic stacks and an equal-parameter residual control.

The invertible boundary is the stack at its native Zipformer resolution. Existing
temporal down/up-sampling stays outside that boundary. Both arms use identical
causal functions, six caches per block, and the ordinary streaming CTC contract.
"""

import torch
from torch import nn


class AcousticFunction(nn.Module):
    """Normalization and nonlinear causal transforms live inside F and G."""

    def __init__(self, channels, hidden, kernel):
        super().__init__()
        if kernel < 2:
            raise ValueError("A causal acoustic function needs a history window")
        self.hidden, self.history = hidden, kernel - 1
        self.norm = nn.LayerNorm(channels)
        self.input = nn.Linear(channels, hidden)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, groups=hidden) for _ in range(3)
        ])
        self.output = nn.Linear(hidden, channels)
        # Both replacement arms start at identity; the shipped prefix is copied.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def init_states(self, batch, device, dtype=torch.float32):
        return [torch.zeros(batch, self.hidden, self.history, device=device, dtype=dtype)
                for _ in self.convs]

    def forward(self, x, states=None):
        if states is None:
            states = self.init_states(x.shape[1], x.device, x.dtype)
        if len(states) != 3:
            raise ValueError("Each acoustic function requires all three causal caches")
        z = self.input(self.norm(x)).permute(1, 2, 0)
        updated = []
        for conv, state in zip(self.convs, states):
            joined = torch.cat((state.to(z.dtype), z), dim=-1)
            updated.append(joined[..., -self.history:])
            z = torch.nn.functional.gelu(conv(joined))
        return self.output(z.permute(2, 0, 1)), updated


class AcousticBlock(nn.Module):
    """Additive coupling, or parallel ordinary residual updates for the control.

    Reversible: y1=x1+F(x2), y2=x2+G(y1).
    Residual:   y1=x1+F(x2), y2=x2+G(x1).
    The control is a normal residual branch with cross-channel connections. It
    can discard information; it uses the same weights and number of operations.
    No dropout is used inside F/G, so inversion needs only the incoming caches.
    """

    def __init__(self, channels, hidden, kernel, mode="reversible"):
        super().__init__()
        if channels % 2 or mode not in ("reversible", "residual"):
            raise ValueError("Expected even channels and reversible/residual mode")
        self.mode = mode
        self.f = AcousticFunction(channels // 2, hidden, kernel)
        self.g = AcousticFunction(channels // 2, hidden, kernel)

    def init_states(self, batch, device, dtype=torch.float32):
        return self.f.init_states(batch, device, dtype) + self.g.init_states(batch, device, dtype)

    def forward(self, x, states=None, mask=1.0):
        if states is None:
            states = self.init_states(x.shape[1], x.device, x.dtype)
        if len(states) != 6:
            raise ValueError("An acoustic block requires six causal caches")
        a, b = x.chunk(2, dim=-1)
        ma, mb = mask.chunk(2, dim=-1) if isinstance(mask, torch.Tensor) and mask.shape[-1] != 1 else (mask, mask)
        delta, sf = self.f(b, states[:3])
        ya = a + delta * ma
        delta, sg = self.g(ya if self.mode == "reversible" else a, states[3:])
        return torch.cat((ya, b + delta * mb), dim=-1), sf + sg

    def inverse(self, y, states=None, mask=1.0):
        if self.mode != "reversible":
            raise ValueError("Ordinary residual blocks have no algebraic inverse")
        if states is None:
            states = self.init_states(y.shape[1], y.device, y.dtype)
        if len(states) != 6:
            raise ValueError("Inversion requires all six incoming causal caches")
        ya, yb = y.chunk(2, dim=-1)
        ma, mb = mask.chunk(2, dim=-1) if isinstance(mask, torch.Tensor) and mask.shape[-1] != 1 else (mask, mask)
        delta, _ = self.g(ya, states[3:])
        b = yb - delta * mb
        delta, _ = self.f(b, states[:3])
        return torch.cat((ya - delta * ma, b), dim=-1)


class AcousticStack(nn.Module):
    """A drop-in native-resolution stack with Zipformer's six-cache routing."""

    def __init__(self, channels, hidden, kernel, num_layers, mode):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([AcousticBlock(channels, hidden, kernel, mode) for _ in range(num_layers)])

    def init_states(self, batch, device):
        return [state for layer in self.layers for state in layer.init_states(batch, device)]

    @staticmethod
    def valid_mask(src, padding):
        if padding is None:
            return 1.0
        return (~padding[..., -src.shape[0]:].bool()).transpose(0, 1).unsqueeze(-1).to(src.dtype)

    def forward(self, src, chunk_size=-1, feature_mask=1.0, attn_mask=None, src_key_padding_mask=None):
        # Mask updates, never the preserved state. Convolutions are strictly
        # causal for every training chunk size, with no future attention.
        mask = feature_mask * self.valid_mask(src, src_key_padding_mask)
        for layer in self.layers:
            src, _ = layer(src, mask=mask)
        return src

    def streaming_forward(self, src, states, left_context_len, src_key_padding_mask):
        if len(states) != 6 * self.num_layers:
            raise ValueError("Incomplete upper acoustic streaming state")
        updated = []
        mask = self.valid_mask(src, src_key_padding_mask)
        for i, layer in enumerate(self.layers):
            src, caches = layer(src, states[i * 6:(i + 1) * 6], mask)
            updated.extend(caches)
        return src, updated


def make_upper_encoder(mode, **kwargs):
    """Retain the first four shipped stacks, replacing only the final two."""
    from zipformer import Zipformer2

    if mode not in ("reversible", "residual"):
        raise ValueError("Unknown upper acoustic block mode")

    class UpperZipformer(Zipformer2):
        def __init__(self):
            super().__init__(**kwargs)
            self.upper_start = len(self.encoders) - 2
            for i in range(self.upper_start, len(self.encoders)):
                module = self.encoders[i]
                stack = AcousticStack(self.encoder_dim[i], kwargs["feedforward_dim"][i] // 2,
                                      self.cnn_module_kernel[i], module.num_layers, mode)
                if hasattr(module, "encoder"):
                    module.encoder = stack
                else:
                    self.encoders[i] = stack

        def get_init_states(self, batch_size=1, device=torch.device("cpu")):
            states = super().get_init_states(batch_size, device)
            offset = sum(module.num_layers for module in self.encoders[:self.upper_start]) * 6
            for module in self.encoders[self.upper_start:]:
                stack = module.encoder if hasattr(module, "encoder") else module
                count = module.num_layers * 6
                states[offset:offset + count] = stack.init_states(batch_size, device)
                offset += count
            return states

    return UpperZipformer()
