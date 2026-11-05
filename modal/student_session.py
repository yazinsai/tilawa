"""Session-conditioned student: frozen acoustic features and identity-initialized BiasNorm FiLM.

The session vector is a mean of frame embeddings from completed streaming chunks.
It carries no text, position, reference identity or test-time fitted parameters.
"""
import copy
import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, frames):
        return self.net(frames)


class SessionStudent(nn.Module):
    def __init__(self, parent, stack=3):
        super().__init__()
        self.acoustic = copy.deepcopy(parent)
        self.frozen = copy.deepcopy(parent).eval().requires_grad_(False)
        self.stack = stack
        # Zipformer stack outputs have time, batch, channel axes.
        channels = self.frozen.encoder.encoder_dim[stack]
        self.frame_encoder = FrameEncoder(channels)
        self.films = nn.ModuleList()
        self.embedding = None
        self.mid = None
        mid_layer = self.frozen.encoder.encoders[stack]
        mid_layer.register_forward_hook(self._capture)
        # Icefall calls streaming_forward directly, bypassing Module.__call__ hooks.
        streaming = mid_layer.streaming_forward
        def capture_stream(*args, **kwargs):
            result = streaming(*args, **kwargs)
            self._capture(mid_layer, args, result)
            return result
        mid_layer.streaming_forward = capture_stream
        for name, layer in self.acoustic.named_modules():
            if layer.__class__.__name__ == 'BiasNorm':
                channels = layer.bias.numel()
                film = nn.Linear(64, channels * 2, bias=False)
                nn.init.zeros_(film.weight)
                self.films.append(film)
                layer.register_forward_hook(self._film(film, layer.channel_dim, 0 if name.startswith("encoder_embed.") else 1))

    def _capture(self, module, inputs, output):
        self.mid = output[0] if isinstance(output, tuple) else output

    def _film(self, film, channel_dim, batch_dim):
        def apply(module, inputs, output):
            if self.embedding is None:
                return output
            scale, shift = film(self.embedding).chunk(2, -1)
            # BiasNorm in subsampling is B,T,C; encoder norms are T,B,C.
            dim = channel_dim % output.ndim
            shape = [1] * output.ndim
            shape[dim] = scale.shape[-1]
            shape[batch_dim] = scale.shape[0]
            return output * (1 + scale.reshape(shape)) + shift.reshape(shape)
        return apply

    def features(self, feats, lens):
        self.frozen.eval()
        with torch.no_grad():
            self.frozen(feats, lens)
        return self.frame_encoder(self.mid.detach())

    def forward(self, feats, lens, embedding):
        self.embedding = embedding
        return self.acoustic(feats, lens)

    def streaming_eval(self, chunk=16, left=128):
        self.acoustic.streaming_eval(chunk, left)
        self.frozen.streaming_eval(chunk, left)


def stable_swoosh_training():
    """Keep icefall's activation formula, using softplus for its overflowing autograd branch."""
    import inspect
    import scaling
    if getattr(scaling, '_session_stable', False):
        return
    for name in ('SwooshRForward', 'SwooshLForward'):
        original = getattr(scaling, name)
        source = inspect.getsource(original)
        needle = '(1.0 + x_offset.exp()).log().to(x.dtype)'
        if source.count(needle) != 1:
            raise ValueError('Installed Swoosh formula changed')
        scope = {}
        exec(compile(source.replace(needle, 'torch.nn.functional.softplus(x_offset).to(x.dtype)'),
                     '<stable-swoosh>', 'exec'), scaling.__dict__, scope)
        stable = scope[name]
        def forward(x, original=original, stable=stable):
            return stable(x) if torch.is_grad_enabled() and x.requires_grad else original(x)
        setattr(scaling, name, forward)
    scaling._session_stable = True
