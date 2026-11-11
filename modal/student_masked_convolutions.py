"""Item boundaries for padded training batches, using the pinned Icefall modules.

Streaming methods and parameter names are inherited unchanged. The encoder's
ConvolutionModule already masks gated inputs before its depthwise convolution;
the missing boundaries are in the embedding and weighted downsampling.
"""

import torch
from subsampling import Conv2dSubsampling
from zipformer import DownsampledZipformer2Encoder, Zipformer2, convert_num_channels


def padding_mask(lengths, frames):
    return torch.arange(frames, device=lengths.device)[None] >= lengths[:, None]


def mask_time(src, lengths):
    """Zero artificial positions in (time, batch, channels) layout."""
    return src.masked_fill(padding_mask(lengths, src.shape[0]).T[..., None], 0)


def mask_conv(src, lengths):
    """Zero artificial positions in (batch, channels, time, frequency) layout."""
    return src.masked_fill(padding_mask(lengths, src.shape[2])[:, None, :, None], 0)


def downsample_valid(module, src, lengths):
    """Weighted pooling with each item's original last-frame boundary padding.

    SimpleDownsample repeats the last frame to fill a partial group. Zeroing
    that group would change intentional boundary behavior. Gather each item's
    last valid frame instead, and zero only completely invalid output groups.
    The ordinary module.forward remains in use by streaming_forward.
    """
    if torch.any(lengths <= 0) or torch.any(lengths > src.shape[0]):
        raise ValueError("Downsampling requires nonempty item lengths within the tensor")
    ds = module.downsample
    frames, batch, channels = src.shape
    groups = (frames + ds - 1) // ds
    index = torch.arange(groups * ds, device=src.device)[:, None].expand(-1, batch)
    index = torch.minimum(index, lengths[None] - 1)
    extended = src.gather(0, index[..., None].expand(-1, -1, channels))
    grouped = extended.reshape(groups, ds, batch, channels)
    out = (grouped * module.bias.softmax(0)[None, :, None, None]).sum(1)
    out_lengths = (lengths + ds - 1) // ds
    return mask_time(out, out_lengths), out_lengths


class MaskedConv2dSubsampling(Conv2dSubsampling):
    def forward(self, x, x_lens):
        lengths = x_lens
        x = x.unsqueeze(1)
        for module in self.conv:
            if isinstance(module, torch.nn.Conv2d):
                # The three embedding convolutions have no temporal padding.
                # Mask after nonlinearities/biases, immediately before mixing.
                x = mask_conv(x, lengths)
                x = module(x)
                lengths = (lengths + 2 * module.padding[0]
                           - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
            else:
                x = module(x)
        if torch.any(lengths <= 0):
            raise ValueError("Input is too short for the embedding convolutions")
        # ConvNeXt has symmetric zero padding of three embedded frames. Zeros
        # here reproduce that boundary for the short item in a larger batch.
        x = self.convnext(mask_conv(x, lengths))
        x = mask_conv(x, lengths)
        batch, channels, frames, freq = x.shape
        x = x.transpose(1, 2).reshape(batch, frames, channels * freq)
        x = self.out(x)
        x = self.out_whiten(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        x = x.masked_fill(padding_mask(lengths, frames)[..., None], 0)
        return x, lengths


class MaskedDownsampledEncoder(DownsampledZipformer2Encoder):
    def forward(self, src, chunk_size=-1, feature_mask=1.0, attn_mask=None,
                src_key_padding_mask=None):
        # Unlike the original wrapper, receive the mask at src's full rate.
        if src_key_padding_mask is None:
            lengths = torch.full((src.shape[1],), src.shape[0], device=src.device, dtype=torch.long)
        else:
            lengths = (~src_key_padding_mask).sum(-1)
        original = mask_time(src, lengths)
        src, down_lengths = downsample_valid(self.downsample, original, lengths)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]
        src = self.encoder(src, chunk_size=chunk_size // ds, feature_mask=feature_mask,
                           attn_mask=attn_mask,
                           src_key_padding_mask=padding_mask(down_lengths, src.shape[0]))
        src = self.upsample(src)[:original.shape[0]]
        return mask_time(self.out_combiner(original, src), lengths)


class MaskedZipformer2(Zipformer2):
    def forward(self, x, x_lens, src_key_padding_mask=None):
        # Derive the full-rate mask from lengths; use those same boundaries for
        # every pooling site and each encoder's existing attention/conv mask.
        mask = padding_mask(x_lens, x.shape[0])
        if src_key_padding_mask is not None and not torch.equal(mask, src_key_padding_mask.bool()):
            raise ValueError("Padding mask must describe the supplied item lengths")
        outputs = []
        feature_masks = self.get_feature_masks(x)
        chunk_size, left_context_chunks = self.get_chunk_info()
        attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)
        for i, module in enumerate(self.encoders):
            x = mask_time(convert_num_channels(x, self.encoder_dim[i]), x_lens)
            x = module(x, chunk_size=chunk_size, feature_mask=feature_masks[i],
                       src_key_padding_mask=mask, attn_mask=attn_mask)
            x = mask_time(x, x_lens)
            outputs.append(x)
        x = self._get_full_dim_output(outputs)
        return downsample_valid(self.downsample_output, x, x_lens)


def install_validity_masks(model):
    """Enable training forward masks without changing weights or streaming APIs."""
    if isinstance(model.encoder_embed, MaskedConv2dSubsampling):
        raise ValueError("Validity masks are already installed")
    if type(model.encoder_embed) is not Conv2dSubsampling or type(model.encoder) is not Zipformer2:
        raise TypeError("Validity masks require the pinned student's embedding and encoder")
    wrappers = [m for m in model.encoder.encoders if isinstance(m, DownsampledZipformer2Encoder)]
    model.encoder_embed.__class__ = MaskedConv2dSubsampling
    model.encoder.__class__ = MaskedZipformer2
    for module in wrappers:
        module.__class__ = MaskedDownsampledEncoder
    return ["encoder_embed.convnext", *[f"encoder.encoders.{i}.downsample" for i, m in
            enumerate(model.encoder.encoders) if isinstance(m, MaskedDownsampledEncoder)],
            "encoder.downsample_output"]
