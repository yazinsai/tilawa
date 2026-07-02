"""QuranCLAP v2: Contrastive audio-text model for Quran verse identification.

Improvements over v1:
- wav2vec2-base audio encoder (better cross-lingual transfer than English HuBERT)
- Deeper projection heads (3 layers with layer norm)
- Hard negative mining support
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer


class QuranCLAPv2(nn.Module):
    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-base",
        text_model_name: str = "aubmindlab/bert-base-arabertv02",
        embed_dim: int = 256,
        freeze_audio: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Audio encoder: wav2vec2-base (95M, language-agnostic SSL)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        audio_dim = self.audio_encoder.config.hidden_size  # 768

        # Text encoder: AraBERT
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Deeper projection heads with layer norm
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        if freeze_audio:
            self._freeze(self.audio_encoder)
        if freeze_text:
            self._freeze(self.text_encoder)

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_audio_encoder(self, last_n_layers: int = 2):
        self._freeze(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - last_n_layers, total):
            for param in self.audio_encoder.encoder.layers[i].parameters():
                param.requires_grad = True

    def unfreeze_text_encoder(self, last_n_layers: int = 2):
        self._freeze(self.text_encoder)
        total = len(self.text_encoder.encoder.layer)
        for i in range(total - last_n_layers, total):
            for param in self.text_encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def encode_audio(self, audio_values, attention_mask=None):
        outputs = self.audio_encoder(audio_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        projected = self.audio_proj(pooled)
        return nn.functional.normalize(projected, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        projected = self.text_proj(cls_output)
        return nn.functional.normalize(projected, dim=-1)

    def forward(self, audio_values, input_ids, text_attention_mask, audio_attention_mask=None):
        audio_embeds = self.encode_audio(audio_values, audio_attention_mask)
        text_embeds = self.encode_text(input_ids, text_attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_audio = logit_scale * audio_embeds @ text_embeds.T
        logits_per_text = logits_per_audio.T
        return logits_per_audio, logits_per_text

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


def contrastive_loss(logits_per_audio, logits_per_text):
    batch_size = logits_per_audio.shape[0]
    labels = torch.arange(batch_size, device=logits_per_audio.device)
    loss_a = nn.functional.cross_entropy(logits_per_audio, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


def get_tokenizer(model_name="aubmindlab/bert-base-arabertv02"):
    return AutoTokenizer.from_pretrained(model_name)
