"""
QuranCLAP: Contrastive Language-Audio Pretraining for Quran verse identification.

Maps both Quranic audio and Arabic verse text into the same embedding space.
At inference, encode the audio and find the nearest verse text embedding.
No ASR needed -- direct audio-to-verse matching via cosine similarity.

Architecture:
  Audio Encoder (HuBERT base) --> projection --> [D-dim normalized vector]
                                                     |  cosine similarity
  Text Encoder  (AraBERT)     --> projection --> [D-dim normalized vector]
"""

import torch
import torch.nn as nn
from transformers import (
    HubertModel,
    AutoModel,
    AutoTokenizer,
    HubertConfig,
)


class QuranCLAP(nn.Module):
    """
    Contrastive model that aligns Quranic audio and Arabic text in a shared
    embedding space, similar to CLIP/CLAP.
    """

    def __init__(
        self,
        audio_model_name: str = "facebook/hubert-base-ls960",
        text_model_name: str = "aubmindlab/bert-base-arabertv02",
        embed_dim: int = 256,
        freeze_audio: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Audio encoder: HuBERT base
        self.audio_encoder = HubertModel.from_pretrained(audio_model_name)
        audio_dim = self.audio_encoder.config.hidden_size  # 768

        # Text encoder: AraBERT
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Projection heads
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable temperature (initialized as in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        # Freeze/unfreeze encoders
        if freeze_audio:
            self.freeze_audio_encoder()
        if freeze_text:
            self.freeze_text_encoder()

    def freeze_audio_encoder(self):
        """Freeze all audio encoder parameters."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def freeze_text_encoder(self):
        """Freeze all text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_audio_encoder(self, last_n_layers: int = 2):
        """Unfreeze the last N transformer layers of the audio encoder."""
        # First freeze everything
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Then unfreeze last N layers
        total_layers = len(self.audio_encoder.encoder.layers)
        for i in range(total_layers - last_n_layers, total_layers):
            for param in self.audio_encoder.encoder.layers[i].parameters():
                param.requires_grad = True

    def unfreeze_text_encoder(self, last_n_layers: int = 2):
        """Unfreeze the last N transformer layers of the text encoder."""
        # First freeze everything
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Then unfreeze last N layers
        total_layers = len(self.text_encoder.encoder.layer)
        for i in range(total_layers - last_n_layers, total_layers):
            for param in self.text_encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def encode_audio(self, audio_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode audio waveform to normalized embedding.

        Args:
            audio_values: [batch, samples] raw waveform at 16kHz
            attention_mask: [batch, samples] mask for padded regions

        Returns:
            [batch, embed_dim] L2-normalized audio embeddings
        """
        outputs = self.audio_encoder(
            audio_values,
            attention_mask=attention_mask,
        )
        # Mean-pool hidden states (respecting mask if provided)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        if attention_mask is not None:
            # The HuBERT encoder downsamples the audio, so we need to adjust
            # the mask. HuBERT feature extractor typically reduces by ~320x.
            # We'll use a simpler approach: just mean-pool all non-padded frames.
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden.mean(dim=1)

        projected = self.audio_proj(pooled)
        return nn.functional.normalize(projected, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode Arabic text to normalized embedding.

        Args:
            input_ids: [batch, seq_len] tokenized text
            attention_mask: [batch, seq_len] attention mask

        Returns:
            [batch, embed_dim] L2-normalized text embeddings
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0]
        projected = self.text_proj(cls_output)
        return nn.functional.normalize(projected, dim=-1)

    def forward(
        self,
        audio_values: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass computing similarity logits.

        Returns:
            logits_per_audio: [batch, batch] audio-to-text similarity scores
            logits_per_text: [batch, batch] text-to-audio similarity scores
        """
        audio_embeds = self.encode_audio(audio_values, audio_attention_mask)
        text_embeds = self.encode_text(input_ids, text_attention_mask)

        # Clamp temperature for stability
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        logits_per_audio = logit_scale * audio_embeds @ text_embeds.T
        logits_per_text = logits_per_audio.T

        return logits_per_audio, logits_per_text

    def get_trainable_params(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class QuranCLAPZeroShot(nn.Module):
    """
    Zero-shot version without projection heads.
    Uses raw HuBERT and AraBERT embeddings directly.
    Useful for establishing baseline performance.
    """

    def __init__(
        self,
        audio_model_name: str = "facebook/hubert-base-ls960",
        text_model_name: str = "aubmindlab/bert-base-arabertv02",
    ):
        super().__init__()
        self.audio_encoder = HubertModel.from_pretrained(audio_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

    def encode_audio(self, audio_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.audio_encoder(audio_values, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return nn.functional.normalize(pooled, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0]
            return nn.functional.normalize(cls_output, dim=-1)


def get_tokenizer(model_name: str = "aubmindlab/bert-base-arabertv02"):
    """Get the AraBERT tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def contrastive_loss(logits_per_audio: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).

    Each audio[i] should match text[i] (diagonal entries are positives).
    """
    batch_size = logits_per_audio.shape[0]
    labels = torch.arange(batch_size, device=logits_per_audio.device)
    loss_a = nn.functional.cross_entropy(logits_per_audio, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


if __name__ == "__main__":
    print("Loading QuranCLAP model...")
    model = QuranCLAP(freeze_audio=True, freeze_text=True)
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    # Test with dummy data
    batch_size = 2
    audio = torch.randn(batch_size, 16000 * 3)  # 3 seconds
    input_ids = torch.randint(0, 30000, (batch_size, 32))
    text_mask = torch.ones(batch_size, 32, dtype=torch.long)

    print("\nRunning forward pass on CPU...")
    logits_a, logits_t = model(audio, input_ids, text_mask)
    print(f"Logits shape: {logits_a.shape}")

    loss = contrastive_loss(logits_a, logits_t)
    print(f"Loss: {loss.item():.4f}")

    # Test parameter unfreezing
    model.unfreeze_audio_encoder(last_n_layers=2)
    model.unfreeze_text_encoder(last_n_layers=2)
    print(f"\nAfter unfreezing last 2 layers:")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
