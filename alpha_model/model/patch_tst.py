"""PatchTST — Patch Time Series Transformer (custom implementation).

Channel-independent patching: each of 23 feature channels is patched and encoded
independently through shared Transformer layers. Output: per-channel representations.

Architecture:
  Input: (B, 23, 250) — 23 channels × 250 trading days
  Patching: unfold into 50 patches of 5 days each
  Per-channel encoding: shared TransformerEncoder (3 layers, 8 heads)
  Output: (B, 23, d_model) — per-channel representations
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        n_channels: int = 23,
        context_len: int = 250,
        patch_len: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.context_len = context_len
        self.patch_len = patch_len
        self.n_patches = context_len // patch_len  # 50
        self.d_model = d_model

        # Patch projection: (patch_len,) → (d_model,)
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Learnable positional embedding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Shared Transformer encoder (channel-independent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm after encoding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_channels, context_len) — e.g., (B, 23, 250)

        Returns:
            (B, n_channels, d_model) — per-channel representations
        """
        B, C, L = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"
        assert L == self.context_len, f"Expected {self.context_len} timesteps, got {L}"

        # Patch: unfold last dimension into patches
        # (B, C, L) → (B, C, n_patches, patch_len)
        x = x.unfold(2, self.patch_len, self.patch_len)

        # Flatten batch and channel dims for shared encoder
        # (B, C, n_patches, patch_len) → (B*C, n_patches, patch_len)
        x = x.reshape(B * C, self.n_patches, self.patch_len)

        # Project patches to d_model and add positional embedding
        x = self.patch_proj(x) + self.pos_embed  # (B*C, 50, 128)

        # Encode with Transformer
        x = self.encoder(x)  # (B*C, 50, 128)

        # Mean pool over patches → single vector per channel
        x = x.mean(dim=1)  # (B*C, 128)

        # Layer norm
        x = self.norm(x)

        # Reshape back to (B, C, d_model)
        x = x.reshape(B, C, self.d_model)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention rollout for interpretability.

        Returns per-patch importance scores averaged across layers and heads.
        Shape: (B, n_channels, n_patches)
        """
        B, C, L = x.shape
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = x.reshape(B * C, self.n_patches, self.patch_len)
        x = self.patch_proj(x) + self.pos_embed

        # Collect attention weights from each layer
        attn_weights = []
        for layer in self.encoder.layers:
            # Forward through self-attention, capture weights
            attn_out, weights = layer.self_attn(x, x, x, need_weights=True)
            attn_weights.append(weights)  # (B*C, n_patches, n_patches)
            # Continue forward pass through rest of layer
            x = layer.norm1(x + layer.dropout1(attn_out))
            x = layer.norm2(x + layer._ff_block(x))

        # Attention rollout: multiply attention matrices across layers
        rollout = attn_weights[0]
        for attn in attn_weights[1:]:
            rollout = torch.bmm(attn, rollout)

        # Sum across columns → per-patch importance
        importance = rollout.sum(dim=1)  # (B*C, n_patches)
        importance = importance / importance.sum(dim=1, keepdim=True)  # normalize

        return importance.reshape(B, C, self.n_patches)
