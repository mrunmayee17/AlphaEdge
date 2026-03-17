"""Cross-channel mixer — combines per-channel PatchTST outputs with static features."""

import torch
import torch.nn as nn


class CrossChannelMixer(nn.Module):
    """MLP that mixes all channel representations with static features.

    Input:
        channel_reps: (B, n_channels, d_channel) — from PatchTST, e.g., (B, 23, 128)
        static_rep: (B, d_static) — from StaticEncoder, e.g., (B, 64)
    Output:
        (B, d_out) — mixed representation for prediction heads
    """

    def __init__(
        self,
        n_channels: int = 23,
        d_channel: int = 128,
        d_static: int = 64,
        d_hidden: int = 512,
        d_out: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        d_in = n_channels * d_channel + d_static  # 23*128 + 64 = 3008
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, channel_reps: torch.Tensor, static_rep: torch.Tensor) -> torch.Tensor:
        # Flatten channel representations: (B, 23, 128) → (B, 2944)
        x = torch.cat([channel_reps.flatten(1), static_rep], dim=-1)  # (B, 3008)
        return self.mlp(x)  # (B, 256)
