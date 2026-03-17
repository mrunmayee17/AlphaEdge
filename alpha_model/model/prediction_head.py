"""Quantile prediction head — separate head per horizon, 3 quantiles each."""

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """Predicts quantiles (q10, q50, q90) at 4 horizons (1d, 5d, 21d, 63d).

    Separate linear head per horizon to prevent interference.

    Input: (B, d_in) — e.g., (B, 256) from CrossChannelMixer
    Output: (B, 4, 3) — 4 horizons × 3 quantiles
    """

    HORIZONS = [1, 5, 21, 63]
    QUANTILES = [0.10, 0.50, 0.90]

    def __init__(self, d_in: int = 256, n_horizons: int = 4, n_quantiles: int = 3):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.heads = nn.ModuleList([nn.Linear(d_in, n_quantiles) for _ in range(n_horizons)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, n_horizons, n_quantiles)
            dim 1 = horizons [1d, 5d, 21d, 63d]
            dim 2 = quantiles [q10, q50, q90]
        """
        return torch.stack([head(x) for head in self.heads], dim=1)


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pinball loss for quantile regression.

    Args:
        predictions: (B, n_horizons, n_quantiles)
        targets: (B, n_horizons) — actual forward returns
    """
    quantiles = torch.tensor(QuantileHead.QUANTILES, device=predictions.device)
    # Expand targets: (B, n_horizons) → (B, n_horizons, 1)
    targets = targets.unsqueeze(-1)
    errors = targets - predictions  # (B, n_horizons, n_quantiles)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()
