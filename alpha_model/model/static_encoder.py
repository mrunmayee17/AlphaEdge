"""Static encoder — maps sector one-hot + market cap bin to a dense vector."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticEncoder(nn.Module):
    """Encodes static features (sector, market cap tier) into a dense vector.

    Inputs:
        sector_onehot: (B, 11) — 11 GICS sectors
        cap_onehot: (B, 5) — mega, large, mid, small, micro
    Output:
        (B, d_out) — dense static representation
    """

    def __init__(self, n_sector: int = 11, n_cap: int = 5, d_out: int = 64):
        super().__init__()
        self.linear = nn.Linear(n_sector + n_cap, d_out)

    def forward(self, sector_onehot: torch.Tensor, cap_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sector_onehot, cap_onehot], dim=-1)  # (B, 16)
        return F.gelu(self.linear(x))  # (B, 64)
