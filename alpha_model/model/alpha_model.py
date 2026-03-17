"""Full alpha model: PatchTST + StaticEncoder + CrossChannelMixer + QuantileHead.

This is the complete model used for both training (on Colab) and inference (on M1 CPU).
"""

import time
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn

from .cross_mixer import CrossChannelMixer
from .patch_tst import PatchTST
from .prediction_head import QuantileHead
from .static_encoder import StaticEncoder


class AlphaModel(nn.Module):
    """End-to-end alpha prediction model.

    Input:
        time_series: (B, 23, 250) — 23 feature channels × 250 trading days
        sector_onehot: (B, 11)
        cap_onehot: (B, 5)

    Output:
        predictions: (B, 4, 3) — 4 horizons × 3 quantiles [q10, q50, q90]
    """

    HORIZONS = [1, 5, 21, 63]
    HORIZON_LABELS = ["1d", "5d", "21d", "63d"]

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
        d_static: int = 64,
        d_mixer_hidden: int = 512,
        d_mixer_out: int = 256,
        mixer_dropout: float = 0.3,
    ):
        super().__init__()
        self.patch_tst = PatchTST(
            n_channels=n_channels,
            context_len=context_len,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.static_encoder = StaticEncoder(n_sector=11, n_cap=5, d_out=d_static)
        self.cross_mixer = CrossChannelMixer(
            n_channels=n_channels,
            d_channel=d_model,
            d_static=d_static,
            d_hidden=d_mixer_hidden,
            d_out=d_mixer_out,
            dropout=mixer_dropout,
        )
        self.quantile_head = QuantileHead(d_in=d_mixer_out)

    def forward(
        self,
        time_series: torch.Tensor,
        sector_onehot: torch.Tensor,
        cap_onehot: torch.Tensor,
    ) -> torch.Tensor:
        channel_reps = self.patch_tst(time_series)  # (B, 23, 128)
        static_rep = self.static_encoder(sector_onehot, cap_onehot)  # (B, 64)
        mixed = self.cross_mixer(channel_reps, static_rep)  # (B, 256)
        return self.quantile_head(mixed)  # (B, 4, 3)

    def predict(
        self,
        time_series: torch.Tensor,
        sector_onehot: torch.Tensor,
        cap_onehot: torch.Tensor,
    ) -> dict:
        """Inference wrapper for single sample — returns a dict with named predictions.

        Expects batch dim but uses only first sample (single-ticker inference).
        If no batch dim, adds one.
        """
        self.eval()
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)
            sector_onehot = sector_onehot.unsqueeze(0)
            cap_onehot = cap_onehot.unsqueeze(0)

        with torch.no_grad():
            start = time.perf_counter()
            preds = self.forward(time_series, sector_onehot, cap_onehot)
            latency_ms = (time.perf_counter() - start) * 1000

        # Use first sample: preds[0] shape: (4, 3) — [q10, q50, q90] per horizon
        p = preds[0]
        result = {"inference_latency_ms": latency_ms}
        for i, label in enumerate(self.HORIZON_LABELS):
            result[f"q10_{label}"] = p[i, 0].item()
            result[f"alpha_{label}"] = p[i, 1].item()  # q50 = point estimate
            result[f"q90_{label}"] = p[i, 2].item()
        return result

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AlphaModel":
        """Load trained model from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        # Support both raw state_dict and checkpoint dict
        if "model_state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            model = cls(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = cls()
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model

    def save(self, path: str, config: dict = None, fold: str = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
            "fold": fold,
            "date": str(date.today()),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
