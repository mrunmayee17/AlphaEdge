"""PatchTST walk-forward training v2 — fixes for prediction collapse + sample redundancy.

Key changes from v1:
  1. Strided sampling (stride=5) — eliminates 250x redundancy, ~5x fewer effective-duplicate samples
  2. Lower learning rate (3e-4) with linear warmup — prevents epoch-1 memorization
  3. Horizon-weighted loss — upweights 21d/63d where model has real signal
  4. Stronger regularization — channel dropout (randomly zero entire feature channels)
  5. Target standardization — per-horizon z-scoring so loss scale is comparable across horizons

Usage (Colab):
    !python -m alpha_model.training.train_patchtst_v2 --fold 9 --device cuda --wandb

Usage (local M1, for testing):
    python -m alpha_model.training.train_patchtst_v2 --fold 9 --device mps --epochs 2
"""

import argparse
import gc
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "alpha_model" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "alpha_model" / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "models"

# Walk-forward fold definitions
FOLDS = {
    1: {"train": (2010, 2014), "val": 2015, "test": 2016},
    2: {"train": (2011, 2015), "val": 2016, "test": 2017},
    3: {"train": (2012, 2016), "val": 2017, "test": 2018},
    4: {"train": (2013, 2017), "val": 2018, "test": 2019},
    5: {"train": (2014, 2018), "val": 2019, "test": 2020},
    6: {"train": (2015, 2019), "val": 2020, "test": 2021},
    7: {"train": (2016, 2020), "val": 2021, "test": 2022},
    8: {"train": (2017, 2021), "val": 2022, "test": 2023},
    9: {"train": (2018, 2024), "val": 2025, "test": 2026},
}

# v2 training hyperparameters — key changes marked with ###
DEFAULT_CONFIG = {
    "n_channels": 23,
    "context_len": 250,
    "patch_len": 5,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 3,
    "d_ff": 256,
    "dropout": 0.1,          ### was 0.2 — reduced since model underfits
    "d_static": 64,
    "d_mixer_hidden": 512,
    "d_mixer_out": 256,
    "mixer_dropout": 0.2,    ### was 0.3 — reduced
    "batch_size": 256,
    "lr": 3e-4,              ### was 1e-3 — lowered to prevent epoch-1 memorization
    "weight_decay": 1e-3,    ### was 1e-4 — stronger L2 reg
    "warmup_epochs": 3,      ### NEW: linear warmup before cosine decay
    "epochs": 50,
    "patience": 10,           ### was 7 — more patience with lower LR
    "seed": 42,
    "sample_stride": 5,      ### NEW: sample every 5th position per ticker
    "channel_dropout": 0.15, ### NEW: randomly zero entire channels during training
    # Per-horizon loss weights: upweight 21d/63d where model has real signal
    "horizon_weights": [0.5, 0.5, 1.5, 2.0],  ### NEW
}

# Sector → one-hot index mapping
SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]
SECTOR_TO_IDX = {s: i for i, s in enumerate(SECTORS)}

CAP_BINS = [0, 2, 10, 50, 200, float("inf")]
CAP_LABELS = ["micro", "small", "mid", "large", "mega"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


FEATURE_COLS = [
    "pct_open", "pct_high", "pct_low", "pct_close",
    "log_volume", "volume_ratio",
    "realized_vol_21d", "garman_klass_vol", "vol_of_vol",
    "beta_Mkt-RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_Mom",
    "spy_return", "vix_level", "vix_change",
    "yield_10y_change", "credit_spread_change", "tlt_return",
    "sector_neutral_return_1d", "sector_neutral_return_5d",
]
TARGET_COLS = ["alpha_1d", "alpha_5d", "alpha_21d", "alpha_63d"]


class AlphaDatasetV2(Dataset):
    """Memory-efficient dataset with strided sampling + target stats for standardization.

    Key difference from v1: sample_stride parameter reduces overlapping windows.
    With stride=5, each ticker contributes ~50 samples/year instead of ~250.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        ticker_meta: dict[str, dict],
        context_len: int = 250,
        sample_stride: int = 1,
        channel_dropout: float = 0.0,
        is_train: bool = False,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):
        self.context_len = context_len
        self.channel_dropout = channel_dropout if is_train else 0.0
        self.is_train = is_train
        self.n_channels = len(FEATURE_COLS)

        self.ticker_features: list[np.ndarray] = []
        self.ticker_targets: list[np.ndarray] = []
        self.ticker_sector_oh: list[np.ndarray] = []
        self.ticker_cap_oh: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []

        # For target standardization
        all_targets_for_stats = []

        tickers = features_df["ticker"].unique()
        ticker_idx = 0

        for ticker in tickers:
            f_ticker = features_df[features_df["ticker"] == ticker].sort_values("date")
            t_ticker = targets_df[targets_df["ticker"] == ticker].sort_values("date")

            if len(f_ticker) < context_len + 1:
                continue

            merged = f_ticker.merge(
                t_ticker[["date"] + TARGET_COLS], on="date", how="left"
            )

            meta = ticker_meta.get(ticker, {})
            sector_idx = SECTOR_TO_IDX.get(meta.get("sector", ""), 7)
            cap_bin = meta.get("cap_bin", 3)

            sector_oh = np.zeros(11, dtype=np.float32)
            sector_oh[sector_idx] = 1.0
            cap_oh = np.zeros(5, dtype=np.float32)
            cap_oh[cap_bin] = 1.0

            feat_values = merged[FEATURE_COLS].values.astype(np.float32)
            target_values = merged[TARGET_COLS].values.astype(np.float32)

            feat_values = np.nan_to_num(feat_values, nan=0.0)

            self.ticker_features.append(feat_values)
            self.ticker_targets.append(target_values)
            self.ticker_sector_oh.append(sector_oh)
            self.ticker_cap_oh.append(cap_oh)

            # Build index with stride
            valid_positions = []
            for i in range(context_len, len(merged)):
                target = target_values[i]
                if not np.any(np.isnan(target)):
                    valid_positions.append(i)

            # Apply stride — take every N-th valid position
            for pos in valid_positions[::sample_stride]:
                self.index.append((ticker_idx, pos))
                all_targets_for_stats.append(target_values[pos])

            ticker_idx += 1

        # Compute or use provided target stats (for standardization)
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean
            self.target_std = target_std
        elif all_targets_for_stats:
            all_tgt = np.stack(all_targets_for_stats)
            self.target_mean = all_tgt.mean(axis=0).astype(np.float32)
            self.target_std = np.clip(all_tgt.std(axis=0), 1e-8, None).astype(np.float32)
        else:
            self.target_mean = np.zeros(4, dtype=np.float32)
            self.target_std = np.ones(4, dtype=np.float32)

        logger.info(
            f"Created dataset: {len(self.index)} samples from {ticker_idx} tickers "
            f"(stride={sample_stride})"
        )
        logger.info(f"Target stats — mean: {self.target_mean}, std: {self.target_std}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker_idx, row_idx = self.index[idx]
        feat = self.ticker_features[ticker_idx]
        window = feat[row_idx - self.context_len : row_idx]  # (250, 23)

        # Raw target (not standardized — standardization applied in loss function)
        target = self.ticker_targets[ticker_idx][row_idx]  # (4,)

        ts = torch.as_tensor(window.T.copy(), dtype=torch.float32)  # (23, 250)

        # Channel dropout: randomly zero entire channels during training
        if self.channel_dropout > 0 and self.is_train:
            mask = torch.bernoulli(
                torch.full((self.n_channels, 1), 1.0 - self.channel_dropout)
            )
            ts = ts * mask / (1.0 - self.channel_dropout)  # scale up to preserve magnitude

        return (
            ts,
            torch.as_tensor(self.ticker_sector_oh[ticker_idx], dtype=torch.float32),
            torch.as_tensor(self.ticker_cap_oh[ticker_idx], dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
        )


def weighted_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_weights: list[float],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Pinball loss with per-horizon weighting + target standardization.

    Standardizing targets ensures equal gradient contribution across horizons.
    Then horizon_weights let us upweight 21d/63d where the model has real signal.
    """
    quantiles = torch.tensor([0.10, 0.50, 0.90], device=predictions.device)
    hw = torch.tensor(horizon_weights, device=predictions.device)

    # Standardize targets so loss scale is comparable across horizons
    # predictions learn to predict standardized targets
    std_targets = (targets - target_mean.to(targets.device)) / target_std.to(targets.device)

    std_targets = std_targets.unsqueeze(-1)  # (B, 4, 1)
    errors = std_targets - predictions       # (B, 4, 3)
    pinball = torch.max(quantiles * errors, (quantiles - 1) * errors)  # (B, 4, 3)

    # Weight by horizon: (B, 4, 3) * (4, 1) → weighted mean
    weighted = pinball * hw.unsqueeze(-1)
    return weighted.mean()


def load_data():
    features_path = PROCESSED_DIR / "features.parquet"
    targets_path = PROCESSED_DIR / "targets.parquet"
    if not features_path.exists() or not targets_path.exists():
        raise FileNotFoundError(f"Data not found at {PROCESSED_DIR}")
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    logger.info(f"Loaded features: {features.shape}, targets: {targets.shape}")
    return features, targets


def split_by_year(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.year >= year_start) & (df["date"].dt.year <= year_end)
    return df[mask].copy()


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0
    corr, _ = spearmanr(predictions[mask], actuals[mask])
    return corr


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    horizon_weights: list[float],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ts, sector, cap, target in dataloader:
            ts = ts.to(device)
            sector = sector.to(device)
            cap = cap.to(device)
            target = target.to(device)

            preds = model(ts, sector, cap)  # (B, 4, 3)
            loss = weighted_quantile_loss(preds, target, horizon_weights, target_mean, target_std)
            total_loss += loss.item()
            n_batches += 1

            # De-standardize q50 predictions for IC computation
            q50_std = preds[:, :, 1]  # (B, 4) in standardized space
            q50_raw = q50_std * target_std.to(device) + target_mean.to(device)
            all_preds.append(q50_raw.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    horizon_labels = ["1d", "5d", "21d", "63d"]
    metrics = {"loss": avg_loss}
    for i, label in enumerate(horizon_labels):
        metrics[f"ic_{label}"] = compute_ic(all_preds[:, i], all_targets[:, i])

    # Decile spread (long-short)
    for i, label in enumerate(horizon_labels):
        order = np.argsort(all_preds[:, i])
        n = len(order)
        top = all_targets[order[-n // 10 :], i].mean()
        bot = all_targets[order[: n // 10], i].mean()
        metrics[f"ls_spread_{label}"] = top - bot

    # Prediction spread check
    for i, label in enumerate(horizon_labels):
        metrics[f"pred_std_{label}"] = all_preds[:, i].std()

    return metrics


def get_lr_with_warmup(epoch: int, warmup_epochs: int, base_lr: float, total_epochs: int) -> float:
    """Linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train_fold(
    fold_num: int,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    ticker_meta: dict[str, dict],
    config: dict,
    device: torch.device,
    use_wandb: bool = False,
) -> tuple:
    fold = FOLDS[fold_num]
    train_start, train_end = fold["train"]
    val_year = fold["val"]

    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_num} (v2): train {train_start}-{train_end}, val {val_year}, test {fold['test']}")
    logger.info(f"{'='*60}")
    logger.info(f"Key v2 changes: stride={config['sample_stride']}, lr={config['lr']}, "
                f"warmup={config['warmup_epochs']}, channel_drop={config['channel_dropout']}, "
                f"horizon_weights={config['horizon_weights']}")

    # Split data
    train_features = split_by_year(features, train_start - 1, train_end)
    train_targets = split_by_year(targets, train_start, train_end)
    val_features = split_by_year(features, val_year - 1, val_year)
    val_targets = split_by_year(targets, val_year, val_year)

    logger.info(f"Train: {len(train_features)} feature rows, {len(train_targets)} target rows")
    logger.info(f"Val: {len(val_features)} feature rows, {len(val_targets)} target rows")

    # Create datasets — stride for train, no stride for val (full evaluation)
    train_dataset = AlphaDatasetV2(
        train_features, train_targets, ticker_meta,
        context_len=config["context_len"],
        sample_stride=config["sample_stride"],
        channel_dropout=config["channel_dropout"],
        is_train=True,
    )
    val_dataset = AlphaDatasetV2(
        val_features, val_targets, ticker_meta,
        context_len=config["context_len"],
        sample_stride=1,  # evaluate on all positions
        channel_dropout=0.0,
        is_train=False,
        target_mean=train_dataset.target_mean,  # use train stats
        target_std=train_dataset.target_std,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Empty dataset — train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)} (was ~{len(train_dataset) * config['sample_stride']} in v1)")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Build model
    from alpha_model.model.alpha_model import AlphaModel

    model_config = {
        "n_channels": config["n_channels"],
        "context_len": config["context_len"],
        "patch_len": config["patch_len"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "d_ff": config["d_ff"],
        "dropout": config["dropout"],
        "d_static": config["d_static"],
        "d_mixer_hidden": config["d_mixer_hidden"],
        "d_mixer_out": config["d_mixer_out"],
        "mixer_dropout": config["mixer_dropout"],
    }
    model = AlphaModel(**model_config).to(device)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Target stats as tensors for loss
    target_mean = torch.tensor(train_dataset.target_mean, dtype=torch.float32)
    target_std = torch.tensor(train_dataset.target_std, dtype=torch.float32)

    # W&B
    if use_wandb:
        import wandb
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "mrunmayeerane09-skystudio-media"),
            project="bam-alpha",
            name=f"fold_{fold_num}_v2",
            config={**config, "fold": fold_num, **fold, "version": "v2"},
        )

    # Training loop
    best_val_loss = float("inf")
    best_val_ic21 = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Manual LR scheduling with warmup
        lr = get_lr_with_warmup(epoch, config["warmup_epochs"], config["lr"], config["epochs"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for ts, sector, cap, target in train_loader:
            ts = ts.to(device)
            sector = sector.to(device)
            cap = cap.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(ts, sector, cap)
            loss = weighted_quantile_loss(
                preds, target, config["horizon_weights"], target_mean, target_std
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate_model(
            model, val_loader, device, config["horizon_weights"], target_mean, target_std
        )
        val_loss = val_metrics["loss"]

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"IC_1d={val_metrics.get('ic_1d', 0):.4f} | "
            f"IC_5d={val_metrics.get('ic_5d', 0):.4f} | "
            f"IC_21d={val_metrics.get('ic_21d', 0):.4f} | "
            f"IC_63d={val_metrics.get('ic_63d', 0):.4f} | "
            f"pred_std_21d={val_metrics.get('pred_std_21d', 0):.5f} | "
            f"LS_21d={val_metrics.get('ls_spread_21d', 0):.5f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
                "lr": lr,
            })

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ic21 = val_metrics.get("ic_21d", 0)
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ★ New best val_loss={val_loss:.6f}, IC_21d={best_val_ic21:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final evaluation
    final_metrics = evaluate_model(
        model, val_loader, device, config["horizon_weights"], target_mean, target_std
    )
    logger.info(f"Final val metrics: {final_metrics}")

    # Save checkpoint — include target stats for inference de-standardization
    ckpt_path = CHECKPOINT_DIR / f"patch_tst_v2_fold{fold_num}.pt"
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "config": model_config,
        "fold": str(fold_num),
        "version": "v2",
        "target_mean": train_dataset.target_mean,
        "target_std": train_dataset.target_std,
        "metrics": final_metrics,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Saved v2 checkpoint to {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.log({"best_val_loss": best_val_loss, **{f"final_{k}": v for k, v in final_metrics.items()}})
        artifact = wandb.Artifact(f"model-v2-fold{fold_num}", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description="PatchTST walk-forward training v2")
    parser.add_argument("--fold", type=int, default=9, help="Fold number (1-9)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None, help="Sample stride (default 5)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--all-folds", action="store_true", help="Train all 9 folds")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr
    if args.stride:
        config["sample_stride"] = args.stride

    set_seed(config["seed"])

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    features, targets = load_data()

    meta_path = PROCESSED_DIR / "ticker_meta.parquet"
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        ticker_meta = {}
        for _, row in meta_df.iterrows():
            ticker_meta[row["ticker"]] = {
                "sector": row.get("sector", "Information Technology"),
                "cap_bin": int(row.get("cap_bin", 3)),
            }
    else:
        logger.warning("No ticker_meta.parquet found — using default sector/cap")
        ticker_meta = {}

    folds_to_train = range(1, 10) if args.all_folds else [args.fold]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_num in folds_to_train:
        model, metrics = train_fold(
            fold_num, features, targets, ticker_meta, config, device, use_wandb=args.wandb
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
