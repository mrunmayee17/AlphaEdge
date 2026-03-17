"""PatchTST walk-forward training script — runs on Google Colab T4 GPU.

Walk-forward folds:
  Fold 1: train 2010-2014, val 2015, test 2016
  Fold 2: train 2011-2015, val 2016, test 2017
  ...
  Fold 8: train 2017-2021, val 2022, test 2023
  Fold 9: train 2018-2022, val 2023, test 2024

Usage (Colab):
    !python -m alpha_model.training.train_patchtst --fold 9 --device cuda

Usage (local M1, for testing):
    python -m alpha_model.training.train_patchtst --fold 9 --device mps --epochs 2
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
    9: {"train": (2010, 2024), "val": 2025, "test": 2026},
}

# Training hyperparameters
DEFAULT_CONFIG = {
    "n_channels": 23,
    "context_len": 250,
    "patch_len": 5,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 3,
    "d_ff": 256,
    "dropout": 0.2,
    "d_static": 64,
    "d_mixer_hidden": 512,
    "d_mixer_out": 256,
    "mixer_dropout": 0.3,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 7,
    "seed": 42,
}

# Sector → one-hot index mapping
SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]
SECTOR_TO_IDX = {s: i for i, s in enumerate(SECTORS)}

# Market cap bins (in billions)
CAP_BINS = [0, 2, 10, 50, 200, float("inf")]  # micro, small, mid, large, mega
CAP_LABELS = ["micro", "small", "mid", "large", "mega"]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
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


class AlphaDataset(Dataset):
    """Memory-efficient dataset for PatchTST training.

    Stores per-ticker numpy arrays and an index of valid sample positions.
    Computes sliding windows on-the-fly in __getitem__ to avoid OOM on Colab.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        ticker_meta: dict[str, dict],
        context_len: int = 250,
    ):
        self.context_len = context_len

        # Per-ticker data stored as contiguous numpy arrays (shared, not copied per sample)
        self.ticker_features: list[np.ndarray] = []   # each (N, 23) float32
        self.ticker_targets: list[np.ndarray] = []    # each (N, 4) float32
        self.ticker_sector_oh: list[np.ndarray] = []  # each (11,) float32
        self.ticker_cap_oh: list[np.ndarray] = []     # each (5,) float32

        # Index: list of (ticker_idx, row_idx) for valid samples
        self.index: list[tuple[int, int]] = []

        tickers = features_df["ticker"].unique()
        ticker_idx = 0

        for ticker in tickers:
            f_ticker = features_df[features_df["ticker"] == ticker].sort_values("date")
            t_ticker = targets_df[targets_df["ticker"] == ticker].sort_values("date")

            if len(f_ticker) < context_len + 1:
                continue

            # Left join: keep all feature rows, targets only where dates match
            merged = f_ticker.merge(
                t_ticker[["date"] + TARGET_COLS], on="date", how="left"
            )

            # Get sector and cap info
            meta = ticker_meta.get(ticker, {})
            sector_idx = SECTOR_TO_IDX.get(meta.get("sector", ""), 7)
            cap_bin = meta.get("cap_bin", 3)

            sector_oh = np.zeros(11, dtype=np.float32)
            sector_oh[sector_idx] = 1.0
            cap_oh = np.zeros(5, dtype=np.float32)
            cap_oh[cap_bin] = 1.0

            feat_values = merged[FEATURE_COLS].values.astype(np.float32)
            target_values = merged[TARGET_COLS].values.astype(np.float32)

            # NaN-fill features once (avoid per-sample cost)
            feat_values = np.nan_to_num(feat_values, nan=0.0)

            # Store arrays
            self.ticker_features.append(feat_values)
            self.ticker_targets.append(target_values)
            self.ticker_sector_oh.append(sector_oh)
            self.ticker_cap_oh.append(cap_oh)

            # Build index of valid samples
            for i in range(context_len, len(merged)):
                target = target_values[i]
                if not np.any(np.isnan(target)):
                    self.index.append((ticker_idx, i))

            ticker_idx += 1

        logger.info(f"Created dataset: {len(self.index)} samples from {ticker_idx} tickers")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker_idx, row_idx = self.index[idx]
        feat = self.ticker_features[ticker_idx]
        window = feat[row_idx - self.context_len : row_idx]  # (250, 23)
        target = self.ticker_targets[ticker_idx][row_idx]     # (4,)

        return (
            torch.as_tensor(window.T.copy(), dtype=torch.float32),  # (23, 250)
            torch.as_tensor(self.ticker_sector_oh[ticker_idx], dtype=torch.float32),
            torch.as_tensor(self.ticker_cap_oh[ticker_idx], dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
        )


def load_data():
    """Load features and targets from parquet files."""
    features_path = PROCESSED_DIR / "features.parquet"
    targets_path = PROCESSED_DIR / "targets.parquet"

    if not features_path.exists() or not targets_path.exists():
        raise FileNotFoundError(
            f"Data not found at {PROCESSED_DIR}. "
            "Run build_features.py and build_targets.py first."
        )

    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)

    logger.info(f"Loaded features: {features.shape}, targets: {targets.shape}")
    return features, targets


def split_by_year(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Filter DataFrame by year range (inclusive)."""
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.year >= year_start) & (df["date"].dt.year <= year_end)
    return df[mask].copy()


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    quantiles = torch.tensor([0.10, 0.50, 0.90], device=predictions.device)
    targets_expanded = targets.unsqueeze(-1)  # (B, 4) → (B, 4, 1)
    errors = targets_expanded - predictions  # (B, 4, 3)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Information Coefficient (Spearman rank correlation)."""
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
) -> dict:
    """Evaluate model on a dataset — returns loss and IC per horizon."""
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
            loss = quantile_loss(preds, target)
            total_loss += loss.item()
            n_batches += 1

            # q50 predictions for IC
            all_preds.append(preds[:, :, 1].cpu().numpy())  # (B, 4)
            all_targets.append(target.cpu().numpy())  # (B, 4)

    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    horizon_labels = ["1d", "5d", "21d", "63d"]
    metrics = {"loss": avg_loss}
    for i, label in enumerate(horizon_labels):
        metrics[f"ic_{label}"] = compute_ic(all_preds[:, i], all_targets[:, i])

    return metrics


def train_fold(
    fold_num: int,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    ticker_meta: dict[str, dict],
    config: dict,
    device: torch.device,
    use_wandb: bool = False,
) -> tuple:
    """Train one fold. Returns (model, metrics)."""
    fold = FOLDS[fold_num]
    train_start, train_end = fold["train"]
    val_year = fold["val"]

    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_num}: train {train_start}-{train_end}, val {val_year}, test {fold['test']}")
    logger.info(f"{'='*60}")

    # Split data — include prior year for context window
    # Training: features from (train_start-1) for context, targets from train_start
    train_features = split_by_year(features, train_start - 1, train_end)
    train_targets = split_by_year(targets, train_start, train_end)
    # Validation: features from (val_year-1) for context, targets from val_year
    val_features = split_by_year(features, val_year - 1, val_year)
    val_targets = split_by_year(targets, val_year, val_year)

    logger.info(f"Train: {len(train_features)} feature rows, {len(train_targets)} target rows")
    logger.info(f"Val: {len(val_features)} feature rows, {len(val_targets)} target rows")

    # Create datasets
    train_dataset = AlphaDataset(train_features, train_targets, ticker_meta, config["context_len"])
    val_dataset = AlphaDataset(val_features, val_targets, ticker_meta, config["context_len"])

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

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # W&B
    if use_wandb:
        import wandb
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "mrunmayeerane09-skystudio-media"),
            project="bam-alpha",
            name=f"fold_{fold_num}",
            config={**config, "fold": fold_num, **fold},
        )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for ts, sector, cap, target in train_loader:
            ts = ts.to(device)
            sector = sector.to(device)
            cap = cap.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(ts, sector, cap)
            loss = quantile_loss(preds, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["loss"]

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"IC_21d={val_metrics.get('ic_21d', 0):.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
                "lr": scheduler.get_last_lr()[0],
            })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    final_metrics = evaluate_model(model, val_loader, device)
    logger.info(f"Final val metrics: {final_metrics}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"patch_tst_fold{fold_num}.pt"
    model.cpu().save(str(ckpt_path), config=model_config, fold=str(fold_num))
    logger.info(f"Saved checkpoint to {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.log({"best_val_loss": best_val_loss, **{f"final_{k}": v for k, v in final_metrics.items()}})
        artifact = wandb.Artifact(f"model-fold{fold_num}", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description="PatchTST walk-forward training")
    parser.add_argument("--fold", type=int, default=9, help="Fold number (1-9)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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

    set_seed(config["seed"])

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load data
    features, targets = load_data()

    # Load ticker metadata (sector + market cap)
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

    # Train
    folds_to_train = range(1, 10) if args.all_folds else [args.fold]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_num in folds_to_train:
        ckpt_path = CHECKPOINT_DIR / f"patch_tst_fold{fold_num}.pt"
        if ckpt_path.exists() and not args.all_folds:
            logger.info(f"Fold {fold_num} already trained, skipping. Delete {ckpt_path} to retrain.")
            continue

        model, metrics = train_fold(
            fold_num, features, targets, ticker_meta, config, device, use_wandb=args.wandb
        )

        # Free memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
