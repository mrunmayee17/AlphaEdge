"""Evaluate trained PatchTST model — IC, ICIR, hit rate, quantile calibration.

Usage:
    python -m alpha_model.evaluation.evaluate --model models/patch_tst_fold9.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "alpha_model" / "data" / "processed"


def compute_daily_ic(preds: np.ndarray, actuals: np.ndarray, dates: np.ndarray) -> dict:
    """Compute daily IC and ICIR (IC / std(IC))."""
    unique_dates = np.unique(dates)
    daily_ics = []

    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 5:
            continue
        p, a = preds[mask], actuals[mask]
        if np.std(p) == 0 or np.std(a) == 0:
            continue
        corr, _ = spearmanr(p, a)
        if not np.isnan(corr):
            daily_ics.append(corr)

    daily_ics = np.array(daily_ics)
    return {
        "ic_mean": daily_ics.mean() if len(daily_ics) > 0 else 0.0,
        "ic_std": daily_ics.std() if len(daily_ics) > 0 else 0.0,
        "icir": daily_ics.mean() / max(daily_ics.std(), 1e-8) if len(daily_ics) > 0 else 0.0,
        "n_days": len(daily_ics),
    }


def compute_hit_rate(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of predictions where sign matches actual."""
    mask = (preds != 0) & (actuals != 0) & ~np.isnan(preds) & ~np.isnan(actuals)
    if mask.sum() == 0:
        return 0.5
    return (np.sign(preds[mask]) == np.sign(actuals[mask])).mean()


def check_quantile_calibration(
    q10_preds: np.ndarray,
    q90_preds: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """Check if quantile predictions are well-calibrated."""
    mask = ~np.isnan(actuals) & ~np.isnan(q10_preds) & ~np.isnan(q90_preds)
    actuals = actuals[mask]
    q10_preds = q10_preds[mask]
    q90_preds = q90_preds[mask]

    pct_below_q10 = (actuals < q10_preds).mean()
    pct_above_q90 = (actuals > q90_preds).mean()
    pct_in_band = ((actuals >= q10_preds) & (actuals <= q90_preds)).mean()

    return {
        "pct_below_q10": pct_below_q10,  # target: 10%
        "pct_above_q90": pct_above_q90,  # target: 10%
        "pct_in_80_band": pct_in_band,   # target: 80%
        "q10_calibration_error": abs(pct_below_q10 - 0.10),
        "q90_calibration_error": abs(pct_above_q90 - 0.10),
    }


def evaluate(model_path: str, test_year: int = 2024, device: str = "cpu"):
    """Full evaluation of trained model on test set. Supports both v1 and v2 models."""
    from alpha_model.model.alpha_model import AlphaModel

    logger.info(f"Loading model from {model_path}")

    # Detect v2 model (has target_mean/target_std in checkpoint)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    is_v2 = "target_mean" in checkpoint if isinstance(checkpoint, dict) else False
    target_mean = checkpoint.get("target_mean") if is_v2 else None
    target_std = checkpoint.get("target_std") if is_v2 else None

    if is_v2:
        logger.info(f"Detected v2 model — target_mean={target_mean}, target_std={target_std}")

    model = AlphaModel.load(model_path, device=device)
    model.eval()

    # Load data
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    targets = pd.read_parquet(PROCESSED_DIR / "targets.parquet")

    features["date"] = pd.to_datetime(features["date"])
    targets["date"] = pd.to_datetime(targets["date"])
    test_features = features[features["date"].dt.year == test_year]
    test_targets = targets[targets["date"].dt.year == test_year]

    logger.info(f"Test set ({test_year}): {len(test_features)} feature rows, {len(test_targets)} target rows")

    # Load ticker metadata
    meta_path = PROCESSED_DIR / "ticker_meta.parquet"
    ticker_meta = {}
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        for _, row in meta_df.iterrows():
            ticker_meta[row["ticker"]] = {
                "sector": row.get("sector", "Information Technology"),
                "cap_bin": int(row.get("cap_bin", 3)),
            }

    # We need context from before test year
    context_start_year = test_year - 1
    features_with_context = features[features["date"].dt.year >= context_start_year]
    targets_with_context = targets[targets["date"].dt.year >= context_start_year]

    # Use v2 dataset if v2 model, else v1
    if is_v2:
        from alpha_model.training.train_patchtst_v2 import AlphaDatasetV2
        dataset = AlphaDatasetV2(
            features_with_context, targets_with_context, ticker_meta,
            context_len=250, sample_stride=1, is_train=False,
            target_mean=target_mean, target_std=target_std,
        )
    else:
        from alpha_model.training.train_patchtst import AlphaDataset
        dataset = AlphaDataset(features_with_context, targets_with_context, ticker_meta, context_len=250)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Collect predictions
    all_q10 = []
    all_q50 = []
    all_q90 = []
    all_targets_arr = []

    with torch.no_grad():
        for ts, sector, cap, target in loader:
            ts = ts.to(device)
            sector = sector.to(device)
            cap = cap.to(device)

            preds = model(ts, sector, cap)  # (B, 4, 3)

            # De-standardize v2 predictions back to raw return space
            if is_v2 and target_mean is not None:
                tm = torch.tensor(target_mean, device=device)
                ts_std = torch.tensor(target_std, device=device)
                for q_idx in range(3):
                    preds[:, :, q_idx] = preds[:, :, q_idx] * ts_std + tm

            all_q10.append(preds[:, :, 0].cpu().numpy())
            all_q50.append(preds[:, :, 1].cpu().numpy())
            all_q90.append(preds[:, :, 2].cpu().numpy())
            all_targets_arr.append(target.numpy())

    if not all_q50:
        logger.error("No predictions generated — check data availability")
        return

    all_q10 = np.concatenate(all_q10, axis=0)
    all_q50 = np.concatenate(all_q50, axis=0)
    all_q90 = np.concatenate(all_q90, axis=0)
    all_targets_arr = np.concatenate(all_targets_arr, axis=0)

    logger.info(f"Generated {len(all_q50)} predictions")

    # Evaluate per horizon
    horizon_labels = ["1d", "5d", "21d", "63d"]
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — Test Year {test_year}" + (" (v2 model)" if is_v2 else " (v1 model)"))
    print(f"{'='*60}\n")

    for i, label in enumerate(horizon_labels):
        preds = all_q50[:, i]
        actuals = all_targets_arr[:, i]

        ic, _ = spearmanr(preds, actuals)
        hit = compute_hit_rate(preds, actuals)
        cal = check_quantile_calibration(all_q10[:, i], all_q90[:, i], actuals)

        # Decile spread
        order = np.argsort(preds)
        n = len(order)
        top_actual = actuals[order[-n // 10 :]].mean()
        bot_actual = actuals[order[: n // 10]].mean()
        ls_spread = top_actual - bot_actual

        # Prediction spread
        pred_std = preds.std()
        tgt_std = actuals.std()

        print(f"Horizon {label}:")
        print(f"  IC (Spearman):    {ic:.4f}")
        print(f"  Hit Rate:         {hit:.1%}")
        print(f"  L/S Spread:       {ls_spread:.5f} (top-bottom decile)")
        print(f"  Pred Std:         {pred_std:.5f} (target std: {tgt_std:.5f}, ratio: {pred_std/tgt_std:.3f})")
        print(f"  Quantile Cal:")
        print(f"    Below q10:      {cal['pct_below_q10']:.1%} (target: 10%)")
        print(f"    Above q90:      {cal['pct_above_q90']:.1%} (target: 10%)")
        print(f"    In 80% band:    {cal['pct_in_80_band']:.1%} (target: 80%)")
        print()

    print(f"Model parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/patch_tst_fold9.pt")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(args.model, args.test_year, args.device)
