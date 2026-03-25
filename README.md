# BAM

## FinCast Fine-Tune Metrics and Evaluation

This README documents the latest local FinCast LoRA run artifacts in `models/fincast_runtime_local`.

### Run Snapshot

- Run timestamp (UTC): `2026-03-24T05:33:30.394853+00:00`
- Base model: frozen FinCast checkpoint (`v1.pth`)
- Fine-tune type: LoRA (`lora_r=8`, `lora_alpha=16`, `dropout=0.05`, `attn_mlp` targets)
- Selection metric: validation `rank_ic` (best epoch: `3`)
- Target: 5-day forward return
- Universe: `ES, NQ, RTY, YM, ZN, ZB, CL, NG, GC, HG`
- Splits:
  - Train: `2017-07-09` to `2023-12-22`
  - Validation: `2024-01-01` to `2024-12-25`
  - Holdout: `2025-01-01` to `2025-12-25`

### Evaluation Protocol

Metrics are computed in `alpha_model/training/fincast_lora_colab.py`:

- `directional_accuracy`: sign agreement between prediction and target
- `rank_ic`: per-date rank correlation, then mean over dates
- `top_bottom_spread`: average top-minus-bottom bucket target spread per date
- `turnover_proxy`: mean sign-change rate through time per symbol

Reporting slices:

- `pooled::all`
- `asset_class::{commodities,equities,rates}`
- `confidence_top_20pct::*` and `confidence_top_10pct::*` using `|prediction|` confidence filters

### Pooled Results (Frozen vs Fine-Tuned)

| Split | Rows | Frozen DA | LoRA DA | DA Delta | Frozen Rank IC | LoRA Rank IC | Rank IC Delta | Frozen Spread | LoRA Spread | Spread Delta | Frozen Turnover | LoRA Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1760 | 0.5222 | 0.5278 | +0.0057 | 0.0335 | 0.0208 | -0.0127 | 0.0034 | 0.0011 | -0.0023 | 0.8080 | 0.1623 |
| Holdout | 1750 | 0.5263 | 0.5806 | +0.0543 | -0.0016 | 0.0692 | +0.0707 | -0.0039 | -0.0001 | +0.0038 | 0.8057 | 0.1080 |

### Holdout Directional Accuracy by Asset Class

| Slice | Frozen DA | LoRA DA | Delta |
|---|---:|---:|---:|
| Commodities | 0.5257 | 0.5629 | +0.0371 |
| Equities | 0.5357 | 0.6514 | +0.1157 |
| Rates | 0.5086 | 0.4743 | -0.0343 |

### Holdout Confidence Slices (Directional Accuracy)

| Slice | Frozen DA | LoRA DA | Delta | Rows (LoRA) |
|---|---:|---:|---:|---:|
| Top 20% confidence (all) | 0.4657 | 0.5914 | +0.1257 | 350 |
| Top 10% confidence (all) | 0.4800 | 0.5886 | +0.1086 | 175 |

### Baseline Context (Holdout, pooled::all)

| Model | Directional Accuracy | Rank IC | Top-Bottom Spread | Turnover Proxy | Rows |
|---|---:|---:|---:|---:|---:|
| Zero | 0.0040 | 0.0000 | 0.6376 | 0.0000 | 3020 |
| Momentum (5d) | 0.4825 | -0.0370 | -0.2793 | 0.3887 | 3020 |
| Linear | 0.5669 | -0.0220 | -0.2698 | 0.0558 | 3020 |
| Fine-tuned LoRA | 0.5806 | 0.0692 | -0.0001 | 0.1080 | 1750 |

Notes:

- Baseline files (`holdout_metrics.json`) and LoRA files (`custom_lora_*_metrics.json`) use different evaluation row counts, so compare as directional context, not a strict apples-to-apples leaderboard.
- Some confidence/asset slices can be very small (especially `rates`), so pooled and major asset-class slices are the most stable.

### Artifact Files

- `models/fincast_runtime_local/training_status.json`
- `models/fincast_runtime_local/frozen_fincast_summary.json`
- `models/fincast_runtime_local/custom_lora_validation_metrics.json`
- `models/fincast_runtime_local/custom_lora_holdout_metrics.json`
- `models/fincast_runtime_local/frozen_vs_lora_comparison.json`
- `models/fincast_runtime_local/custom_lora_history.csv`
- `models/fincast_runtime_local/config_manifest.json`
