# Alpha Edge

Alpha Edge is an AI investment research app with two layers:

1. Forecasting layer: foundation-model forecasts (`chronos` or `fincast_lora`).
2. Committee layer: 5 specialist agents (`quant`, `fundamentals`, `sentiment`, `risk`, `macro`) that debate and produce a final investment memo.

## End-to-End Flow

1. Frontend calls `POST /api/v1/analysis` with a ticker and `forecast_model`.
2. Backend generates `1d`, `5d`, `21d`, `63d` alpha forecasts.
3. 5 agents run in parallel with tool data (Yahoo, Brave, BrightData, FRED).
4. Claims are checked, debated, and synthesized into an investment memo.
5. Session state is stored in Redis and exposed via polling/WebSocket.

## Forecast Engines

### Chronos (`forecast_model="chronos"`)

- Runtime model: `amazon/chronos-bolt-base` via `ChronosBoltPipeline`.
- Input: recent log-return context (`context_length=512`).
- Output: cumulative forecasts for `1d/5d/21d/63d` from Chronos quantiles.
- Metadata: `model_version=chronos-2:<model_name>`, `training_fold=pretrained`.

### FinCast LoRA (`forecast_model="fincast_lora"`)

- Base checkpoint: `Vincent05R/FinCast` (`v1.pth`).
- Adapter type: LoRA (`attn_mlp`, `r=8`, `alpha=16`, `dropout=0.05`, `train_base=false`).
- Trained futures universe: `ES NQ RTY YM ZN ZB CL NG GC HG`.
- Inference: iterative 5-step rollout to reach 63 trading days.

## What Was Fine-Tuned

### PatchTST work (custom alpha model stack)

Implemented in `alpha_model/`:

- Architecture: `PatchTST + StaticEncoder + CrossChannelMixer + QuantileHead`.
- Targets: `alpha_1d`, `alpha_5d`, `alpha_21d`, `alpha_63d`.
- Features: 23 channels (price/volume/vol/factor/macro + sector-neutral returns).
- Context window: 250 days, patch length: 5.

Training scripts:

- `alpha_model/training/train_patchtst.py` (v1, walk-forward folds).
- `alpha_model/training/train_patchtst_v2.py` (improved: stride sampling, warmup, horizon-weighted loss, channel dropout, target standardization).

Checked-in artifact:

- `models/patch_tst_fold9.pt` (saved on `2026-03-17`, fold `9`, v1-style config metadata).

Note: PatchTST training/evaluation pipeline is in-repo, but current API inference routing is `chronos` or `fincast_lora`.

### FinCast LoRA fine-tuning run (saved artifacts)

Artifacts in `models/fincast_runtime_local/` show a completed custom LoRA run:

- Run time (UTC): `2026-03-24T05:33:30.394853+00:00`
- Trainer: `custom_peft_loop`
- Best epoch: `3`
- Selection metric: `rank_ic`
- Best validation loss: `0.0003303606`
- Split coverage: `2017-07-09` to `2026-03-17` (`2703` rows)
- Splits:
  - Train: `2017-07-09` to `2023-12-22`
  - Validation: `2024-01-01` to `2024-12-25` (`1760` examples)
  - Holdout: `2025-01-01` to `2025-12-25` (`1750` examples)
  - Forward: `2025-12-26` to `2026-03-17`

Holdout pooled metrics (frozen -> LoRA):

- Directional accuracy: `0.5263 -> 0.5806` (`+0.0543`)
- Rank IC: `-0.0016 -> 0.0692` (`+0.0707`)
- Turnover proxy: `0.8057 -> 0.1080`

## Repository Layout

- `backend/`: FastAPI APIs, inference routing, committee orchestration, memo generation.
- `frontend/`: React dashboard and backtest UI.
- `alpha_model/`: PatchTST training/evaluation and FinCast data/training utilities.
- `models/fincast_runtime_local/`: FinCast LoRA metrics and adapter artifacts.
- `external/FinCast-fts/`: vendored upstream FinCast code.
- `docs/`: guides and generated figures.

## API Surface

- `GET /health`
- `POST /api/v1/analysis`
- `GET /api/v1/analysis/{analysis_id}`
- `GET /api/v1/analysis/{analysis_id}/memo`
- `POST /api/v1/backtest`
- `WS /api/v1/ws/analysis/{analysis_id}`

## Quick Start

1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Frontend

```bash
npm install
npm run dev --prefix frontend
```

3. Open `http://localhost:5173` and run an analysis.

Required env vars are defined in `backend/app/config.py` (Nemotron, Redis, Brave, BrightData, FRED, and optional FinCast artifact paths).

## More Detail

- FinCast guide: `docs/fincast_futures_lora_guide.md`
- FinCast LoRA runner: `alpha_model/training/fincast_lora_colab.py`
- Forecast inference routing: `backend/app/services/prediction/inference.py`
- Committee pipeline: `backend/app/api/endpoints/analysis.py`
