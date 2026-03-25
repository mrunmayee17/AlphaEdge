# Vercel Deployment Guide

This project should be deployed as:
- Frontend on Vercel
- Backend on a long-running host (Render/Railway/Fly/VM)

The backend should not run on Vercel Functions because the FinCast checkpoint is too large for serverless function packaging/runtime constraints.

## 1. Deploy Frontend to Vercel

Create a Vercel project with root directory set to `frontend/`.

Set frontend environment variables in Vercel:
- `VITE_API_URL=https://<your-backend-domain>/api/v1`
- `VITE_BASE_URL=/`

Build settings:
- Build command: `npm run build`
- Output directory: `dist`

`frontend/vercel.json` already contains SPA rewrite to `index.html`.

## 2. Deploy Backend on a Host

Run the existing FastAPI backend on a host that supports:
- Persistent disk for model artifacts
- Enough RAM/CPU for inference
- Outbound network access to APIs used by the app

Set backend environment variables:
- Existing required keys from `.env` (`NVIDIA_*`, `BRAVE_API_KEY`, `FRED_API_KEY`, Redis, etc.)
- `CORS_ORIGINS=https://<your-vercel-app-domain>,http://localhost:3000,http://localhost:5173`
- FinCast vars:
  - `FINCAST_CHECKPOINT_PATH=<path to v1.pth>`
  - `FINCAST_RESULTS_ZIP_PATH=<path to fincast_results.zip>`
  - OR URL-based bootstrap (no shell upload needed):
    - `FINCAST_CHECKPOINT_URL=<https://.../v1.pth>`
    - `FINCAST_RESULTS_ZIP_URL=<https://.../fincast_results.zip>`
    - optional `FINCAST_DOWNLOAD_TIMEOUT_SECONDS=1800`
    - if URL is private on Hugging Face: `HF_TOKEN=<token with read access>`
  - optional: `FINCAST_ADAPTER_SUBDIR=lora_adapter_best`

## 3. Smoke Test

1. Open the Vercel frontend URL.
2. Run one analysis with model `Chronos-2`.
3. Run one analysis with model `Fincast LoRA`.
4. Verify backend response contains:
   - `forecast_model: "chronos"` for Chronos runs
   - `forecast_model: "fincast_lora"` for FinCast runs
