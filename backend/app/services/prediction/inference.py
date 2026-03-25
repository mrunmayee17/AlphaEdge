"""Forecast inference providers for Chronos and fine-tuned FinCast LoRA."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from backend.app.services.data.yahoo_finance import FUTURES_ALIASES

if TYPE_CHECKING:
    from backend.app.config import Settings

logger = logging.getLogger(__name__)

# Cached runtimes to avoid reloading on every request
_pipeline_cache: dict[str, Any] = {}
_fincast_runtime_cache: dict[str, "FincastRuntime"] = {}
_fincast_runtime_lock = threading.Lock()

# Forecast horizons (trading days)
HORIZONS = [1, 5, 21, 63]
HORIZON_LABELS = ["1d", "5d", "21d", "63d"]

# Chronos-Bolt outputs 9 quantiles: [0.1, 0.2, ..., 0.9]
Q10_IDX, Q50_IDX, Q90_IDX = 0, 4, 8

FINCAST_SUPPORTED_SYMBOLS = ("ES", "NQ", "RTY", "YM", "ZN", "ZB", "CL", "NG", "GC", "HG")
FINCAST_ASSET_CLASS_MAP = {
    "ES": "equities",
    "NQ": "equities",
    "RTY": "equities",
    "YM": "equities",
    "ZN": "rates",
    "ZB": "rates",
    "CL": "commodities",
    "NG": "commodities",
    "GC": "commodities",
    "HG": "commodities",
}


@dataclass
class FincastRuntime:
    model: Any
    device: torch.device
    context_len: int
    step_horizon: int
    q10_idx: int | None
    q50_idx: int | None
    q90_idx: int | None
    adapter_path: Path
    training_status: dict[str, Any]


def _load_pipeline(model_id: str):
    """Load Chronos pipeline from HuggingFace (cached after first call)."""
    if model_id in _pipeline_cache:
        return _pipeline_cache[model_id]

    from chronos import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    _pipeline_cache[model_id] = pipeline
    logger.info("Loaded Chronos pipeline: %s", model_id)
    return pipeline


def _fetch_returns(ticker: str, context_length: int) -> np.ndarray:
    """Fetch recent daily close-price returns for a ticker."""
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=int(context_length * 1.8))

    for attempt in range(3):
        try:
            df = yf.download(ticker, start=str(start), end=str(end), progress=False)
            break
        except Exception as exc:
            if "Rate" in str(exc) and attempt < 2:
                wait = 2 ** (attempt + 1)
                logger.warning("Yahoo Finance rate limited in inference, retrying in %ss...", wait)
                time.sleep(wait)
            else:
                raise

    if df.empty:
        raise ValueError(f"No price data found for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_series = pd.Series(df["Close"], dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    close = close_series.to_numpy(dtype=np.float64)
    returns = np.diff(np.log(close))
    if len(returns) < context_length:
        raise ValueError(
            f"Insufficient return history for {ticker}: {len(returns)} rows (need {context_length})"
        )
    return returns[-context_length:]


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_fincast_symbol(ticker: str) -> str | None:
    upper = ticker.upper()
    if upper in FINCAST_SUPPORTED_SYMBOLS:
        return upper

    for alias, (yf_ticker, _display_name) in FUTURES_ALIASES.items():
        if alias in FINCAST_SUPPORTED_SYMBOLS and upper == yf_ticker.upper():
            return alias

    return None


def _default_fincast_artifact_dir(settings: "Settings") -> Path:
    return Path(settings.fincast_extract_dir).expanduser().resolve().parent / "artifacts"


def _resolve_artifact_path(configured_path: str, default_name: str, settings: "Settings") -> Path:
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (_default_fincast_artifact_dir(settings) / default_name).resolve()


def _download_artifact(url: str, destination: Path, *, timeout_seconds: int, label: str, hf_token: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    logger.info("Downloading %s from %s to %s", label, url, destination)
    try:
        headers = {"User-Agent": "alpha-edge-fincast/1.0"}
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response, partial_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial_path.replace(destination)
    except Exception as exc:  # noqa: BLE001
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {label} from {url}: {exc}") from exc


def _resolve_fincast_adapter_dir(settings: "Settings") -> Path:
    if settings.fincast_adapter_dir:
        adapter_path = Path(settings.fincast_adapter_dir).expanduser().resolve()
        if not adapter_path.exists():
            raise RuntimeError(f"FINCAST_ADAPTER_DIR does not exist: {adapter_path}")
        return adapter_path

    if not settings.fincast_results_zip_path and not settings.fincast_results_zip_url:
        raise RuntimeError(
            "Fine-tuned FinCast requires FINCAST_ADAPTER_DIR, FINCAST_RESULTS_ZIP_PATH, or FINCAST_RESULTS_ZIP_URL."
        )

    zip_path = _resolve_artifact_path(settings.fincast_results_zip_path, "fincast_results.zip", settings)
    if not zip_path.exists():
        if not settings.fincast_results_zip_url:
            raise RuntimeError(f"FINCAST_RESULTS_ZIP_PATH does not exist: {zip_path}")
        _download_artifact(
            settings.fincast_results_zip_url,
            zip_path,
            timeout_seconds=settings.fincast_download_timeout_seconds,
            label="FINCAST_RESULTS_ZIP",
            hf_token=settings.hf_token,
        )

    extract_root = Path(settings.fincast_extract_dir).expanduser().resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    adapter_path = extract_root / settings.fincast_adapter_subdir

    if not adapter_path.exists():
        with zipfile.ZipFile(zip_path) as bundle:
            bundle.extractall(extract_root)

    if not adapter_path.exists():
        raise RuntimeError(
            f"Adapter subdirectory '{settings.fincast_adapter_subdir}' was not found inside {zip_path}"
        )

    return adapter_path


def _nearest_quantile_index(quantiles: list[float], target: float) -> int | None:
    if not quantiles:
        return None
    return min(range(len(quantiles)), key=lambda idx: abs(float(quantiles[idx]) - target))


def _load_fincast_runtime(settings: "Settings") -> FincastRuntime:
    if not settings.fincast_checkpoint_path and not settings.fincast_checkpoint_url:
        raise RuntimeError(
            "Fine-tuned FinCast requires FINCAST_CHECKPOINT_PATH or FINCAST_CHECKPOINT_URL."
        )

    checkpoint_path = _resolve_artifact_path(settings.fincast_checkpoint_path, "v1.pth", settings)
    if not checkpoint_path.exists():
        if not settings.fincast_checkpoint_url:
            raise RuntimeError(f"FINCAST_CHECKPOINT_PATH does not exist: {checkpoint_path}")
        _download_artifact(
            settings.fincast_checkpoint_url,
            checkpoint_path,
            timeout_seconds=settings.fincast_download_timeout_seconds,
            label="FINCAST_CHECKPOINT",
            hf_token=settings.hf_token,
        )

    adapter_path = _resolve_fincast_adapter_dir(settings)
    cache_key = f"{checkpoint_path}:{adapter_path}"
    if cache_key in _fincast_runtime_cache:
        return _fincast_runtime_cache[cache_key]

    with _fincast_runtime_lock:
        if cache_key in _fincast_runtime_cache:
            return _fincast_runtime_cache[cache_key]

        try:
            from ffm import FFM, FFmHparams
        except ImportError as exc:
            raise RuntimeError(
                "FinCast runtime dependency missing: install the package that provides `ffm.FFM`."
            ) from exc

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("FinCast LoRA requires the `peft` package in the backend environment.") from exc

        backend_name = "gpu" if settings.fincast_device == "cuda" else "cpu"
        hparams = FFmHparams(
            backend=backend_name,
            per_core_batch_size=1,
            context_len=settings.fincast_context_length,
            horizon_len=settings.fincast_step_horizon,
            num_experts=settings.fincast_num_experts,
            gating_top_n=settings.fincast_gating_top_n,
            load_from_compile=True,
            point_forecast_mode="mean",
        )
        api = FFM(hparams=hparams, checkpoint=str(checkpoint_path), loading_mode=0)
        base_model = api._model
        model = PeftModel.from_pretrained(base_model, adapter_path)

        device = torch.device(settings.fincast_device)
        model.to(device)
        model.eval()
        api._model = model

        quantiles = [float(q) for q in getattr(hparams, "quantiles", [])]
        runtime = FincastRuntime(
            model=model,
            device=device,
            context_len=settings.fincast_context_length,
            step_horizon=settings.fincast_step_horizon,
            q10_idx=_nearest_quantile_index(quantiles, 0.1),
            q50_idx=_nearest_quantile_index(quantiles, 0.5),
            q90_idx=_nearest_quantile_index(quantiles, 0.9),
            adapter_path=adapter_path,
            training_status=_load_json_if_exists(adapter_path.parent / "training_status.json"),
        )
        _fincast_runtime_cache[cache_key] = runtime
        logger.info("Loaded FinCast LoRA runtime from %s", adapter_path)
        return runtime


def _pick_quantile_column(quantile_chunk: np.ndarray, index: int | None, fallback: np.ndarray) -> np.ndarray:
    if index is None or quantile_chunk.ndim != 2 or quantile_chunk.shape[1] <= index:
        return fallback
    return quantile_chunk[:, index]


def _rollout_fincast_forecast(runtime: FincastRuntime, returns: np.ndarray, max_horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively roll a 5-step FinCast model forward to longer horizons."""
    current_context = torch.tensor(returns, dtype=torch.float32, device=runtime.device).unsqueeze(0)
    q10_values: list[float] = []
    q50_values: list[float] = []
    q90_values: list[float] = []

    with torch.no_grad():
        while len(q50_values) < max_horizon:
            padding = torch.zeros_like(current_context, dtype=torch.float32)
            freq = torch.zeros((1, 1), dtype=torch.long, device=runtime.device)
            outputs, _aux_loss = runtime.model(current_context, padding, freq)

            point_chunk = outputs[:, -1, : runtime.step_horizon, 0].detach().cpu().numpy()[0]
            quantile_chunk = outputs[:, -1, : runtime.step_horizon, 1:].detach().cpu().numpy()[0]

            q10_chunk = _pick_quantile_column(quantile_chunk, runtime.q10_idx, point_chunk)
            q50_chunk = _pick_quantile_column(quantile_chunk, runtime.q50_idx, point_chunk)
            q90_chunk = _pick_quantile_column(quantile_chunk, runtime.q90_idx, point_chunk)

            remaining = max_horizon - len(q50_values)
            take = min(runtime.step_horizon, remaining)
            q10_values.extend(float(v) for v in q10_chunk[:take])
            q50_values.extend(float(v) for v in q50_chunk[:take])
            q90_values.extend(float(v) for v in q90_chunk[:take])

            predicted_tail = torch.tensor(point_chunk[:take], dtype=torch.float32, device=runtime.device).unsqueeze(0)
            current_context = torch.cat([current_context[:, take:], predicted_tail], dim=1)

    return (
        np.asarray(q10_values, dtype=np.float64),
        np.asarray(q50_values, dtype=np.float64),
        np.asarray(q90_values, dtype=np.float64),
    )


async def run_selected_inference(
    *,
    ticker: str,
    sector: str,
    sector_etf: str,
    forecast_model: str,
    settings: "Settings",
) -> dict[str, Any]:
    if forecast_model == "chronos":
        return await run_chronos_inference(ticker, sector, sector_etf, settings.chronos_model_id)
    if forecast_model == "fincast_lora":
        return await run_fincast_lora_inference(ticker, sector, sector_etf, settings)
    raise ValueError(f"Unsupported forecast model: {forecast_model}")


async def run_inference(ticker: str, sector: str, sector_etf: str, model_id: str) -> dict[str, Any]:
    """Backward-compatible wrapper for Chronos tests and callers."""
    return await run_chronos_inference(ticker, sector, sector_etf, model_id)


async def run_chronos_inference(ticker: str, sector: str, sector_etf: str, model_id: str) -> dict[str, Any]:
    """Run Chronos-2 inference for a single ticker."""
    start_time = time.time()

    pipeline = await asyncio.to_thread(_load_pipeline, model_id)
    returns = await asyncio.to_thread(_fetch_returns, ticker, 512)
    context = torch.tensor(returns, dtype=torch.float32)

    max_horizon = max(HORIZONS)
    quantiles = await asyncio.to_thread(
        lambda: pipeline.predict(context, prediction_length=max_horizon).squeeze(0).numpy()
    )

    result: dict[str, Any] = {
        "ticker": ticker,
        "prediction_date": str(date.today()),
        "sector": sector,
        "sector_etf": sector_etf,
    }

    for horizon, label in zip(HORIZONS, HORIZON_LABELS):
        result[f"alpha_{label}"] = float(quantiles[Q50_IDX, :horizon].sum())
        result[f"q10_{label}"] = float(quantiles[Q10_IDX, :horizon].sum())
        result[f"q90_{label}"] = float(quantiles[Q90_IDX, :horizon].sum())

    result.update({
        "patch_attention": [],
        "top_features": [],
        "model_version": f"chronos-2:{model_id.split('/')[-1]}",
        "training_fold": "pretrained",
        "inference_latency_ms": round((time.time() - start_time) * 1000, 1),
    })
    return result


async def run_fincast_lora_inference(
    ticker: str,
    sector: str,
    sector_etf: str,
    settings: "Settings",
) -> dict[str, Any]:
    """Run fine-tuned FinCast LoRA inference for supported futures symbols."""
    symbol = _resolve_fincast_symbol(ticker)
    if symbol is None:
        supported = ", ".join(FINCAST_SUPPORTED_SYMBOLS)
        raise ValueError(
            f"FinCast LoRA currently supports only the trained futures universe: {supported}. "
            f"Received {ticker}."
        )

    start_time = time.time()
    runtime = await asyncio.to_thread(_load_fincast_runtime, settings)
    returns = await asyncio.to_thread(_fetch_returns, ticker, runtime.context_len)
    q10_series, q50_series, q90_series = await asyncio.to_thread(
        _rollout_fincast_forecast,
        runtime,
        returns,
        max(HORIZONS),
    )

    inferred_sector = sector if sector and sector != "Unknown" else FINCAST_ASSET_CLASS_MAP.get(symbol, "futures")
    training_status = runtime.training_status or {}
    best_epoch = training_status.get("best_epoch")
    training_fold = f"best_epoch:{best_epoch}" if best_epoch is not None else runtime.adapter_path.name

    result: dict[str, Any] = {
        "ticker": ticker,
        "prediction_date": str(date.today()),
        "sector": inferred_sector,
        "sector_etf": sector_etf,
    }
    for horizon, label in zip(HORIZONS, HORIZON_LABELS):
        result[f"alpha_{label}"] = float(q50_series[:horizon].sum())
        result[f"q10_{label}"] = float(q10_series[:horizon].sum())
        result[f"q90_{label}"] = float(q90_series[:horizon].sum())

    result.update({
        "patch_attention": [],
        "top_features": [],
        "model_version": f"fincast-lora:{runtime.adapter_path.name}",
        "training_fold": training_fold,
        "inference_latency_ms": round((time.time() - start_time) * 1000, 1),
    })
    return result
