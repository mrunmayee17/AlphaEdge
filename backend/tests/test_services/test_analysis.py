"""Test risk/macro analysis modules."""

import numpy as np
import pandas as pd

from backend.app.services.analysis.macro import fit_regime_model, predict_regime
from backend.app.services.analysis.risk import compute_var_cvar, position_size


def test_var_cvar():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
    result = compute_var_cvar(returns)
    assert result["var_95"] < 0
    assert result["cvar_95"] < result["var_95"]
    assert result["var_99"] < result["var_95"]


def test_var_insufficient_data():
    returns = pd.Series([0.01, -0.01, 0.005])
    result = compute_var_cvar(returns, lookback=500)
    assert "error" in result


def test_position_size():
    size = position_size(alpha_q50=0.03, realized_vol_21d=0.25, conviction=0.8)
    assert 0 < size <= 0.10


def test_position_size_cap():
    size = position_size(alpha_q50=0.50, realized_vol_21d=0.05, conviction=1.0)
    assert size == 0.10


def test_position_size_zero():
    size = position_size(alpha_q50=0.0, realized_vol_21d=0.25, conviction=0.8)
    assert size == 0.0


def test_hmm_regime():
    np.random.seed(42)
    segments = []
    for _ in range(20):
        regime = np.random.choice(3)
        length = np.random.randint(30, 80)
        if regime == 0:
            segments.append((np.random.normal(14, 2, length), np.random.normal(2, 0.3, length), np.random.normal(1.5, 0.2, length)))
        elif regime == 1:
            segments.append((np.random.normal(22, 4, length), np.random.normal(4, 0.8, length), np.random.normal(0.3, 0.4, length)))
        else:
            segments.append((np.random.normal(35, 6, length), np.random.normal(7, 1.5, length), np.random.normal(-0.5, 0.3, length)))

    vix = np.concatenate([s[0] for s in segments])
    spread = np.concatenate([s[1] for s in segments])
    yc = np.concatenate([s[2] for s in segments])

    hmm, state_order = fit_regime_model(vix, spread, yc)

    r1 = predict_regime(hmm, state_order, vix=13, hyg_lqd_spread=2, yield_2s10s=1.5)
    assert r1["regime"] == "risk_on"

    r3 = predict_regime(hmm, state_order, vix=38, hyg_lqd_spread=7, yield_2s10s=-0.5)
    assert r3["regime"] == "risk_off"

    # Probabilities should sum to ~1
    total = sum(r1["probabilities"].values())
    assert abs(total - 1.0) < 0.001
