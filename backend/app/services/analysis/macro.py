"""HMM regime detection — 3-state Gaussian HMM on VIX/credit/yield curve."""

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "risk_on", 1: "transition", 2: "risk_off"}


def fit_regime_model(
    vix: np.ndarray,
    hyg_lqd_spread: np.ndarray,
    yield_2s10s: np.ndarray,
    random_state: int = 42,
) -> GaussianHMM:
    """Fit 3-state GaussianHMM on [VIX, credit spread, yield curve].

    States are labeled by sorting VIX emission means:
    lowest VIX → risk_on, middle → transition, highest → risk_off.
    """
    features = np.column_stack([vix, hyg_lqd_spread, yield_2s10s])

    # Remove NaNs
    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]

    if len(features) < 100:
        raise ValueError(f"Insufficient data for HMM: {len(features)} rows (need ≥100)")

    # K-means initialization
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    kmeans.fit(features)

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=random_state,
        init_params="stmc",
    )
    hmm.fit(features)

    # Label states by VIX mean (column 0)
    vix_means = hmm.means_[:, 0]
    state_order = np.argsort(vix_means)

    return hmm, state_order


def predict_regime(
    hmm: GaussianHMM,
    state_order: np.ndarray,
    vix: float,
    hyg_lqd_spread: float,
    yield_2s10s: float,
    recent_history: np.ndarray | None = None,
) -> dict:
    """Predict current regime from latest macro values.

    For single-point prediction, uses emission probability (score_samples)
    instead of Viterbi decoding, which is more reliable without temporal context.
    If recent_history is provided (N, 3), appends current observation and uses
    full Viterbi decoding on the sequence.
    """
    current = np.array([[vix, hyg_lqd_spread, yield_2s10s]])

    if recent_history is not None and len(recent_history) >= 5:
        # Use full sequence for Viterbi decoding
        sequence = np.vstack([recent_history, current])
        raw_state = hmm.predict(sequence)[-1]  # last state
        probs = hmm.predict_proba(sequence)[-1]
    else:
        # Single-point: use emission log-likelihood per component
        # Score each component's emission probability
        log_probs = np.array([
            hmm._compute_log_likelihood(current)[0, i]
            if hasattr(hmm, '_compute_log_likelihood')
            else -0.5 * np.sum((current[0] - hmm.means_[i]) ** 2 / np.diag(hmm.covars_[i]))
            for i in range(3)
        ])
        raw_state = int(np.argmax(log_probs))
        # Softmax for probabilities
        log_probs -= log_probs.max()
        exp_probs = np.exp(log_probs)
        probs = exp_probs / exp_probs.sum()

    # Map raw state to labeled state via VIX-sorted order
    label_idx = int(np.where(state_order == raw_state)[0][0])
    regime = REGIME_LABELS[label_idx]

    return {
        "regime": regime,
        "probabilities": {
            REGIME_LABELS[i]: float(probs[state_order[i]])
            for i in range(3)
        },
        "inputs": {
            "vix": vix,
            "hyg_lqd_spread": hyg_lqd_spread,
            "yield_2s10s": yield_2s10s,
        },
    }
