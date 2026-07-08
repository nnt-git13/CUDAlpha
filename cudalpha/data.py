"""Synthetic market data via geometric Brownian motion (GBM).

Synthetic keeps the project reproducible and dependency-free (no yfinance / API
keys). Swap in real tickers later if you want — the rest of the pipeline is
agnostic to where the arrays come from.
"""
from __future__ import annotations

import numpy as np

from .config import SEED


def gbm_prices(
    n_assets: int,
    n_steps: int,
    mu: float = 0.05,
    sigma: float = 0.20,
    dt: float = 1.0 / 252.0,
    s0: float = 100.0,
    seed: int = SEED,
) -> np.ndarray:
    """Return a (n_steps, n_assets) float64 array of simulated prices."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_steps, n_assets))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    log_paths = np.cumsum(drift + diffusion, axis=0)
    return s0 * np.exp(log_paths)


def log_returns(prices: np.ndarray) -> np.ndarray:
    """(n_steps-1, n_assets) log returns from a price array."""
    return np.diff(np.log(prices), axis=0)


def mean_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annualized expected returns and covariance for the optimizer."""
    mu = returns.mean(axis=0) * 252.0
    cov = np.cov(returns, rowvar=False) * 252.0
    # Nudge onto the PSD cone so the QP is well-posed for small samples.
    cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(cov.shape[0])
    return mu, cov
