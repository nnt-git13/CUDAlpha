"""GBM data invariants — shapes, determinism, and a well-posed covariance."""
import numpy as np

from cudalpha.data import gbm_prices, log_returns, mean_cov


def test_gbm_shape_and_positivity():
    p = gbm_prices(n_assets=5, n_steps=200)
    assert p.shape == (200, 5)
    assert (p > 0).all(), "GBM prices are exp(...) and must stay positive"


def test_gbm_is_seeded_deterministic():
    a = gbm_prices(n_assets=4, n_steps=100, seed=7)
    b = gbm_prices(n_assets=4, n_steps=100, seed=7)
    c = gbm_prices(n_assets=4, n_steps=100, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_log_returns_shape():
    p = gbm_prices(n_assets=3, n_steps=50)
    r = log_returns(p)
    assert r.shape == (49, 3)


def test_mean_cov_is_symmetric_psd():
    p = gbm_prices(n_assets=6, n_steps=300)
    mu, cov = mean_cov(log_returns(p))
    assert mu.shape == (6,)
    assert cov.shape == (6, 6)
    assert np.allclose(cov, cov.T), "covariance must be symmetric"
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > 0, "the +eps ridge must keep cov strictly PSD"
