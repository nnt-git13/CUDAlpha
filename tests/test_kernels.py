"""The rolling-mean contract both CUDA kernels implement (verified on CPU).

The GPU kernels are validated on-device at runtime; here we pin down the NumPy
reference they must match so the math is provable without a GPU.
"""
import numpy as np

from cudalpha.workloads.kernels import rolling_mean_reference


def _brute_force(x, window):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.size, dtype=np.float32)
    for i in range(window - 1, x.size):
        out[i] = x[i - window + 1 : i + 1].mean()
    return out


def test_reference_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    for w in (1, 2, 25, 50, 999):
        ref = rolling_mean_reference(x, w)
        bf = _brute_force(x, w)
        assert np.allclose(ref, bf, atol=1e-4), f"window={w}"


def test_leading_window_is_zero():
    x = np.arange(10, dtype=np.float32)
    out = rolling_mean_reference(x, 4)
    assert (out[:3] == 0).all()
    assert np.isclose(out[3], (0 + 1 + 2 + 3) / 4)


def test_window_larger_than_series_is_all_zero():
    x = np.ones(5, dtype=np.float32)
    assert (rolling_mean_reference(x, 10) == 0).all()


def test_prefix_sum_identity_holds():
    # out[i] == (P[i] - P[i-w]) / w  — the exact identity the fast kernel uses.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500).astype(np.float32)
    w = 30
    P = np.cumsum(x.astype(np.float64))
    ref = rolling_mean_reference(x, w)
    for i in range(w, x.size):
        assert np.isclose(ref[i], (P[i] - P[i - w]) / w, atol=1e-4)
