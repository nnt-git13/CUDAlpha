"""CPU-path smoke tests for the workloads (no GPU required).

These exercise the reference (CPU) side of each workload and the validator wiring
that the runner depends on. GPU paths are covered by on-device validation.
"""
import numpy as np
import pytest

from cudalpha.workloads.backtester import LONG_W, BacktesterWorkload, _sma_cumsum
from cudalpha.workloads.kernels import rolling_mean_reference


def test_backtester_cpu_runs_and_returns_finite_rolling_mean():
    wl = BacktesterWorkload()
    call = wl.cpu({"series_len": 5000})
    out = call.fn()                      # now the long-window rolling-mean array
    assert out.shape == (5000,)
    assert np.isfinite(out).all(), "float32 overflow / nan regression guard"
    assert call.backend == "numpy"


def test_backtester_long_series_does_not_overflow():
    # The horizon cap must keep a 1e6-step series inside float32 range.
    wl = BacktesterWorkload()
    price = wl._prices({"series_len": 1_000_000})
    assert np.isfinite(price).all(), "prices overflowed float32 (horizon not capped)"


def test_sma_cumsum_matches_kernel_reference():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000).astype(np.float32)
    got = _sma_cumsum(x, LONG_W, np)
    ref = rolling_mean_reference(x, LONG_W)
    assert np.allclose(got, ref, atol=1e-3)


def test_backtester_validate_matches_identical_outputs():
    wl = BacktesterWorkload()
    assert wl.validate(1.2345, 1.2345)["passed"]
    assert not wl.validate(1.0, 2.0)["passed"]


def test_forecaster_validate_tolerates_recurrent_fp32_near_zero():
    # Mirrors real cuDNN-vs-CPU LSTM behaviour: outputs ~1e-3, agreeing to ~1e-6
    # absolute, with one near-zero output. Must pass; a 50% divergence must fail.
    # (No torch needed — this exercises only the tolerance policy.)
    from cudalpha.workloads.forecaster import ForecasterWorkload

    wl = ForecasterWorkload()
    cpu = np.array([0.00778832, -0.00060792, 0.00483714, 0.0], dtype=np.float32)
    gpu = cpu + np.array([2.6e-6, 1.2e-6, -1.4e-6, 3e-7], dtype=np.float32)
    assert wl.validate(cpu, gpu)["passed"]
    bad = cpu.copy()
    bad[0] *= 1.5
    assert not wl.validate(cpu, bad)["passed"]


torch = pytest.importorskip("torch")


def test_forecaster_cpu_inference_is_deterministic():
    from cudalpha.workloads.forecaster import ForecasterWorkload

    wl = ForecasterWorkload()
    call = wl.cpu({"batch": 16})
    a = call.fn()
    b = wl.cpu({"batch": 16}).fn()
    assert a.shape == (16, 1)
    assert torch.allclose(a, b), "seeded model + input must be reproducible"


def test_forecaster_validate_self_consistent():
    from cudalpha.workloads.forecaster import ForecasterWorkload

    wl = ForecasterWorkload()
    out = wl.cpu({"batch": 8}).fn()
    assert wl.validate(out, out)["passed"]
