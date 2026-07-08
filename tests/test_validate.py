"""Validation logic: fp32 tightness, the fp16 tolerance path, feasibility."""
import numpy as np

from cudalpha.config import FP16_RTOL, FP32_RTOL
from cudalpha.validate import check_portfolio, compare_arrays


def test_fp32_identical_passes():
    a = np.arange(10, dtype=np.float32)
    rec = compare_arrays(a, a.copy())
    assert rec["passed"] and rec["max_abs_err"] == 0.0


def test_fp32_small_deviation_fails_tight_tolerance():
    a = np.ones(10, dtype=np.float32)
    b = a + 0.05  # far outside FP32_RTOL/ATOL
    assert not compare_arrays(a, b)["passed"]


def test_fp16_deviation_within_loose_tolerance_passes():
    a = np.ones(100, dtype=np.float32)
    b = a * (1.0 + 0.5 * FP32_RTOL + 0.4 * FP16_RTOL)  # fails fp32, passes fp16
    assert not compare_arrays(a, b)["passed"]
    rec16 = compare_arrays(a, b, fp16=True)
    assert rec16["passed"] and rec16["fp16"]


def test_explicit_tolerance_override():
    a = np.zeros(4, dtype=np.float32)
    b = a + 5e-5
    assert not compare_arrays(a, b)["passed"]                 # default atol=1e-6 fails
    rec = compare_arrays(a, b, atol=1e-4)                      # scale-appropriate override
    assert rec["passed"] and rec["atol"] == 1e-4


def test_shape_mismatch_is_reported_not_crashed():
    rec = compare_arrays(np.zeros(3), np.zeros(4))
    assert not rec["passed"] and "shape" in rec["reason"]


def test_check_portfolio_feasible_and_infeasible():
    good = check_portfolio(np.array([0.5, 0.5]))
    assert good["passed"] and abs(good["sum"] - 1.0) < 1e-9
    bad_sum = check_portfolio(np.array([0.5, 0.7]))
    assert not bad_sum["passed"]
    short = check_portfolio(np.array([1.3, -0.3]))
    assert not short["passed"], "long-only must reject negative weights"
