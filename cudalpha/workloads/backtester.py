"""Backtester workload — moving-average crossover strategy.

Depth role: CUSTOM KERNEL story. A simple SMA-crossover backtest is dominated by
rolling-window means over long series, which is embarrassingly parallel. Three
paths:
  - CPU baseline (numpy cumsum)
  - GPU vectorized baseline (cupy cumsum)         backend: cupy
  - GPU custom kernel (kernels.rolling_mean_*)    backend: cupy-rawkernel

Validation compares strategy returns across paths on the same series.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import MA_WINDOW, SERIES_LENGTHS
from ..data import gbm_prices
from .base import Callable_, Workload

SHORT_W = MA_WINDOW // 2
LONG_W = MA_WINDOW


def _sma_cumsum(x, window: int, xp):
    """Simple moving average via cumulative sum (vectorized), xp = numpy|cupy.

    The cumulative sum MUST accumulate in float64: over a long series the prefix
    sum reaches ~1e9, and the windowed difference `csum[i] - csum[i-window]`
    subtracts two nearly-equal large numbers. In float32 that cancellation
    destroys all precision (and made the vectorized GPU path disagree with the
    CPU reference); float64 keeps it exact. Returns a float64 array.
    """
    csum = xp.cumsum(x, dtype=xp.float64)
    out = xp.zeros(x.shape[0], dtype=xp.float64)
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def _crossover_return(price, short, long_):
    """Strategy: long when short SMA > long SMA. Return total strategy log-return."""
    xp = np.get_array_module(price) if hasattr(np, "get_array_module") else np
    signal = (short > long_).astype(price.dtype)  # 1 when in the market
    rets = xp.diff(xp.log(price))
    # yesterday's signal applies to today's return
    return float((signal[:-1] * rets).sum())


class BacktesterWorkload(Workload):
    name = "backtester"

    def sizes(self) -> list[dict[str, Any]]:
        return [{"series_len": n} for n in SERIES_LENGTHS]

    # Fix the simulated horizon (~30 years) regardless of series length, so a
    # longer series is a FINER sampling of the same horizon rather than a longer
    # one. Without this, GBM over ~1e6 daily steps drifts to exp(~120), which
    # overflows float32 on cast (-> inf -> nan) and corrupts every downstream sum.
    HORIZON_YEARS = 30.0

    def _prices(self, size: dict[str, Any]) -> np.ndarray:
        n = size["series_len"]
        dt = self.HORIZON_YEARS / n
        return gbm_prices(n_assets=1, n_steps=n, dt=dt).ravel().astype(np.float32)

    # --- CPU -----------------------------------------------------------------
    def cpu(self, size: dict[str, Any]) -> Callable_:
        price = self._prices(size)

        def run():
            short = _sma_cumsum(price, SHORT_W, np)
            long_ = _sma_cumsum(price, LONG_W, np)
            signal = (short > long_).astype(np.float32)
            rets = np.diff(np.log(price))
            _ = float((signal[:-1] * rets).sum())   # strategy return (derived; timed for realism)
            return long_                             # the long SMA — what validation compares

        return Callable_(fn=run, backend="numpy", throughput_items=price.size)

    # --- GPU vectorized baseline --------------------------------------------
    def gpu(self, size: dict[str, Any]) -> Callable_:
        import cupy as cp

        price = cp.asarray(self._prices(size))

        def run():
            short = _sma_cumsum(price, SHORT_W, cp)
            long_ = _sma_cumsum(price, LONG_W, cp)
            signal = (short > long_).astype(cp.float32)
            rets = cp.diff(cp.log(price))
            _ = float((signal[:-1] * rets).sum())
            return long_

        return Callable_(fn=run, backend="cupy",
                         synchronize=cp.cuda.Stream.null.synchronize,
                         throughput_items=int(price.size))

    # --- GPU custom kernel paths --------------------------------------------
    def _gpu_kernel_path(self, size: dict[str, Any], rolling_mean, backend: str) -> Callable_:
        """Build a backtest callable that computes both SMAs with a custom
        kernel. `rolling_mean` is rolling_mean_naive (the memory-bound "before")
        or rolling_mean_fast (the O(1)-per-output "after")."""
        import cupy as cp

        price = cp.asarray(self._prices(size))

        def run():
            short = rolling_mean(price, SHORT_W)
            long_ = rolling_mean(price, LONG_W)
            signal = (short > long_).astype(cp.float32)
            rets = cp.diff(cp.log(price))
            _ = float((signal[:-1] * rets).sum())
            return long_

        return Callable_(fn=run, backend=backend,
                         synchronize=cp.cuda.Stream.null.synchronize,
                         throughput_items=int(price.size))

    def gpu_naive_kernel(self, size: dict[str, Any]) -> Callable_:
        from .kernels import rolling_mean_naive

        return self._gpu_kernel_path(size, rolling_mean_naive, "cupy-rawkernel-naive")

    def gpu_fast_kernel(self, size: dict[str, Any]) -> Callable_:
        from .kernels import rolling_mean_fast

        return self._gpu_kernel_path(size, rolling_mean_fast, "cupy-rawkernel-fast")

    def gpu_paths(self, size: dict[str, Any]) -> list[Callable_]:
        # Vectorized CuPy baseline, then the naive kernel (bottleneck #2 "before")
        # and the prefix-sum kernel ("after"). Benchmarking all three side by side
        # is what makes the kernel-optimization delta a measured claim.
        return [
            self.gpu(size),
            self.gpu_naive_kernel(size),
            self.gpu_fast_kernel(size),
        ]

    def validate(self, cpu_out: Any, gpu_out: Any, *, fp16: bool = False) -> dict[str, Any]:
        """Compare the long-window rolling-mean arrays produced by each path.

        We validate the rolling mean (the kernel's actual output), not the
        downstream strategy return: the return is a sign-sensitive reduction
        (signal = short > long) whose value flips on sub-ULP differences near SMA
        crossings, so it is ill-conditioned for cross-backend comparison. The
        rolling mean is continuous and directly tests the kernel; compare_arrays
        applies the standard fp32 tolerances."""
        from ..validate import compare_arrays

        return compare_arrays(cpu_out, gpu_out)
