"""Rolling-mean kernel study — the H2 crossover (bottleneck #2).

Sweeps the moving-average window at a fixed series length and times the naive
re-sum kernel against the O(1)-per-output prefix-sum kernel. The naive kernel
does O(window) cached global reads per output, so it is cheap for small windows;
the prefix-sum kernel pays a fixed float64 scan but is O(1) per output, so it
wins once the window is large enough. This script finds the crossover and writes
one artifact per (window, backend) so the dashboard can plot it.

    python -m bench.kernel_sweep                       # series_len=1e6, default windows
    python -m bench.kernel_sweep --series-len 500000 --windows 50 500 5000

Requires a CUDA GPU + CuPy.
"""
from __future__ import annotations

import argparse

import numpy as np

from cudalpha.benchmark import time_callable
from cudalpha.config import KERNEL_WINDOWS, RESULTS_DIR, set_seed
from cudalpha.metrics import BenchmarkResult
from cudalpha.workloads.backtester import BacktesterWorkload
from cudalpha.workloads.base import gpu_available
from cudalpha.workloads.kernels import (
    rolling_mean_fast,
    rolling_mean_naive,
    rolling_mean_reference,
)

KERNELS = [("cupy-rawkernel-naive", rolling_mean_naive),
           ("cupy-rawkernel-fast", rolling_mean_fast)]


def main(argv=None):
    p = argparse.ArgumentParser(description="rolling-mean kernel window sweep (H2)")
    p.add_argument("--series-len", type=int, default=1_000_000)
    p.add_argument("--windows", type=int, nargs="*", default=KERNEL_WINDOWS)
    p.add_argument("--job-id", default="local")
    args = p.parse_args(argv)

    if not gpu_available():
        raise SystemExit("kernel sweep needs a CUDA GPU + CuPy.")

    try:
        import cupy as cp
    except ImportError:
        raise SystemExit("kernel sweep needs CuPy: pip install cupy-cuda12x")

    set_seed()
    price_host = BacktesterWorkload()._prices({"series_len": args.series_len})
    price = cp.asarray(price_host)

    print(f"=== rolling-mean kernel window sweep, series_len={args.series_len} ===")
    print(f"{'window':>8} {'naive_ms':>10} {'fast_ms':>10} {'speedup':>9} {'winner':>7}  valid")
    crossover = None
    for w in args.windows:
        ref = rolling_mean_reference(price_host, w)
        med = {}
        for label, fn in KERNELS:
            out = fn(price, w)                                   # one eager call for validation
            valid = bool(np.allclose(cp.asnumpy(out), ref, rtol=1e-4, atol=1e-3))
            t = time_callable(lambda fn=fn, w=w: fn(price, w),
                              synchronize=cp.cuda.Stream.null.synchronize)
            med[label] = t["median_ms"]
            BenchmarkResult(
                workload="backtester-kernel", device="gpu", backend=label,
                size={"series_len": args.series_len, "window": w},
                job_id=str(args.job_id), **t, passed_validation=valid,
            ).save(RESULTS_DIR)
        naive_ms, fast_ms = med["cupy-rawkernel-naive"], med["cupy-rawkernel-fast"]
        speedup = naive_ms / fast_ms if fast_ms else float("nan")
        winner = "fast" if fast_ms < naive_ms else "naive"
        if crossover is None and winner == "fast":
            crossover = w
        print(f"{w:>8} {naive_ms:>10.3f} {fast_ms:>10.3f} {speedup:>8.2f}x {winner:>7}  {valid}")

    if crossover is not None:
        print(f"\ncrossover: prefix-sum overtakes the naive re-sum at window >= {crossover}")
    else:
        print("\nno crossover in the swept range — prefix-sum never wins here "
              "(try larger windows).")


if __name__ == "__main__":
    main()
