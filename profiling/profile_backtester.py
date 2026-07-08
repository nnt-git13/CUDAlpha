"""Bottleneck #2 — profile the backtester rolling-mean kernels (naive vs fast).

Times the memory-bound naive kernel against the O(1)-per-output prefix-sum kernel
with the project harness and prints the speedup. The fine-grained memory-throughput
/ occupancy story comes from Nsight Compute; this script prints the ncu commands to
run for the deep dive. Requires a CUDA GPU + CuPy.

    python -m profiling.profile_backtester --series-len 1000000
"""
from __future__ import annotations

import argparse

from cudalpha.benchmark import time_callable
from cudalpha.config import set_seed
from cudalpha.workloads.backtester import BacktesterWorkload
from cudalpha.workloads.base import gpu_available


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--series-len", type=int, default=1_000_000)
    args = p.parse_args(argv)

    if not gpu_available():
        raise SystemExit("No GPU visible — this profiling recipe needs CUDA + CuPy.")

    set_seed()
    size = {"series_len": args.series_len}
    wl = BacktesterWorkload()

    cpu_out = wl.cpu(size).fn()
    print(f"=== backtester kernel profiling, series_len={args.series_len} ===\n")

    results = {}
    for label, call in [("naive", wl.gpu_naive_kernel(size)),
                        ("fast (prefix-sum)", wl.gpu_fast_kernel(size))]:
        gpu_out = call.fn()
        t = time_callable(call.fn, synchronize=call.synchronize)
        val = wl.validate(cpu_out, gpu_out)
        results[label] = t["median_ms"]
        print(f"{label:20s}  median={t['median_ms']:8.3f} ms  valid={val['passed']}")

    if "naive" in results and "fast (prefix-sum)" in results:
        sp = results["naive"] / results["fast (prefix-sum)"]
        print(f"\nprefix-sum kernel is {sp:.1f}x faster than the naive re-sum kernel")

    print("\nFor the per-kernel deep dive (memory throughput / achieved occupancy):")
    print("  ncu -k rolling_mean_naive -o traces/rolling_mean_naive \\")
    print(f"      python -m bench.run_all --workload backtester --sizes {args.series_len}")
    print("  ncu -k rolling_mean_fast  -o traces/rolling_mean_fast  \\")
    print(f"      python -m bench.run_all --workload backtester --sizes {args.series_len}")


if __name__ == "__main__":
    main()
