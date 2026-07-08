"""Run benchmarks and write one result artifact per (workload, size, device).

This is what a single Slurm array task calls. Example:
    python -m bench.run_all --workload backtester --sizes 100000 1000000 --job-id $SLURM_ARRAY_TASK_ID
    python -m bench.run_all --workload all
"""
from __future__ import annotations

import argparse
import sys
import traceback

from cudalpha.benchmark import GpuUtilSampler, time_callable, torch_peak_mem_mb, torch_reset_peak_mem
from cudalpha.config import RESULTS_DIR, TRIALS, WARMUP_ITERS, set_seed
from cudalpha.metrics import BenchmarkResult
from cudalpha.workloads.backtester import BacktesterWorkload
from cudalpha.workloads.base import Workload, gpu_available
from cudalpha.workloads.forecaster import ForecasterWorkload
from cudalpha.workloads.optimizer import OptimizerWorkload

WORKLOADS: dict[str, type[Workload]] = {
    "forecaster": ForecasterWorkload,
    "backtester": BacktesterWorkload,
    "optimizer": OptimizerWorkload,
}


def _bench_one(call, *, warmup, trials):
    """Time a prepared Callable_, sampling GPU mem + util. Returns (timing, mem, util, output)."""
    output = call.fn()  # one eager call to get an output for validation
    torch_reset_peak_mem()
    with GpuUtilSampler() as sampler:
        timing = time_callable(call.fn, warmup=warmup, trials=trials, synchronize=call.synchronize)
    mem = torch_peak_mem_mb()
    return timing, mem, sampler.mean_util, output


def _run_size(wl, name, size, have_gpu, job_id, warmup, trials):
    """Benchmark one problem size (CPU baseline + every GPU backend)."""
    if True:  # noqa: SIM108 - kept as a block so the size body reads as one unit
        # --- CPU ---
        cpu_call = wl.cpu(size)
        cpu_t, _, _, cpu_out = _bench_one(cpu_call, warmup=warmup, trials=trials)
        BenchmarkResult(
            workload=name, device="cpu", backend=cpu_call.backend, size=size,
            job_id=str(job_id),
            **cpu_t,
            throughput=(cpu_call.throughput_items / (cpu_t["median_ms"] / 1000.0)
                        if cpu_call.throughput_items else None),
        ).save(RESULTS_DIR)
        print(f"[{name}] cpu  {size}  median={cpu_t['median_ms']:.3f}ms")

        # --- GPU (one artifact per backend in gpu_paths) ---
        if not have_gpu:
            print(f"[{name}] gpu  {size}  SKIPPED (no GPU visible)")
            return
        try:
            gpu_calls = wl.gpu_paths(size)
        except NotImplementedError as e:
            gpu_calls = []
            print(f"[{name}] gpu  {size}  NOT IMPLEMENTED ({e})")
        if not gpu_calls:
            return
        for gpu_call in gpu_calls:
            try:
                gpu_t, gpu_mem, gpu_util, gpu_out = _bench_one(gpu_call, warmup=warmup, trials=trials)
            except NotImplementedError as e:
                print(f"[{name}] gpu  {size}  {gpu_call.backend}  NOT IMPLEMENTED ({e})")
                continue
            val = wl.validate(cpu_out, gpu_out, fp16=gpu_call.fp16)
            speedup = cpu_t["median_ms"] / gpu_t["median_ms"] if gpu_t["median_ms"] else None
            BenchmarkResult(
                workload=name, device="gpu", backend=gpu_call.backend, size=size,
                job_id=str(job_id),
                **gpu_t,
                throughput=(gpu_call.throughput_items / (gpu_t["median_ms"] / 1000.0)
                            if gpu_call.throughput_items else None),
                peak_mem_mb=gpu_mem, gpu_util_pct=gpu_util,
                passed_validation=val.get("passed"), validation_detail=val,
                speedup_vs_cpu=speedup,
            ).save(RESULTS_DIR)
            sp = f"{speedup:.1f}x" if speedup else "n/a"
            print(f"[{name}] gpu  {size}  {gpu_call.backend:22s}  median={gpu_t['median_ms']:.3f}ms  "
                  f"speedup={sp}  util={gpu_util}  valid={val.get('passed')}")


def run_workload(name: str, sizes_filter=None, job_id="local", *, warmup=WARMUP_ITERS, trials=TRIALS):
    wl = WORKLOADS[name]()
    have_gpu = gpu_available()
    sizes = wl.sizes()
    if sizes_filter:
        key = next(iter(sizes[0]))
        sizes = [s for s in sizes if s[key] in sizes_filter]

    failures = 0
    for size in sizes:
        # Isolate each size: a solver blow-up or OOM at one size must not lose
        # the artifacts from every other size in the sweep.
        try:
            _run_size(wl, name, size, have_gpu, job_id, warmup=warmup, trials=trials)
        except Exception:
            failures += 1
            print(f"[{name}] {size} FAILED (continuing):\n{traceback.format_exc()}",
                  file=sys.stderr)
    return failures


def main(argv=None):
    p = argparse.ArgumentParser(description="CUDAlpha benchmark sweep driver")
    p.add_argument("--workload", default="all", choices=[*WORKLOADS, "all"])
    p.add_argument("--sizes", type=int, nargs="*", default=None,
                   help="filter to these size values (n_assets / batch / series_len)")
    p.add_argument("--job-id", default="local")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    p.add_argument("--trials", type=int, default=TRIALS)
    args = p.parse_args(argv)

    set_seed()
    names = list(WORKLOADS) if args.workload == "all" else [args.workload]
    failures = 0
    for name in names:
        try:
            failures += run_workload(name, args.sizes, args.job_id,
                                     warmup=args.warmup, trials=args.trials)
        except Exception:
            failures += 1
            print(f"[{name}] FAILED:\n{traceback.format_exc()}", file=sys.stderr)
    if failures:
        print(f"\n{failures} size(s) failed; artifacts for the successful sizes were "
              f"still written to results/.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
