"""Bottleneck #1 — profile the forecaster and show the before/after deltas.

Runs a sequence of lever configurations on the GPU, times each with the project
harness, and dumps a torch.profiler operator table for the baseline and the
fully-optimized config into traces/. Requires a CUDA GPU.

    python -m profiling.profile_forecaster            # default batch 512
    python -m profiling.profile_forecaster --batch 2048

For the system-level timeline, wrap the same run in Nsight Systems:
    nsys profile -o traces/forecaster_opt python -m profiling.profile_forecaster
"""
from __future__ import annotations

import argparse

from cudalpha.benchmark import GpuUtilSampler, time_callable, torch_peak_mem_mb
from cudalpha.config import TRACES_DIR, set_seed
from cudalpha.workloads.base import gpu_available
from cudalpha.workloads.forecaster import ForecasterWorkload

# (label, lever kwargs) — the optimization arc, worst to best.
CONFIGS = [
    ("baseline (host copy, pageable)", dict(use_host_input=True)),
    ("pinned + non_blocking",          dict(use_host_input=True, use_pinned=True)),
    ("pinned + AMP (fp16)",            dict(use_host_input=True, use_pinned=True, use_amp=True)),
    ("pinned + AMP + torch.compile",   dict(use_host_input=True, use_pinned=True, use_amp=True,
                                            use_compile=True)),
    ("CUDA graph (pinned + AMP)",      dict(use_cuda_graphs=True, use_host_input=True,
                                            use_pinned=True, use_amp=True)),
]


def _config(size, **levers):
    wl = ForecasterWorkload()
    for k, v in levers.items():
        setattr(wl, k, v)
    return wl.gpu(size)


def _torch_profile(call, tag: str):
    import torch
    from torch.profiler import ProfilerActivity, profile

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            call.fn()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    out = TRACES_DIR / f"forecaster_{tag}.json"
    prof.export_chrome_trace(str(out))
    print(f"  chrome trace -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args(argv)

    if not gpu_available():
        raise SystemExit("No GPU visible — this profiling recipe needs CUDA.")

    set_seed()
    size = {"batch": args.batch}
    print(f"=== forecaster profiling, batch={args.batch} ===\n")
    rows = []
    for label, levers in CONFIGS:
        try:
            call = _config(size, **levers)
        except NotImplementedError as e:
            print(f"{label:34s}  SKIPPED ({e})")
            continue
        with GpuUtilSampler() as sampler:
            t = time_callable(call.fn, synchronize=call.synchronize)
        mem = torch_peak_mem_mb()
        rows.append((label, t["median_ms"], sampler.mean_util, mem))
        util = f"{sampler.mean_util:.0f}%" if sampler.mean_util is not None else "n/a"
        print(f"{label:34s}  median={t['median_ms']:8.3f} ms  util={util:>5}  peak={mem}")

    if rows:
        base = rows[0][1]
        best = min(rows, key=lambda r: r[1])
        print(f"\nbest: {best[0]}  ->  {base / best[1]:.1f}x faster than baseline")
        print("\n--- torch.profiler: baseline ---")
        _torch_profile(_config(size, **CONFIGS[0][1]), "baseline")
        print("\n--- torch.profiler: best ---")
        _torch_profile(_config(size, **best_levers(best[0])), "optimized")


def best_levers(label: str) -> dict:
    return dict(CONFIGS)[label]


if __name__ == "__main__":
    main()
