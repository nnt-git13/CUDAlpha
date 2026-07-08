"""Statistically-rigorous timing + GPU resource sampling.

The runner is framework-agnostic: you hand it a zero-arg callable and (for GPU
work) a `synchronize` function so the timer waits for the device before stopping
the clock. Do NOT time a single run — warmup then N steady-state trials, report
median / p95 / std.
"""
from __future__ import annotations

import statistics
import subprocess
import threading
import time
from typing import Callable

from .config import TRIALS, WARMUP_ITERS


def time_callable(
    fn: Callable[[], object],
    warmup: int = WARMUP_ITERS,
    trials: int = TRIALS,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Time `fn` and return {median_ms, p95_ms, std_ms}.

    `synchronize` (e.g. torch.cuda.synchronize or cupy device sync) is called
    after each iteration so GPU kernels actually finish before we stop timing.
    """
    for _ in range(max(0, warmup)):
        fn()
    if synchronize:
        synchronize()

    samples: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        if synchronize:
            synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": p95,
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


# --- GPU resource helpers --------------------------------------------------

def torch_peak_mem_mb(reset: bool = True) -> float | None:
    """Peak CUDA allocation (MB) since the last reset, via torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        mb = torch.cuda.max_memory_allocated() / (1024**2)
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return mb
    except ImportError:
        return None


def torch_reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class GpuUtilSampler:
    """Poll `nvidia-smi` for utilization in a background thread while a run
    executes. Coarse (nvidia-smi granularity) but good enough to show the
    before/after utilization story; Nsight is used for the fine-grained view.
    """

    def __init__(self, interval_s: float = 0.05, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                self._samples.append(float(out.decode().strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuUtilSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def mean_util(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None
