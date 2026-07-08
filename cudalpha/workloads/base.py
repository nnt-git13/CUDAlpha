"""Common interface for the three workloads.

Each workload exposes CPU and GPU callables for a given problem size plus a
validator that compares their outputs. The runner (bench/run_all.py) drives them
uniformly: build callable -> time it -> validate -> record BenchmarkResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Callable_:
    """A prepared, zero-arg unit of work plus what's needed to time/validate it."""
    fn: Callable[[], Any]                 # the thing to time (returns its output)
    backend: str                          # numpy / torch / cupy / cupy-rawkernel / cvxpy / cuopt
    synchronize: Callable[[], None] | None = None
    fp16: bool = False                    # tells the validator to use loose tolerances
    throughput_items: float | None = None # items processed per call (for throughput)


class Workload:
    name: str = "base"

    def sizes(self) -> list[dict[str, Any]]:
        """Return the list of size-parameter dicts to sweep over."""
        raise NotImplementedError

    def cpu(self, size: dict[str, Any]) -> Callable_:
        raise NotImplementedError

    def gpu(self, size: dict[str, Any]) -> Callable_:
        raise NotImplementedError

    def gpu_paths(self, size: dict[str, Any]) -> list[Callable_]:
        """All GPU backends to benchmark for this size, each validated against
        the same CPU output. Default is the single `gpu()` path; workloads with
        several GPU implementations (e.g. a vectorized baseline plus a custom
        kernel) override this to return one Callable_ per backend."""
        try:
            return [self.gpu(size)]
        except NotImplementedError:
            return []

    def validate(self, cpu_out: Any, gpu_out: Any, *, fp16: bool = False) -> dict[str, Any]:
        """Return a pass/fail record comparing one CPU output to one GPU output."""
        raise NotImplementedError


def gpu_available() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
