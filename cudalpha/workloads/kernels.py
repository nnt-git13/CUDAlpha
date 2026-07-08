"""Custom GPU kernels for the backtester.

This is the "custom CUDA kernel" story. We use CuPy RawKernel, which compiles
actual CUDA C at runtime (via NVRTC) — so calling it a custom CUDA kernel is
honest (unlike a Numba @cuda.jit fallback, which should be described as a
"Numba CUDA kernel").

The arc for bottleneck #2:
  1. `rolling_mean_naive`: every thread re-sums a full window -> O(N * window)
     global-memory reads, low arithmetic intensity. Profile it with Nsight
     Compute; it is memory-bound (redundant global loads).
  2. `rolling_mean_fast`: an O(1)-per-output version. We build an inclusive
     prefix sum once (in float64, for numerical stability over long series),
     then a kernel emits out[i] = (P[i] - P[i-window]) / window — a single
     subtract per output instead of `window` adds. Re-profile: global-load
     traffic collapses and memory throughput / achieved occupancy climb. THAT
     delta is bottleneck #2.

`rolling_mean_reference` is the pure-NumPy definition both kernels implement; the
CPU test suite checks the kernel math against it without needing a GPU.
"""
from __future__ import annotations

import numpy as np

# --- semantics --------------------------------------------------------------
# out[i] = mean(x[i-window+1 .. i])   for i >= window-1
# out[i] = 0                          for i <  window-1  (not enough history yet)


def rolling_mean_reference(x: np.ndarray, window: int) -> np.ndarray:
    """Pure-NumPy trailing rolling mean — the contract both kernels implement.

    Uses a float64 prefix sum so it is a stable reference for validating the
    float32 GPU kernels on long series.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.zeros(n, dtype=np.float32)
    if window <= 0 or window > n:
        return out
    csum = np.cumsum(x)                      # inclusive prefix sum, float64
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


# --- naive kernel: O(window) global reads per output (bottleneck #2 "before") ---
_ROLLING_MEAN_NAIVE_SRC = r"""
extern "C" __global__
void rolling_mean_naive(const float* x, float* out, const int n, const int window) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // one thread per output element
    if (i >= n) return;
    if (i < window - 1) { out[i] = 0.0f; return; }   // not enough history yet
    float acc = 0.0f;
    for (int k = 0; k < window; ++k) {               // naive: re-sum the whole window
        acc += x[i - k];
    }
    out[i] = acc / window;
}
"""

# --- fast kernel: O(1) per output from a precomputed prefix sum ("after") -----
# Consumes an inclusive prefix sum P (float64). out[i] = (P[i] - P[i-window]) / w.
# For the first valid index (i == window-1) there is no P[i-window]; the whole
# window is P[i] itself.
_ROLLING_MEAN_FAST_SRC = r"""
extern "C" __global__
void rolling_mean_fast(const double* prefix, float* out, const int n, const int window) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    if (i < window - 1) { out[i] = 0.0f; return; }
    double lo = (i >= window) ? prefix[i - window] : 0.0;   // exclusive lower bound
    out[i] = (float)((prefix[i] - lo) / (double)window);    // one subtract, not `window` adds
}
"""

_kernel_cache: dict[str, object] = {}


def _get_kernel(name: str, src: str):
    """Compile (once) and cache a RawKernel — NVRTC compilation is not free."""
    if name not in _kernel_cache:
        import cupy as cp

        _kernel_cache[name] = cp.RawKernel(src, name)
    return _kernel_cache[name]


def _launch(kernel, n: int, args: tuple, threads: int = 256) -> None:
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), args)


def rolling_mean_naive(x, window: int):
    """Naive custom-kernel rolling mean over a 1-D cupy float32 array."""
    import cupy as cp

    x = cp.ascontiguousarray(x, dtype=cp.float32)
    out = cp.empty_like(x)
    n = x.size
    _launch(_get_kernel("rolling_mean_naive", _ROLLING_MEAN_NAIVE_SRC),
            n, (x, out, cp.int32(n), cp.int32(window)))
    return out


def rolling_mean_fast(x, window: int):
    """Optimized rolling mean — O(1) per output via a prefix sum (bottleneck #2).

    The prefix sum is built once on-device with cupy.cumsum (float64 for
    stability); the kernel then does a single subtract per output. Compared with
    `rolling_mean_naive` this cuts global-load traffic from O(N*window) to O(N),
    which is exactly what the Nsight Compute before/after is meant to show.
    """
    import cupy as cp

    x = cp.ascontiguousarray(x, dtype=cp.float32)
    n = int(x.size)
    out = cp.empty(n, dtype=cp.float32)
    if window <= 0 or window > n:
        out.fill(0)
        return out
    prefix = cp.cumsum(x, dtype=cp.float64)   # inclusive prefix sum, one O(N) pass
    _launch(_get_kernel("rolling_mean_fast", _ROLLING_MEAN_FAST_SRC),
            n, (prefix, out, cp.int32(n), cp.int32(window)))
    return out
