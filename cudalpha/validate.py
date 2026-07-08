"""Correctness validation: prove CPU and GPU agree before trusting the speedups.

Without this the project is "speed numbers only". Key subtlety: mixed-precision
(fp16) GPU output will NOT match fp32 CPU output exactly. That's expected, not a
bug — so fp16 comparisons use deliberately looser tolerances (config.FP16_*) and
the deviation is recorded rather than treated as a failure.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import FP16_ATOL, FP16_RTOL, FP32_ATOL, FP32_RTOL


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    elif hasattr(x, "get"):           # cupy array
        x = x.get()
    return np.asarray(x)


def compare_arrays(cpu: Any, gpu: Any, *, fp16: bool = False,
                   rtol: float | None = None, atol: float | None = None) -> dict[str, Any]:
    """Compare two arrays and return a structured pass/fail record.

    Set fp16=True when the GPU path used mixed precision, so the looser
    tolerances apply and the result stays honest about expected deviation.
    `rtol`/`atol` override the defaults when a workload needs a scale-appropriate
    tolerance (e.g. a recurrent net whose outputs sit near zero) — pass them
    explicitly and document why at the call site.
    """
    a, b = _to_numpy(cpu), _to_numpy(gpu)
    d_rtol, d_atol = (FP16_RTOL, FP16_ATOL) if fp16 else (FP32_RTOL, FP32_ATOL)
    rtol = d_rtol if rtol is None else rtol
    atol = d_atol if atol is None else atol

    if a.shape != b.shape:
        return {"passed": False, "reason": f"shape mismatch {a.shape} vs {b.shape}"}

    abs_err = np.abs(a - b)
    max_abs = float(abs_err.max()) if a.size else 0.0
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, abs_err / denom, 0.0)
    max_rel = float(rel.max()) if a.size else 0.0
    passed = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    return {
        "passed": passed,
        "fp16": fp16,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "rtol": rtol,
        "atol": atol,
        "note": "fp16 vs fp32 deviation is expected" if fp16 and not passed else "",
    }


def check_portfolio(weights: Any, *, expected_return: float | None = None,
                    mu: Any = None, long_only: bool = True) -> dict[str, Any]:
    """Feasibility checks for an optimizer solution (independent of speed)."""
    w = _to_numpy(weights).ravel()
    detail: dict[str, Any] = {
        "sum": float(w.sum()),
        "min_weight": float(w.min()) if w.size else None,
    }
    ok = abs(w.sum() - 1.0) < 1e-3
    if long_only:
        ok = ok and bool((w >= -1e-6).all())
    if mu is not None:
        detail["achieved_return"] = float(_to_numpy(mu).ravel() @ w)
    detail["passed"] = ok
    return detail
