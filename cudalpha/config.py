"""Central configuration for CUDAlpha.

Everything that a benchmark run depends on lives here so runs are reproducible:
seeds, the sweep grids, validation tolerances, and default trial counts.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TRACES_DIR = ROOT / "traces"

# --- reproducibility -------------------------------------------------------
SEED = 1337

# --- benchmark defaults ----------------------------------------------------
WARMUP_ITERS = 5          # discarded before timing (critical for torch.compile / JIT)
TRIALS = 30               # steady-state timed runs; report median / p95 / std

# --- sweep grids (kept small so a full sweep finishes in minutes) ----------
# Portfolio optimizer: number of assets.
ASSET_COUNTS = [50, 200, 500, 1000, 2000, 5000]
# Forecaster: inference batch sizes.
BATCH_SIZES = [32, 128, 512, 2048]
# Backtester: length of the price series (per asset) and asset count.
SERIES_LENGTHS = [10_000, 100_000, 1_000_000]
MA_WINDOW = 50
# Rolling-mean kernel study: window sizes swept at a fixed series length to find
# where the O(1)-per-output prefix-sum kernel overtakes the O(window) naive one
# (bottleneck #2 / H2). Small windows favor the cache-cheap naive re-sum; the
# prefix-sum wins once the window is large enough to amortize its scan.
KERNEL_WINDOWS = [50, 100, 200, 500, 1000, 2000, 5000]

# --- validation tolerances -------------------------------------------------
# fp32 CPU-vs-GPU should match tightly; fp16 (mixed precision) will NOT match
# fp32 exactly, so it gets deliberately looser tolerances and is documented as
# expected in validate.py. A naive allclose against fp32 here would fail.
FP32_RTOL, FP32_ATOL = 1e-4, 1e-6
FP16_RTOL, FP16_ATOL = 3e-2, 1e-2


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we might touch. Call once at the top of every entrypoint."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Trade a little speed for determinism in the correctness runs.
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import cupy as cp

        cp.random.seed(seed)
    except ImportError:
        pass
