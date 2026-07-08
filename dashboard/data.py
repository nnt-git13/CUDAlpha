"""Shared data access for the dashboard pages.

All pages read the same artifacts the benchmark harness writes: the aggregated
`results/summary.parquet` if present, else the raw per-run `results/*.json`.
Kept import-light (pandas only) so the app starts before any benchmark has run,
and every helper degrades gracefully to an empty frame.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Put the repo root on the path so `bench` / `cudalpha` import when the app is
# launched as `python dashboard/app.py` (script dir, not repo root, is sys.path[0]).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "results"

# Backend -> which optimization stage it represents, for the profiling story.
BACKEND_STAGE = {
    "cupy-rawkernel-naive": "before (memory-bound)",
    "cupy-rawkernel-fast": "after (prefix-sum, O(1)/output)",
    "torch-hostcopy": "before (pageable H2D)",
    "torch-pinned": "after (pinned + non_blocking)",
    "torch-cudagraph": "after (captured graph)",
}


def load_raw(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Every per-run result JSON as one flat DataFrame (one row per artifact)."""
    jsons = sorted(results_dir.glob("*.json"))
    if not jsons:
        return pd.DataFrame()
    rows = []
    for p in jsons:
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    from bench.aggregate import add_size_columns

    df = add_size_columns(pd.json_normalize(rows))
    for c in ("median_ms", "p95_ms", "std_ms", "throughput", "peak_mem_mb",
              "gpu_util_pct", "speedup_vs_cpu", "size_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_summary(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """The aggregated summary if `make aggregate` has run, else built on the fly."""
    parquet = results_dir / "summary.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    # Fall back to aggregating in-process so the dashboard works without the step.
    try:
        from bench.aggregate import build_frame, summary_table

        return summary_table(build_frame(results_dir))
    except Exception:
        return pd.DataFrame()


def empty_state(message: str):
    """A consistent 'no data yet' placeholder for every page."""
    import dash_bootstrap_components as dbc

    return dbc.Alert(
        [message, " Run ", dbc.Badge("make bench && make aggregate", color="dark"),
         " on a CUDA host to populate this view."],
        color="secondary",
    )
