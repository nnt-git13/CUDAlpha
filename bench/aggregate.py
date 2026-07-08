"""Aggregate result artifacts into a summary table.

Reads results/*.json (one per job), builds a tidy DataFrame, and prints the
markdown table the README wants (workload, size, CPU time, GPU time, speedup,
GPU util, bottleneck, fix). Also writes results/summary.parquet for the dashboard.

    python -m bench.aggregate
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cudalpha.config import RESULTS_DIR
from cudalpha.metrics import load_results


def _int_if_whole(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def add_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give `df` a readable `size_label` and a numeric `size_value`, whether the
    size arrived as a nested `size` dict column or as flattened `size.*` columns
    (pandas.json_normalize expands nested dicts by default)."""
    if df.empty:
        return df
    size_cols = [c for c in df.columns if c.startswith("size.")]
    if size_cols:
        def label(row):
            parts = [(c.split(".", 1)[1], row[c]) for c in size_cols if pd.notna(row[c])]
            return ", ".join(f"{k}={_int_if_whole(v)}" for k, v in parts)

        def value(row):
            for c in size_cols:
                if pd.notna(row[c]):
                    return _int_if_whole(row[c])
            return None

        df["size_label"] = df.apply(label, axis=1)
        df["size_value"] = df.apply(value, axis=1)
    elif "size" in df.columns:
        df["size_label"] = df["size"].apply(
            lambda d: ", ".join(f"{k}={v}" for k, v in d.items()) if isinstance(d, dict) else str(d)
        )
        df["size_value"] = df["size"].apply(
            lambda d: next(iter(d.values())) if isinstance(d, dict) and d else None
        )
    return df


def build_frame(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = load_results(results_dir)
    if not rows:
        return pd.DataFrame()
    return add_size_columns(pd.json_normalize(rows))


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (workload, size) pairing the CPU and GPU medians."""
    if df.empty:
        return df
    keep = ["workload", "size_label", "device", "median_ms", "gpu_util_pct",
            "speedup_vs_cpu", "passed_validation", "backend"]
    df = df[[c for c in keep if c in df.columns]].copy()
    cpu = df[df.device == "cpu"][["workload", "size_label", "median_ms"]].rename(
        columns={"median_ms": "cpu_ms"})
    gpu = df[df.device == "gpu"][["workload", "size_label", "median_ms", "gpu_util_pct",
                                  "speedup_vs_cpu", "passed_validation", "backend"]].rename(
        columns={"median_ms": "gpu_ms"})
    out = cpu.merge(gpu, on=["workload", "size_label"], how="outer")
    return out.sort_values(["workload", "size_label"]).reset_index(drop=True)


def to_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No results yet. Run `make bench` first._"
    cols = {
        "workload": "Workload", "size_label": "Size", "cpu_ms": "CPU (ms)",
        "gpu_ms": "GPU (ms)", "speedup_vs_cpu": "Speedup", "gpu_util_pct": "GPU util %",
        "passed_validation": "Valid",
    }
    disp = summary.rename(columns=cols)
    for c in ("CPU (ms)", "GPU (ms)", "Speedup", "GPU util %"):
        if c in disp:
            disp[c] = disp[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    disp["Bottleneck found"] = ""   # fill from your Nsight notes
    disp["Fix"] = ""
    try:
        return disp.to_markdown(index=False)
    except Exception:
        return disp.to_string(index=False)


def main():
    df = build_frame()
    summary = summary_table(df)
    if not summary.empty:
        out = RESULTS_DIR / "summary.parquet"
        try:
            summary.to_parquet(out)
            print(f"wrote {out}\n")
        except Exception as e:
            # parquet needs pyarrow/fastparquet; the markdown table is the primary
            # output, so don't fail the whole aggregate step over the cache file.
            print(f"(skipped {out}: {e})\n")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
