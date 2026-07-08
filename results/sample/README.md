# Sample results — checked-in evidence snapshot

These are **real benchmark artifacts** from a full run on an **NVIDIA GeForce
RTX 5060 Laptop GPU** (Blackwell, 8 GB; driver 580.159.03 / CUDA 13.0; PyTorch
cu128, CuPy 12.x, CVXPY 1.5 / CLARABEL; Python 3.11). They are committed so the
README's scoreboard is backed by inspectable data, not just prose.

- One JSON per `(workload, device, backend, size)` — the exact schema in
  [`cudalpha/metrics.py`](../../cudalpha/metrics.py), including the timing
  distribution, throughput, GPU memory/util, `passed_validation`, `speedup_vs_cpu`,
  and a captured `env` block (GPU name, driver, framework versions, git SHA).
- [`SCOREBOARD.md`](SCOREBOARD.md) — the aggregated table `make aggregate` prints.
- `summary.parquet` — the same table the dashboard reads.

Live runs write to `results/` (git-ignored); this `sample/` folder is the
tracked snapshot. Reproduce with `make bench && make aggregate` on a CUDA host.
