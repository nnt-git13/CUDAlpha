# README assets

Clean Plotly chart exports (dashboard → camera "download plot as png"), all live
from `results/` on an RTX 5060. Referenced by `README.md`.

| File | Chart | Used in README |
|---|---|---|
| `best_speedup_per_workload.png` | Best GPU speedup per workload (61.6× / 5.8× / 48.8×) | Sneak peek |
| `h2_crossover.png` | H2: kernel runtime vs MA window (naive O(window) vs flat O(1)) | Sneak peek + Results |
| `optimizer_speedup.png` | Optimizer GPU speedup vs size (crosses parity) | Sneak peek + Results |
| `backtester_runtime.png` / `_speedup.png` / `_throughput.png` | Backtester across series length | Results (gallery) |
| `forecaster_runtime.png` / `_speedup.png` / `_throughput.png` / `_peak_memory.png` | Forecaster across batch size | Results (gallery) |
| `optimizer_runtime.png` / `_throughput.png` | Optimizer across portfolio size | Results (gallery) |

## Optional (not yet made) — architecture diagrams
Three `<!-- diagram: ... -->` slots remain in `README.md` for hand-drawn diagrams
(hero banner, system overview, data-to-evidence flow). They're optional; the
measured charts above carry the results story on their own.
