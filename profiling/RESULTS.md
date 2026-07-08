# Profiling results — measured bottlenecks (RTX 5060)

The two bottlenecks, each **root-caused and fixed with before/after numbers**.
Raw `torch.profiler` chrome traces are checked in under
[`traces/sample/`](../traces/sample/); the charts referenced below are in
[`figures/readme_assets/`](../figures/readme_assets/). Reproduce with
`make profile-forecaster` and `make bench-kernel-sweep`.

## Bottleneck #1 — forecaster inference is host-bound

`torch.profiler` on the baseline (batch 512) shows the wall-clock is **not** in
the model — it's in the host→device copy:

| Operator | Self CPU time | Share |
|---|---:|---:|
| `aten::copy_` (H2D transfer) | 58.955 ms | **92.17%** |
| `aten::_cudnn_rnn` (the LSTM) | 1.805 ms | 2.82% |

So the GPU is starved by a synchronous, pageable-memory copy. Fixing that — not
tuning FLOPs — is what moves the needle. The optimization arc:

| Stage | median (ms) | GPU util | Note |
|---|---:|---:|---|
| baseline (pageable H2D) | 7.375 | 28% | copy-bound |
| + pinned / non-blocking | 7.313 | 28% | barely moves — the copy *is* the work |
| + AMP (fp16) | 1.466 | 44% | **the big win (5×)** — half the bytes |
| + `torch.compile` | 1.310 | — | small extra |
| + CUDA graph | **0.927** | — | **8.0× vs baseline** — removes launch overhead |

Traces: `traces/sample/forecaster_baseline.json`,
`traces/sample/forecaster_optimized.json` (open in `chrome://tracing` or Perfetto).

> Note: `torch.profiler` reported CPU-side activity only on this box
> (`CUPTI_ERROR_INVALID_DEVICE` — profiler counters need elevated permission on
> the laptop). The CPU-side view is sufficient to establish the copy-bound cause;
> the GPU kernel timeline would come from Nsight Systems with counters enabled.

## Bottleneck #2 — rolling-mean kernel scaling (H2 crossover)

`make bench-kernel-sweep` at a fixed 1e6-element series. The naive kernel does
O(window) cached global reads per output; the prefix-sum kernel is O(1) per
output after one float64 scan:

| MA window | naive (ms) | prefix-sum (ms) | winner |
|---:|---:|---:|---|
| 50 | 0.065 | 0.176 | naive |
| 100 | 0.110 | 0.175 | naive |
| 200 | 0.206 | 0.174 | **prefix-sum** |
| 500 | 0.444 | 0.160 | prefix-sum |
| 1000 | 0.890 | 0.161 | prefix-sum |
| 2000 | 1.890 | 0.166 | prefix-sum |
| 5000 | 4.741 | 0.159 | **prefix-sum (~30×)** |

Naive scales linearly with the window; the prefix-sum kernel stays flat. They
cross at **window ≥ 200** — the finding is *where* an O(1) algorithm's fixed
overhead is amortized, not that it's unconditionally faster. Chart:
`figures/readme_assets/h2_crossover.png`.
