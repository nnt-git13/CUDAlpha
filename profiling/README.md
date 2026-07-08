# Profiling evidence

This is where the "I can do performance engineering" proof lives: before/after
traces for **two** bottlenecks, each with an identified cause, a fix, and a
measured result. Store traces here and link them from the dashboard Profiling
page.

## Bottleneck #1 — forecaster inference (PyTorch)
Likely findings: GPU idle waiting on the dataloader / small batches, and fp32
compute where fp16 would do.

```bash
# system-level timeline (find CPU<->GPU gaps, kernel timeline)
nsys profile -o traces/forecaster_baseline \
    python -m bench.run_all --workload forecaster --sizes 512

# torch operator view
python - <<'PY'
import torch
from torch.profiler import profile, ProfilerActivity
from cudalpha.workloads.forecaster import ForecasterWorkload
call = ForecasterWorkload().gpu({"batch": 512})
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10): call.fn(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
PY
```

Then flip the levers in `ForecasterWorkload` (`use_amp`, bigger batch, pinned
memory, `use_compile`, then CUDA graphs), re-profile, and record the
latency + utilization delta. Example claim to substantiate:
"dataloader stalls → pinned memory + larger batch → GPU util 42% → 83%".

## Bottleneck #2 — backtester rolling-mean kernel (CuPy RawKernel)
`rolling_mean_naive` is memory-bound (re-sums the whole window per output).

```bash
# per-kernel deep dive
ncu -o traces/rolling_mean_naive \
    python -m bench.run_all --workload backtester --sizes 1000000
```

Implement `rolling_mean_fast` (prefix-sum / sliding-window), re-run `ncu`, and
report the speedup + the change in memory throughput / achieved occupancy.

## For the README
Drop three screenshots into the repo: the dashboard, the Slurm `squeue`/`sacct`
view, and one Nsight before/after comparison.
