# HPC-style orchestration with Slurm

This runs the benchmark sweep as a **Slurm job array** — each array task runs one
problem size and emits its own result artifact, which `bench/aggregate.py`
collects into the summary table.

> **Honest framing:** running this locally against a containerized Slurm cluster
> *simulates the cluster workflow* (`sbatch`, job arrays, `squeue`, `sacct`,
> per-job artifacts, failed-job handling). It is **not** a real HPC cluster
> unless you submit `sweep.sbatch` on one. Describe it as "HPC-style Slurm
> orchestration," not "ran on an HPC cluster."

## Option A — local Slurm cluster in Docker

Use an existing open-source **slurm-docker-cluster** project (a controller + a
few compute nodes via docker-compose). To run GPU jobs you also need the NVIDIA
Container Toolkit configured for the compute-node containers. Check the specific
project's license and follow its setup, then:

```bash
# from inside the cluster's controller node, with this repo mounted:
sbatch slurm/sweep.sbatch
squeue                     # watch the array run
sacct -j <jobid>           # inspect per-task status (see failed tasks)
python -m bench.aggregate  # collect artifacts into the summary table
```

## Option B — single-node Slurm

Install `slurm-wlm` on your Linux box, configure a single-node `slurm.conf` with
`Gres=gpu:1`, then submit the same script. Simpler cluster, same workflow.

## What to capture for the writeup
- A `squeue`/`sacct` screenshot showing the array (this is the "at scale"
  evidence for the README).
- The aggregated results table across sizes.
- At least one deliberately-failed task to show failed-job handling works.
