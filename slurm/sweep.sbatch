#!/bin/bash
# HPC-style benchmark sweep via a Slurm job array. Each array task runs ONE
# problem size and writes its own result artifact (one artifact per job), which
# aggregate.py later collects. Run locally against a slurm-docker-cluster (see
# slurm/README.md) — that simulates the cluster workflow, it is not a real HPC
# cluster unless you submit this on one.
#
#   sbatch slurm/sweep.sbatch
#
#SBATCH --job-name=cudalpha-sweep
#SBATCH --array=0-5                 # one task per size below (0..len-1)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%A_%a.out
#SBATCH --error=results/slurm-%A_%a.err

set -euo pipefail

# Map the array index -> (workload, size). Keep in sync with config.py grids.
WORKLOAD="backtester"
SIZES=(10000 100000 1000000 10000 100000 1000000)   # example grid
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"

echo "task ${SLURM_ARRAY_TASK_ID}: ${WORKLOAD} size=${SIZE} on $(hostname)"

# Fail the task cleanly if the run errors, so failed jobs are visible in sacct.
srun python3 -m bench.run_all \
    --workload "${WORKLOAD}" \
    --sizes "${SIZE}" \
    --job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task ${SLURM_ARRAY_TASK_ID}: done"
