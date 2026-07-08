# Running on Windows

The NVIDIA posting mentions performance testing across **Linux and Windows**
environments. CUDAlpha is Linux-first, but it runs on Windows through **WSL2**:

1. Install WSL2 with a recent Ubuntu, and install the NVIDIA GPU driver for WSL
   on the Windows host (the driver bridges the GPU into WSL2).
2. Install **Docker Desktop** with the WSL2 backend and enable GPU support.
3. From the WSL2 shell, use the repo exactly as on native Linux:
   `make setup`, `make bench`, `make docker-bench`, `make dashboard`.

Native Windows (non-WSL) is **not** a target — CUDA tooling, containers, and
Slurm all assume Linux. This note exists to acknowledge the environment, not to
claim native Windows support.
