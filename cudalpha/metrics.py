"""Benchmark results schema + environment capture + IO.

One BenchmarkResult == one (workload, device, backend, size) measurement.
Each is written to results/ as a single JSON file (one artifact per Slurm job),
then aggregate.py collects them into a table. Keeping the schema explicit is what
makes the benchmarks look serious rather than "some speed numbers".
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _nvidia_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip().splitlines()[0]
    except Exception:
        return "unknown"


def capture_environment() -> dict[str, Any]:
    """Programmatically capture hardware/software metadata for every run."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "nvidia_driver": _nvidia_driver(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
        )
    except ImportError:
        env["torch"] = None
    return env


@dataclass
class BenchmarkResult:
    workload: str                 # forecaster | backtester | optimizer
    device: str                   # cpu | gpu
    backend: str                  # e.g. numpy, cvxpy, torch, cupy, cupy-rawkernel, cuopt
    size: dict[str, Any]          # size params, e.g. {"n_assets": 500}

    # timing (milliseconds) over TRIALS steady-state runs
    median_ms: float
    p95_ms: float
    std_ms: float

    throughput: float | None = None      # items/sec, workload-defined
    peak_mem_mb: float | None = None      # GPU peak allocation
    gpu_util_pct: float | None = None     # sampled during the run

    # correctness + comparison (filled by the runner)
    passed_validation: bool | None = None
    validation_detail: dict[str, Any] = field(default_factory=dict)
    speedup_vs_cpu: float | None = None

    # provenance
    job_id: str = "local"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)
    env: dict[str, Any] = field(default_factory=capture_environment)

    def save(self, out_dir: Path | str = RESULTS_DIR) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        size_tag = "_".join(f"{k}{v}" for k, v in self.size.items())
        fname = f"{self.workload}_{self.backend}_{self.device}_{size_tag}_{self.run_id}.json"
        path = out_dir / fname
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def load_results(results_dir: Path | str = RESULTS_DIR) -> list[dict[str, Any]]:
    """Read every result JSON in a directory into a list of plain dicts."""
    results_dir = Path(results_dir)
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out
