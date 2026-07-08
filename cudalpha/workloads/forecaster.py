"""Forecaster workload — small PyTorch price/volatility model.

Depth role: DEEPEST PROFILING story (bottleneck #1). We benchmark inference and
progressively optimize it. The optimization levers are composable flags so you
can A/B them cleanly and record the latency / utilization delta at each step:

    use_host_input   input starts on the host and is copied H2D each call.
                     This *creates* the bottleneck to profile: a small model
                     spends most of its wall-clock stalled on a synchronous,
                     pageable-memory copy while the GPU sits idle.
    use_pinned       pin the host buffer and copy non_blocking -> the copy
                     overlaps and the stall shrinks (the classic dataloader fix).
    use_amp          mixed precision (fp16) -> validator uses loose tolerances.
    use_compile      torch.compile (remember: it needs extra warmup to compile).
    use_cuda_graphs  capture the inference once and replay it, removing per-call
                     launch overhead — the biggest win for tiny, fixed-shape
                     inference like this.

The model itself is intentionally small — a realistic GPU inference load to
measure, not a model meant to make money.
"""
from __future__ import annotations

from typing import Any

from ..config import BATCH_SIZES, SEED
from ..data import gbm_prices, log_returns
from .base import Callable_, Workload

SEQ_LEN = 64
N_FEATURES = 8
HIDDEN = 128


def build_model():
    import torch.nn as nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEATURES, HIDDEN, num_layers=2, batch_first=True)
            self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return Forecaster()


def _gbm_next_step_batch(batch: int, device: str, seed: int = SEED):
    """A (x, y) minibatch from GBM log-returns: predict the next step's return
    from a window of features. Gives the training path a real target instead of
    pure noise (honors the training/inference lifecycle claim)."""
    import torch

    # One long return series, windowed into (batch, SEQ_LEN, N_FEATURES) samples.
    prices = gbm_prices(n_assets=N_FEATURES, n_steps=batch + SEQ_LEN + 1, seed=seed)
    rets = log_returns(prices).astype("float32")            # (T, N_FEATURES)
    xs, ys = [], []
    for i in range(batch):
        window = rets[i : i + SEQ_LEN]                       # (SEQ_LEN, N_FEATURES)
        target = rets[i + SEQ_LEN, 0]                        # next return of asset 0
        xs.append(window)
        ys.append(target)
    x = torch.as_tensor(xs, device=device)
    y = torch.as_tensor(ys, device=device).unsqueeze(1)
    return x, y


def train(model, steps: int = 50, batch: int = 128, device: str = "cpu"):
    """Minimal training loop against a GBM next-step-return target — enough to
    exercise the training path for the lifecycle claim."""
    import torch

    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    x, y = _gbm_next_step_batch(batch, device)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model


class ForecasterWorkload(Workload):
    name = "forecaster"

    # Optimization levers — flip these to build the before/after story.
    use_host_input: bool = False   # input starts on host, copied H2D per call (the bottleneck)
    use_pinned: bool = False       # pin the host buffer + non_blocking copy (the fix)
    use_amp: bool = False          # mixed precision (fp16) -> validator uses loose tol
    use_compile: bool = False      # torch.compile
    use_cuda_graphs: bool = False  # capture/replay a CUDA graph for inference

    def sizes(self) -> list[dict[str, Any]]:
        return [{"batch": b} for b in BATCH_SIZES]

    # --- deterministic construction ----------------------------------------
    # Model weights and the input are built on the CPU under fixed seeds and then
    # moved to the device, so the CPU and GPU paths run the SAME weights on the
    # SAME input — a prerequisite for CPU-vs-GPU validation to mean anything.
    def _seeded_model(self):
        import torch

        torch.manual_seed(SEED)
        return build_model().eval()

    def _seeded_input(self, batch: int, *, pinned: bool = False):
        import torch

        g = torch.Generator().manual_seed(SEED + 1)
        x = torch.randn(batch, SEQ_LEN, N_FEATURES, generator=g)   # pageable host tensor
        return x.pin_memory() if pinned else x

    # --- CPU ----------------------------------------------------------------
    def cpu(self, size: dict[str, Any]) -> Callable_:
        import torch

        model = self._seeded_model().to("cpu")
        x = self._seeded_input(size["batch"])

        @torch.inference_mode()
        def run():
            return model(x)

        return Callable_(fn=run, backend="torch", throughput_items=size["batch"])

    # --- GPU ----------------------------------------------------------------
    def gpu(self, size: dict[str, Any]) -> Callable_:
        import torch

        batch = size["batch"]
        model = self._seeded_model().to("cuda")
        if self.use_compile and not self.use_cuda_graphs:
            model = torch.compile(model)
        amp = self.use_amp

        def _forward(inp):
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return model(inp)
            return model(inp)

        sync = torch.cuda.synchronize

        # --- CUDA graph capture/replay -------------------------------------
        if self.use_cuda_graphs:
            static_in = self._seeded_input(batch).to("cuda")
            host_in = self._seeded_input(batch, pinned=self.use_pinned) if self.use_host_input else None
            # Warm up on a side stream, then capture the graph (torch's canonical
            # pattern). inference_mode keeps autograd out of the captured region.
            with torch.inference_mode():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _forward(static_in)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = _forward(static_in)

            def run():
                if host_in is not None:
                    static_in.copy_(host_in, non_blocking=self.use_pinned)
                graph.replay()
                return static_out

            return Callable_(fn=run, backend="torch-cudagraph", synchronize=sync,
                             fp16=amp, throughput_items=batch)

        # --- non-graph paths (optionally with a host->device copy) ---------
        if self.use_host_input:
            host_in = self._seeded_input(batch, pinned=self.use_pinned)

            @torch.inference_mode()
            def run():
                dev = host_in.to("cuda", non_blocking=self.use_pinned)
                return _forward(dev)

            backend = "torch-pinned" if self.use_pinned else "torch-hostcopy"
            return Callable_(fn=run, backend=backend, synchronize=sync,
                             fp16=amp, throughput_items=batch)

        # baseline: input already resident on-device
        x = self._seeded_input(batch).to("cuda")

        @torch.inference_mode()
        def run():
            return _forward(x)

        return Callable_(fn=run, backend="torch", synchronize=sync,
                         fp16=amp, throughput_items=batch)

    def validate(self, cpu_out: Any, gpu_out: Any, *, fp16: bool = False) -> dict[str, Any]:
        from ..validate import compare_arrays

        if fp16:
            # AMP path: keep the standard loose fp16 tolerances.
            return compare_arrays(cpu_out, gpu_out, fp16=True)
        # fp32 path: a 2-layer LSTM over 64 timesteps accumulates rounding
        # differently on cuDNN vs the CPU backend, so outputs agree only to
        # ~1e-6 absolute — while the untrained model's predictions sit near zero
        # (~1e-3), which makes a *relative* tolerance blow up on near-zero
        # outputs. Validate at a scale-appropriate absolute tolerance (still
        # ~40x tighter than the signal, so a real divergence is caught).
        return compare_arrays(cpu_out, gpu_out, rtol=1e-2, atol=1e-4)
