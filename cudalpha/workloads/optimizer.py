"""Portfolio optimizer workload — Markowitz mean-variance (a QP).

Depth role: BENCHMARK / VALIDATION story (a scaling study, not a "GPU always
wins" claim). We solve the same QP two ways and compare solve time, feasibility,
and objective quality across portfolio sizes:

    minimize    xᵀ Q x  -  γ · μᵀ x        (Q = covariance, μ = expected returns)
    subject to  Σ xᵢ = 1,  x ≥ 0           (fully-invested, long-only)

  - CPU baseline: CVXPY (tries CLARABEL, then OSQP/SCS).        backend: cvxpy
  - GPU path (shipped): CuPy Frank-Wolfe solver over the simplex. backend: cupy-fw
  - GPU path (planned): NVIDIA cuOpt direct-QP barrier solver — currently STUBBED
    (`_cuopt_solve` raises NotImplementedError; `gpu()` falls back to cupy-fw).

cuOpt notes (for when the stubbed path is wired — verify against your installed
cuOpt docs first):
  * cuOpt solves QP of the form  min ½ xᵀQx + cᵀx  with linear constraints and
    bounds; the BARRIER (interior-point) method is currently the only method
    that supports QPs.
  * The QP solver is reached through cuOpt's Python SDK / direct QP interface —
    NOT through CVXPY or another third-party modeling language.
  * cuOpt's Q is specified WITHOUT the 1/2 factor some solvers assume — mind the
    scaling so the two objectives are comparable.

Expectation to TEST, not assume: barrier has fixed overhead, so CPU may win on
small portfolios and GPU may pull ahead only at larger sizes. Report the
crossover; don't pre-declare a winner.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ASSET_COUNTS, SEED
from ..data import gbm_prices, log_returns, mean_cov
from .base import Callable_, Workload

GAMMA = 1.0  # risk-aversion trade-off


def _problem(n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    prices = gbm_prices(n_assets=n_assets, n_steps=756, seed=SEED)
    mu, cov = mean_cov(log_returns(prices))
    return mu, cov


def objective(cov: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    """The QP objective  wᵀ cov w − γ·μᵀw  (no ½ on the quadratic term, matching
    the CVXPY formulation). Used to compare CPU and GPU solutions by *value*
    rather than by weights, since two QP solvers won't return identical weights."""
    w = np.asarray(w, dtype=np.float64).ravel()
    return float(w @ cov @ w - GAMMA * (mu @ w))


def _frank_wolfe_simplex(cov, mu, *, xp, iters: int = 500):
    """Conditional-gradient (Frank–Wolfe) QP solve over the probability simplex
    {x : Σx = 1, x ≥ 0} — exactly the long-only fully-invested constraint set.

    `xp` is numpy or cupy, so the same code runs on CPU or GPU; on GPU the O(n²)
    `cov @ x` matvec that dominates each iteration runs on the device. This is the
    fallback GPU QP path when cuOpt is unavailable (labelled backend `cupy-fw`).
    """
    n = mu.shape[0]
    x = xp.full(n, 1.0 / n, dtype=xp.float64)
    two_cov = 2.0 * cov
    for t in range(iters):
        grad = two_cov @ x - GAMMA * mu          # ∇(xᵀcov x − γμᵀx)
        s = int(xp.argmin(grad))                 # LMO over the simplex: a vertex
        eta = 2.0 / (t + 2.0)                     # standard FW step size
        x = (1.0 - eta) * x
        x[s] += eta
    return x


class OptimizerWorkload(Workload):
    name = "optimizer"

    def sizes(self) -> list[dict[str, Any]]:
        return [{"n_assets": n} for n in ASSET_COUNTS]

    # --- CPU baseline (CVXPY) ------------------------------------------------
    # Try a robust interior-point solver first. The covariance is estimated from
    # ~756 samples, so at large n it is heavily rank-deficient (only the +eps
    # ridge makes it PD) and OSQP's ADMM can fail to converge; CLARABEL (a proper
    # interior-point method) handles the ill-conditioning. We fall through the
    # list so one solver's failure at one size doesn't kill the baseline.
    _SOLVERS = ("CLARABEL", "OSQP", "SCS")

    def cpu(self, size: dict[str, Any]) -> Callable_:
        import cvxpy as cp

        mu, cov = _problem(size["n_assets"])
        n = size["n_assets"]

        def run():
            w = cp.Variable(n)
            # psd_wrap asserts cov is PSD so cvxpy skips its own (slow, and here
            # numerically fragile) PSD check on the near-singular matrix.
            objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)) - GAMMA * mu @ w)
            prob = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
            last = None
            for solver in self._SOLVERS:
                if not hasattr(cp, solver):
                    continue
                try:
                    prob.solve(solver=getattr(cp, solver))
                    if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                        return np.asarray(w.value)
                except Exception as e:  # noqa: BLE001 - try the next solver
                    last = e
            raise RuntimeError(f"all CVXPY solvers failed for n={n} (last: {last})")

        return Callable_(fn=run, backend="cvxpy", throughput_items=n)

    # --- GPU path: CuPy Frank–Wolfe (probes the stubbed cuOpt path first) ---
    def _cuopt_solve(self, cov: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Solve the QP with cuOpt's direct QP (barrier) interface.

        cuOpt's QP is  min ½ xᵀ Q x + cᵀx ; our objective is  xᵀ cov x − γ μᵀx,
        so Q = 2·cov (to cancel the ½) and c = −γ·μ. Equality constraint Σx = 1,
        bounds x ≥ 0. cuOpt's Python API differs across versions, so this is kept
        behind a try in `gpu()` — verify the exact call against your installed
        cuOpt docs; the CuPy Frank–Wolfe path below is the portable fallback.
        """
        from cuopt.linear_programming import solver  # noqa: F401  (import shape varies by version)

        raise NotImplementedError(
            "wire the cuOpt QP call for your installed cuOpt version; "
            "Q=2*cov, c=-GAMMA*mu, sum(x)=1, x>=0, method=barrier"
        )

    def gpu(self, size: dict[str, Any]) -> Callable_:
        mu, cov = _problem(size["n_assets"])

        # Prefer cuOpt; fall back to the CuPy Frank–Wolfe solver if cuOpt's QP
        # interface isn't available on this box. The backend label records which
        # path actually ran so the scoreboard stays honest.
        try:
            self._cuopt_solve(cov, mu)   # probe: raises NotImplementedError/ImportError
            backend = "cuopt"

            def run():
                return self._cuopt_solve(cov, mu)

            sync = None
        except (ImportError, NotImplementedError):
            import cupy as cp

            cov_d, mu_d = cp.asarray(cov), cp.asarray(mu)
            backend = "cupy-fw"

            def run():
                w = _frank_wolfe_simplex(cov_d, mu_d, xp=cp)
                return cp.asnumpy(w)

            sync = cp.cuda.Stream.null.synchronize

        return Callable_(fn=run, backend=backend, synchronize=sync,
                         throughput_items=size["n_assets"])

    def validate(self, cpu_out: Any, gpu_out: Any, *, fp16: bool = False) -> dict[str, Any]:
        """Compare two solutions: feasibility of each + objective-value closeness.

        Two QP solvers won't return identical weights, so we check feasibility and
        objective-value agreement rather than element-wise weight equality. The
        problem is re-derived from the solution length so validate stays a pure
        function of its inputs."""
        from ..validate import check_portfolio

        cpu_w = np.asarray(cpu_out, dtype=np.float64).ravel()
        n = cpu_w.size
        mu, cov = _problem(n)
        rec: dict[str, Any] = {
            "cpu": check_portfolio(cpu_out, mu=mu),
            "gpu": check_portfolio(gpu_out, mu=mu),
        }
        obj_cpu = objective(cov, mu, cpu_out)
        obj_gpu = objective(cov, mu, gpu_out)
        denom = max(abs(obj_cpu), 1e-9)
        rec["obj_cpu"], rec["obj_gpu"] = obj_cpu, obj_gpu
        rec["obj_rel_err"] = abs(obj_cpu - obj_gpu) / denom
        rec["passed"] = bool(
            rec["cpu"]["passed"] and rec["gpu"]["passed"] and rec["obj_rel_err"] < 1e-2
        )
        return rec
