"""Optimizer math: the Frank-Wolfe simplex solver and the objective helper.

Runs the GPU fallback solver on NumPy (xp=numpy) so its correctness is provable
without a GPU — the same code path executes with xp=cupy on-device.
"""
import numpy as np

from cudalpha.workloads.optimizer import (
    GAMMA,
    _frank_wolfe_simplex,
    _problem,
    objective,
)


def test_frank_wolfe_returns_feasible_simplex_point():
    mu, cov = _problem(50)
    w = _frank_wolfe_simplex(cov, mu, xp=np, iters=500)
    assert abs(w.sum() - 1.0) < 1e-6, "must stay on the simplex (sum = 1)"
    assert (w >= -1e-9).all(), "long-only: no negative weights"


def test_frank_wolfe_beats_equal_weight():
    mu, cov = _problem(80)
    w = _frank_wolfe_simplex(cov, mu, xp=np, iters=800)
    eq = np.full(mu.shape[0], 1.0 / mu.shape[0])
    # FW minimizes the objective, so it should not be worse than equal-weight.
    assert objective(cov, mu, w) <= objective(cov, mu, eq) + 1e-6


def test_objective_matches_manual():
    mu = np.array([0.1, 0.2, 0.3])
    cov = np.eye(3)
    w = np.array([0.2, 0.3, 0.5])
    expected = float(w @ cov @ w - GAMMA * (mu @ w))
    assert np.isclose(objective(cov, mu, w), expected)


def test_frank_wolfe_is_near_optimal_on_diagonal_problem():
    # Diagonal cov + linear term has a closed-form-ish interior optimum; FW should
    # place most mass on the highest-return / lowest-variance asset.
    mu = np.array([0.0, 0.0, 1.0])
    cov = np.diag([1.0, 1.0, 1.0])
    w = _frank_wolfe_simplex(cov, mu, xp=np, iters=1000)
    assert w.argmax() == 2
