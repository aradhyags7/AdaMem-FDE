"""
Unit tests for multi-tolerance sensitivity and Pareto mode monotonicity.
"""

import numpy as np
import pytest
import torch

from benchmarks.systems import MittagLefflerDecay
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


def test_tolerance_mode_monotonicity():
    """
    Verifies that tightening the tolerance \epsilon_{tol} leads to a non-decreasing
    average number of active memory modes: \bar{K}(\epsilon_{tight}) >= \bar{K}(\epsilon_{loose}).
    """
    system = MittagLefflerDecay(beta=0.7, decay_rate=1.0, x0=1.0)
    t_grid = torch.linspace(0.0, 3.0, 151, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)

    # Loose tolerance
    solver_loose = AdaptiveSOEFDESolver(
        beta=0.7, tol=5e-2, K_init=4, K_min=4, K_max=32, delta_K=4, prune_ratio=0.1
    )
    sol_loose = solver_loose.solve(system.rhs, z0, t_grid)

    # Tight tolerance
    solver_tight = AdaptiveSOEFDESolver(
        beta=0.7, tol=5e-4, K_init=4, K_min=4, K_max=32, delta_K=4, prune_ratio=0.1
    )
    sol_tight = solver_tight.solve(system.rhs, z0, t_grid)

    print(f"\n[Monotonicity Check] Loose (5e-2) K_avg: {sol_loose.avg_modes:.2f}, Tight (5e-4) K_avg: {sol_tight.avg_modes:.2f}")
    assert sol_tight.avg_modes >= sol_loose.avg_modes, (
        f"Tight tolerance modes ({sol_tight.avg_modes}) should be >= loose ({sol_loose.avg_modes})"
    )


def test_mode_allocation_ceiling():
    """Verifies that the controller strictly enforces the K_max ceiling."""
    system = MittagLefflerDecay(beta=0.8, decay_rate=1.5, x0=1.0)
    t_grid = torch.linspace(0.0, 3.0, 101, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)

    K_max_allowed = 20
    solver = AdaptiveSOEFDESolver(
        beta=0.8, tol=1e-6, K_init=8, K_min=4, K_max=K_max_allowed, delta_K=4
    )
    sol = solver.solve(system.rhs, z0, t_grid)

    assert sol.max_modes <= K_max_allowed, (
        f"Maximum modes {sol.max_modes} exceeded allowable ceiling {K_max_allowed}"
    )


def test_error_decreases_with_tighter_tolerance():
    """Verifies that tightening tolerance produces lower forward error against analytical ground truth."""
    system = MittagLefflerDecay(beta=0.75, decay_rate=1.0, x0=1.0)
    t_grid = torch.linspace(0.0, 3.0, 151, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)
    z_exact = system.exact_solution(t_grid)
    exact_norm = torch.norm(z_exact).item()

    solver_loose = AdaptiveSOEFDESolver(beta=0.75, tol=1e-2, K_init=4, K_min=4, K_max=32)
    sol_loose = solver_loose.solve(system.rhs, z0, t_grid)
    err_loose = torch.norm(sol_loose.z.squeeze(-1) - z_exact).item() / exact_norm

    solver_tight = AdaptiveSOEFDESolver(beta=0.75, tol=1e-4, K_init=8, K_min=4, K_max=32)
    sol_tight = solver_tight.solve(system.rhs, z0, t_grid)
    err_tight = torch.norm(sol_tight.z.squeeze(-1) - z_exact).item() / exact_norm

    print(f"\n[Error Check] Loose Err: {err_loose:.4e}, Tight Err: {err_tight:.4e}")
    assert err_tight < err_loose, (
        f"Tight error ({err_tight}) should be strictly less than loose error ({err_loose})"
    )
