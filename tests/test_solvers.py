"""
Tests for forward solvers on Phase I Mittag-Leffler analytical benchmark.
"""

import numpy as np
import pytest
import torch

from benchmarks.systems import MittagLefflerDecay
from core.solvers.full_history import FullHistoryFDESolver
from core.solvers.fixed_soe import FixedSOEFDESolver
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


def test_full_history_vs_analytical():
    """Full-history Adams-Bashforth-Moulton solver vs exact Mittag-Leffler."""
    benchmark = MittagLefflerDecay(beta=0.7, decay_rate=1.0, x0=1.0)
    t_grid = torch.linspace(0.0, 3.0, 151, dtype=torch.float64)

    z0 = torch.tensor([1.0], dtype=torch.float64)
    solver = FullHistoryFDESolver(beta=0.7)
    z_num = solver.solve(benchmark.rhs, z0, t_grid).squeeze(-1)

    z_exact = benchmark.exact_solution(t_grid)

    rel_error = torch.norm(z_num - z_exact) / torch.norm(z_exact)
    assert rel_error < 0.02, f"Full history relative error {rel_error.item()} is too high"


def test_fixed_soe_vs_analytical():
    """Fixed-order SOE solver vs exact Mittag-Leffler."""
    benchmark = MittagLefflerDecay(beta=0.75, decay_rate=1.0, x0=1.0)
    t_grid = torch.linspace(0.0, 4.0, 201, dtype=torch.float64)

    z0 = torch.tensor([1.0], dtype=torch.float64)
    solver = FixedSOEFDESolver(beta=0.75, num_modes=20)
    z_num = solver.solve(benchmark.rhs, z0, t_grid).squeeze(-1)

    z_exact = benchmark.exact_solution(t_grid)

    rel_error = torch.norm(z_num - z_exact) / torch.norm(z_exact)
    assert rel_error < 0.03, f"Fixed SOE relative error {rel_error.item()} is too high"


def test_adaptive_soe_vs_analytical():
    """AdaMem-FDE adaptive solver vs exact Mittag-Leffler."""
    benchmark = MittagLefflerDecay(beta=0.7, decay_rate=1.2, x0=1.0)
    t_grid = torch.linspace(0.0, 3.0, 151, dtype=torch.float64)

    z0 = torch.tensor([1.0], dtype=torch.float64)
    solver = AdaptiveSOEFDESolver(beta=0.7, tol=5e-4, K_init=8, K_min=4, K_max=32)
    sol = solver.solve(benchmark.rhs, z0, t_grid)
    z_num = sol.z.squeeze(-1)

    z_exact = benchmark.exact_solution(t_grid)

    rel_error = torch.norm(z_num - z_exact) / torch.norm(z_exact)
    assert rel_error < 0.03, f"Adaptive SOE relative error {rel_error.item()} is too high"
    assert sol.avg_modes <= sol.max_modes
