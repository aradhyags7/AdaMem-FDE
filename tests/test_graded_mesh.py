"""
Unit tests for Graded Temporal Meshes and Caputo Singularity Quenching.

Verifies:
1. Mathematical properties of graded mesh (monotonicity, exact boundaries, grading exponent).
2. Singularity quenching on analytical Mittag-Leffler benchmark:
   Demonstrates that graded mesh achieves substantially lower error than uniform mesh
   for beta < 1 where solution has weak singularity at t = 0.
"""

import pytest
import torch
import numpy as np

from core.fractional.graded_mesh import (
    generate_graded_mesh,
    optimal_grading_exponent,
    analyze_mesh_characteristics,
)
from core.fractional.mittag_leffler import mittag_leffler
from core.solvers.fixed_soe import FixedSOEFDESolver


def test_grading_exponent():
    """Verifies r = (2 - beta) / beta values."""
    assert np.isclose(optimal_grading_exponent(1.0), 1.0)
    assert np.isclose(optimal_grading_exponent(0.5), 3.0)
    assert np.isclose(optimal_grading_exponent(0.8), (2.0 - 0.8) / 0.8)

    with pytest.raises(ValueError):
        optimal_grading_exponent(0.0)
    with pytest.raises(ValueError):
        optimal_grading_exponent(1.2)


def test_graded_mesh_properties():
    """Verifies endpoints, monotonicity, and node clustering near t=0."""
    T = 2.0
    N = 100
    beta = 0.7
    t_grid = generate_graded_mesh(T=T, N=N, beta=beta)

    assert len(t_grid) == N + 1
    assert torch.isclose(t_grid[0], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(t_grid[-1], torch.tensor(T, dtype=torch.float64))

    # Strict monotonicity
    dts = t_grid[1:] - t_grid[:-1]
    assert torch.all(dts > 0.0)

    # First step should be much smaller than uniform step T / N
    uniform_dt = T / N
    assert dts[0] < uniform_dt * 0.1

    # Analysis characteristics
    chars = analyze_mesh_characteristics(t_grid)
    assert chars["N"] == N
    assert not chars["is_uniform"]
    assert chars["ratio"] > 10.0


def test_graded_vs_uniform_mittag_leffler_error():
    r"""
    Verifies that graded mesh outperforms uniform mesh on fractional relaxation
        ^C D^\beta z(t) = - \lambda z(t),  z(0) = 1.0
    whose solution is z(t) = E_\beta(-\lambda t^\beta), exhibiting dot{z}(0) = -\infty.
    """
    beta = 0.7
    lam_scalar = 1.5
    T = 1.0
    N = 250
    K = 32

    # Vector field f(z, t) = -1.5 * z
    def f(z, t):
        return -lam_scalar * z

    z0 = torch.tensor([1.0], dtype=torch.float64)

    # 1. Uniform mesh
    t_uniform = torch.linspace(0.0, T, N + 1, dtype=torch.float64)
    solver_uniform = FixedSOEFDESolver(beta=beta, num_modes=K)
    z_uniform = solver_uniform.solve(f, z0, t_uniform).squeeze(-1)

    # 2. Graded mesh
    t_graded = generate_graded_mesh(T=T, N=N, beta=beta, dtype=torch.float64)
    solver_graded = FixedSOEFDESolver(beta=beta, num_modes=K)
    z_graded = solver_graded.solve(f, z0, t_graded).squeeze(-1)

    # Exact analytical solution at mesh points
    z_exact_uniform = torch.tensor(
        mittag_leffler(beta, 1.0, -lam_scalar * (t_uniform.numpy() ** beta)),
        dtype=torch.float64,
    )
    z_exact_graded = torch.tensor(
        mittag_leffler(beta, 1.0, -lam_scalar * (t_graded.numpy() ** beta)),
        dtype=torch.float64,
    )

    err_uniform = torch.max(torch.abs(z_uniform - z_exact_uniform)).item()
    err_graded = torch.max(torch.abs(z_graded - z_exact_graded)).item()

    # Graded mesh error should be at least 1.8x lower than uniform mesh!
    assert err_graded < err_uniform
    assert err_uniform / err_graded >= 1.8, (
        f"Expected graded mesh error to be at least 1.8x lower than uniform: "
        f"err_uniform={err_uniform:.2e}, err_graded={err_graded:.2e}"
    )

