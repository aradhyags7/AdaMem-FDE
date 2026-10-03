"""
Tests for L-BFGS Quasi-Newton sensitivity and adjoint gradient directionality.
"""

import pytest
import torch
import numpy as np

from benchmarks.systems import FractionalDuffing, FractionalVanDerPol
from core.adjoint.adamem_adjoint import adamem_integrate
from core.solvers.fixed_soe import FixedSOEFDESolver
from models.neural_fde import NeuralFDE
from experiments.run_phase4_training import StructuredDuffingField, generate_ground_truth_data
from experiments.run_phase4_lbfgs_stiff_ablation import (
    ParametricVDP,
    evaluate_gradient_angular_error,
)


def test_gradient_angular_directionality():
    """Verify that proposed adjoint preserves true descent direction whereas naive suffers angular deflection."""
    beta = 0.85
    t_grid, z0, z_true = generate_ground_truth_data(beta=beta, T=2.0, num_steps=30)

    torch.manual_seed(42)
    vf = StructuredDuffingField(hidden_dim=32)
    model = NeuralFDE(vector_field=vf, beta=beta, default_tol=1e-3, K_init=8, K_min=4, K_max=24)

    met_prop = evaluate_gradient_angular_error(model, z0, t_grid, z_true, beta=beta, method="adamem", n_params=10)
    met_naive = evaluate_gradient_angular_error(model, z0, t_grid, z_true, beta=beta, method="naive_adaptive", n_params=10)

    # Proposed must be close to exact descent direction (angle < 5 deg, cos_sim > 0.99)
    assert met_prop["angle_deg"] < 5.0
    assert met_prop["cos_sim"] > 0.99
    assert met_prop["rel_error"] < 0.05

    # Naive must exhibit noticeable angular corruption
    assert met_naive["angle_deg"] > 8.0
    assert met_naive["rel_error"] > 0.30


def test_lbfgs_optimization_step():
    """Verify that L-BFGS with Strong Wolfe line search makes progress with proposed adjoint."""
    beta = 0.85
    t_grid, z0, z_true = generate_ground_truth_data(beta=beta, T=2.0, num_steps=30)

    torch.manual_seed(42)
    vf = StructuredDuffingField(hidden_dim=32)
    model = NeuralFDE(vector_field=vf, beta=beta, default_tol=1e-3, K_init=8, K_min=4, K_max=24)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=5,
        line_search_fn="strong_wolfe",
    )

    losses = []
    for _ in range(4):
        def closure():
            optimizer.zero_grad()
            z_pred = model(z0, t_grid, method="adamem", tol=1e-3)
            loss = torch.mean((z_pred - z_true) ** 2)
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        losses.append(loss_val.item())

    # Loss must decrease monotonically across L-BFGS steps
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.01


def test_van_der_pol_stiff_gradient_comparison():
    """Verify that across non-linear stiffness boundary layers, R^T jump prevents gradient error explosion."""
    beta = 0.80
    mu = 1.5
    t_grid = torch.linspace(0.0, 3.0, 60, dtype=torch.float64)
    z0 = torch.tensor([1.5, 0.0], dtype=torch.float64)

    sys_vdp = FractionalVanDerPol(beta=beta, mu=mu)
    solver_ref = FixedSOEFDESolver(beta=beta, num_modes=32)
    z_true = solver_ref.solve(sys_vdp.rhs, z0, t_grid).detach()

    h = 1e-6
    with torch.no_grad():
        vf_p = ParametricVDP(mu_init=mu + h)
        zp = adamem_integrate(vf_p, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
        lp = 0.5 * torch.sum((zp - z_true) ** 2).item()

        vf_m = ParametricVDP(mu_init=mu - h)
        zm = adamem_integrate(vf_m, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
        lm = 0.5 * torch.sum((zm - z_true) ** 2).item()

    g_fd = (lp - lm) / (2.0 * h)

    # Proposed
    vf_prop = ParametricVDP(mu_init=mu)
    zp_prop = adamem_integrate(vf_prop, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True, parameters=(vf_prop.mu,))
    (0.5 * torch.sum((zp_prop - z_true) ** 2)).backward()
    g_prop = vf_prop.mu.grad.item()

    # Naive
    vf_naive = ParametricVDP(mu_init=mu)
    zp_naive = adamem_integrate(vf_naive, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=False, parameters=(vf_naive.mu,))
    (0.5 * torch.sum((zp_naive - z_true) ** 2)).backward()
    g_naive = vf_naive.mu.grad.item()

    err_prop = abs(g_prop - g_fd) / abs(g_fd)
    err_naive = abs(g_naive - g_fd) / abs(g_fd)

    # Proposed error must be substantially smaller than naive
    assert err_prop < 0.30
    assert err_naive > err_prop * 2.0
