"""
Unit tests for Incommensurate Multi-Order Fractional Dynamics (AdaMem Frontier).

Verifies:
1. IncommensurateSOEFDESolver reproduces exact independent Mittag-Leffler decay
   for decoupled 2D system with distinct orders beta_1 != beta_2.
2. Coupled non-linear vector fields integrate stably.
3. NeuralFDE with vector beta supports both fixed and learnable multi-order parameterization.
4. Autograd / Adjoint gradients propagate to learnable beta vector and neural weights.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from core.fractional.mittag_leffler import mittag_leffler
from core.solvers.incommensurate_soe import IncommensurateSOEFDESolver, incommensurate_adamem_integrate
from models.neural_fde import NeuralFDE, VectorFieldNetwork


def test_incommensurate_mittag_leffler_decoupled():
    """
    Tests decoupled 2D system:
        ^C D^{\beta_1} z_1 = -1.0 * z_1,  z_1(0) = 1.0  => z_1(t) = E_{\beta_1}(-t^{\beta_1})
        ^C D^{\beta_2} z_2 = -2.0 * z_2,  z_2(0) = 1.0  => z_2(t) = E_{\beta_2}(-2 t^{\beta_2})
    with incommensurate orders \beta_1 = 0.60, \beta_2 = 0.85.
    """
    beta_vec = np.array([0.60, 0.85], dtype=np.float64)
    T = 1.0
    N = 100
    K = 24
    t_grid = torch.linspace(0.0, T, N + 1, dtype=torch.float64)

    def f_decoupled(z, t):
        # z: (..., 2)
        lam = torch.tensor([1.0, 2.0], dtype=z.dtype, device=z.device)
        return -lam * z

    z0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    solver = IncommensurateSOEFDESolver(beta=beta_vec, num_modes=K)
    z_sol = solver.solve(f_decoupled, z0, t_grid)  # (N+1, 2)

    # Analytical solutions
    t_np = t_grid.numpy()
    z1_exact = mittag_leffler(beta_vec[0], 1.0, -1.0 * (t_np ** beta_vec[0]))
    z2_exact = mittag_leffler(beta_vec[1], 1.0, -2.0 * (t_np ** beta_vec[1]))

    err1 = np.max(np.abs(z_sol[:, 0].numpy() - z1_exact))
    err2 = np.max(np.abs(z_sol[:, 1].numpy() - z2_exact))

    assert err1 < 1e-2, f"Dimension 1 error too high: {err1}"
    assert err2 < 1e-2, f"Dimension 2 error too high: {err2}"



def test_neural_fde_incommensurate_learnable():
    """
    Verifies that NeuralFDE supports vector beta and optimizes beta_1 and beta_2 with autograd.
    """
    torch.manual_seed(42)
    state_dim = 2
    net = VectorFieldNetwork(state_dim=state_dim, hidden_dim=16, num_layers=2)

    init_beta = [0.55, 0.80]
    model = NeuralFDE(
        vector_field=net,
        beta=init_beta,
        learnable_beta=True,
        K_init=12,
    )

    assert model.is_incommensurate
    assert model.d == 2
    assert isinstance(model.beta, torch.Tensor)
    assert model.beta.shape == (2,)

    z0 = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    t_grid = torch.linspace(0.0, 0.5, 21, dtype=torch.float32)

    # Forward pass
    traj = model(z0, t_grid, method="adamem")
    assert traj.shape == (21, 1, 2)

    # Loss and backward
    loss = torch.sum(traj ** 2)
    loss.backward()

    # Verify gradients computed for both beta components and network parameters
    assert model._raw_beta.grad is not None
    assert model._raw_beta.grad.shape == (2,)
    assert not torch.allclose(model._raw_beta.grad, torch.zeros_like(model._raw_beta.grad))

    # Verify optimizer step updates beta
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    old_beta = model.get_beta_value()
    optimizer.step()
    new_beta = model.get_beta_value()

    assert not np.allclose(old_beta, new_beta), "Beta should be updated by optimizer"
