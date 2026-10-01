"""
Unit Tests for Learnable Fractional Order (Beta) Sensitivity and Recovery.

Validates:
1. Parameter registration and unconstrained logit mapping in NeuralFDE.
2. Adjoint gradient propagation to self._raw_beta.
3. Parameter recovery: Convergence from beta_0 = 0.50 to true beta* = 0.75.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.neural_fde import NeuralFDE, VectorFieldNetwork
from core.adjoint.adamem_adjoint import adamem_integrate


class SyntheticDecayField(nn.Module):
    """Simple linear vector field f(z) = theta * z for recovery tests."""
    def __init__(self, init_theta: float = -1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([init_theta], dtype=torch.float64))

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        return self.theta * z


def test_learnable_beta_parameter_registration():
    """Verifies that _raw_beta is properly registered as an nn.Parameter when learnable_beta=True."""
    vf = VectorFieldNetwork(state_dim=1, hidden_dim=16)
    
    # Non-learnable model
    fixed_model = NeuralFDE(vf, beta=0.75, learnable_beta=False)
    param_names_fixed = [name for name, _ in fixed_model.named_parameters()]
    assert "_raw_beta" not in param_names_fixed
    assert abs(fixed_model.get_beta_value() - 0.75) < 1e-5

    # Learnable model
    learn_model = NeuralFDE(vf, beta=0.75, learnable_beta=True, beta_min=0.05, beta_max=0.98)
    param_names_learn = [name for name, _ in learn_model.named_parameters()]
    assert "_raw_beta" in param_names_learn
    assert abs(learn_model.get_beta_value() - 0.75) < 1e-4

    # Test setter
    learn_model.beta = 0.60
    assert abs(learn_model.get_beta_value() - 0.60) < 1e-4


def test_learnable_beta_gradient_flow():
    """Verifies backward gradient flow through custom adjoint into _raw_beta."""
    vf = VectorFieldNetwork(state_dim=1, hidden_dim=16)
    model = NeuralFDE(vf, beta=0.65, learnable_beta=True, default_tol=1e-3, K_init=8, K_min=4, K_max=24)

    z0 = torch.tensor([1.0], dtype=torch.float32)
    t_grid = torch.linspace(0.0, 1.0, 21, dtype=torch.float32)

    traj = model(z0, t_grid)
    loss = torch.sum((traj - 0.5) ** 2)
    loss.backward()

    assert model._raw_beta.grad is not None, "Expected _raw_beta.grad to be populated"
    grad_val = model._raw_beta.grad.item()
    assert not np.isnan(grad_val), "Gradient must not be NaN"
    assert not np.isinf(grad_val), "Gradient must not be Infinite"
    assert abs(grad_val) > 1e-6, f"Expected non-zero gradient, got {grad_val}"


def test_learnable_beta_convergence_synthetic():
    """
    Validates parameter recovery:
    Generates synthetic trajectory with true beta* = 0.75 and theta* = -1.2.
    Initializes model at beta_0 = 0.50 and theta_0 = -0.80.
    Verifies that Adam optimizer recovers beta within 0.03 tolerance in <= 35 steps.
    """
    torch.manual_seed(42)
    t_grid = torch.linspace(0.0, 1.5, 31, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)

    # 1. Ground truth rollout with beta* = 0.75, theta* = -1.2
    true_field = SyntheticDecayField(init_theta=-1.2)
    with torch.no_grad():
        true_traj = adamem_integrate(
            true_field, z0, t_grid, beta=0.75, tol=1e-3, K_init=8, K_min=4, K_max=24
        )

    # 2. Learnable model initialized at beta_0 = 0.50, theta_0 = -0.8
    test_field = SyntheticDecayField(init_theta=-0.8)
    model = NeuralFDE(
        test_field,
        beta=0.50,
        learnable_beta=True,
        default_tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    initial_loss = None
    final_loss = None
    for step in range(35):
        optimizer.zero_grad()
        traj = model(z0, t_grid)
        loss = torch.mean((traj - true_traj) ** 2)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    recovered_beta = model.get_beta_value()
    beta_error = abs(recovered_beta - 0.75)
    print(f"\n[Recovery] Initial Loss: {initial_loss:.4e} -> Final Loss: {final_loss:.4e}")
    print(f"[Recovery] Target beta: 0.7500, Recovered: {recovered_beta:.4f}, Error: {beta_error:.4f}")

    assert beta_error < 0.03, f"Recovered beta {recovered_beta:.4f} is not within 0.03 of 0.75"
    assert final_loss < 0.05 * initial_loss, f"Expected significant loss reduction, got {final_loss:.4e} from {initial_loss:.4e}"
