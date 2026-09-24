"""
Phase V: Gradient Verification and Ablation Tests.

Compares adjoint gradients computed via AdaMemAdjointFunction against
central finite differences:
    g_FD = (L(\theta + h) - L(\theta - h)) / (2h)
and validates that the proposed adjoint jump \lambda^- = R^T \lambda^+ produces
superior gradient fidelity compared to naive representation resizing (Baseline 4).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.adjoint.adamem_adjoint import adamem_integrate


class SimpleLinearField(nn.Module):
    """Linear field f(z) = -theta * z for clean gradient validation."""
    def __init__(self, init_theta: float = 1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([init_theta], dtype=torch.float64))

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        return -self.theta * z


def test_adjoint_gradient_vs_finite_difference():
    """
    Validates adjoint gradient against two-sided finite difference.
    At discrete step dt = 0.025, continuous adjoint error scales as O(dt) ~ 0.05.
    """
    beta = 0.8
    t_grid = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)

    # Synthetic target
    z_target = torch.exp(-0.8 * t_grid).unsqueeze(-1)

    model = SimpleLinearField(init_theta=1.0)

    # Forward + Adjoint backward
    z_traj = adamem_integrate(
        f=model,
        z0=z0,
        t_grid=t_grid,
        beta=beta,
        tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
        use_adjoint_jump=True,
        parameters=tuple(model.parameters()),
    )
    loss = 0.5 * torch.sum((z_traj - z_target) ** 2)
    loss.backward()
    grad_adj = model.theta.grad.clone().item()

    # Two-sided Central Finite Difference
    h = 1e-6
    with torch.no_grad():
        model.theta.data.fill_(1.0 + h)
        z_plus = adamem_integrate(
            f=model,
            z0=z0,
            t_grid=t_grid,
            beta=beta,
            tol=1e-3,
            K_init=8,
            K_min=4,
            K_max=24,
            use_adjoint_jump=True,
        )
        loss_plus = 0.5 * torch.sum((z_plus - z_target) ** 2).item()

        model.theta.data.fill_(1.0 - h)
        z_minus = adamem_integrate(
            f=model,
            z0=z0,
            t_grid=t_grid,
            beta=beta,
            tol=1e-3,
            K_init=8,
            K_min=4,
            K_max=24,
            use_adjoint_jump=True,
        )
        loss_minus = 0.5 * torch.sum((z_minus - z_target) ** 2).item()

    grad_fd = (loss_plus - loss_minus) / (2.0 * h)
    rel_error = abs(grad_adj - grad_fd) / max(abs(grad_fd), 1e-7)

    print(f"\n[Gradient Check] Adjoint: {grad_adj:.6f}, FD: {grad_fd:.6f}, Rel Error: {rel_error:.6e}")
    assert rel_error < 0.08, f"Gradient relative error {rel_error} exceeds discretization tolerance"


def test_ablation_adjoint_jump_superiority():
    """
    Ablation Study C (Section 18):
    Verifies that AdaMem-FDE with R^T adjoint jump achieves significantly lower
    relative gradient error than naive resizing without the jump operator (Baseline 4).
    """
    beta = 0.8
    t_grid = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)
    z_target = torch.exp(-0.8 * t_grid).unsqueeze(-1)

    # 1. Proposed with R^T
    m_prop = SimpleLinearField(1.0)
    z_prop = adamem_integrate(
        m_prop, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24,
        use_adjoint_jump=True, parameters=tuple(m_prop.parameters())
    )
    (0.5 * torch.sum((z_prop - z_target) ** 2)).backward()
    grad_prop = m_prop.theta.grad.item()

    # 2. Baseline 4 ablation without R^T
    m_naive = SimpleLinearField(1.0)
    z_naive = adamem_integrate(
        m_naive, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24,
        use_adjoint_jump=False, parameters=tuple(m_naive.parameters())
    )
    (0.5 * torch.sum((z_naive - z_target) ** 2)).backward()
    grad_naive = m_naive.theta.grad.item()

    # 3. Reference FD
    h = 1e-6
    with torch.no_grad():
        m_ref = SimpleLinearField(1.0 + h)
        zp = adamem_integrate(m_ref, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
        lp = 0.5 * torch.sum((zp - z_target) ** 2).item()
        m_ref.theta.data.fill_(1.0 - h)
        zm = adamem_integrate(m_ref, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
        lm = 0.5 * torch.sum((zm - z_target) ** 2).item()
    grad_ref = (lp - lm) / (2 * h)

    err_prop = abs(grad_prop - grad_ref) / abs(grad_ref)
    err_naive = abs(grad_naive - grad_ref) / abs(grad_ref)

    print(f"\n[Ablation C] Proposed Error: {err_prop:.4e}, Naive Error: {err_naive:.4e}")
    assert err_prop < err_naive, (
        f"Proposed adjoint jump error ({err_prop}) was expected to be lower than naive ({err_naive})"
    )
