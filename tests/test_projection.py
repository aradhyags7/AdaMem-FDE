"""
Tests for state transition map R and adjoint jump R^T.
"""

import numpy as np
import pytest
import torch
from core.soe.dyadic_quadrature import DyadicSOEGenerator
from core.soe.projection import (
    compute_projection_matrix,
    apply_state_transition,
    apply_adjoint_transition,
)


def test_projection_matrix_shape():
    """Projection matrix R must have shape (K_new, K_old)."""
    generator = DyadicSOEGenerator()
    soe_old = generator.generate(beta=0.7, delta_t=0.01, T=5.0, num_modes=8)
    soe_new = generator.generate(beta=0.7, delta_t=0.01, T=5.0, num_modes=16)

    R = compute_projection_matrix(soe_old.lambdas, soe_new.lambdas)
    assert R.shape == (16, 8)
    assert not torch.isnan(R).any()
    assert not torch.isinf(R).any()


def test_adjoint_inner_product_identity():
    """
    Validates fundamental adjoint property:
        < \lambda^+, m^+ > = < \lambda^+, R m^- > = < R^T \lambda^+, m^- > = < \lambda^-, m^- >
    Must hold to numerical machine precision.
    """
    torch.manual_seed(42)
    K_old = 10
    K_new = 18

    generator = DyadicSOEGenerator()
    soe_old = generator.generate(beta=0.75, delta_t=0.01, T=5.0, num_modes=K_old)
    soe_new = generator.generate(beta=0.75, delta_t=0.01, T=5.0, num_modes=K_new)

    K_old_actual = len(soe_old.lambdas)
    K_new_actual = len(soe_new.lambdas)

    R = compute_projection_matrix(soe_old.lambdas, soe_new.lambdas)

    # Random test vectors matching actual modes
    m_old = torch.randn(K_old_actual, dtype=torch.float64)
    lambda_new = torch.randn(K_new_actual, dtype=torch.float64)

    # Forward map: m^+ = R m^-
    m_new = apply_state_transition(m_old, R)

    # Adjoint map: \lambda^- = R^T \lambda^+
    lambda_old = apply_adjoint_transition(lambda_new, R)

    # Inner products
    ip_new = torch.dot(lambda_new, m_new)
    ip_old = torch.dot(lambda_old, m_old)

    diff = torch.abs(ip_new - ip_old)
    assert diff < 1e-12, f"Adjoint duality gap {diff.item()} exceeds machine precision"


def test_identity_projection():
    """Projecting onto identical modes should yield identity matrix."""
    generator = DyadicSOEGenerator()
    soe = generator.generate(beta=0.7, delta_t=0.01, T=5.0, num_modes=12)

    R = compute_projection_matrix(soe.lambdas, soe.lambdas, regularization=1e-10)
    eye = torch.eye(12, dtype=torch.float64)
    assert torch.allclose(R, eye, atol=1e-4)
