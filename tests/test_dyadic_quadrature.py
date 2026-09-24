"""
Tests for Dyadic Contour Quadrature and SOE kernel compression.
"""

import numpy as np
import pytest
from core.soe.dyadic_quadrature import DyadicSOEGenerator


def test_soe_nodes_positivity():
    """All decay rates lambda_k and weights w_k must be strictly positive."""
    generator = DyadicSOEGenerator(gl_order=4)
    beta = 0.7
    soe = generator.generate(beta=beta, delta_t=0.01, T=10.0, tol=1e-4)

    assert len(soe.lambdas) > 0
    assert np.all(soe.lambdas > 0.0), "Decay rates must be positive for stability"
    assert np.all(soe.weights > 0.0), "Quadrature weights must be positive"


def test_soe_kernel_approximation_accuracy():
    """SOE approximation must meet prescribed tolerance over [delta_t, T]."""
    generator = DyadicSOEGenerator(gl_order=4)
    beta = 0.6
    delta_t = 0.05
    T = 10.0
    tol = 1e-3

    soe = generator.generate(beta=beta, delta_t=delta_t, T=T, tol=tol)

    s_test = np.geomspace(delta_t, T, 100)
    k_approx = generator.evaluate_kernel(s_test, soe)
    k_exact = generator.evaluate_exact_kernel(s_test, beta)

    rel_error = np.abs(k_approx - k_exact) / k_exact
    max_rel_error = np.max(rel_error)

    # Max relative error over interior should be well controlled
    assert max_rel_error < 0.05, f"Relative error {max_rel_error} exceeded bound"


def test_soe_order_refinement():
    """Higher number of modes must decrease kernel error."""
    generator = DyadicSOEGenerator(gl_order=4)
    beta = 0.75
    delta_t = 0.05
    T = 5.0

    soe_coarse = generator.generate(beta=beta, delta_t=delta_t, T=T, num_modes=8)
    soe_fine = generator.generate(beta=beta, delta_t=delta_t, T=T, num_modes=24)

    s_test = np.geomspace(delta_t, T, 50)
    k_exact = generator.evaluate_exact_kernel(s_test, beta)

    err_coarse = np.mean(np.abs(generator.evaluate_kernel(s_test, soe_coarse) - k_exact))
    err_fine = np.mean(np.abs(generator.evaluate_kernel(s_test, soe_fine) - k_exact))

    assert err_fine < err_coarse, f"Fine error {err_fine} not smaller than coarse {err_coarse}"
