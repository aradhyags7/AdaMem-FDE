"""
Tests for Mittag-Leffler function implementation.
"""

import numpy as np
import pytest
from core.fractional.mittag_leffler import mittag_leffler


def test_mittag_leffler_exponential():
    """For alpha=1, beta=1, E_{1,1}(z) must match exp(z)."""
    z_test = np.linspace(-10.0, 2.0, 50)
    ml_vals = mittag_leffler(alpha=1.0, beta=1.0, z=z_test)
    exact_vals = np.exp(z_test)
    assert np.allclose(ml_vals, exact_vals, rtol=1e-12, atol=1e-12)


def test_mittag_leffler_zero():
    """E_{alpha, beta}(0) = 1 / Gamma(beta)."""
    import scipy.special as sp
    for alpha in [0.3, 0.5, 0.8, 1.0]:
        val = mittag_leffler(alpha=alpha, beta=1.0, z=0.0)
        assert np.isclose(val, 1.0, atol=1e-12)


def test_mittag_leffler_half_analytical():
    """For alpha=0.5, beta=1: E_{0.5, 1}(z) = exp(z^2) * erfc(-z)."""
    import scipy.special as sp
    z = -1.5
    val = mittag_leffler(alpha=0.5, beta=1.0, z=z)
    expected = np.exp(z**2) * sp.erfc(-z)
    assert np.isclose(val, expected, rtol=1e-4)


def test_mittag_leffler_vectorized():
    """Ensures input shapes are preserved."""
    z = np.array([[-1.0, -2.0], [-5.0, -10.0]])
    out = mittag_leffler(alpha=0.7, beta=1.0, z=z)
    assert out.shape == z.shape
    assert np.all(out > 0.0)  # Monotonic positive decay for real negative inputs
