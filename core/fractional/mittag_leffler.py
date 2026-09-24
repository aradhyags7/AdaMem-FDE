"""
Mittag-Leffler function evaluation with high numerical precision.

Supports E_{alpha, beta}(z) for real inputs, with specialized routines for
decay dynamics z <= 0 commonly encountered in fractional relaxation benchmarks:
    {}^C D_t^\beta x(t) = -lambda x(t) => x(t) = x_0 E_beta(-lambda t^\beta)
"""

import math
from typing import Union
import numpy as np
import scipy.special as sp


def mittag_leffler(
    alpha: float,
    beta: float = 1.0,
    z: Union[float, np.ndarray] = 0.0,
    tol: float = 1e-14,
    max_terms: int = 150,
) -> Union[float, np.ndarray]:
    """
    Computes the two-parameter Mittag-Leffler function E_{alpha, beta}(z).

    Args:
        alpha: Fractional order, 0 < alpha <= 1 (or general alpha > 0).
        beta: Second parameter, typically 1.0.
        z: Scalar or NumPy array of evaluation points.
        tol: Convergence tolerance for series summation.
        max_terms: Maximum number of terms to sum.

    Returns:
        Value(s) of E_{alpha, beta}(z) matching input shape.
    """
    is_scalar = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    res = np.zeros_like(z_arr)

    # For alpha == 1, beta == 1: E_{1,1}(z) = exp(z)
    if abs(alpha - 1.0) < 1e-12 and abs(beta - 1.0) < 1e-12:
        out = np.exp(z_arr)
        return float(out[0]) if is_scalar else out

    # Partition points: small/moderate |z| vs large negative z
    # Threshold for switching between series expansion and asymptotic expansion
    r_cutoff = 18.0

    mask_series = np.abs(z_arr) <= r_cutoff
    mask_asymptotic = (z_arr < -r_cutoff)
    mask_remainder = ~mask_series & ~mask_asymptotic

    # 1. Taylor Series for |z| <= r_cutoff:
    # E_{alpha, beta}(z) = \sum_{k=0}^\infty \frac{z^k}{\Gamma(alpha * k + beta)}
    if np.any(mask_series):
        z_sub = z_arr[mask_series]
        acc = np.zeros_like(z_sub)
        for k in range(max_terms):
            gamma_val = sp.gamma(alpha * k + beta)
            if np.isinf(gamma_val) or gamma_val == 0:
                continue
            term = (z_sub ** k) / gamma_val
            acc += term
            if np.max(np.abs(term)) < tol and k > 5:
                break
        res[mask_series] = acc

    # 2. Asymptotic expansion for z < -r_cutoff (arg(z) = pi):
    # E_{alpha, beta}(z) = - \sum_{k=1}^M \frac{z^{-k}}{\Gamma(beta - alpha * k)}
    if np.any(mask_asymptotic):
        z_sub = z_arr[mask_asymptotic]
        acc = np.zeros_like(z_sub)
        for k in range(1, 20):
            gamma_val = sp.gamma(beta - alpha * k)
            if np.isinf(gamma_val) or gamma_val == 0:
                continue
            term = - (z_sub ** (-k)) / gamma_val
            acc += term
            if np.max(np.abs(term)) < tol:
                break
        res[mask_asymptotic] = acc

    # 3. For any remaining points (large positive or general complex):
    if np.any(mask_remainder):
        # Use numerical contour integral or high-order series
        z_sub = z_arr[mask_remainder]
        acc = np.zeros_like(z_sub)
        for k in range(max_terms * 2):
            gamma_val = sp.gamma(alpha * k + beta)
            term = (z_sub ** k) / gamma_val
            acc += term
            if np.max(np.abs(term)) < tol and k > 10:
                break
        res[mask_remainder] = acc

    return float(res[0]) if is_scalar else res
