"""
Full-History Fractional Differential Equation Solver (Diethelm Predictor-Corrector).

Implements the classical Adams-Bashforth-Moulton PECE (Predict, Evaluate, Correct, Evaluate)
method for Caputo FDEs:
    {}^C D_t^\beta z(t) = f(z(t), t), \quad z(0) = z_0, \quad 0 < \beta \le 1

Complexity:
    Time: O(N^2)
    Memory: O(N * dim(z))
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.special as sp
import torch


class FullHistoryFDESolver:
    """
    Gold-standard O(N^2) Fractional Adams-Bashforth-Moulton Predictor-Corrector Solver.
    """

    def __init__(self, beta: float):
        """
        Args:
            beta: Fractional order 0 < beta <= 1.
        """
        assert 0.0 < beta <= 1.0, f"Expected 0 < beta <= 1, got {beta}"
        self.beta = beta

    def solve(
        self,
        f: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrates the FDE over t_grid using the Diethelm Adams-Bashforth-Moulton scheme.

        Args:
            f: Right-hand side vector field f(z, t), returning tensor of same shape as z.
            z0: Initial state tensor, shape (...).
            t_grid: 1D strictly increasing uniform time grid [t_0, ..., t_N].

        Returns:
            z_trajectory: Trajectory tensor of shape (N+1, ...).
        """
        N = len(t_grid) - 1
        dt = float((t_grid[1] - t_grid[0]).item())
        beta = self.beta

        # Precompute common factors
        gamma_beta = sp.gamma(beta)
        gamma_beta1 = sp.gamma(beta + 1.0)
        gamma_beta2 = sp.gamma(beta + 2.0)

        dt_beta = dt ** beta
        coeff_p = dt_beta / gamma_beta1
        coeff_c = dt_beta / gamma_beta2

        device = z0.device
        dtype = z0.dtype

        # Pre-allocate trajectory and evaluation history
        shape = (N + 1,) + z0.shape
        z = torch.zeros(shape, dtype=dtype, device=device)
        f_evals = torch.zeros(shape, dtype=dtype, device=device)

        z[0] = z0
        f_evals[0] = f(z0, float(t_grid[0].item()))

        # Precompute weights table for corrector
        # a_{j, n+1} weights
        for n in range(N):
            t_next = float(t_grid[n + 1].item())

            # 1. Predictor step (Fractional Adams-Bashforth)
            # b_{j, n+1} = (n + 1 - j)^\beta - (n - j)^\beta, for j = 0, ..., n
            j_arr = np.arange(n + 1, dtype=np.float64)
            b = (n + 1.0 - j_arr) ** beta - (n - j_arr) ** beta
            b_tensor = torch.tensor(b, dtype=dtype, device=device)
            # Reshape for broadcasting
            b_shape = [n + 1] + [1] * (z0.ndim)
            b_tensor = b_tensor.reshape(b_shape)

            predictor_sum = torch.sum(b_tensor * f_evals[: n + 1], dim=0)
            z_pred = z0 + coeff_p * predictor_sum

            # Evaluate at predicted point
            f_pred = f(z_pred, t_next)

            # 2. Corrector step (Fractional Adams-Moulton)
            # a_{0, n+1} = n^{\beta+1} - (n - \beta) * (n+1)^\beta
            # a_{j, n+1} = (n - j + 2)^{\beta+1} - 2*(n - j + 1)^{\beta+1} + (n - j)^{\beta+1}, j = 1, ..., n
            # a_{n+1, n+1} = 1
            if n == 0:
                a_0 = 1.0
                a_weights = np.array([a_0], dtype=np.float64)
            else:
                j_mid = np.arange(1, n + 1, dtype=np.float64)
                a_mid = (
                    (n - j_mid + 2.0) ** (beta + 1.0)
                    - 2.0 * (n - j_mid + 1.0) ** (beta + 1.0)
                    + (n - j_mid) ** (beta + 1.0)
                )
                a_0 = (n) ** (beta + 1.0) - (n - beta) * ((n + 1.0) ** beta)
                a_weights = np.concatenate([[a_0], a_mid])

            a_tensor = torch.tensor(a_weights, dtype=dtype, device=device)
            a_shape = [n + 1] + [1] * (z0.ndim)
            a_tensor = a_tensor.reshape(a_shape)

            corrector_history = torch.sum(a_tensor * f_evals[: n + 1], dim=0)
            z_corr = z0 + coeff_c * (corrector_history + 1.0 * f_pred)

            # Update state
            z[n + 1] = z_corr
            f_evals[n + 1] = f(z_corr, t_next)

        return z
