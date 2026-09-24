"""
Fixed-Order SOE Fractional Differential Equation Solver (Baseline 2).

Integrates the augmented memory state system:
    \dot{m}_k(t) = -\lambda_k m_k(t) + f(z(t), t), \quad k = 1, \dots, K
with algebraic state reconstruction:
    z(t) = z_0 + \frac{1}{\Gamma(\beta)} \sum_{k=1}^K w_k m_k(t)

Uses second-order Exponential Time Differencing (ETD-RK2) for exact linear decay
and unconditional numerical stability across stiff SOE decay poles \lambda_k.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import scipy.special as sp
import torch

from core.soe.dyadic_quadrature import DyadicSOEGenerator, SOEWeights


class FixedSOEFDESolver:
    """
    Fixed-Order Sum-of-Exponentials FDE Solver with ETD-RK2 integration.
    """

    def __init__(
        self,
        beta: float,
        num_modes: int = 16,
        delta_t: Optional[float] = None,
        T: Optional[float] = None,
        soe_weights: Optional[SOEWeights] = None,
    ):
        """
        Args:
            beta: Fractional order 0 < beta <= 1.
            num_modes: Number of memory modes K.
            delta_t: Smallest time step (if generating weights).
            T: Horizon (if generating weights).
            soe_weights: Pre-computed SOEWeights instance (optional).
        """
        self.beta = beta
        self.num_modes = num_modes
        self.generator = DyadicSOEGenerator()

        self.soe_weights = soe_weights
        self.delta_t = delta_t
        self.T = T

    def _ensure_weights(self, dt: float, T_horizon: float, dtype: torch.dtype, device: torch.device):
        if self.soe_weights is None:
            dt_ref = self.delta_t or dt
            T_ref = self.T or T_horizon
            self.soe_weights = self.generator.generate(
                beta=self.beta, delta_t=dt_ref, T=T_ref, num_modes=self.num_modes
            )
        self.lambdas_t, self.weights_t = self.soe_weights.to_torch(dtype=dtype, device=device)

    def solve(
        self,
        f: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrates the FDE using fixed K memory modes over t_grid.

        Args:
            f: Right-hand side f(z, t).
            z0: Initial state, shape (..., d).
            t_grid: Uniform time grid [t_0, ..., t_N].

        Returns:
            z_trajectory: State trajectory of shape (N+1, ..., d).
        """
        N = len(t_grid) - 1
        dt = float((t_grid[1] - t_grid[0]).item())
        T_horizon = float((t_grid[-1] - t_grid[0]).item())

        device = z0.device
        dtype = z0.dtype
        gamma_factor = 1.0 / sp.gamma(self.beta)

        self._ensure_weights(dt, T_horizon, dtype, device)
        K = self.num_modes

        # Precompute ETD operators:
        # For dot{m}_k = -\lambda_k m_k + F:
        # E = exp(-\lambda_k * dt)
        # phi_1 = (1 - exp(-\lambda_k * dt)) / (\lambda_k * dt)
        # phi_2 = (exp(-\lambda_k * dt) - 1 + \lambda_k * dt) / ((\lambda_k * dt)^2)
        lam = self.lambdas_t  # (K,)
        lam_dt = lam * dt
        E = torch.exp(-lam_dt)  # (K,)

        # Stable calculation of phi_1 and phi_2 near zero
        small_mask = lam_dt < 1e-4
        phi_1 = torch.where(
            small_mask,
            1.0 - 0.5 * lam_dt + (1.0 / 6.0) * (lam_dt ** 2),
            (1.0 - E) / lam_dt,
        )
        phi_2 = torch.where(
            small_mask,
            0.5 - (1.0 / 6.0) * lam_dt + (1.0 / 24.0) * (lam_dt ** 2),
            (E - 1.0 + lam_dt) / (lam_dt ** 2),
        )

        # Allocate trajectory
        shape = (N + 1,) + z0.shape
        z = torch.zeros(shape, dtype=dtype, device=device)
        z[0] = z0

        # Auxiliary memory state m_k: shape (..., d, K)
        # Initial condition m_k(0) = 0
        m_shape = z0.shape + (K,)
        m = torch.zeros(m_shape, dtype=dtype, device=device)

        w = self.weights_t  # (K,)

        for n in range(N):
            t_curr = float(t_grid[n].item())
            t_next = float(t_grid[n + 1].item())

            # Evaluate f(z(t_n), t_n)
            F_curr = f(z[n], t_curr)  # shape (..., d)
            # Unsqueeze to broadcast over K modes: (..., d, 1)
            F_curr_exp = F_curr.unsqueeze(-1)

            # Stage 1: Predictor for m_{n+1}
            # a = E * m_n + dt * phi_1 * F_curr
            m_pred = E * m + dt * (phi_1 * F_curr_exp)

            # Evaluate state at predictor
            # z_pred = z0 + gamma_factor * sum_{k=1}^K w_k * m_pred_k
            z_pred = z0 + gamma_factor * torch.sum(m_pred * w, dim=-1)
            F_pred = f(z_pred, t_next).unsqueeze(-1)

            # Stage 2: ETD-RK2 Corrector
            # m_{n+1} = a + dt * phi_2 * (F_pred - F_curr)
            m = m_pred + dt * (phi_2 * (F_pred - F_curr_exp))

            # Reconstruct exact state at t_{n+1}
            z[n + 1] = z0 + gamma_factor * torch.sum(m * w, dim=-1)

        return z
