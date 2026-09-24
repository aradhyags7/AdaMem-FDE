"""
Standardized benchmark dynamical systems for Phases I, II, IV, VI.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from core.fractional.mittag_leffler import mittag_leffler


@dataclass
class MittagLefflerDecay:
    """
    Phase I Benchmark: Linear fractional decay with closed-form analytical solution.
        {}^C D_t^\beta x(t) = -\lambda x(t), \quad x(0) = x_0
        Exact: x(t) = x_0 * E_\beta(-\lambda t^\beta)
    """
    beta: float = 0.7
    decay_rate: float = 1.0
    x0: float = 1.0

    def rhs(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """Right-hand side f(x, t) = -\lambda x."""
        return -self.decay_rate * z

    def exact_solution(self, t_grid: torch.Tensor) -> torch.Tensor:
        """Computes exact analytical solution using Mittag-Leffler function."""
        t_np = t_grid.detach().cpu().numpy()
        arg = -self.decay_rate * (t_np ** self.beta)
        ml_vals = mittag_leffler(alpha=self.beta, beta=1.0, z=arg)
        x_exact = self.x0 * ml_vals
        return torch.tensor(x_exact, dtype=t_grid.dtype, device=t_grid.device)


@dataclass
class FractionalDuffing:
    """
    Nonlinear fractional Duffing oscillator with hereditary memory:
        {}^C D_t^\beta x_1 = x_2
        {}^C D_t^\beta x_2 = -\delta x_2 - \alpha x_1 - \mu x_1^3 + \gamma \cos(\omega t)
    """
    beta: float = 0.85
    alpha: float = -1.0
    mu: float = 1.0
    delta: float = 0.2
    gamma: float = 0.3
    omega: float = 1.0

    def rhs(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """
        z has shape (..., 2) with z[..., 0] = x_1, z[..., 1] = x_2.
        """
        x1 = z[..., 0]
        x2 = z[..., 1]

        dx1 = x2
        forcing = self.gamma * float(np.cos(self.omega * t))
        dx2 = -self.delta * x2 - self.alpha * x1 - self.mu * (x1 ** 3) + forcing

        return torch.stack([dx1, dx2], dim=-1)


@dataclass
class FractionalVanDerPol:
    """
    Nonlinear fractional Van der Pol oscillator:
        {}^C D_t^\beta x_1 = x_2
        {}^C D_t^\beta x_2 = \mu (1 - x_1^2) x_2 - x_1
    """
    beta: float = 0.8
    mu: float = 1.2

    def rhs(self, z: torch.Tensor, t: float) -> torch.Tensor:
        x1 = z[..., 0]
        x2 = z[..., 1]

        dx1 = x2
        dx2 = self.mu * (1.0 - x1 ** 2) * x2 - x1
        return torch.stack([dx1, dx2], dim=-1)


@dataclass
class FractionalLorenz:
    """
    Phase VI Benchmark: Fractional Lorenz chaotic attractor for long-horizon scalability.
        {}^C D_t^\beta x = \sigma (y - x)
        {}^C D_t^\beta y = x (\rho - z) - y
        {}^C D_t^\beta z = x y - \beta_L z
    """
    beta: float = 0.99
    sigma: float = 10.0
    rho: float = 28.0
    beta_L: float = 8.0 / 3.0

    def rhs(self, state: torch.Tensor, t: float) -> torch.Tensor:
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta_L * z
        return torch.stack([dx, dy, dz], dim=-1)
