"""
Neural Fractional Differential Equation (Neural FDE) Architecture.

Combines a parameterized neural vector field f_\theta(z, t) with the AdaMem-FDE
error-adaptive solver and baseline comparison modes.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from core.adjoint.adamem_adjoint import adamem_integrate
from core.solvers.fixed_soe import FixedSOEFDESolver
from core.solvers.full_history import FullHistoryFDESolver
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


class VectorFieldNetwork(nn.Module):
    """
    MLP parameterizing vector field f_\theta(z, t).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        time_dependent: bool = False,
        activation: str = "tanh",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.time_dependent = time_dependent

        in_dim = state_dim + (1 if time_dependent else 0)

        act_cls = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
        }.get(activation.lower(), nn.Tanh)

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_cls())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """Evaluates f_\theta(z, t)."""
        param_dtype = next(self.parameters()).dtype
        param_device = next(self.parameters()).device
        z_cast = z.to(dtype=param_dtype, device=param_device)

        if self.time_dependent:
            t_tensor = torch.full_like(z_cast[..., :1], fill_value=t)
            inp = torch.cat([z_cast, t_tensor], dim=-1)
        else:
            inp = z_cast

        out = self.net(inp)
        return out.to(dtype=z.dtype, device=z.device)


class NeuralFDE(nn.Module):
    """
    Neural Fractional Differential Equation wrapper.
    """

    def __init__(
        self,
        vector_field: nn.Module,
        beta: float = 0.8,
        default_tol: float = 1e-4,
        K_init: int = 12,
        K_min: int = 4,
        K_max: int = 48,
    ):
        super().__init__()
        assert 0.0 < beta <= 1.0, f"Expected 0 < beta <= 1, got {beta}"
        self.vector_field = vector_field
        self.beta = beta
        self.default_tol = default_tol
        self.K_init = K_init
        self.K_min = K_min
        self.K_max = K_max

    def forward(
        self,
        z0: torch.Tensor,
        t_grid: torch.Tensor,
        method: str = "adamem",
        tol: Optional[float] = None,
        fixed_modes: int = 16,
    ) -> torch.Tensor:
        """
        Integrates Neural FDE trajectory.

        Args:
            z0: Initial condition (..., state_dim).
            t_grid: Time grid [t_0, ..., t_N].
            method: 'adamem' (proposed), 'fixed' (fixed SOE),
                    'full' (Adams-Bashforth-Moulton reference),
                    'naive_adaptive' (ablation: adaptive without adjoint jump).
            tol: Tolerance for adaptive methods (defaults to self.default_tol).
            fixed_modes: Modes K for method='fixed'.

        Returns:
            z_trajectory: shape (N+1, ..., state_dim).
        """
        tol_val = tol or self.default_tol

        if method == "adamem":
            return adamem_integrate(
                f=self.vector_field,
                z0=z0,
                t_grid=t_grid,
                beta=self.beta,
                tol=tol_val,
                K_init=self.K_init,
                K_min=self.K_min,
                K_max=self.K_max,
                use_adjoint_jump=True,
                parameters=tuple(self.vector_field.parameters()),
            )

        elif method == "naive_adaptive":
            # Baseline 4 Ablation: dynamic adaptation without adjoint jump
            return adamem_integrate(
                f=self.vector_field,
                z0=z0,
                t_grid=t_grid,
                beta=self.beta,
                tol=tol_val,
                K_init=self.K_init,
                K_min=self.K_min,
                K_max=self.K_max,
                use_adjoint_jump=False,
                parameters=tuple(self.vector_field.parameters()),
            )

        elif method == "fixed":
            # Baseline 2: Fixed SOE
            solver = FixedSOEFDESolver(beta=self.beta, num_modes=fixed_modes)
            return solver.solve(self.vector_field, z0, t_grid)

        elif method == "full":
            # Baseline 1: Full History
            solver = FullHistoryFDESolver(beta=self.beta)
            return solver.solve(self.vector_field, z0, t_grid)

        else:
            raise ValueError(f"Unknown integration method: '{method}'")
