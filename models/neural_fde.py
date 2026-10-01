"""
Neural Fractional Differential Equation (Neural FDE) Architecture.

Combines a parameterized neural vector field f_\theta(z, t) with the AdaMem-FDE
error-adaptive solver and baseline comparison modes.
"""

from typing import Optional, Tuple, Union
import numpy as np
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
    Neural Fractional Differential Equation wrapper with optional learnable fractional order beta.
    """

    def __init__(
        self,
        vector_field: nn.Module,
        beta: float = 0.8,
        learnable_beta: bool = False,
        beta_min: float = 0.05,
        beta_max: float = 0.98,
        default_tol: float = 1e-4,
        K_init: int = 12,
        K_min: int = 4,
        K_max: int = 48,
    ):
        super().__init__()
        assert 0.0 < beta <= 1.0, f"Expected 0 < beta <= 1, got {beta}"
        assert 0.0 < beta_min < beta_max <= 1.0, f"Invalid beta bounds: [{beta_min}, {beta_max}]"
        self.vector_field = vector_field
        self.learnable_beta = learnable_beta
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.default_tol = default_tol
        self.K_init = K_init
        self.K_min = K_min
        self.K_max = K_max

        if learnable_beta:
            # Invert sigmoid mapping: beta = beta_min + (beta_max - beta_min) * sigmoid(eta)
            clipped_beta = max(min(beta, beta_max - 1e-4), beta_min + 1e-4)
            normalized = (clipped_beta - beta_min) / (beta_max - beta_min)
            init_logit = float(np.log(normalized / (1.0 - normalized)))
            self._raw_beta = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
            self._fixed_beta = None
        else:
            self.register_parameter("_raw_beta", None)
            self._fixed_beta = float(beta)

    @property
    def beta(self) -> Union[float, torch.Tensor]:
        """
        Fractional derivative order beta in (beta_min, beta_max).
        If learnable_beta=True, returns a 0-dim torch.Tensor with autograd tracking.
        If learnable_beta=False, returns a float.
        """
        if self.learnable_beta and self._raw_beta is not None:
            return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self._raw_beta)
        return self._fixed_beta

    @beta.setter
    def beta(self, value: Union[float, torch.Tensor]):
        """Sets beta, adjusting internal parameter if learnable."""
        val_float = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        if self.learnable_beta and self._raw_beta is not None:
            clipped = max(min(val_float, self.beta_max - 1e-4), self.beta_min + 1e-4)
            normalized = (clipped - self.beta_min) / (self.beta_max - self.beta_min)
            new_logit = float(np.log(normalized / (1.0 - normalized)))
            with torch.no_grad():
                self._raw_beta.copy_(
                    torch.tensor(new_logit, dtype=self._raw_beta.dtype, device=self._raw_beta.device)
                )
        else:
            self._fixed_beta = val_float

    def get_beta_value(self) -> float:
        """Returns the scalar numerical value of current beta."""
        b = self.beta
        return float(b.item()) if isinstance(b, torch.Tensor) else float(b)

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
        current_beta = self.beta

        if method == "adamem":
            return adamem_integrate(
                f=self.vector_field,
                z0=z0,
                t_grid=t_grid,
                beta=current_beta,
                tol=tol_val,
                K_init=self.K_init,
                K_min=self.K_min,
                K_max=self.K_max,
                use_adjoint_jump=True,
                parameters=tuple(p for p in self.vector_field.parameters() if p.requires_grad),
            )

        elif method == "naive_adaptive":
            # Baseline 4 Ablation: dynamic adaptation without adjoint jump
            return adamem_integrate(
                f=self.vector_field,
                z0=z0,
                t_grid=t_grid,
                beta=current_beta,
                tol=tol_val,
                K_init=self.K_init,
                K_min=self.K_min,
                K_max=self.K_max,
                use_adjoint_jump=False,
                parameters=tuple(p for p in self.vector_field.parameters() if p.requires_grad),
            )

        elif method == "fixed":
            # Baseline 2: Fixed SOE
            beta_float = float(current_beta.item()) if isinstance(current_beta, torch.Tensor) else float(current_beta)
            solver = FixedSOEFDESolver(beta=beta_float, num_modes=fixed_modes)
            return solver.solve(self.vector_field, z0, t_grid)

        elif method == "full":
            # Baseline 1: Full History
            beta_float = float(current_beta.item()) if isinstance(current_beta, torch.Tensor) else float(current_beta)
            solver = FullHistoryFDESolver(beta=beta_float)
            return solver.solve(self.vector_field, z0, t_grid)

        else:
            raise ValueError(f"Unknown integration method: '{method}'")
