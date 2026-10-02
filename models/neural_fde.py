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
from core.solvers.incommensurate_soe import IncommensurateSOEFDESolver, incommensurate_adamem_integrate


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
    r"""
    Neural Fractional Differential Equation wrapper with optional learnable fractional order beta.
    Supports both scalar commensurate beta and multi-order incommensurate vector \vec{\beta} \in (0, 1)^d.
    """

    def __init__(
        self,
        vector_field: nn.Module,
        beta: Union[float, torch.Tensor, list, np.ndarray] = 0.8,
        learnable_beta: bool = False,
        beta_min: float = 0.05,
        beta_max: float = 0.98,
        default_tol: float = 1e-4,
        K_init: int = 12,
        K_min: int = 4,
        K_max: int = 48,
    ):
        super().__init__()
        assert 0.0 < beta_min < beta_max <= 1.0, f"Invalid beta bounds: [{beta_min}, {beta_max}]"
        self.vector_field = vector_field
        self.learnable_beta = learnable_beta
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.default_tol = default_tol
        self.K_init = K_init
        self.K_min = K_min
        self.K_max = K_max

        # Detect incommensurate vector beta
        is_vec = False
        if isinstance(beta, (list, tuple, np.ndarray)):
            b_arr = np.asarray(beta, dtype=np.float32)
            if b_arr.ndim == 1 and len(b_arr) > 1:
                is_vec = True
        elif isinstance(beta, torch.Tensor) and beta.ndim == 1 and len(beta) > 1:
            is_vec = True
            b_arr = beta.detach().cpu().numpy().astype(np.float32)

        self.is_incommensurate = is_vec

        if is_vec:
            assert np.all((b_arr > 0.0) & (b_arr <= 1.0)), f"All beta elements must be in (0, 1], got {b_arr}"
            self.d = len(b_arr)
            if learnable_beta:
                clipped = np.clip(b_arr, beta_min + 1e-4, beta_max - 1e-4)
                norm = (clipped - beta_min) / (beta_max - beta_min)
                logits = np.log(norm / (1.0 - norm))
                self._raw_beta = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
                self._fixed_beta = None
            else:
                self.register_parameter("_raw_beta", None)
                self._fixed_beta = torch.tensor(b_arr, dtype=torch.float32)
        else:
            b_val = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
            assert 0.0 < b_val <= 1.0, f"Expected 0 < beta <= 1, got {b_val}"
            self.d = 1
            if learnable_beta:
                clipped_beta = max(min(b_val, beta_max - 1e-4), beta_min + 1e-4)
                normalized = (clipped_beta - beta_min) / (beta_max - beta_min)
                init_logit = float(np.log(normalized / (1.0 - normalized)))
                self._raw_beta = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                self._fixed_beta = None
            else:
                self.register_parameter("_raw_beta", None)
                self._fixed_beta = float(b_val)

    @property
    def beta(self) -> Union[float, torch.Tensor]:
        """
        Fractional derivative order beta in (beta_min, beta_max).
        If learnable_beta=True, returns torch.Tensor with autograd tracking.
        If learnable_beta=False and scalar, returns float.
        """
        if self.learnable_beta and self._raw_beta is not None:
            return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self._raw_beta)
        return self._fixed_beta

    @beta.setter
    def beta(self, value: Union[float, torch.Tensor, list, np.ndarray]):
        """Sets beta, adjusting internal parameter if learnable."""
        if self.is_incommensurate:
            if isinstance(value, torch.Tensor):
                val_np = value.detach().cpu().numpy().astype(np.float32)
            else:
                val_np = np.asarray(value, dtype=np.float32)
            if self.learnable_beta and self._raw_beta is not None:
                clipped = np.clip(val_np, self.beta_min + 1e-4, self.beta_max - 1e-4)
                norm = (clipped - self.beta_min) / (self.beta_max - self.beta_min)
                new_logits = np.log(norm / (1.0 - norm))
                with torch.no_grad():
                    self._raw_beta.copy_(torch.tensor(new_logits, dtype=self._raw_beta.dtype, device=self._raw_beta.device))
            else:
                self._fixed_beta = torch.tensor(val_np, dtype=torch.float32)
        else:
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

    def get_beta_value(self) -> Union[float, np.ndarray]:
        """Returns the scalar or array numerical value of current beta."""
        b = self.beta
        if self.is_incommensurate:
            return b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
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
            t_grid: Time grid [t_0, ..., t_N] (uniform or graded).
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

        # If incommensurate multi-order system
        if self.is_incommensurate:
            beta_vec = current_beta if isinstance(current_beta, torch.Tensor) else torch.tensor(current_beta, dtype=z0.dtype, device=z0.device)
            if method in ("adamem", "naive_adaptive"):
                return incommensurate_adamem_integrate(
                    f=self.vector_field,
                    z0=z0,
                    t_grid=t_grid,
                    beta_vec=beta_vec,
                    num_modes=self.K_init,
                    parameters=tuple(p for p in self.vector_field.parameters() if p.requires_grad),
                )
            elif method == "fixed":
                solver = IncommensurateSOEFDESolver(beta=beta_vec, num_modes=fixed_modes)
                return solver.solve(self.vector_field, z0, t_grid)
            else:
                raise NotImplementedError(f"Method '{method}' is not supported for incommensurate systems.")

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

