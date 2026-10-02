r"""
Error-Adaptive SOE Fractional Differential Equation Solver (AdaMem-FDE).

Dynamically adapts memory representation K(t) and auxiliary states m(t)
subject to user-prescribed error tolerance \epsilon_{tol}, and logs representation
transitions R for exact adjoint gradient propagation.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import scipy.special as sp
import torch

from core.controllers.embedded_controller import MemoryErrorController, AdaptationEvent
from core.soe.dyadic_quadrature import DyadicSOEGenerator, SOEWeights
from core.soe.projection import compute_projection_matrix, apply_state_transition


@dataclass
class SolverSolution:
    """Stores full solver trajectory and adaptation history."""
    z: torch.Tensor                     # Trajectory (N+1, ..., d)
    t: torch.Tensor                     # Time grid
    m_final: torch.Tensor               # Final auxiliary memory state
    events: List[AdaptationEvent]       # Recorded adaptation transitions
    modes_history: List[int]            # Active modes K at each step
    avg_modes: float                    # \bar{K}
    max_modes: int                      # K_{max}
    num_adaptations: int                # N_{adapt}
    m_history: Optional[List[torch.Tensor]] = None # History of auxiliary states


class AdaptiveSOEFDESolver:
    """
    Error-Adaptive Dynamic Memory Solver for Neural FDEs.
    """

    def __init__(
        self,
        beta: float,
        tol: float = 1e-4,
        K_init: int = 12,
        K_min: int = 4,
        K_max: int = 48,
        delta_K: int = 4,
        prune_ratio: float = 0.05,
        patience: int = 3,
    ):
        r"""
        Args:
            beta: Fractional order 0 < beta <= 1.
            tol: Prescribed memory error tolerance \epsilon_{tol}.
            K_init: Initial number of modes.
            K_min: Minimum modes floor.
            K_max: Maximum modes ceiling.
            delta_K: Modes step size for expansion/pruning.
            prune_ratio: Relative prune threshold factor.
            patience: Step patience for pruning.
        """
        self.beta = beta
        self.tol = tol
        self.K_init = K_init
        self.K_min = K_min
        self.K_max = K_max
        self.delta_K = delta_K
        self.prune_ratio = prune_ratio
        self.patience = patience

    @staticmethod
    def _compute_etd_operators(lambdas: torch.Tensor, dt: float):
        """Precomputes exact ETD operators E, phi_1, phi_2 for given decay poles."""
        lam_dt = lambdas * dt
        E = torch.exp(-lam_dt)
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
        return E, phi_1, phi_2

    def solve(
        self,
        f: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> SolverSolution:
        """
        Integrates the FDE adaptively over t_grid while logging all representation jumps.

        Args:
            f: Right-hand side vector field f(z, t).
            z0: Initial condition, shape (..., d).
            t_grid: Uniform time grid [t_0, ..., t_N].

        Returns:
            SolverSolution containing trajectory, events, and metrics.
        """
        N = len(t_grid) - 1
        dts = t_grid[1:] - t_grid[:-1]
        dt_min = float(torch.min(dts).item())
        T_horizon = float((t_grid[-1] - t_grid[0]).item())
        is_uniform = bool(torch.allclose(dts, dts[0], rtol=1e-4, atol=1e-7))

        device = z0.device
        dtype = z0.dtype
        beta_val = float(self.beta.item()) if isinstance(self.beta, torch.Tensor) else float(self.beta)
        gamma_factor = 1.0 / sp.gamma(beta_val)

        # Initialize embedded memory controller
        controller = MemoryErrorController(
            beta=beta_val,
            delta_t=dt_min,
            T=T_horizon,
            tol=self.tol,
            K_init=self.K_init,
            K_min=self.K_min,
            K_max=self.K_max,
            delta_K=self.delta_K,
            prune_ratio=self.prune_ratio,
            patience=self.patience,
            device=str(device),
            dtype=dtype,
        )

        # Allocate trajectory
        shape = (N + 1,) + z0.shape
        z = torch.zeros(shape, dtype=dtype, device=device)
        z[0] = z0

        # Initial auxiliary memory states: m_k(0) = 0, shape (..., d, K_init)
        m_shape = z0.shape + (controller.current_K,)
        m = torch.zeros(m_shape, dtype=dtype, device=device)
        m_history = []

        # ETD operators for initial modes
        dt_0 = float(dts[0].item())
        E, phi_1, phi_2 = self._compute_etd_operators(controller.lambdas_t, dt_0)

        for n in range(N):
            t_curr = float(t_grid[n].item())
            t_next = float(t_grid[n + 1].item())
            dt_n = float(dts[n].item())

            # 1. Error check and dynamic memory adaptation at step boundary
            m, event = controller.step_adaptation(
                step_idx=n, t=t_curr, m_state=m, z_curr=z[n]
            )
            m_history.append(m.clone())

            # If adaptation occurred or mesh is non-uniform, update ETD operators
            if event is not None or not is_uniform:
                E, phi_1, phi_2 = self._compute_etd_operators(controller.lambdas_t, dt_n)

            # 2. Vector field evaluation
            w = controller.weights_t
            F_curr = f(z[n], t_curr).unsqueeze(-1)  # (..., d, 1)

            # 3. ETD-RK2 Step
            # Predictor
            m_pred = E * m + dt_n * (phi_1 * F_curr)
            z_pred = z0 + gamma_factor * torch.sum(m_pred * w, dim=-1)
            F_pred = f(z_pred, t_next).unsqueeze(-1)

            # Corrector
            m = m_pred + dt_n * (phi_2 * (F_pred - F_curr))

            # 4. State reconstruction at t_{n+1}
            z[n + 1] = z0 + gamma_factor * torch.sum(m * w, dim=-1)


        # Log final mode and auxiliary state
        controller.active_modes_history.append(controller.current_K)
        m_history.append(m.clone())

        return SolverSolution(
            z=z,
            t=t_grid,
            m_final=m,
            events=controller.events,
            modes_history=controller.active_modes_history,
            avg_modes=controller.average_modes,
            max_modes=controller.max_modes,
            num_adaptations=len(controller.events),
            m_history=m_history,
        )
