"""
Embedded Error Controller for Dynamic Fractional Memory Adaptation.

Implements an error-controlled mechanism analogous to adaptive Runge-Kutta step control,
but operating on the memory dimension K(t) and kernel representation \mathcal{M}(t).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import scipy.special as sp
import torch

from core.soe.dyadic_quadrature import DyadicSOEGenerator, SOEWeights
from core.soe.projection import compute_projection_matrix, apply_state_transition


@dataclass
class AdaptationEvent:
    """Records a single representation adaptation event for the adjoint solver."""
    step_idx: int
    t: float
    K_old: int
    K_new: int
    R: torch.Tensor          # Projection matrix (K_new, K_old)
    error_est: float
    event_type: str          # 'expand', 'prune', or 'reconfigure'


class MemoryErrorController:
    """
    Monitors memory truncation error and dynamically adapts K(t) and \mathcal{M}(t).
    """

    def __init__(
        self,
        beta: float,
        delta_t: float,
        T: float,
        tol: float = 1e-4,
        K_init: int = 12,
        K_min: int = 4,
        K_max: int = 48,
        delta_K: int = 4,
        prune_ratio: float = 0.05,
        patience: int = 3,
        gl_order: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            beta: Fractional order 0 < beta < 1.
            delta_t: Minimum time interval (typically dt).
            T: Total simulation horizon.
            tol: Target error tolerance \epsilon_{tol}.
            K_init: Initial number of modes.
            K_min: Minimum mode floor.
            K_max: Maximum mode ceiling.
            delta_K: Number of modes to add or remove per adaptation.
            prune_ratio: Lower relative threshold factor (\tau_{prune} = prune_ratio * tol).
            patience: Number of consecutive steps error must remain below prune threshold before pruning.
            gl_order: Order of Gauss-Legendre quadrature for generator.
            device: PyTorch device ('cpu' or 'cuda').
            dtype: PyTorch float precision (default float64).
        """
        self.beta = beta
        self.delta_t = delta_t
        self.T = T
        self.tol = tol
        self.K_min = K_min
        self.K_max = K_max
        self.delta_K = delta_K
        self.prune_tol = prune_ratio * tol
        self.patience = patience
        self.device = device
        self.dtype = dtype

        self.generator = DyadicSOEGenerator(gl_order=gl_order)

        self.current_soe = self.generator.generate(
            beta=beta, delta_t=delta_t, T=T, tol=tol, num_modes=K_init
        )
        self.current_K = len(self.current_soe.lambdas)
        self.lambdas_t, self.weights_t = self.current_soe.to_torch(dtype=dtype, device=device)

        # Higher-order shadow representation for embedded error estimation
        self.shadow_delta = max(4, delta_K)
        self.shadow_soe = self.generator.generate(
            beta=beta, delta_t=delta_t, T=T, tol=tol * 0.1, num_modes=self.current_K + self.shadow_delta
        )
        self.shadow_lambdas, self.shadow_weights = self.shadow_soe.to_torch(dtype=dtype, device=device)
        self.R_to_shadow = compute_projection_matrix(
            self.current_soe.lambdas, self.shadow_soe.lambdas
        ).to(dtype=dtype, device=device)

        # Adaptation history for adjoint backprop
        self.events: List[AdaptationEvent] = []
        self._consecutive_low_error = 0
        self._cooldown = 0
        self.active_modes_history: List[int] = []

    def compute_memory_contribution(self, m_state: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs memory contribution: M(t) = \frac{1}{\Gamma(\beta)} \sum_{k=1}^K w_k m_k(t).
        Args:
            m_state: Auxiliary state tensor of shape (..., K).
        Returns:
            Memory contribution tensor of shape (...).
        """
        # (..., K) * (K,) -> sum over last dim
        gamma_factor = 1.0 / sp.gamma(self.beta)
        return gamma_factor * torch.sum(m_state * self.weights_t, dim=-1)

    def estimate_error(
        self, m_state: torch.Tensor, z_curr: Optional[torch.Tensor] = None
    ) -> float:
        """
        Estimates local memory error using embedded shadow modes:
            \widehat{\epsilon}_M(t) = \| M_K(t) - M_{shadow}(t) \| / (\| z(t) \| + 1.0)
        """
        with torch.no_grad():
            m_curr = self.compute_memory_contribution(m_state)

            # Project current m to shadow modes to estimate what shadow representation would yield
            m_shadow_approx = apply_state_transition(m_state, self.R_to_shadow)
            gamma_factor = 1.0 / sp.gamma(self.beta)
            m_shadow = gamma_factor * torch.sum(m_shadow_approx * self.shadow_weights, dim=-1)

            diff = torch.norm(m_curr - m_shadow)
            denom = (torch.norm(z_curr) if z_curr is not None else torch.norm(m_curr)) + 1.0
            rel_error = float((diff / denom).cpu().item())
            return rel_error

    def step_adaptation(
        self,
        step_idx: int,
        t: float,
        m_state: torch.Tensor,
        z_curr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[AdaptationEvent]]:
        """
        Checks memory error at current step and reconfigures memory modes if necessary.

        Args:
            step_idx: Current discrete time step index.
            t: Current time.
            m_state: Current auxiliary states, shape (..., K_old).
            z_curr: Current physical state tensor, shape (...).

        Returns:
            (m_new, event): Updated state and AdaptationEvent (if transition occurred, else None).
        """
        self.active_modes_history.append(self.current_K)
        error_est = self.estimate_error(m_state, z_curr)

        event = None

        if self._cooldown > 0:
            self._cooldown -= 1
        elif step_idx >= 3:
            if error_est > self.tol and self.current_K < self.K_max:
                # Under-resolved: Expand memory modes
                new_K = min(self.K_max, self.current_K + self.delta_K)
                event = self._reconfigure(step_idx, t, new_K, error_est, event_type="expand")
                m_state = apply_state_transition(m_state, event.R)
                self._consecutive_low_error = 0
                self._cooldown = 6

            elif error_est < self.prune_tol and self.current_K > self.K_min:
                self._consecutive_low_error += 1
                if self._consecutive_low_error >= self.patience:
                    # Over-resolved: Prune redundant modes
                    new_K = max(self.K_min, self.current_K - self.delta_K)
                    event = self._reconfigure(step_idx, t, new_K, error_est, event_type="prune")
                    m_state = apply_state_transition(m_state, event.R)
                    self._consecutive_low_error = 0
                    self._cooldown = 8
            else:
                self._consecutive_low_error = 0

        if event is not None:
            self.events.append(event)

        return m_state, event

    def _reconfigure(
        self,
        step_idx: int,
        t: float,
        new_K: int,
        error_est: float,
        event_type: str,
    ) -> AdaptationEvent:
        """Internal helper to reconfigure SOE representation and compute transition operator."""
        old_soe = self.current_soe
        old_K = self.current_K

        # Generate new SOE configuration
        new_soe = self.generator.generate(
            beta=self.beta, delta_t=self.delta_t, T=self.T, tol=self.tol, num_modes=new_K
        )
        new_K = len(new_soe.lambdas)

        # Compute Cauchy-Gram projection matrix R: R^{old_K} -> R^{new_K}
        R = compute_projection_matrix(old_soe.lambdas, new_soe.lambdas).to(
            dtype=self.dtype, device=self.device
        )

        # Update active state
        self.current_K = new_K
        self.current_soe = new_soe
        self.lambdas_t, self.weights_t = new_soe.to_torch(dtype=self.dtype, device=self.device)

        # Update shadow modes
        self.shadow_soe = self.generator.generate(
            beta=self.beta, delta_t=self.delta_t, T=self.T, tol=self.tol * 0.1, num_modes=new_K + self.shadow_delta
        )
        self.shadow_lambdas, self.shadow_weights = self.shadow_soe.to_torch(
            dtype=self.dtype, device=self.device
        )
        self.R_to_shadow = compute_projection_matrix(
            new_soe.lambdas, self.shadow_soe.lambdas
        ).to(dtype=self.dtype, device=self.device)

        return AdaptationEvent(
            step_idx=step_idx,
            t=t,
            K_old=old_K,
            K_new=new_K,
            R=R,
            error_est=error_est,
            event_type=event_type,
        )

    @property
    def average_modes(self) -> float:
        """Calculates \\bar{K} = (1/N) \\sum K_n."""
        if not self.active_modes_history:
            return float(self.current_K)
        return float(np.mean(self.active_modes_history))

    @property
    def max_modes(self) -> int:
        """Calculates K_{\\max}."""
        if not self.active_modes_history:
            return self.current_K
        return int(np.max(self.active_modes_history))
