"""
Custom PyTorch Autograd Adjoint Solver with Dynamic Memory Representation Transitions.

Implements backward adjoint ODE propagation for AdaMem-FDE:
Between transitions:
    \dot{a}_{m, k}(t) = \lambda_k a_{m, k}(t) - \frac{w_k}{\Gamma(\beta)} \left(\frac{\partial f_\theta}{\partial z}\right)^T \sum_{j=1}^K a_{m, j}(t)
    \frac{d\mathcal{L}}{d\theta} = - \int_0^T \left(\sum_{j=1}^K a_{m, j}(t)\right)^T \frac{\partial f_\theta}{\partial \theta} dt

Across transitions (at recorded event boundaries t_j):
    a_m(t_j^-) = R_j^T a_m(t_j^+)
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
import scipy.special as sp
import torch

from core.solvers.adaptive_soe import AdaptiveSOEFDESolver, SolverSolution
from core.soe.projection import apply_adjoint_transition


class AdaMemAdjointFunction(torch.autograd.Function):
    """
    Custom torch.autograd.Function that performs error-adaptive forward integration
    and backward adjoint sensitivity propagation with discrete R^T jumps.
    """

    @staticmethod
    def forward(
        ctx,
        f_func: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
        beta: float,
        tol: float,
        K_init: int,
        K_min: int,
        K_max: int,
        use_adjoint_jump: bool,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward integration saving adaptation history for backward adjoint.
        """
        ctx.f_func = f_func
        ctx.beta = beta
        ctx.use_adjoint_jump = use_adjoint_jump
        ctx.t_grid = t_grid
        ctx.num_params = len(params)

        # Solve adaptively
        solver = AdaptiveSOEFDESolver(
            beta=beta,
            tol=tol,
            K_init=K_init,
            K_min=K_min,
            K_max=K_max,
        )

        with torch.no_grad():
            sol: SolverSolution = solver.solve(f_func, z0, t_grid)

        # Save trajectory, events, and parameters in context
        ctx.save_for_backward(z0, sol.z, *params)
        ctx.events = sol.events
        ctx.dt = float((t_grid[1] - t_grid[0]).item())
        ctx.modes_history = sol.modes_history

        return sol.z

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward adjoint sensitivity integration.
        """
        z0 = ctx.saved_tensors[0]
        z_traj = ctx.saved_tensors[1]
        params = ctx.saved_tensors[2:]
        f_func = ctx.f_func
        beta = ctx.beta
        use_adjoint_jump = ctx.use_adjoint_jump
        t_grid = ctx.t_grid
        dt = ctx.dt
        events = ctx.events
        gamma_factor = 1.0 / sp.gamma(beta)

        N = len(t_grid) - 1
        device = z0.device
        dtype = z0.dtype

        # Index events by step_idx for O(1) lookup during backward pass
        events_by_step = {ev.step_idx: ev for ev in events}

        # Parameter gradients accumulator
        grad_params = [torch.zeros_like(p) for p in params]
        grad_z0 = torch.zeros_like(z0)

        # Generator to re-obtain SOE weights for each interval
        from core.soe.dyadic_quadrature import DyadicSOEGenerator
        generator = DyadicSOEGenerator()
        T_horizon = float((t_grid[-1] - t_grid[0]).item())

        # Start adjoint at t_N with terminal condition
        # grad_output has shape (N+1, ..., d)
        current_K = ctx.modes_history[-1]
        current_soe = generator.generate(
            beta=beta, delta_t=dt, T=T_horizon, num_modes=current_K
        )
        lambdas_t, weights_t = current_soe.to_torch(dtype=dtype, device=device)

        # a_m has shape (..., d, current_K)
        # Terminal condition: from z(t_N) = z_0 + gamma_factor * sum(w * m)
        # dL/dm = (dL/dz) * (w / Gamma(beta))
        dL_dz_N = grad_output[N]
        # (..., d, 1) * (current_K,) -> (..., d, current_K)
        w_exp = weights_t.unsqueeze(0).expand(z0.shape + (current_K,))
        a_m = gamma_factor * dL_dz_N.unsqueeze(-1) * w_exp
        grad_z0 = grad_z0 + dL_dz_N

        # Backward integration from N-1 down to 0
        for n in range(N - 1, -1, -1):
            t_curr = float(t_grid[n].item())
            z_n = z_traj[n].detach().requires_grad_(True)

            # 1. Sum of adjoint modes: v = \sum_j a_{m, j}
            # Shape (..., d)
            v = torch.sum(a_m, dim=-1)

            # 2. Compute vector-Jacobian products (VJP) at t_n
            with torch.enable_grad():
                f_n = f_func(z_n, t_curr)
                # Compute VJP for parameters: v * df/dtheta
                if len(params) > 0:
                    vjp_params = torch.autograd.grad(
                        outputs=f_n,
                        inputs=params,
                        grad_outputs=v,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for i, g in enumerate(vjp_params):
                        if g is not None:
                            grad_params[i] += dt * g

                # Compute VJP for state: v * df/dz
                vjp_z = torch.autograd.grad(
                    outputs=f_n,
                    inputs=z_n,
                    grad_outputs=v,
                    retain_graph=False,
                    allow_unused=True,
                )[0]

            # 3. Adjoint ODE step backward using ETD:
            # For d a_m / d\tau = -\lambda_k a_m + coupling:
            # a_m^n = E * a_m^{n+1} + dt * phi_1 * coupling
            lam_dt = lambdas_t * dt
            E = torch.exp(-lam_dt)
            small_mask = lam_dt < 1e-4
            phi_1 = torch.where(
                small_mask,
                1.0 - 0.5 * lam_dt + (1.0 / 6.0) * (lam_dt ** 2),
                (1.0 - E) / lam_dt,
            )

            coupling = gamma_factor * (vjp_z.unsqueeze(-1) if vjp_z is not None else 0.0) * w_exp
            a_m = E * a_m + dt * (phi_1 * coupling)

            # Add observation gradient at t_n
            if torch.any(grad_output[n] != 0.0):
                dL_dz_n = grad_output[n]
                grad_z0 = grad_z0 + dL_dz_n
                a_m = a_m + gamma_factor * dL_dz_n.unsqueeze(-1) * w_exp

            # 4. Check for representation transition at step n
            # If step n was an event, forward transition was: m^+ = R m^-
            # Backward adjoint jump is: a_m^- = R^T a_m^+
            if n in events_by_step:
                ev = events_by_step[n]
                if use_adjoint_jump:
                    # Apply exact adjoint jump
                    a_m = apply_adjoint_transition(a_m, ev.R)
                else:
                    # Ablation: naive resizing without R^T jump
                    if ev.K_old > ev.K_new:
                        pad = torch.zeros(
                            z0.shape + (ev.K_old - ev.K_new,), dtype=dtype, device=device
                        )
                        a_m = torch.cat([a_m, pad], dim=-1)
                    else:
                        a_m = a_m[..., : ev.K_old]

                # Update current active modes to K_old
                current_K = ev.K_old
                current_soe = generator.generate(
                    beta=beta, delta_t=dt, T=T_horizon, num_modes=current_K
                )
                lambdas_t, weights_t = current_soe.to_torch(dtype=dtype, device=device)
                w_exp = weights_t.unsqueeze(0).expand(z0.shape + (current_K,))

        return (None, grad_z0, None, None, None, None, None, None, None, *grad_params)


def adamem_integrate(
    f: Callable[[torch.Tensor, float], torch.Tensor],
    z0: torch.Tensor,
    t_grid: torch.Tensor,
    beta: float,
    tol: float = 1e-4,
    K_init: int = 12,
    K_min: int = 4,
    K_max: int = 48,
    use_adjoint_jump: bool = True,
    parameters: Optional[Tuple[torch.nn.Parameter, ...]] = None,
) -> torch.Tensor:
    """
    High-level integration function for AdaMem-FDE with custom adjoint backprop.

    Args:
        f: Vector field f(z, t).
        z0: Initial state (..., d).
        t_grid: Uniform time grid [t_0, ..., t_N].
        beta: Fractional order 0 < beta <= 1.
        tol: Memory error tolerance \epsilon_{tol}.
        K_init: Initial number of modes.
        K_min: Minimum modes floor.
        K_max: Maximum modes ceiling.
        use_adjoint_jump: True for AdaMem-FDE; False for Baseline 4 ablation.
        parameters: Tuple of trainable model parameters for gradient propagation.

    Returns:
        Trajectory tensor of shape (N+1, ..., d).
    """
    params = tuple(parameters) if parameters is not None else ()
    return AdaMemAdjointFunction.apply(
        f,
        z0,
        t_grid,
        beta,
        tol,
        K_init,
        K_min,
        K_max,
        use_adjoint_jump,
        *params,
    )
