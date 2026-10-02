r"""
Incommensurate Multi-Order Fractional Differential Equation Solver (AdaMem-FDE Frontier).

Generalizes fractional dynamics to decoupled multi-order systems where each state variable
z_i(t) exhibits a distinct anomalous memory exponent \beta_i \in (0, 1):
    {}^C D_0^{\vec{\beta}} z(t) = f(z(t), t) \iff {}^C D_0^{\beta_i} z_i(t) = f_i(z(t), t), \quad i = 1, \dots, d

Each dimension i is discretized via an independent dyadic Sum-of-Exponentials:
    K_i(s) = \frac{s^{\beta_i - 1}}{\Gamma(\beta_i)} \approx \frac{1}{\Gamma(\beta_i)} \sum_{k=1}^{K_i} w_{i, k} e^{-\lambda_{i, k} s}

yielding auxiliary memory states m_{i, k}(t) with block-diagonal structure:
    \dot{m}_{i, k}(t) = -\lambda_{i, k} m_{i, k}(t) + f_i(z(t), t)
    z_i(t) = z_{i, 0} + \frac{1}{\Gamma(\beta_i)} \sum_{k=1}^{K_i} w_{i, k} m_{i, k}(t)

Full vectorization over (d, K) is achieved via stacked ETD-RK2 tensor operations.
Analytical parameter sensitivities \nabla_{\vec{\beta}} \mathcal{L} \in \mathbb{R}^d are computed
via exact digamma-based kernel differentiation.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import scipy.special as sp
import torch
import torch.nn as nn

from core.soe.dyadic_quadrature import DyadicSOEGenerator, SOEWeights


class IncommensurateSOEFDESolver:
    r"""
    Fixed-Order Multi-Order Incommensurate SOE FDE Solver.
    Integrates d-dimensional systems with state-dependent fractional orders \vec{\beta} = (\beta_1, \dots, \beta_d).
    """

    def __init__(
        self,
        beta: Union[torch.Tensor, List[float], np.ndarray],
        num_modes: int = 16,
        delta_t: Optional[float] = None,
        T: Optional[float] = None,
    ):
        """
        Args:
            beta: Vector of fractional orders (\beta_1, ..., \beta_d), with 0 < \beta_i <= 1.
            num_modes: Modes K per state dimension.
            delta_t: Smallest time step cutoff for SOE kernel.
            T: Horizon for SOE kernel.
        """
        if isinstance(beta, torch.Tensor):
            self.beta_vec = beta.detach().cpu().numpy().astype(np.float64)
        else:
            self.beta_vec = np.asarray(beta, dtype=np.float64)

        assert self.beta_vec.ndim == 1, f"beta must be a 1D vector of length d, got shape {self.beta_vec.shape}"
        assert np.all((self.beta_vec > 0.0) & (self.beta_vec <= 1.0)), f"All beta_i must be in (0, 1], got {self.beta_vec}"

        self.d = len(self.beta_vec)
        self.num_modes = num_modes
        self.delta_t = delta_t
        self.T = T
        self.generator = DyadicSOEGenerator()

        self._weights_initialized = False
        self.lambdas_t: Optional[torch.Tensor] = None   # shape (d, K)
        self.weights_t: Optional[torch.Tensor] = None   # shape (d, K)
        self.gamma_factors: Optional[torch.Tensor] = None # shape (d,)

    def _ensure_weights(self, dt: float, T_horizon: float, dtype: torch.dtype, device: torch.device):
        if not self._weights_initialized:
            dt_ref = self.delta_t or dt
            T_ref = self.T or T_horizon

            lambdas_list = []
            weights_list = []

            for i in range(self.d):
                b_i = float(self.beta_vec[i])
                soe = self.generator.generate(
                    beta=b_i, delta_t=dt_ref, T=T_ref, num_modes=self.num_modes
                )
                lambdas_list.append(soe.lambdas)
                weights_list.append(soe.weights)

            # Stack into (d, K)
            lambdas_arr = np.stack(lambdas_list, axis=0)
            weights_arr = np.stack(weights_list, axis=0)

            self.lambdas_t = torch.tensor(lambdas_arr, dtype=dtype, device=device)
            self.weights_t = torch.tensor(weights_arr, dtype=dtype, device=device)

            gamma_vals = [1.0 / sp.gamma(float(b)) for b in self.beta_vec]
            self.gamma_factors = torch.tensor(gamma_vals, dtype=dtype, device=device)
            self._weights_initialized = True

    def solve(
        self,
        f: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Integrates incommensurate FDE over t_grid (uniform or graded).

        Args:
            f: Right-hand side f(z, t), mapping (..., d) -> (..., d).
            z0: Initial state of shape (..., d).
            t_grid: Time points [t_0, ..., t_N].

        Returns:
            z_trajectory: shape (N+1, ..., d).
        """
        N = len(t_grid) - 1
        dts = t_grid[1:] - t_grid[:-1]  # shape (N,)
        dt_min = float(torch.min(dts).item())
        T_horizon = float((t_grid[-1] - t_grid[0]).item())

        device = z0.device
        dtype = z0.dtype
        assert z0.shape[-1] == self.d, f"Expected state dimension d={self.d}, got z0 shape {z0.shape}"

        self._ensure_weights(dt_min, T_horizon, dtype, device)
        K = self.num_modes
        d = self.d

        # Precompute vectorized ETD operators:
        # dts: (N, 1, 1), lambdas_t: (1, d, K) -> lam_dt: (N, d, K)
        lam_dt = dts.view(N, 1, 1) * self.lambdas_t.unsqueeze(0)  # (N, d, K)
        E = torch.exp(-lam_dt)

        small_mask = lam_dt < 1e-4
        phi_1 = torch.where(
            small_mask,
            1.0 - 0.5 * lam_dt + (1.0 / 6.0) * (lam_dt ** 2),
            (1.0 - E) / lam_dt,
        )  # (N, d, K)
        phi_2 = torch.where(
            small_mask,
            0.5 - (1.0 / 6.0) * lam_dt + (1.0 / 24.0) * (lam_dt ** 2),
            (E - 1.0 + lam_dt) / (lam_dt ** 2),
        )  # (N, d, K)

        # Allocate trajectory
        shape = (N + 1,) + z0.shape
        z = torch.zeros(shape, dtype=dtype, device=device)
        z[0] = z0

        # Auxiliary memory state m_{i, k}: shape (..., d, K)
        m_shape = z0.shape + (K,)
        m = torch.zeros(m_shape, dtype=dtype, device=device)
        m_history = [m.clone()]

        w = self.weights_t            # (d, K)
        gammas = self.gamma_factors    # (d,)

        for n in range(N):
            t_curr = float(t_grid[n].item())
            t_next = float(t_grid[n + 1].item())
            dt_n = dts[n]

            E_n = E[n]       # (d, K)
            p1_n = phi_1[n]  # (d, K)
            p2_n = phi_2[n]  # (d, K)

            # Evaluate vector field f(z_n, t_n)
            F_curr = f(z[n], t_curr)        # (..., d)
            F_curr_exp = F_curr.unsqueeze(-1) # (..., d, 1)

            # Stage 1: ETD Predictor
            m_pred = E_n * m + dt_n * (p1_n * F_curr_exp)

            # Reconstruct predictor state
            weighted_sum_pred = torch.sum(m_pred * w, dim=-1)
            z_pred = z0 + gammas * weighted_sum_pred

            # Evaluate at predictor
            F_pred = f(z_pred, t_next)
            F_pred_exp = F_pred.unsqueeze(-1)

            # Stage 2: ETD-RK2 Corrector
            m = m_pred + dt_n * (p2_n * (F_pred_exp - F_curr_exp))

            # Reconstruct state at t_{n+1}
            weighted_sum = torch.sum(m * w, dim=-1)
            z[n + 1] = z0 + gammas * weighted_sum
            m_history.append(m.clone())

        self.m_history = m_history
        return z


class IncommensurateAdaMemAdjointFunction(torch.autograd.Function):
    r"""
    Custom PyTorch autograd function for multi-order incommensurate systems.
    Computes vector parameter sensitivities \nabla_{\vec{\beta}} \mathcal{L} \in \mathbb{R}^d
    and neural vector field gradients \nabla_\theta \mathcal{L}.
    """

    @staticmethod
    def forward(
        ctx,
        f_func: Callable[[torch.Tensor, float], torch.Tensor],
        z0: torch.Tensor,
        t_grid: torch.Tensor,
        beta_vec: torch.Tensor,  # 1D tensor of shape (d,)
        num_modes: int,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.f_func = f_func
        ctx.t_grid = t_grid
        ctx.num_modes = num_modes
        ctx.d = z0.shape[-1]

        solver = IncommensurateSOEFDESolver(
            beta=beta_vec,
            num_modes=num_modes,
        )

        with torch.no_grad():
            z_traj = solver.solve(f_func, z0, t_grid)

        ctx.save_for_backward(z0, z_traj, beta_vec, *params)
        ctx.lambdas_t = solver.lambdas_t
        ctx.weights_t = solver.weights_t
        ctx.gamma_factors = solver.gamma_factors
        ctx.m_history = solver.m_history
        return z_traj

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z0, z_traj, beta_vec = ctx.saved_tensors[:3]
        params = ctx.saved_tensors[3:]
        f_func = ctx.f_func
        t_grid = ctx.t_grid
        d = ctx.d
        K = ctx.num_modes
        N = len(t_grid) - 1
        device = z0.device
        dtype = z0.dtype

        lambdas = ctx.lambdas_t  # (d, K)
        weights = ctx.weights_t  # (d, K)
        gammas = ctx.gamma_factors  # (d,)

        beta_np = beta_vec.detach().cpu().numpy()
        psi_beta = torch.tensor([sp.digamma(b) for b in beta_np], dtype=dtype, device=device) # (d,)
        psi_1_beta = torch.tensor([sp.digamma(1.0 - b) for b in beta_np], dtype=dtype, device=device) # (d,)

        grad_params = [torch.zeros_like(p) for p in params]
        grad_z0 = torch.zeros_like(z0)
        grad_beta = torch.zeros_like(beta_vec)

        # Terminal condition at t_N
        dL_dz_N = grad_output[N]  # (..., d)
        grad_z0 = grad_z0 + dL_dz_N

        # dw_dbeta: (d, K)
        dw_dbeta = weights * (psi_1_beta.unsqueeze(-1) - torch.log(lambdas))

        # Terminal beta sensitivity at t_N
        m_N = ctx.m_history[N]
        sum_wm_N = torch.sum(m_N * weights, dim=-1)
        sum_dwm_N = torch.sum(m_N * dw_dbeta, dim=-1)
        dz_dbeta_N = (-psi_beta * gammas) * sum_wm_N + gammas * sum_dwm_N
        if dL_dz_N.ndim > 1:
            grad_beta = grad_beta + torch.sum(dL_dz_N * dz_dbeta_N, dim=list(range(dL_dz_N.ndim - 1)))
        else:
            grad_beta = grad_beta + (dL_dz_N * dz_dbeta_N)

        # Adjoint memory state a_m has shape (..., d, K)
        coupling_coeff = (gammas.unsqueeze(-1) * weights).unsqueeze(0)  # (1, d, K)
        a_m = coupling_coeff * dL_dz_N.unsqueeze(-1)

        # Backward integration from N-1 down to 0
        for n in range(N - 1, -1, -1):
            t_curr = float(t_grid[n].item())
            dt_n = float((t_grid[n + 1] - t_grid[n]).item())
            z_n = z_traj[n].detach().requires_grad_(True)

            # Sum of adjoint modes per coordinate: v_i = \sum_{k=1}^K a_{m, i, k}
            v = torch.sum(a_m, dim=-1)  # (..., d)

            # Vector-Jacobian products
            with torch.enable_grad():
                f_n = f_func(z_n, t_curr)
                trainable = [p for p in params if p.requires_grad]
                if len(trainable) > 0:
                    vjp_params = torch.autograd.grad(
                        outputs=f_n,
                        inputs=trainable,
                        grad_outputs=v,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    idx = 0
                    for i, p in enumerate(params):
                        if p.requires_grad:
                            g = vjp_params[idx]
                            idx += 1
                            if g is not None:
                                grad_params[i] += dt_n * g

                vjp_z = torch.autograd.grad(
                    outputs=f_n,
                    inputs=z_n,
                    grad_outputs=v,
                    retain_graph=False,
                    allow_unused=True,
                )[0]

            # ETD operator backward for each dimension i:
            lam_dt = lambdas * dt_n  # (d, K)
            E = torch.exp(-lam_dt)
            small_mask = lam_dt < 1e-4
            phi_1 = torch.where(
                small_mask,
                1.0 - 0.5 * lam_dt + (1.0 / 6.0) * (lam_dt ** 2),
                (1.0 - E) / lam_dt,
            )

            # Coupling: \frac{w_{i, k}}{\Gamma(\beta_i)} (vjp_z)_i
            coupling = coupling_coeff * (vjp_z.unsqueeze(-1) if vjp_z is not None else 0.0)
            a_m = E * a_m + dt_n * (phi_1 * coupling)

            # Add observation gradient at t_n
            if torch.any(grad_output[n] != 0.0):
                dL_dz_n = grad_output[n]
                grad_z0 = grad_z0 + dL_dz_n
                a_m = a_m + coupling_coeff * dL_dz_n.unsqueeze(-1)

            # Beta gradient contribution at step n
            m_n = ctx.m_history[n]
            sum_wm = torch.sum(m_n * weights, dim=-1)
            sum_dwm = torch.sum(m_n * dw_dbeta, dim=-1)
            dz_dbeta_n = (-psi_beta * gammas) * sum_wm + gammas * sum_dwm
            total_sens = grad_output[n] + dt_n * (vjp_z if vjp_z is not None else 0.0)
            if total_sens.ndim > 1:
                grad_beta = grad_beta + torch.sum(total_sens * dz_dbeta_n, dim=list(range(total_sens.ndim - 1)))
            else:
                grad_beta = grad_beta + (total_sens * dz_dbeta_n)

        return (None, grad_z0, None, grad_beta, None, *grad_params)



def incommensurate_adamem_integrate(
    f: Callable[[torch.Tensor, float], torch.Tensor],
    z0: torch.Tensor,
    t_grid: torch.Tensor,
    beta_vec: torch.Tensor,
    num_modes: int = 16,
    parameters: Optional[Tuple[torch.nn.Parameter, ...]] = None,
) -> torch.Tensor:
    r"""
    Integrates an incommensurate multi-order Neural FDE with vector order \vec{\beta} \in (0, 1)^d.
    """
    params = tuple(parameters) if parameters is not None else ()
    return IncommensurateAdaMemAdjointFunction.apply(
        f,
        z0,
        t_grid,
        beta_vec,
        num_modes,
        *params,
    )
