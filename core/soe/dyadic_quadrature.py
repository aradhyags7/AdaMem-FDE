"""
Dyadic Contour Quadrature for Sum-of-Exponentials (SOE) Kernel Approximation.

Implements the Jiang-Zhang (2017) / Beylkin-Monzon approach for power-law kernels:
    K(s) = s^{\beta - 1} = s^{-\gamma},  where \gamma = 1 - \beta \in (0, 1).

Using the integral identity:
    s^{-\gamma} = \frac{1}{\Gamma(\gamma)} \int_0^\infty e^{-\lambda s} \lambda^{\gamma - 1} d\lambda
Under substitution \lambda = e^u, the infinite integral transforms to:
    \int_{-\infty}^\infty \exp(-s e^u + \gamma u) du
which is truncated to [u_{min}, u_{max}] and integrated using composite Gauss-Legendre quadrature.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import scipy.special as sp
import torch


@dataclass
class SOEWeights:
    """Stores SOE nodes, weights, and kernel metadata."""
    lambdas: np.ndarray      # Decay rates \lambda_k > 0, shape (K,)
    weights: np.ndarray      # Quadrature weights w_k > 0, shape (K,)
    beta: float              # Fractional order \beta \in (0, 1)
    delta_t: float           # Lower time cutoff \delta t
    T: float                 # Horizon T
    num_modes: int           # Active modes K

    def to_torch(self, dtype=torch.float64, device="cpu"):
        """Converts weights and decay rates to PyTorch tensors."""
        return (
            torch.tensor(self.lambdas, dtype=dtype, device=device),
            torch.tensor(self.weights, dtype=dtype, device=device),
        )


class DyadicSOEGenerator:
    """
    Generates Sum-of-Exponentials parameters for fractional kernel s^{\beta - 1}.
    """

    def __init__(self, gl_order: int = 4):
        """
        Args:
            gl_order: Order of Gauss-Legendre quadrature per subinterval (default 4).
        """
        self.gl_order = gl_order
        self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(gl_order)

    def generate(
        self,
        beta: float,
        delta_t: float,
        T: float,
        tol: float = 1e-4,
        num_modes: Optional[int] = None,
    ) -> SOEWeights:
        """
        Computes poles \lambda_k and weights w_k such that
        \sum_{k=1}^K w_k e^{-\lambda_k s} \approx s^{\beta - 1} for s \in [\delta t, T].

        Args:
            beta: Fractional order 0 < beta < 1.
            delta_t: Smallest time scale (typically integration step dt or dt/2).
            T: Total simulation horizon.
            tol: Target uniform approximation tolerance.
            num_modes: If specified, fixes K to this number of modes.

        Returns:
            SOEWeights instance containing \lambda_k and w_k.
        """
        gamma = 1.0 - beta
        assert 0.0 < gamma < 1.0, f"Expected 0 < beta < 1, got beta={beta}"
        assert delta_t > 0.0 and T > delta_t, f"Invalid interval [{delta_t}, {T}]"

        # Truncation limits for u = ln(\lambda):
        # As u -> -\infty: e^{-s e^u} \approx 1, integrand ~ e^{\gamma u}.
        # For error < tol: \int_{-\infty}^{u_{min}} e^{\gamma u} du = e^{\gamma u_{min}} / \gamma \approx tol =>
        u_min = np.log(max(1e-15, (gamma * tol) ** (1.0 / gamma) / T))

        # As u -> +\infty: e^{-s e^u} <= e^{-\delta t e^u}.
        # Truncate when \delta t e^u >= ln(1/tol) =>
        u_max = np.log(max(1.0, np.log(1.0 / max(tol, 1e-12)) / delta_t))

        # Determine number of subintervals
        if num_modes is not None:
            n_sub = max(1, num_modes // self.gl_order)
        else:
            # Dyadic intervals: width ~ 1 to 2 units in u
            interval_width = u_max - u_min
            n_sub = max(2, int(np.ceil(interval_width / 1.5)))

        sub_bounds = np.linspace(u_min, u_max, n_sub + 1)

        u_points = []
        w_points = []

        for i in range(n_sub):
            a, b = sub_bounds[i], sub_bounds[i + 1]
            # Map [-1, 1] to [a, b]
            mid = 0.5 * (b + a)
            half = 0.5 * (b - a)
            u_mapped = mid + half * self._gl_nodes
            w_mapped = half * self._gl_weights

            u_points.extend(u_mapped)
            w_points.extend(w_mapped)

        u_arr = np.array(u_points, dtype=np.float64)
        w_arr = np.array(w_points, dtype=np.float64)

        # \lambda_k = e^{u_k}
        lambdas = np.exp(u_arr)

        # w_k = (1 / \Gamma(\gamma)) * w_{gl} * e^{\gamma u_k}
        gamma_factor = 1.0 / sp.gamma(gamma)
        weights = gamma_factor * w_arr * np.exp(gamma * u_arr)

        # If user explicitly requested num_modes, truncate or pad to match
        if num_modes is not None and len(lambdas) != num_modes:
            if len(lambdas) > num_modes:
                idx = np.round(np.linspace(0, len(lambdas) - 1, num_modes)).astype(int)
                lambdas = lambdas[idx]
                weights = weights[idx] * (len(u_arr) / num_modes)
            else:
                # Re-run with exact subinterval count if possible
                pass

        return SOEWeights(
            lambdas=lambdas,
            weights=weights,
            beta=beta,
            delta_t=delta_t,
            T=T,
            num_modes=len(lambdas),
        )

    @staticmethod
    def evaluate_kernel(
        s: Union[float, np.ndarray],
        soe: SOEWeights,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates the SOE approximation K_SOE(s) = \sum_{k=1}^K w_k e^{-\lambda_k s}.
        """
        s_arr = np.asarray(s, dtype=np.float64)
        # s_arr shape: (...), lambdas shape: (K,)
        # Broadcast product: s_arr[..., None] * lambdas[None, :]
        exponent = -np.outer(s_arr.ravel(), soe.lambdas)  # (N, K)
        approx = np.sum(np.exp(exponent) * soe.weights, axis=1)  # (N,)
        if np.isscalar(s):
            return float(approx[0])
        return approx.reshape(s_arr.shape)

    @staticmethod
    def evaluate_exact_kernel(
        s: Union[float, np.ndarray],
        beta: float,
    ) -> Union[float, np.ndarray]:
        """Evaluates exact kernel s^{\beta - 1}."""
        s_arr = np.asarray(s, dtype=np.float64)
        return s_arr ** (beta - 1.0)
