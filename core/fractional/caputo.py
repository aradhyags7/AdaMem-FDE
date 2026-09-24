"""
Caputo fractional derivative and Riemann-Liouville fractional integral computation.

Provides:
- L1 discretization of Caputo derivative: O(N^2) full-history reference
- Fractional integral I^beta (Volterra convolution): O(N^2) reference
"""

import numpy as np
import scipy.special as sp
import torch
from typing import Union


def caputo_l1_weights(n_steps: int, beta: float) -> np.ndarray:
    """
    Computes L1 convolution weights:
        b_j = (j+1)^{1-beta} - j^{1-beta},  j = 0, ..., n_steps - 1
    """
    j = np.arange(n_steps, dtype=np.float64)
    b = (j + 1.0) ** (1.0 - beta) - j ** (1.0 - beta)
    return b


def caputo_l1_derivative(
    t: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    beta: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Evaluates Caputo fractional derivative {}^C D_t^beta y(t) using the standard L1 scheme.

    Args:
        t: 1D array of uniform time points [t_0, ..., t_N].
        y: Array of shape (N+1, ...) containing state trajectory.
        beta: Fractional order, 0 < beta < 1.

    Returns:
        Fractional derivative array of same shape as y. At t=t_0, derivative is 0.
    """
    is_torch = isinstance(y, torch.Tensor)
    if is_torch:
        y_np = y.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
    else:
        y_np = np.asarray(y, dtype=np.float64)
        t_np = np.asarray(t, dtype=np.float64)

    N = len(t_np) - 1
    dt = t_np[1] - t_np[0]
    coeff = 1.0 / (sp.gamma(2.0 - beta) * (dt ** beta))

    deriv = np.zeros_like(y_np)
    diff_y = np.diff(y_np, axis=0)  # Shape (N, ...)

    b_all = caputo_l1_weights(N, beta)

    for n in range(1, N + 1):
        # sum_{j=0}^{n-1} b_j * (y_{n-j} - y_{n-j-1})
        # diff_y[n-1 - j] corresponds to y_{n-j} - y_{n-j-1}
        weights = b_all[:n]  # shape (n,)
        slice_diff = diff_y[:n][::-1]  # shape (n, ...)
        
        # Reshape weights for broadcasting
        w_shape = [n] + [1] * (diff_y.ndim - 1)
        w_expanded = weights.reshape(w_shape)
        
        deriv[n] = coeff * np.sum(w_expanded * slice_diff, axis=0)

    if is_torch:
        return torch.tensor(deriv, dtype=y.dtype, device=y.device)
    return deriv


def fractional_integral_l1(
    t: Union[np.ndarray, torch.Tensor],
    f_vals: Union[np.ndarray, torch.Tensor],
    beta: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes Riemann-Liouville fractional integral:
        I^beta f(t_n) = 1/Gamma(beta) * int_0^{t_n} (t_n - s)^{beta-1} f(s) ds
    using piecewise-linear convolution quadrature (O(N^2)).

    Args:
        t: Uniform time grid [t_0, ..., t_N].
        f_vals: Function values at each time point, shape (N+1, ...).
        beta: Fractional integration order (0 < beta <= 1).

    Returns:
        Fractional integral values at each time point, shape (N+1, ...).
    """
    is_torch = isinstance(f_vals, torch.Tensor)
    if is_torch:
        f_np = f_vals.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
    else:
        f_np = np.asarray(f_vals, dtype=np.float64)
        t_np = np.asarray(t, dtype=np.float64)

    N = len(t_np) - 1
    dt = t_np[1] - t_np[0]
    out = np.zeros_like(f_np)

    coeff = (dt ** beta) / sp.gamma(beta + 2.0)

    for n in range(1, N + 1):
        # Quadrature weights for piecewise linear approximation:
        # a_{0, n} = (n-1)^{beta+1} - (n - 1 - beta) * n^beta
        # a_{j, n} = (n-j+1)^{beta+1} - 2*(n-j)^{beta+1} + (n-j-1)^{beta+1}
        # a_{n, n} = 1
        j = np.arange(1, n, dtype=np.float64)
        a_j = (n - j + 1.0) ** (beta + 1.0) - 2.0 * (n - j) ** (beta + 1.0) + (n - j - 1.0) ** (beta + 1.0)
        a_0 = (n - 1.0) ** (beta + 1.0) - (n - 1.0 - beta) * (n ** beta)
        a_n = 1.0

        weights = np.concatenate([[a_0], a_j, [a_n]])  # length n+1
        w_shape = [n + 1] + [1] * (f_np.ndim - 1)
        w_expanded = weights.reshape(w_shape)

        out[n] = coeff * np.sum(w_expanded * f_np[:n + 1], axis=0)

    if is_torch:
        return torch.tensor(out, dtype=f_vals.dtype, device=f_vals.device)
    return out
