r"""
Graded Temporal Mesh Generation for Caputo Fractional Differential Equations.

Quenches the Caputo initial weak singularity at t = 0 (where \dot{z}(t) ~ t^{\beta - 1})
by clustering discretization nodes near the origin according to:
    t_n = T * (n / N)^r, \quad n = 0, 1, \dots, N

where the optimal grading parameter is:
    r = (2 - \beta) / \beta \ge 1

Under this grading, the non-uniform step sizes satisfy:
    \Delta t_n \sim \frac{r T}{N} \left(\frac{n}{N}\right)^{r-1}
which restores optimal second-order global convergence \mathcal{O}(N^{-2})
for ETD-RK2 and fractional linear multistep methods.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


def optimal_grading_exponent(beta: Union[float, torch.Tensor]) -> float:
    """
    Computes optimal grading exponent r = (2 - beta) / beta for Caputo singularity quenching.

    Args:
        beta: Fractional order 0 < beta <= 1.

    Returns:
        r: Grading exponent >= 1.0. For beta=1 (standard ODE), r=1 (uniform mesh).
    """
    beta_val = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    if not (0.0 < beta_val <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta_val}")
    return (2.0 - beta_val) / beta_val


def generate_graded_mesh(
    T: float,
    N: int,
    beta: float,
    r: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Generates a graded time grid t_n = T * (n / N)^r on [0, T].

    Args:
        T: Integration horizon (T > 0).
        N: Number of time intervals (grid has N + 1 points).
        beta: Fractional derivative order 0 < beta <= 1.
        r: Grading exponent. If None, uses optimal r = (2 - beta) / beta.
        device: PyTorch device.
        dtype: PyTorch dtype (defaults to torch.float64 for numerical precision).

    Returns:
        t_grid: 1D torch.Tensor of shape (N + 1,) with t_0 = 0 and t_N = T.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    if r is None:
        r = optimal_grading_exponent(beta)
    else:
        if r < 1.0:
            raise ValueError(f"Grading exponent r must be >= 1.0, got {r}")

    n = torch.arange(N + 1, dtype=dtype, device=device)
    t_grid = T * torch.pow(n / N, r)
    # Ensure exact endpoints
    t_grid[0] = 0.0
    t_grid[-1] = T
    return t_grid


def analyze_mesh_characteristics(
    t_grid: torch.Tensor,
) -> dict:
    """
    Analyzes step size variation, dynamic ratio, and singularity resolution.

    Args:
        t_grid: 1D torch.Tensor of time points.

    Returns:
        Dictionary of mesh metrics:
        - N: Number of intervals
        - dt_min: Smallest step size (at origin)
        - dt_max: Largest step size (at terminal time)
        - ratio: dt_max / dt_min
        - is_uniform: True if max(dt) - min(dt) < 1e-7
    """
    dts = t_grid[1:] - t_grid[:-1]
    dt_min = float(torch.min(dts).item())
    dt_max = float(torch.max(dts).item())
    ratio = dt_max / max(dt_min, 1e-15)
    is_uniform = bool(torch.allclose(dts, dts[0], rtol=1e-4, atol=1e-7))

    return {
        "N": len(t_grid) - 1,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "ratio": ratio,
        "is_uniform": is_uniform,
    }
