"""
State transition operators and adjoint jump maps for dynamic memory adaptation.

Mathematical Derivation:
Each auxiliary state represents a projection of past history f(t - \tau) against e^{-\lambda_k \tau}:
    m_k(t) = \int_0^t e^{-\lambda_k \tau} f(t - \tau) d\tau

When transitioning from K^- modes with decay rates \lambda^- to K^+ modes with \lambda^+,
we find the linear operator R \in \mathbb{R}^{K^+ \times K^-} such that:
    e^{-\lambda_j^+ \tau} \approx \sum_{k=1}^{K^-} R_{j, k} e^{-\lambda_k^- \tau}

Using the L_2([0, \infty)) inner product:
    \langle e^{-a \tau}, e^{-b \tau} \rangle = \int_0^\infty e^{-(a+b)\tau} d\tau = \frac{1}{a + b}

This yields the Cauchy-Gram system:
    G^-_{k, l} = \frac{1}{\lambda_k^- + \lambda_l^-} \in \mathbb{R}^{K^- \times K^-}
    H_{j, k} = \frac{1}{\lambda_j^+ + \lambda_k^-} \in \mathbb{R}^{K^+ \times K^-}
    R = H (G^- + \sigma I)^{-1}

Consequently:
    Forward Transition:  m^+ = R m^-
    Adjoint Transition:  \lambda^- = R^T \lambda^+
"""

import numpy as np
import torch
from typing import Tuple, Union


def compute_projection_matrix(
    lambdas_old: Union[np.ndarray, torch.Tensor],
    lambdas_new: Union[np.ndarray, torch.Tensor],
    regularization: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the optimal least-squares transition matrix R: R^{K^-} -> R^{K^+}.

    Args:
        lambdas_old: Decay rates of old memory representation \lambda^-, shape (K^-).
        lambdas_new: Decay rates of new memory representation \lambda^+, shape (K^+).
        regularization: Ridge parameter \sigma for numerical stability of Cauchy matrix inversion.

    Returns:
        R: Transition matrix of shape (K^+, K^-) as a PyTorch tensor (float64).
    """
    if isinstance(lambdas_old, np.ndarray):
        l_old = torch.tensor(lambdas_old, dtype=torch.float64)
    else:
        l_old = lambdas_old.to(dtype=torch.float64)

    if isinstance(lambdas_new, np.ndarray):
        l_new = torch.tensor(lambdas_new, dtype=torch.float64)
    else:
        l_new = lambdas_new.to(dtype=torch.float64)

    # Cauchy Gram matrix for old basis: G_{k, l} = 1 / (lambda_k^- + lambda_l^-)
    # Shape: (K^-, K^-)
    G = 1.0 / (l_old.unsqueeze(1) + l_old.unsqueeze(0))

    # Cross Cauchy Gram matrix: H_{j, k} = 1 / (lambda_j^+ + lambda_k^-)
    # Shape: (K^+, K^-)
    H = 1.0 / (l_new.unsqueeze(1) + l_old.unsqueeze(0))

    # Solve R = H * (G + sigma * I)^{-1}
    K_old = l_old.shape[0]
    reg_eye = regularization * torch.eye(K_old, dtype=torch.float64, device=l_old.device)
    G_reg = G + reg_eye

    # Using torch.linalg.solve: (G_reg^T R^T = H^T) => R = (solve(G_reg, H^T))^T
    R = torch.linalg.solve(G_reg, H.T).T

    return R


def apply_state_transition(
    m_old: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Applies forward memory state transition: m^+ = R m^-.

    Args:
        m_old: Old memory state vector, shape (..., K^-).
        R: Transition matrix, shape (K^+, K^-).

    Returns:
        m_new: New memory state vector, shape (..., K^+).
    """
    # Matrix multiplication on the last dimension: (..., K^-) @ (K^+, K^-)^T => (..., K^+)
    return torch.matmul(m_old, R.T.to(m_old.dtype))


def apply_adjoint_transition(
    lambda_new: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Applies backward adjoint transition: \lambda^- = R^T \lambda^+.

    Args:
        lambda_new: Adjoint state vector after transition, shape (..., K^+).
        R: Transition matrix, shape (K^+, K^-).

    Returns:
        lambda_old: Adjoint state vector before transition, shape (..., K^-).
    """
    # lambda^- = lambda^+ @ R => shape (..., K^-)
    return torch.matmul(lambda_new, R.to(lambda_new.dtype))
