"""
Sum-of-Exponentials (SOE) kernel compression and state transition operators.
"""

from .dyadic_quadrature import DyadicSOEGenerator, SOEWeights
from .projection import compute_projection_matrix, apply_state_transition, apply_adjoint_transition

__all__ = [
    "DyadicSOEGenerator",
    "SOEWeights",
    "compute_projection_matrix",
    "apply_state_transition",
    "apply_adjoint_transition",
]
