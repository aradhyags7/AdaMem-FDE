"""
Fractional Differential Equation Solvers:
- FullHistorySolver: O(N^2) reference baseline (Adams-Bashforth-Moulton / L1)
- FixedSOESolver: O(N K) fixed memory representation baseline
- AdaptiveSOESolver: Error-adaptive dynamic memory solver with event logging
"""

from .full_history import FullHistoryFDESolver
from .fixed_soe import FixedSOEFDESolver
from .adaptive_soe import AdaptiveSOEFDESolver, SolverSolution
from .incommensurate_soe import IncommensurateSOEFDESolver, incommensurate_adamem_integrate

__all__ = [
    "FullHistoryFDESolver",
    "FixedSOEFDESolver",
    "AdaptiveSOEFDESolver",
    "SolverSolution",
    "IncommensurateSOEFDESolver",
    "incommensurate_adamem_integrate",
]

