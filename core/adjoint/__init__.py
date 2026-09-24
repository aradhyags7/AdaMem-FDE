"""
Adjoint sensitivity engine with discrete representation transitions for AdaMem-FDE.
"""

from .adamem_adjoint import AdaMemAdjointFunction, adamem_integrate

__all__ = ["AdaMemAdjointFunction", "adamem_integrate"]
