"""
causalig: Causal Integrated Gradients (CIG) and Causal Expected Gradients (CEG).
"""

__version__ = "0.1.0"

from causalig.attr import (
    CausalExpectedGradients,
    CausalExpectedGradientsMultiStep,
    CausalIntegratedGradients,
)

__all__ = [
    "__version__",
    "CausalIntegratedGradients",
    "CausalExpectedGradients",
    "CausalExpectedGradientsMultiStep",
]
