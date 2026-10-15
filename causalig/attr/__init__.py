"""
Attribution methods module.

This module provides attribution methods following Captum's architecture.

The SHAP-based methods (``CausalSHAP``, ``ConditionalSHAP``) are imported lazily
because they require the optional ``shap`` dependency (``pip install causalig[shap]``);
importing this module does not pull in ``shap``.
"""

from causalig.attr._core.causal_expected_gradients import (
    CausalExpectedGradients,
    CausalExpectedGradientsMultiStep,
)
from causalig.attr._core.causal_integrated_gradients import CausalIntegratedGradients

__all__ = [
    "CausalIntegratedGradients",
    "CausalExpectedGradients",
    "CausalExpectedGradientsMultiStep",
    "CausalSHAP",
    "ConditionalSHAP",
]


def __getattr__(name):
    # Lazily import the SHAP-based methods so the optional ``shap`` dependency is
    # only required when they are actually used, not on ``import causalig``.
    if name == "CausalSHAP":
        from causalig.attr._core.causal_shap import CausalSHAP

        return CausalSHAP
    if name == "ConditionalSHAP":
        from causalig.attr._core.conditional_shap import ConditionalSHAP

        return ConditionalSHAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
