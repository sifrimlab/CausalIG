"""
Model definitions module.
"""

from causalig.models.scm import DifferentiableSCM
from causalig.models.surrogate import (
    MaskedGlobalLinear,
    create_autoencoder_from_scm,
    train_surrogate_model,
)

__all__ = [
    "DifferentiableSCM",
    "MaskedGlobalLinear",
    "train_surrogate_model",
    "create_autoencoder_from_scm",
]
