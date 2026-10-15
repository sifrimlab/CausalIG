"""Tests for the surrogate models in :mod:`causalig.models.surrogate`.

Covers the training entry point and
:func:`create_autoencoder_from_scm`.

``MaskedGlobalLinear`` and its analytic gradients are covered separately in
``tests/test_masked_global_linear.py`` and are not re-tested here.

All tests run on CPU with tiny dimensions so the whole module executes in well
under a second.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from causalig.models.scm import DifferentiableSCM
from causalig.models.surrogate import (
    MaskedGlobalLinear,
    create_autoencoder_from_scm,
    train_surrogate_model,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_dag(n=6):
    """Small chain-ish DAG on ``n`` nodes. Rows are parents, columns are
    children (A[i, j] = 1 means i -> j). Node 0 has no parents; every other node
    has parent ``j - 1`` so the graph is valid for any ``n >= 2``."""
    A = np.zeros((n, n), dtype=np.float32)
    for j in range(1, n):
        A[j - 1, j] = 1.0
    return A


# --------------------------------------------------------------------------- #
# training entry points
# --------------------------------------------------------------------------- #
def test_train_surrogate_model_returns_module():
    """A tiny CPU training run of ``train_surrogate_model`` returns an
    ``nn.Module`` without error.

    ``train_surrogate_model`` reads ``surrogate.predictable_children`` and so is
    only compatible with the masked surrogate family; it is exercised here on a
    small :class:`MaskedGlobalLinear`.
    """
    d, n = 6, 48
    X = torch.randn(n, d)
    A = _make_dag(d)
    surrogate = MaskedGlobalLinear(A, use_basis_expansion=False)

    logger = logging.getLogger("test_train_surrogate_model")
    logger.addHandler(logging.NullHandler())

    trained = train_surrogate_model(
        surrogate,
        X,
        epochs=2,
        batch_size=8,
        device="cpu",
        num_workers=0,
        seed=0,
        logger=logger,
    )

    assert isinstance(trained, nn.Module)
    with torch.no_grad():
        out = trained(X[:4])
    assert out.shape == (4, d)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# create_autoencoder_from_scm
# --------------------------------------------------------------------------- #
def _build_simple_scm():
    # DAG: x0 -> x1, x0 -> x2, x1 -> x2
    adjacency = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    assignments = [
        None,
        lambda Xp: 2.0 * Xp[:, 0],
        lambda Xp: Xp[:, 0] + Xp[:, 1],
    ]
    return DifferentiableSCM(
        adjacency_matrix=adjacency, assignments=assignments, topological_order=[0, 1, 2]
    )


def test_create_autoencoder_from_scm_returns_callable_with_right_shape():
    scm = _build_simple_scm()
    autoencoder = create_autoencoder_from_scm(scm, structural=True)

    assert callable(autoencoder)

    batch, d = 4, 3
    x = torch.randn(batch, d)
    out = autoencoder(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch, d)  # forward gives (batch, n_features)
    assert torch.isfinite(out).all()


def test_create_autoencoder_from_scm_matches_structural_mechanisms():
    """In structural mode the wrapper is exactly ``scm(x, structural=True)``,
    so the deterministic mechanisms are reproduced (root copied, children from
    their direct parents)."""
    scm = _build_simple_scm()
    autoencoder = create_autoencoder_from_scm(scm, structural=True)

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]])
    out = autoencoder(x)

    # Structural mode applies each mechanism to the *input* parent values (not
    # to the recomputed ones): x0 is a root and is copied through; x1 = 2*x0;
    # x2 = x0 + x1 using the given inputs x0, x1.
    expected = torch.stack([x[:, 0], 2.0 * x[:, 0], x[:, 0] + x[:, 1]], dim=-1)
    torch.testing.assert_close(out, expected)
