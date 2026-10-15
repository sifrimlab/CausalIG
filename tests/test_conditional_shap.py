"""Tests for ConditionalSHAP and its Gaussian conditional maskers/samplers."""

import numpy as np
import pytest
import torch

# ConditionalSHAP depends on the optional `shap` extra; skip the whole module
# cleanly (rather than erroring at collection) when it is not installed.
pytest.importorskip("shap")

from causalig.attr._core.conditional_shap import (  # noqa: E402
    ConditionalMasker,
    ConditionalSHAP,
    GaussianConditionalSampler,
    gaussian_conditional_sampler,
)


def _gaussian_background(n=200, d=3):
    # Correlated Gaussian data so conditioning is non-trivial.
    mean = np.zeros(d)
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
    return np.random.multivariate_normal(mean, cov, size=n).astype(np.float32)


def test_gaussian_sampler_output_shape():
    X = _gaussian_background()
    sampler = GaussianConditionalSampler(X)
    mask = np.array([True, False, False])  # observe feature 0, sample 1 and 2
    out = sampler(np.array([0.5, 0.0, 0.0], dtype=np.float32), mask, n_samples=8)
    assert out.shape == (8, 2)  # two missing features
    assert np.isfinite(out).all()


def test_gaussian_sampler_all_observed_returns_empty():
    X = _gaussian_background()
    sampler = GaussianConditionalSampler(X)
    mask = np.array([True, True, True])
    out = sampler(np.array([1.0, 2.0, 3.0], dtype=np.float32), mask, n_samples=5)
    assert out.shape == (5, 0)


def test_gaussian_sampler_no_observed_uses_marginal():
    X = _gaussian_background()
    sampler = GaussianConditionalSampler(X)
    mask = np.array([False, False, False])
    out = sampler(np.zeros(3, dtype=np.float32), mask, n_samples=6)
    assert out.shape == (6, 3)
    assert np.isfinite(out).all()


def test_gaussian_sampler_rejects_wrong_mask_length():
    sampler = GaussianConditionalSampler(_gaussian_background())
    with pytest.raises(ValueError):
        sampler(np.zeros(3, dtype=np.float32), np.array([True, False]), n_samples=2)


def test_gaussian_sampler_rejects_1d_train():
    with pytest.raises(ValueError):
        GaussianConditionalSampler(np.zeros(10, dtype=np.float32))


def test_convenience_wrapper_requires_xtrain():
    with pytest.raises(ValueError):
        gaussian_conditional_sampler(
            np.zeros(3, dtype=np.float32), np.array([True, False, False]), n_samples=2
        )


def test_conditional_masker_fixes_observed_features():
    # A trivial deterministic sampler that returns zeros for the missing block.
    def zero_sampler(x, mask, n_samples):
        n_missing = int((~np.asarray(mask, dtype=bool)).sum())
        return np.zeros((n_samples, n_missing), dtype=np.float32)

    masker = ConditionalMasker(zero_sampler, n_samples=5)
    mask = np.array([True, False, True])
    x = np.array([1.5, -4.0, 7.25], dtype=np.float32)
    out = masker(mask, x)

    assert out.shape == (5, 3)
    # observed features (mask True) are fixed to x
    assert np.allclose(out[:, 0], x[0])
    assert np.allclose(out[:, 2], x[2])
    # masked feature filled by the sampler (zeros here)
    assert np.allclose(out[:, 1], 0.0)


def test_conditional_masker_all_observed_is_constant():
    def sampler(x, mask, n_samples):
        return np.zeros((n_samples, 0), dtype=np.float32)

    masker = ConditionalMasker(sampler, n_samples=4)
    mask = np.array([True, True, True])
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    out = masker(mask, x)
    assert out.shape == (4, 3)
    assert np.allclose(out, x)


def test_conditional_shap_attribute_smoke():
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor([[1.0, 0.5, -0.25]], dtype=torch.float32)
                )

        def forward(self, x):
            return self.linear(x).squeeze(-1)

    model = LinearModel().to(torch.device("cpu"))
    background = _gaussian_background(n=150, d=3)
    explainer = ConditionalSHAP(model, background, n_samples=10, npermutations=5)

    inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=torch.float32)
    attrs = np.asarray(explainer.attribute(inputs))

    assert attrs.shape[0] == inputs.shape[0]
    assert attrs.shape[1] == inputs.shape[1]
    assert np.isfinite(attrs).all()


def test_conditional_shap_reproducible_with_seed():
    # shap reseeds NumPy's global RNG when the explainer is built (from fresh
    # entropy if no seed is given), so the seed has to reach the explainer.
    model = torch.nn.Linear(3, 1)
    background = _gaussian_background(n=150, d=3)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])

    def run():
        np.random.seed(None)  # scramble global state between the two runs
        explainer = ConditionalSHAP(
            model, background, n_samples=10, npermutations=5, seed=0
        )
        return np.asarray(explainer.attribute(inputs))

    np.testing.assert_array_equal(run(), run())
